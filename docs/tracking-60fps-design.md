# 60 fps 推論パイプライン設計

現行の本番経路 (`FusionProvider`) を 60 fps 推論に載せるための設計。
**As-Is の仕様は [tracking-v2-design.md](tracking-v2-design.md)**、本書は「16.67 ms
予算に対して各段をどう配置するか」だけを書く。

数字は全て `diagnose_fusion_replay` の実測 (2026-09-28)。計測手順は AGENTS.md
「Dependency provisioning」節の下、hand 系の項を参照。

---

## 1. 予算と現状

60 fps = **16.67 ms/frame**。キャプチャ側は既に 60 fps を選択可能
(Tracking インスペクタ `camera_framerate_index` 1 = 60 fps、D435 は 640×480 で
color/depth とも 60 fps を出せる)。

現行は **2 段パイプライン** (`fusion/detector_thread.rs`)。throughput は和ではなく
`max(detector, solver)`:

```
capture ─> LatestCell ─> solver thread ──submit──> detector thread
                          │                        (YOLO26 + face)
                          │<──result(M ≤ N)────────
                          ▼
                    solve(M) → publish
```

実測 (中央値、30 Hz プロファイル、dense 有効):

| 段 | s1789246274 (chin) | s1789246660 (最悪) | スレッド |
|---|---|---|---|
| detector (`rtmw`) | 8.2 ms | 8.1 ms | detector ✓ |
| `vis` | 1.5 | 1.4 | solver |
| **`hands`** | **14.2** | **31.3** | solver |
| **estimator** (`main`+`seeds`+`reacc`+`finish`) | **28.9** | **41.4** | solver |
| `head`/`dense`/`facefit`/`output` | 0.4 | 0.2 | solver |
| **solver 段 合計** | **45.0** | **74.3** | |
| **throughput** | **22 fps** | **13 fps** | |

detector 段は 8 ms で 90% 遊んでいる。**ボトルネックは solver 段の一本槍**で、
その中身は hands と estimator の 2 つだけ (他は合計 2 ms 未満)。

> 注意: `acc` / `eval` / `lin` は `main`+`seeds`+`reacc` を**横断して積算**される
> 内訳なので、段の合計に足してはいけない (`estimator/mod.rs` の `SolveTimings`)。

---

## 2. estimator の 95% は dense 表面点

`VULVATAR_FUSION_NO_DENSE` ablation (s1789246274、中央値):

| | `acc` | `main` | estimator 計 |
|---|---|---|---|
| dense 有効 | 19.3 ms | 27.7 | **28.9** |
| dense 無効 | 0.95 | 1.7 | **~3.1** |

`dense` phase (サンプリング自体) は 0.09 ms。**点を作るのは無料に近く、
残差に積むのが高い** — `acc` は「dense 点数 × LM 反復回数」に比例する。

したがって estimator の予算調整は `VULVATAR_DENSE_STRIDE_MUL` が主ノブになる。
既存の 60 Hz プロファイル (`4680fc9`、`capture_fps >= 50` で自動) の掃引:

estimator 計 = `main` + `seeds`(中央値) + `finish`:

| config | s1789246274 | s1789246660 | dense 点数 (med / min) 274 | 同 660 | 200 点未満 |
|---|---|---|---|---|---|
| 30Hz 既定 (stride 8) | **29.1** | **35.7** | 2433 / 1832 | 2355 / 173 | 0% / 0% |
| 60Hz 自動 (×1.5 → 12) | **13.4** | **19.5** | 1083 / 803 | 1047 / 362 | 0% / 0% |
| 60Hz ×2 (→ 16) | **9.4** | **13.8** | 608 / 447 | 588 / 165 | 0% / 0% |
| 60Hz ×3 (→ 24) | 6.2 | 9.7 | 270 / 201 | 260 / **39** | 0% / **12%** |

**×3 は床を割るので却下**。最悪録画で 94/782 フレーム (12%) が 200 点未満になり、
そこでは dense 項が丸ごと落ちる。**×2 が安全な上限**。

`MIN_DENSE_SURFACE_POINTS = 200` を割ると段階的劣化ではなく崖になるので、stride を
触るときは `diagnose_fusion_replay` の `dense surface:` 行 (med / min / 床未満の割合)
を必ず見ること。`frames.csv` の `n3d` は**疎**観測数 (≈40) でこれは見えない。

### 品質: chin 系は横ばい、最悪系は代償がある

| | s1789246274 (chin) | s1789246660 (最悪) |
|---|---|---|
| torso yaw err (30Hz → 60Hz×2) | +2.4 → +2.2 std 1.3→1.4 | +2.1 → +1.8 |
| **同 err の最大** | +5.5 → +4.8 | **+43.6 → +71.1** |
| R wrist max jump | 0.021 → 0.031 m | **0.464 → 0.765 m** |
| R snaps | 0 → 0 | 17 → 11 |
| L duty | 0.72 → 0.70 | **0.40 → 0.34** |
| hand crops | 217 → 217 | 19/119 → 21/114 |

chin 系は実質無害。**最悪系は右手首の最大ジャンプが 0.46 → 0.77 m、torso yaw の
誤差裾が 43.6° → 71.1° に悪化する。** しかもこれは ×1.5 の時点で既に出ており
(0.766 m / 68.3°)、×2・×3 で増えない — つまり**劣化の出所は dense stride ではなく
60 Hz プロファイルの他の 2 機構 (seed cadence 3 / LM 反復上限)**。
`VULVATAR_FUSION_SEED_EVERY_N` と `VULVATAR_LM_MAX_ITERS` を個別に振って
切り分けること (未実施)。

---

## 3. 提案: 3 段パイプライン

solver 段を hands と solver に割る。detector 段が遊んでいるのと同じ理由で、
hands は独立段にできる (入力は color frame + 前フレームのロック、出力は
`HandResult`)。

```
capture ─> LatestCell ─┬─> detector thread   (YOLO26 + face sidecar)
                       ├─> hand thread       (crop ladder → HandResult)
                       └─> solver thread     (vis → estimator → output)
                                │
                                ▼  60 Hz publish
```

段別予算 (60 Hz プロファイル + dense ×2、中央値):

| 段 | chin | 最悪 | 予算 | 判定 |
|---|---|---|---|---|
| detector (`rtmw`) | 8.95 ms | 8.71 ms | 16.67 | ✓ |
| solver (`vis` + estimator + `head`/`facefit`/`output`) | 11.3 | 15.4 | 16.67 | ✓ |
| hands | 13.8 | 28.2 | **66** (自分の cadence) | ✓ |

**要点: hands を独立段にすると、その予算はフレーム周期 (16.67 ms) ではなく
自分の cadence (60 ms floor ≈ 15 Hz) になる。** 28 ms は 66 ms の予算に対して
余裕がある。段分離しない限り hands は 16.67 ms を割らねばならず、それは
(§2 の計測が示す通り) 推論回数を減らす以外に方法が無く、そのすべてが
ロックを失う — AGENTS.md の該当節に 4 つの却下記録がある。

したがって 60 fps の binding constraint は solver 段の 15.4 ms になり、予算内。

### 各段の契約

- **detector 段**: 現行のまま。latest-wins request / freshest-only result
  (`detector_thread.rs` の invariant)。
- **hand 段**: 同じ latest-only セマンティクス。solver は「直近に完成した
  `HandResult`」を読むだけで、無ければ前回値を据え置く (現行の `prev_hands` と
  同じ扱い)。**hand 段が遅れても solver 段は止まらない** — これが段分離の本質で、
  最悪録画の 34 ms スパイクが publish を止めなくなる。
- **solver 段**: 60 Hz で回す。dense stride ×2、seed cadence 3、LM 反復上限は
  既存 60 Hz プロファイルのまま。
- **publish**: solver 完了ごと。GUI 側は既に `pose_interp` で描画レートへ補間する
  ので、solver が一時的に落ちても描画は 60 Hz を保つ。

---

## 4. 未実装: cadence の時間ベース化

**フレーム数ベースの間引きは capture rate で意味が変わる。** 顔チェーンは既に
これを学んで壁時計に移してある (`yolo26.rs`、`face_interval_ms` 既定 66 ms ≈
15 Hz、フレーム数は timestamp 欠損時の fallback のみ。コードのコメントに
"frame-count modulo would double it at 60 fps" と明記)。

**hand チェーンは未移行**: `VULVATAR_HAND_EVERY_N`(既定 2) はフレーム数ベースなので

| capture | hand 更新 | hand の秒あたりコスト |
|---|---|---|
| 30 fps | 15 Hz | 1.0× |
| 60 fps | 30 Hz | **2.0×** |

60 fps にすると hand が勝手に 30 Hz へ上がり、60 Hz プロファイルが estimator で
稼いだ分を食い潰す。**顔と同じ `VULVATAR_HAND_MIN_INTERVAL_MS` (既定 66) に移すこと。**
フレーム数ルールは timestamp が無い最初の数フレームの fallback として残す。

この移行で hand の秒あたりコストは capture rate に対して不変になり、
段別予算の表がそのまま成立する。

---

## 5. 実装済み / 未実装

| 項目 | 状態 |
|---|---|
| 60 fps キャプチャ選択 | ✓ GUI (`camera_framerate_index`) |
| detector 段の分離 | ✓ `detector_thread.rs` |
| 60 Hz solver プロファイル (seed/3, stride×1.5, LM 上限) | ✓ `4680fc9`、`capture_fps >= 50` で自動 |
| 60 Hz プロファイルの offline 計測 | ✓ `VULVATAR_REPLAY_CAPTURE_FPS` (本作業で追加。これが無く**誰も測れなかった**) |
| dense 点数の可観測化 | ✓ replay の `dense surface:` 行 (本作業で追加) |
| 顔 cadence の時間ベース化 | ✓ `face_interval_ms` |
| hand cadence の時間ベース化 | ✓ `VULVATAR_HAND_MIN_INTERVAL_MS` (既定 60 ms、本作業で追加) |
| **hand 段の分離** | **未** (§3) |
| publish 60 Hz + 補間 | ✓ GUI `pose_interp` |

---

## 6. 未検証のリスク

測っていないことを測ったように書かないための一覧。

1. **replay のクロックは名目 30 fps**。`VULVATAR_REPLAY_CAPTURE_FPS` が選ぶのは
   **プロファイルだけ**で、フレーム間隔は 30 fps のまま。実際の 60 fps キャプチャは
   フレーム間の動きが半分になるので **LM の warm start はもっと効く** (反復が減る)
   はず — つまり本書の estimator 値は**悲観側**の見積り。ただし未実測。
2. **GPU 競合**。hand landmark は単体ベンチ 6.68 ms に対し実パイプラインで
   11-12 ms (`hands::batch_bench`)。差は detector セッションとの DirectML 競合。
   60 fps では detector の秒あたり負荷が倍になり、hand 段を別スレッドにすると
   競合はさらに増える。`GpuExclusiveGuard` があるので、段分離と排他の相互作用は
   実測が必要。**バッチ化は既に却下済み** (DirectML で batch3 が batch1 の 7.4 倍)。
3. **dense stride は 2 録画でしか検証していない**。AGENTS.md の要求通り
   デスク系 + 正面系 (wave/palms/namaste) でも回すこと。特に正面で腕が伸びる
   ポーズは表面点が主観測なので、stride の影響が chin 系より大きいはず。
   ×3 が最悪録画で床を割った (12% のフレーム) ことからも、点数の下限は
   録画内容に強く依存する。
4. **hand refiner 3 本が欠けた状態での数字**。presence/palm net が戻れば
   ladder の候補数が変わり、hand 段のコストも変わる (AGENTS.md の該当節)。
5. **段を 3 つにするとレイテンシが 1 フレーム増える**。現行は capture → publish が
   約 1 フレーム遅れ (`detector_thread.rs` の説明)。hand 段を挟むと最悪 2 フレーム。
   60 fps なら 33 ms で、30 fps 時代の 1 フレーム (33 ms) と同じなので実用上は
   等価だが、配信の口パク同期には効くので確認が要る。

---

## 7. 作業順序

1. ~~hand cadence を時間ベースへ~~ — 済 (§4)。
2. **60 Hz プロファイルの品質劣化を切り分ける** (§2 後半)。最悪録画で
   R wrist max jump 0.46 → 0.77 m、torso yaw 誤差裾 43.6° → 71.1°。dense stride は
   無罪 (×1.5 で既に出て ×3 で増えない) なので、`VULVATAR_FUSION_SEED_EVERY_N` と
   `VULVATAR_LM_MAX_ITERS` を個別に振る。**これが 60 fps 化の最大の未解決事項** —
   速度は届いているが、この劣化を抱えたままでは出せない。
3. **デスク系 + 正面系で dense stride ×2 を検証** (§6-3)。本書の数字は 2 録画のみ。
4. **hand 段の分離** (§3)。2 と 3 が済んでいれば、残る作業はこれだけ。
5. **実機 60 fps で計測** (§6-1, §6-2)。`debug_gui.json` の `render_cpu_ms` と
   `debug_state.json` の `rig.diag` を見て、GPU 競合が予測通りか確認。
