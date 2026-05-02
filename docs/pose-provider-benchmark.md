# Pose Provider ベンチマークレポート / 課題

トラッキングパイプライン3種 (`rtmw3d`, `rtmw3d-with-depth`,
`cigpose-metric-depth`) を同条件で計測した結果と、既知の課題を
まとめる。設計思想と各 provider の責務は
[`onnx-tracking-pipeline.md`](onnx-tracking-pipeline.md) を参照。

---

## 1. 計測条件

| 項目 | 値 |
|---|---|
| 計測ツール | `src/bin/bench_provider.rs` |
| ビルド | dev (`vulvatar` opt-level 2 / 3rd-party opt-level 0) |
| ORT | `onnxruntime` 2.0.0-rc.12, DirectML EP |
| OS | Windows 11 Pro 26200 |
| ウォームアップ | 2 フレーム (DirectML カーネルコンパイル消化) |
| 計測フレーム数 | 3–5 |
| 統計 | median (中央値) を採用、min/max/mean も出力 |
| 入力画像 | `validation_images/basic_pose_samples/photorealistic/` の 1600×900 PNG |

実行例:

```bash
cargo run --bin bench_provider --features inference -- \
    validation_images/basic_pose_samples/photorealistic/<image>.png \
    rtmw3d-with-depth 5
```

---

## 2. 結果サマリ

T-pose (1600×900) における中央値:

| Provider | median | mean | max | **mean fps** | 2D pose 部 | Depth 部 | YOLOX (main) | FaceMesh | skel ほか |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rtmw3d` (Step 4 async) | 18.1 ms | 20.7 ms | 28.3 ms | **48.3** | 14.7 ms | — | **~1 ms** (worker async) | 3.4 ms (DirectML) | — |
| **`rtmw3d-with-depth`** (Step 1+3+4 + FaceMesh CPU) | **23.9 ms** | **24.5 ms** | **35.3 ms** | **40.8** | 14.7 ms | 48 ms (worker async) | **~1 ms** (worker async) | 7.5 ms (CPU) | wait+calib+metric+skel ≈ 0.6 ms |
| `cigpose-metric-depth`| 220 ms | 220 ms | 223 ms | **4.5** | 17.7 ms (CIGPose) | 175.0 ms (MoGe-2 ViT-S) | 22.0 ms (毎フレーム sync) | 3.4 ms (CPU) | skel ≈ 0.1 ms |

サイドランジ画像 (full_body_three_quarter_left_side_lunge…) の同設定 `rtmw3d-with-depth`:
median = mean = 23.8 ms / max 25.2 ms / **42.0 fps** — フレーム時間が極めて安定。

### 改善履歴 (`rtmw3d-with-depth` 上)

| ステップ | median | mean | max | mean fps | 累積増分 |
|---|---:|---:|---:|---:|---:|
| 開始 (sync, 518×392) | 106.2 ms | 106.2 | 106 | 9.4 | (基準) |
| Step 1 (DAv2 392×294) | 84.5 ms | 84.5 | 85 | 11.8 | +25% |
| Step 1 + 3 (DAv2 async) | 51.5 ms | 51.5 | 52 | 19.4 | +106% |
| + FaceMesh CPU | 45.3 ms | 45.3 | 47 | 22.1 | +135% |
| + Step 2 (depth half-rate) | 45.1 ms | 45.1 | 47 | 22.2 | +136% |
| Step 4 sync (YOLOX 1/4) | 23.9 ms | 28.9 | **55** ⚠️ | 34.6 | +268% |
| **Step 4 async (YOLOX worker)** | **23.9 ms** | **24.5** | **35** ✅ | **40.8** | **+334%** |

**30 fps 目標達成 + フレーム時間も均質化**。median ≈ mean (フレーム間ばらつき小)、max は 30Hz budget (33 ms) を 2 ms 上回る程度。

横ランジ画像でも `rtmw3d-with-depth` は 108.1 ms / 9.3 fps と
ほぼ同等。DAv2 は dynamic shape のため入力解像度依存性は
パディング後固定 (518×392) であり、画像内容に依存しない。

### Phase 別内訳 (rtmw3d-with-depth)

```
total=106.2ms
 ├─ rtmw3d  : 38.8ms  (YOLOX 21.7 + pre 3.3 + RTMW3D run 9.8 + decode 0.2 + face 3.4)
 ├─ dav2    : 66.5ms  (preprocess + ort run + sample)
 ├─ calib   :  0.01ms (closed-form 1-DoF solve)
 ├─ metric  :  0.6ms  (DAv2 grid 全画素 back-projection ≈ 200K pixels)
 └─ skel    :  0.1ms  (133 keypoints sample + COCO body skeleton 構築)
```

### Phase 別内訳 (cigpose-metric-depth)

```
total=218ms
 ├─ yolox   :  22.0ms (人物 bbox)
 ├─ cigpose :  17.7ms (CIGPose-x 288×384 + SimCC 復号)
 ├─ moge    : 175.0ms (MoGe-2 ViT-S preprocess + ort run + scale*points)
 ├─ face    :   3.4ms (FaceMesh + Blendshape カスケード)
 └─ skel    :   0.1ms (resolve_origin + build_skeleton)
```

注意: CIGPose は事前情報の「150ms+」は誤りで、CIGPose-x DirectML
EP 単体では ~18ms と RTMW3D-x の同 phase (~10ms) とほぼ同等。
重さの主因は CIGPose 単体ではなく後段 MoGe-2 ViT-S の inference。

### Calibration 動作確認

`shoulder-span` (RTMW3D 5↔6) を第1優先、失敗時 `hip-span`
(11↔12) にフォールバック。実測値:

| 画像 | anchor | d_raw (a, b) | scale `c` |
|---|---|---|---|
| T-pose front       | shoulder-span | 2.71, 2.62 | 3.40 |
| Side lunge 3/4     | shoulder-span | 2.89, 2.98 | 4.18 |

被写体距離が遠い側ランジでは `c` が大きく出ており、
`z = c / d_raw` が単調に伸びる方向に校正されていることが確認できる。

---

## 3. 結論

- **動作確認**: 3 provider すべて end-to-end で出力を生成。
- **回帰なし**: `rtmw3d` 単体は 38ms / 26.1 fps を維持。
- **設計通りの依存関係**: アクセント `c` は身体距離に応じて
  単調変化、metric backprojection は ~1ms と無視できるコスト。
- **30 fps 目標未達**: `rtmw3d-with-depth` は逐次 9.4 fps。
  並列化前提でも DAv2 がボトルネック (66ms) のため 15 fps 程度。

---

## 4. 課題

### 4.1 [優先] 30 fps 目標とのギャップ

**現状**: `rtmw3d-with-depth` 逐次 9.4 fps (518×392) →
**Step 1 適用後 11.8 fps (392×294)**。

**根本原因**: Depth 推論が支配的。RTMW3D (38ms) は問題ない。

**段階的対策ロードマップ** (採用順):

#### Step 1 [✅ 適用済み] DAv2 入力解像度を低下

`INPUT_W` / `INPUT_H` を `392×294` (DPT patch size 14 の倍数) に固定。
DAv2 ONNX は完全 dynamic-shape なのでリビルド不要、定数 2 行の変更のみ。

実測 (T-pose 1600×900, dev build, DirectML, median):

| W×H | tokens | DAv2 単体 | total | fps | calibration `c` | z_anchor (m) |
|---|---:|---:|---:|---:|---:|---:|
| 518×392 | 1036 | 66.5 ms | 106.2 ms |  9.4 | 3.40 | 1.255 |
| **392×294** | **588** | **42.0 ms** | **84.5 ms** | **11.8** | **3.18** | **1.260** |
| 308×224 |  352 | 29.2 ms |  70.2 ms | 14.2 | 2.80 | 1.235 |
| 252×196 |  252 | 23.0 ms |  63.2 ms | 15.8 | 2.64 | 1.216 |

`d_raw` は解像度依存で変動するが、`c = D/√geom` が解像度依存性を
吸収するため、最終 metric `z_anchor = c/d_raw` は全解像度で
1.22–1.26 m の範囲に収まり (ばらつき <4%)、shoulder-span 校正は
解像度に対して頑健であることが確認できた。

更に下げる余地はあるが (308×224 で 14fps、252×196 で 16fps)、
低解像度ほど指先・顔の depth detail が荒れて hand pose で破綻する
リスクが上がる。30 fps 目標は次の Step 2/3 と組み合わせて達成し、
DAv2 自体は 392×294 で確定する。

#### Step 2 [次] Depth フレームスキップ (depth refresh 15 Hz)

毎 2 フレームに 1 回だけ DAv2 を走らせ、間のフレームは前回の
metric_frame と calibration `c` を再利用。被写体–カメラ距離は
高々秒オーダーでしか変わらないため、15 Hz refresh で破綻しない。

期待値: pose 30 Hz で更新し続けつつ、平均コスト
`(38 + 42)/2 + 38/2 = 59 ms` → 約 17 fps。並列化と組み合わせる
のが本質的に必要。

#### Step 3 [✅ 適用済み] RTMW3D / DAv2 非同期化

`rtmw3d_with_depth.rs` 内に DAv2 専用ワーカースレッドを spawn。
`mpsc::channel` の inbox + `Mutex<Option<Arc<DepthResult>>>` の
sticky outbox で動かす。設計:

- **drain-old**: ワーカーは `recv()` 後に `try_recv()` ループで
  キューを最新に巻き戻し、古いフレームを破棄。被写体が動いて
  いる時に depth が「過去をなぞる」事態を回避。
- **sticky outbox**: outbox は最新完了結果を `Arc` で保持。
  メインは Arc を clone するだけ (refcount bump、~ns) で 100K
  pixel buffer のディープコピーを避ける。
- **コールドスタート**: 初回フレームのみ `Condvar` で depth
  到着を待機。以降は非ブロックで latest を読む。
- **`Drop`**: Sender を drop → ワーカーの `recv()` が `Err` →
  ループ脱出 → join。

**実測結果** (T-pose 1600×900, 392×294 DAv2):

| 指標 | 同期 (Step 1 後) | 非同期 (Step 1+3) | 改善 |
|---|---:|---:|---:|
| total median | 84.5 ms | **51.5 ms** | -39% |
| fps | 11.8 | **19.4** | +65% |
| `depth_age` | (n/a) | 0 frames | — |

depth_age=0 はワーカーがメインスレッドより先にフレーム N を
完成させているため (DAv2 48ms < total 51ms)。1 フレーム遅延を
当初想定していたが、実際には現フレームの depth が間に合っている。

##### 想定外の発見: GPU 競合が FaceMesh に集中

理論上限は `max(rtmw3d=38ms, dav2=42ms) = 42ms` だったが、
実測 51ms と 9ms の余剰。原因は DirectML EP 上での GPU 競合:

| 計装名 | 同期時 | 非同期時 | 差 |
|---|---:|---:|---:|
| YOLOX | 22 ms | 22 ms | ±0 |
| RTMW3D pre+run+decode | 14 ms | 14 ms | ±0 |
| **FaceMesh** | **3.4 ms** | **13.3 ms** | **+10 ms** |
| DAv2 | 42 ms | 48 ms | +6 ms |

YOLOX や RTMW3D 本体の Conv は影響を受けないのに、相対的に
小さい FaceMesh + Blendshape カスケードが 4 倍に伸びる。
DirectML が GPU command queue を時分割するときに、小モデルの
セットアップ/起動コストが他モデルの大きな batch に押し負けて
いる、という推測。

対策候補:
- FaceMesh を CPU EP に切り替える (4.8 + 1.8 MB なので CPU でも
  数 ms で完走するはず → 競合フリー)
- Step 2 のフレームスキップで DAv2 を 1/2 に減らせば、off-tick
  フレームでは FaceMesh が解放される

#### Step 4 [✅ 適用済み] YOLOX を低頻度化

`Rtmw3dInference` 内に `BboxCache` を追加し、
`YOLOX_REFRESH_PERIOD = 4` フレームに 1 回だけ YOLOX を再検出。
中間フレームは前回 bbox を再利用 (downstream の 25% pad が
被写体の数 px / フレームの動きを吸収する)。

**実測**:

| 指標 | 適用前 | 適用後 | 改善 |
|---|---:|---:|---:|
| `rtmw3d-with-depth` median | 45.1 ms | 23.9 ms | -47% |
| `rtmw3d-with-depth` mean | 45.1 ms | 28.9 ms | -36% |
| `rtmw3d-with-depth` mean fps | 22.2 | **34.6** | **+56%** |
| `rtmw3d` standalone median | 39.1 ms | 18.4 ms | -53% |
| `rtmw3d` standalone mean | 39.1 ms | 22.6 ms | -42% |
| `rtmw3d` standalone mean fps | 25.5 | **44.2** | **+73%** |

##### Phase 別内訳

```
Skip frame (3 / 4 frames):
  yolox=  0.9 ms  ← cache check のみ
  pre  =  3.3 ms
  run  = 10.0 ms
  decode= 0.2 ms
  face =  7.5 ms (CPU) / 3.4 ms (DirectML)
  ──────────
  RTMW3D ≈ 22 ms

YOLOX-refresh frame (1 / 4 frames):
  yolox = 33 ms   ← 通常通り検出
  ...
  RTMW3D ≈ 54 ms
```

##### YOLOX 非同期化 [✅ 適用済み]

同期版のヒッチ (4 フレームに 1 回 54 ms) を解消するため、
`yolox_worker.rs` に DAv2 と同設計のワーカースレッドを実装:

- mpsc inbox + drain-old + sticky `Arc<DetectResult>` outbox
- メインスレッドは `submit()` (cheap clone + send) と
  `wait_latest()` (cold start のみブロック、以降は Arc clone) の
  2 操作のみ。critical path 上の YOLOX コストは ~1 ms。
- `YoloxWorker::Drop` で sender を閉じてワーカーを join。

実測効果 (`rtmw3d-with-depth`, T-pose):

| 指標 | sync 版 | async 版 | 改善 |
|---|---:|---:|---:|
| median | 23.9 ms | 23.9 ms | ±0 |
| mean | 28.9 ms | 24.5 ms | -15% |
| max | 54.9 ms | 35.3 ms | **-36%** |
| mean fps | 34.6 | **40.8** | +18% |

副次効果として bbox 取得自体が 1–2 フレーム分過去になるが、
被写体の数 px / フレームの動きは downstream の 25% pad で吸収
されるため tracking 精度には体感影響なし。サイドランジ画像
では median = mean = 23.8 ms と完全に均質化。

#### Step 5 [最後] EP 差替 / 別 depth モデル

CUDA / TensorRT EP に切り出して計測、または DAv2 Tiny / Mobile 系
への置き換え。Step 1–4 の設計改善で 30 fps が達成できるなら
モデル変更まで踏み込む必要はない。

#### 組み合わせの効果まとめ

| 構成 | 想定 fps | 実測 mean fps |
|---|---:|---:|
| 現状 (518×392, 逐次) |  9.4 |  **9.4** ✅ |
| Step 1 単体 (392×294, 逐次) | 11.8 | **11.8** ✅ |
| Step 1 + 3 (392×294, 並列) | ~24 | **19.4** ⚠️ (GPU 競合で下振れ) |
| Step 1 + 3 + FaceMesh CPU | ~25 | **22.1** ✅ (競合解消) |
| Step 1 + 2 + 3 + FaceMesh CPU | ~25 | **22.2** ⚠️ (Step 2 は no-op) |
| Step 1 + 3 + 4 sync + FaceMesh CPU | 30+α | 34.6 ✅ (但し 1/4 ヒッチ) |
| **Step 1 + 3 + 4 async + FaceMesh CPU** | **45** | **40.8** ✅ **均質、ヒッチなし** |

### 4.2 [中] DAv2 校正の前提仮定が hard-coded

**現状**:

```rust
const ASSUMED_SHOULDER_SPAN_M: f32 = 0.40;
const ASSUMED_HIP_SPAN_M: f32 = 0.32;
const ASSUMED_VERTICAL_FOV_DEG: f32 = 60.0;
```

成人男性平均値。子供・小柄被写体や広角/望遠カメラで絶対距離に
誤差が乗る。アバターソルバは相対骨方向しか使わないため
最終出力への影響は小さいが、Z の絶対値で何かする
(例: VRM のスケール推定) なら問題化する。

**対策候補**:

- VRM 側の `Hips ↔ LeftShoulder/RightShoulder` 距離を測り、
  `ASSUMED_SHOULDER_SPAN_M` を被写体ごとに置き換える。
- カメラ FOV を `MfFrameSource` のメタから取れるなら使う。
  webcam は EXIF 相当を出さないことが多いので保留。

### 4.3 [低] DetectionAnnotation の keypoints が 17 → 133 に拡張済み

**経緯**: `rtmw3d-with-depth` で DAv2 path が走らない原因が
`build_annotation` の `take(17)` だった。修正で 133 全部を
emit するように変更。

**影響範囲**:

- `skeleton` エッジリストは body-17 のままなので
  `diagnose_pose` のエッジ描画は不変。
- ただし keypoint dot 描画はループが全 133 を回るため、
  顔・手の dot が overlay 画像に追加で描かれる。
  視認性に問題があれば `i < 17` などの描画フィルタを
  GUI / diagnostic 側で追加する。
- GUI/renderer 本体は `DetectionAnnotation.keypoints` を
  消費していないので回帰なし (確認済み)。

### 4.4 [低] head pose の横顔フォールバック

ear score が両側とも `KEYPOINT_VISIBILITY_FLOOR (=0.05)` 未満の
場合 `derive_face_pose_metric` が `None` を返し、後段で
FaceMesh confidence のみで補正される。意図通りの安全動作で
バグではないが、横顔のヨー精度が落ちるという制約は記録しておく。

---

## 5. 関連ファイル

- `src/bin/bench_provider.rs` — 計測ツール
- `src/tracking/provider.rs` — provider 切替
- `src/tracking/rtmw3d_with_depth.rs` — RTMW3D + DAv2 実装
- `src/tracking/cigpose_metric_depth.rs` — CIGPose + MoGe-2 実装
- `src/tracking/depth_anything/` — DAv2-Small 推論
- `src/tracking/metric_depth/` — MoGe-2 推論
- `src/tracking/skeleton_from_depth.rs` — depth-aware provider 共通ヘルパ
- `src/tracking/rtmw3d/annotation.rs` — DetectionAnnotation 構築 (17 → 133 修正済み)

---

## 6. 再計測手順

```bash
# RTMW3D 単体
cargo run --bin bench_provider --features inference -- \
    validation_images/basic_pose_samples/photorealistic/<img>.png rtmw3d 10

# RTMW3D + DAv2
cargo run --bin bench_provider --features inference -- \
    validation_images/basic_pose_samples/photorealistic/<img>.png rtmw3d-with-depth 10

# CIGPose + MoGe-2 (重いので少なめ)
cargo run --bin bench_provider --features inference -- \
    validation_images/basic_pose_samples/photorealistic/<img>.png cigpose-metric-depth 5
```

`RUST_LOG=vulvatar=debug` を付けると `RTMW3D+DAv2 timing:` 行で
phase 別内訳が見える。
