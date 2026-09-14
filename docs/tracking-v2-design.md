# Tracking v2 — 観測融合型ボディフィッタ (現行仕様)

本番経路は `FusionProvider` (`src/tracking/fusion/provider.rs`) の一本のみ。
本書は **As-Is の仕様** を書く。変更の経緯・計測記録は git log と `diagnostics/` に置き、ここには残さない。

関連: [architecture.md](architecture.md), [threading-model.md](threading-model.md),
運用手順 (ライブ計測・録画・リプレイ) は `CLAUDE.md`。

---

## 1. 前提と目標

- 入力は RealSense D435 のみ (単眼フォールバック無し、多人数非対応)。現行運用はカラー 640×480、深度は color に align 済み (`MetricDepthFrame::points_m`、穴は NaN)。
- 主戦場は **デスク配信エンベロープ**: 頭 + 肩のみ、カメラ斜め、手は常時デスク下。この状態で腕・脚・root が暴れないことが第一条件。フレーミングの寄り/引きはモード切替ではなく「観測可能性の変化」として同じ推定器が扱う。
- 出力は関節位置ではなく **関節体モデルのパラメータ + 共分散** (`TrackingRigPose`)。アバター側は不確かさで駆動/休止をブレンドする。
- 設計原則: 観測は不確かさ付きで一つの推定器に入れ、外れ値の扱いはロバスト核・事前・σ に吸収させる。**症状ごとの判定ゲートを足さない**。新しい症状は「どの観測の p_vis / σ / 事前が誤っているか」をリプレイで測ってから直す。

---

## 2. パイプライン

```
D435 color 640×480 + aligned depth
  │
  ├─ crop 決定 (§3.1) ── RTMW3D wholebody 133 (SimCC) ── decode (§3.2)
  │                                                      │
  │      ┌───────────────────────────────────────────────┘
  │      ▼
  ├─ 可視性層 (§3.3): p_vis × 深度シルエット → score を較正済み可視確率に置換
  │
  ├─ 顔: FaceMesh 478 (state-driven crop) → 重心アンカー + 方位観測 OriObs (§3.4)
  ├─ 手: hand landmarker ×2 (state-driven crop、検出器/予測駆動) (§3.5)
  ├─ 深度リフト / 肩 yaw 観測 (§3.6)
  ├─ 密表面点 = 主観測 (§3.7): シルエット画素 → 胴・頭カプセル表面へ
  ▼
推定器 (§5): sparse LM、ロバスト核、共分散、predict(t)
  ▼
リターゲット (§6): 関節角 → VRM ボーン、σ でフェード
```

GUI/レンダースレッドとの接続は `app/render.rs` (`rig` 経路)。ONNX セッションは GPU 逐次実行 (Arc ドライバ TDR 対策)。

---

## 3. 観測生成

### 3.1 人物 crop (`src/tracking/rtmw3d/mod.rs`)

crop は **状態の関数** であり、フレーム毎の検出結果がフィードバックしない構造にする (手クロップと同じ原則)。

| 優先 | ソース | 内容 |
|---|---|---|
| 1 | **crop ヒント** (`set_crop_hint`) | 推定器が頭を追跡中 (`joint_data_sigma(head) < 0.5`) は FK 予測の頭頂・頭中心・両肩 ±0.12 m・肩中点 +0.30 m (胸) を投影した bbox。予測腕は含めない (含めると胴 yaw 合計が悪化する実測あり) |
| 2 | 自己追跡 bbox | 可視 (p_vis ≥ 0.5) キーポイントの min/max。body 17 点中 8 点以上、または「顔 3 点以上 + 片肩以上」で成立。フレームにクランプ |
| 3 | YOLOX sticky | 1/2 が無い時のみ。`YOLOX_REFRESH_PERIOD` フレームおきに非同期検出、stale (2 秒) は破棄 |
| 4 | 全フレーム letterbox | 上記すべて無し |

- 1 と 2 は **union** (ヒステリシス無し。自己追跡 bbox 単体の更新には 10% マージン / 60% 縮小のヒステリシスがある)。
- crop は 25% パディング後に 288:384 へアスペクト補正し、フレーム外はゼロ埋め (mmpose `TopdownAffine` 互換)。

### 3.2 SimCC decode (`src/tracking/rtmw3d/decode.rs`)

`DecodedJoint` に以下を持つ。

| フィールド | 定義 |
|---|---|
| `nx, ny, nz` | サブビン精度 argmax (放物線補間) をフレーム正規化 |
| `score` | `sigmoid(min(peak_x, peak_y))` — **可視性判定には使わない** (幻覚 0.52〜0.60 / 実在 0.71 に圧縮される) |
| `sx, sy` | ピーク FWHM から求めた位置 σ (公称 44 bin で 2.5 bin、幅比の二乗でスケール) |
| `second_x, second_y` | 主ピーク ±44 bin 外の最大応答 / 主ピーク。幻覚の最強指標 |
| `half_x, half_y` | 半値以上のビン割合 |
| `zscore` | z 軸ピークの sigmoid |

### 3.3 可視性層 (`src/tracking/fusion/visibility.rs`, `provider.rs` の `keypoint visibility` 節)

「この関節は追跡中の人物の上に本当にあるか」を一箇所で決め、結果を `score` に書き戻す。下流 (σ 膨張・手クロップ種・自己追跡・GUI 表示) はすべてこの値を読む。

1. **p_vis** = ロジスティック較正 `σ(w·[score, σxy, second, half, zscore, ln σxy] + b)` (`VisPolicy::default`、23 録画 12.6 万点で較正、セッション分割 AUC 0.985)。`min_p = min_p_obs = 0.5`。
2. **画面外**: nx/ny が [−0.05, 1.05] 外、または crop 端 2% 以内 → 不可視。1〜2% 外の鼻先は保持する (FaceMesh の顔枠種になる)。
3. **深度シルエット** (`Silhouette`): 可視な顔キーポイント 2 点以上をシード (無ければ、直近 30 フレーム以内に顔が見えていて頭が追跡中なら予測頭中心)。シード深度の中央値 z_ref を基準に深度帯 [z_ref − 0.60, z_ref + 0.45] m かつ 4 近傍深度差 < 0.03 m/m の画素をフラッドフィルし、chamfer 3-4 距離場を持つ。
   - 輪郭から 0.06 m 超 → 不可視。画素が深度ホールなら 0.20 m まで許容 (ホールは情報欠落であって不在の証拠ではない)。
   - **フレーム端 2% 以内 × ホール** → 検証不能として不可視。
   - 輪郭外でも、人物深度帯の独立ブロブで面積 0.003〜0.08 m² (手・前腕) の上なら可視 (`blob_area_m2`)。
   - シードが無い (顔が見えない) フレームは **全関節不可視** (無人フレームはここで観測ゼロになる)。
4. **脚**: シルエットの可視高さ (頭頂〜最下行) < 0.65 m なら関節 11..22 は構成上画面外 → 不可視。
5. 旧ゲート (crop 境界カリング、reach フィルタ、脚ゲート、腕/脚コヒーレンス) は既定 OFF。`filter_duplicate_wrists` (両手首が幅 6% 以内なら低スコア側を落とす) と手クロップによる手ブロック置換は残る。

### 3.4 顔 (`fusion/head_ori.rs`, `fusion/canonical_face.rs`, `rtmw3d/face.rs`)

- FaceMesh は state-driven crop。478 点は **重心 1 点** の位置アンカーとして 2D 項に入れる (ランドマーク配列は 30° 超で前額化し方位ソースを圧殺するため)。深度リフト点は lateral σ ×6。
- 方位は `OriObs` (FaceMesh transformation-matrix の SO(3) 直接観測、全チェーンヤコビアン)。|yaw| > 0.35 rad の主張は深度頬プロファイル (顔ボックス列中央値の勾配、|Δz| > 0.025 m、手と重なる間は 0.040 m、符号一致) の裏付け必須。裏付けは連続 target 限定で 20 フレームの grace。
- 顔オクルージョン: 鼻が手矩形内なら 5 フレームのクールダウンで方位を止める。顎に手はブロックしない。
- 個人差は canonical テンプレートのスケール + オフセットのみ EMA 学習 (`FaceFit`)。
- SimCC 顔 kp (0..4) は mesh 重心がある間 σ ×(1 + 2·conf 連続補間)、さらに ×0.5 (2026-09-14、29 録画ベンチ: 胴 yaw |err| 合計 156°→74°、頭 roll の GT 利得改善。`VULVATAR_HEAD_KP_SCALE` で上書き)。頭の depth_at 窓は予測頭 ±0.20 m。

### 3.5 手 (`fusion/hands.rs`, `provider.rs` の hand crops 節)

- crop 候補は順に: 前フレームのロック手を自身のランドマークで再クロップ → 検出器の手ブロック (p_vis ≥ 0.35 が 6 点以上) → 予測手首。presence ≥ 0.6 でロック、≥ 0.5 で採用。
- handedness 拒否帯: 左スロットは > 0.70、右スロットは < 0.30 を拒否 ([0.30, 0.70] は通す)。
- body 検出器の裏付け (手首 p_vis ≥ 0.3 が crop 近傍、または手ブロック 6 点) を 3 フレーム欠くロックは破棄 (`hand_unsupported`)。
- 両 crop が同一の手を掴んだら handedness で所属を決め、負けた側は棄却せず σ ×4。
- 手ブロック L/R 入替は、両手首が追跡中で入替コストが 0.6 倍未満かつ差 60 px 超のときのみ。
- 観測は 2D 21 点 (crop スケールから σ)。手クロップが走ったフレームは body の手ブロック (手首以外) を置換。

### 3.6 深度リフトと胴 yaw (`fusion/observe.rs`)

- body 17 点 + 手首は、キーポイント直下 (3 px 窓) の深度中央値を人物帯でゲートし、皮膚→関節オフセットで視線方向へ押し込んで 3D 観測にする。手矩形内の画素は顔の深度に使わない。予測腕カプセルがレイを遮る場合は表面点に格下げ。
- σ: 2D は `KpSigma { floor 2.8 px, simcc_gain 1.9, score_inflate 3.0 }`、3D は `SCORE_INFLATE_BASE_3D = 1.3` × (1 + (1 − p_vis))。gain/floor/base は較正済み p_vis (鋭いピークで ≈ 0.99) 前提の値。
- `ShoulderYawObs`: 胸部ストリップの列中央値深度の LS 勾配 (手クロップ画素のみ除外、胸が塞がる姿勢では肩上面バンド) を 1-DoF の胴 yaw 観測として注入。胴カプセルは自軸対称なので表面項は yaw 情報を持たない。σ スケール既定 0.5 (2026-09-14、29 録画ベンチ: 胴 yaw sd 概ね半減 s1789349252 12.7→4.9 / s1789088238 37.4→6.0、|err| 合計 ~103°→~60°、手首 snap 中立 195→191、独立指標の 2D 再投影は 11/11 不変。0.25 は腕と闘って snap 5→22)。`VULVATAR_CHESTYAW_SIGMA` で上書き。

### 3.7 密表面点 (主観測、`Silhouette::sample_points` → `Estimator::surface_term`)

深度が見ている体表面そのものを主観測にする。2D キーポイントは面が決められない自由度 (左右、面に沿った関節位置、手) だけを担当する。

- サンプル: シルエット (§3.3、2 倍間引きグリッド) の主成分画素を 8 px 間隔 (1280 幅では 12 px) で採り、手クロップ矩形内は除外。640×480 のデスク framing で約 1,300 点。σ = 0.004 + 0.001·z² m (D435 レンジノイズ) に推定器側で形状誤差 σ 0.012 m を二乗和。
- 対応付け (毎 LM 反復、`CapGeom::dist`、AABB で早期棄却):
  - 手カプセルは対象外。ただし最近傍が手なら点は棄却 (前腕へ渡さない)。
  - 胴 (楕円柱)・首・頭 = コア。**前方ゲート 0.03 m**: コア表面より手前に 3 cm 超離れた点は遮蔽物 (胸前の手・前腕・机縁) として却下。後方/内側は 0.20 m (×GNC) まで許容。
  - 腕・脚カプセルは `FrameObs::surf_allow` で許可された時だけ点を取り、許可されていない肢が最近傍なら点は棄却 (未観測の腕は依然として胸を遮る)。既定で腕は不許可 (`VULVATAR_DENSE_ARMS=1` で「肘/手首が追跡中なら許可」)、脚はシルエット可視高さ ≥ 0.65 m かつ膝追跡中のみ。
  - 許可された肢は最近傍で公平に競合する (胸前で構えた腕は自分の点を持つ)。
- 重み: カプセルごとの実効点数 `n_eff` (胴 30、頭 8、肢 15) で各点の重みを min(1, n_eff/n) 倍。形状誤差はカプセル内で相関するので N 点は N 倍の情報を持たない。頭は FaceMesh 重心が既に画素精度で位置を決めるため小さい。
- ヤコビアン: 圧縮正規方程式 (カプセルごとに端点 a・b・側方点 c・log 半径 の 10 列)。円断面は解析、楕円断面は前進差分。
- 追跡健全性: 対応点 ≥ 150 かつ平均符号付き距離 < 3 cm なら、2D 顔点の残差が跳んでも lost にしない (掌で顔を覆うと検出器は鼻・目を手の上に描き、旧判定はそこへ再捕捉していた)。
- 無効化: `VULVATAR_FUSION_NO_DENSE=1`。

### 3.8 未観測手首のホールド (`Estimator::accumulate_at` の wrist-hold 項)

そのフレームに手首観測 (2D/3D いずれか) が**ゼロ**の手首は、root 並進相対の予測位置へ σ 0.02 m の非ロバスト疑似観測でピン止めする。肘は観測され手は机下、のデスク構図では前腕方向 (屈曲+捻り 2-DoF) は全データ項の零方向であり、事前の浅い盆地間をソルバがフレーム単位で跳ぶ (s1789219959 で snap 29/31 は観測変化ゼロ・temporal コスト 11.5×)。

- ゲートはフレーム毎: 手首に観測があれば直ちに無効 → 復帰観測や再 seed とは競合しない (`q_hold_floor` を下げる方式は復帰観測と闘って劣化測定済み)。
- 実測 (s1789219959): R 手首 snap >15 cm 12→0、最大 jump 0.41→0.13 m、頭 yaw sd 23.8→15.1。代償は胴 yaw sd +0.2〜0.4°。
- 狭い拘束は効かない測定済み: swivel 角のみ (snap 13/15 残存 — 屈曲側が主因)、肩アンカー位置 (胴 yaw err sd 2.7→4.5 と干渉)。
- 無効化/調整: `VULVATAR_FUSION_NO_WHOLD=1` / `VULVATAR_FUSION_WHOLD_SIGMA=<m>`。

---

## 4. 関節体モデル (`fusion/model.rs`)

- VRM T-pose 基準、約 120 DoF: root 6DoF、脊椎 3・首・頭・鎖骨 2・肩 2・肘 2 (屈曲 + 回内)・手首 2・指 15×2・股 2・膝 2・足首 2。
- 形状 β = 骨長 + 分節カプセル半径。セッション内で推定し収束後は強い事前で固定 (`shape_frozen`)。**半径は骨長スケールと独立** (`rad_mul = exp(rad[g])`): 身長と胴回りは比例しないし、密表面点が半径を直接測る (連動させると胸を広げるために全骨長が 13% 伸びた)。
- 各分節はカプセル。**胴は 1 本の楕円柱** (`capsule_ellipse`: 軸 `torso_lo`→`torso_hi`、側方基準点 `torso_l_hi`、側方半径 0.17 m・奥行 0.105 m)。平らな胸面の向きが yaw を決める (円柱 2 本では平面パッチにどの yaw でも同じ精度で当たり、30° 外れた解に落ちた)。頭・首・四肢は円柱カプセル。サイト (`head_top`, `head_center` など) は関節に固定したオフセット点。解析ヤコビアン。
- 関節限界はソフト境界 (hinge)、姿勢事前は neutral 周りの弱いガウス。

---

## 5. 推定器 (`fusion/estimator.rs`, `fusion/seed.rs`)

```
E = Σ ρ_C(‖π(J(x)) − u‖²/σ²)      2D 再投影 (body / face 重心 / hand)、Cauchy
  + Σ ρ_C(‖J(x) − p‖²/σ²)          深度リフト 3D 点
  + Σ ρ_C(d(p, surface)²/σ²)       表面点 (gate 0.20 m × GNC)
  + OriObs / ShoulderYawObs         SO(3) / 1-DoF 方位観測
  + hinge(q) + ‖q − q̄‖²           関節限界・姿勢事前 (q̄ はモデル relaxed pose)
  + (root up · x, z)²/σ²           root 直立事前 (σ 0.12)
  + (trunk axis · z)²/σ²           胴軸事前 (σ 0.15): 骨盤が画面外のとき面項が下部胴をカメラ側へ振る (root 深度 0.79→0.43 m) のを止める
  + ‖x_t − f(x_{t−1})‖²_Q          時間事前 (q_joint 2.0 rad²/s、頭は trunk に含め q 0.15)
  + ‖β − β̂‖²                       形状定常
```

- sparse LM、warm start = `predict(t)`、GNC ブートストラップ。共分散は H⁻¹ の対角ブロック → 関節ごとの周辺 σ (`joint_world_sigma`) と観測到達度 (`joint_data_sigma`、< 0.5 で「追跡中」)。
- `predict(t)` は任意時刻の状態 + 共分散 (レンダースレッドが 60 Hz で呼ぶ)。1€ フィルタは持たない。**観測ゼロ帧は速度をゼロに落とす** (`finish`): データ項が無い帧で速度差分を取り直すと予測自身の外挿 (≈0.96×旧速度/帧) を再摂取して減衰がほぼ打ち消され、ジャンク検出 1 枚の速度スパイク (root 3 m/s) がドロップアウト全体を 0.1–0.2 m/帧で暴れさせる (実測 s1789279985: root_z 0.93→1.32 m、203 snap 中 172 がこのモード。修正で 12 録画 snap 合計 244→109、胴 yaw |err| は不変)。
- 健全性: スパース主要関節の 2D 残差中央値 (`med_sparse_2d_px`) が `lost_rms_px` (40 px) を 2 フレーム連続で超えたら lost。頭/肩の 3D アンカーが追従中 (`n_kp3d ≥ 4` かつ `mean_3d_m < 0.12`) か、密表面が当たっている (§3.7) なら lost にしない。lost 時も直前深度を引き継ぐ。
- 未観測肢のプロセスノイズ縮小 (`q_hold_floor`) は 0.25 で悪化 (保持された腕が戻ってきた観測と喧嘩し胴が代償を払う) を実測し、既定 1.0 (無効)。
- 再シード: 解析的アームシード候補 (`update_with_arm_seeds`) は現解のコストの 0.95 倍未満で勝った時だけ採用 (僅差採用は手首瞬間移動になる)。**勝敗比較は密表面コストを除外**する (2026-09-15): シードは metric キーポイント由来であり、面項を比較に入れるとシード自身が変えた対応付けの利得が票になる — 腕カプセルが点を取る構成では自分の画素を説明し直すだけで勝つ (wave で seed wins 3→15、手首 0.5–0.8 m)。ソルブ自体は全候補で面項込み、accept 判定だけ除外 (12 録画ベンチ: 単独では中立)。**定常ゲート**: 手首 data-σ が追跡閾値 (0.5) 以下の腕はシード候補をスキップする — 左手首常時観測のデスク構図では seed_l + seed_both が毎フレーム走って推定器時間の ~2/3 を占め、勝率 1/600 だった (s1789219959: seed評価 612→3、estimator 68.7→26.3 ms、品質指標は全桁同一。勝った 1 フレームは右手首の復帰直後で data-σ EMA がまだ高く、ゲートは自然に開く)。ブートストラップは σ=10 で常時シード。`VULVATAR_FUSION_SEEDGATE_SIGMA` で閾値変更、0 で無効化 (常時シード = 旧動作)。
- root は 1:1 metric、recenter 無し (`root_recenter_horizon_s = None`)。

---

## 6. リターゲット (`src/avatar/retarget.rs`)

- 入力 `TrackingRigPose { t, 関節回転 (親相対), root, 手関節角, blendshape, チャンネル別 σ }`。
- rest offset を掛けた回転コピー (world-delta)。σ がしきい値を超えた関節は idle へ smoothstep (上腕の idle は A ポーズ、他は bind)、root はアンカー。
- 表示レストはプロシージャル A ポーズ (`src/avatar/relax.rs`)。bind (T ポーズ) は skinning・cloth bind・SDF の数学的参照なので書き換えず、上腕を rest 形状 (上腕→肘方向) から水平 45° 下ろすオーバーレイを表示側に掛ける。既に A ポーズで bind された rig は不変。ドライバー (追跡・アニメ) が外れた最初のフレームで最終ポーズをキャプチャし、0.9 秒 smoothstep でレストへイージングする (即時ジャンプしない)。`VULVATAR_APOSE_DEG` (下ろし角) / `VULVATAR_RELAX_S` (遷移秒、0 = 従来の即時) で調整。
- 表情・視線・スプリング・クロスは従来どおり。

---

## 7. 計測基盤

| ツール | 用途 |
|---|---|
| `diagnose_fusion_replay <dir> [out] [--render N] [--avatar]` | 録画を本番プロバイダで無人リプレイ。`frames.csv` (毎フレームのコスト内訳・σ・関節位置)、summary (胴 yaw std / 肩深度参照との誤差 / 手首・膝ジャンプと snap / data-σ duty / root jump / seed wins / 再捕捉)。GPU なら 400 フレーム約 24 秒 |
| `VULVATAR_REPLAY_VISDUMP=1` | `kps.csv`: 133 関節 × 全フレームの SimCC 統計、p_vis、シルエット距離、crop、crop ヒント、各段階のスコア |
| `diagnostics/visibility/` | `analyze_vis.py` (較正・混同行列)、`bench_compare.py <runsA> <runsB>` (セッション別 + 合計、crop 遷移数、顔可視率)、`overlay_vis.py` (判定オーバーレイ)、`ablate.ps1` / `dense_ablate.ps1` / `dense_multi.ps1` / `dense_sweep.ps1` (アブレーション行列、密表面項の off/on 比較) |
| `VULVATAR_FUSION_OBSDUMP=<frame>` | 観測とモデルの対応ダンプ。カプセル幾何 (`cap N ... allow=`) と各表面点の推定器側対応 (`est=<capsule>`, −1 = 棄却) を含む |
| `validate_gt` | 合成 GT (既知ポーズをレンダ → 追跡 → 復元) |
| ライブ debug チャネル | `CLAUDE.md` 参照 (`debug_state.json` / `debug_avatar.json` / `debug_depth.bin`) |

判断は **23 録画の合計** で行う。単一セッション、特に手を上げた短い録画 (`s1787219804`) は推定器の崩壊モード (root がカメラ側へ寄り胴 yaw −60° 級) に落ちるか否かが σ や crop の微差で反転し、指標として再現しない。

環境変数 (アブレーション):

| 変数 | 効果 |
|---|---|
| `VULVATAR_FUSION_NO_VIS` | 可視性層を無効化 (旧ゲートが有効になる) |
| `VULVATAR_FUSION_OLDGATES` | 可視性層に加えて旧ゲートも有効 |
| `VULVATAR_FUSION_NO_HINT` | crop ヒントを渡さない |
| `VULVATAR_FUSION_OLDSIGMA` | σ 再表現前の gain/floor/base (1.0 / 1.5 px / 1.0) |
| `VULVATAR_FUSION_NO_{SURF,3D,BURNIN,REACH,CHESTYAW}` | 各項の無効化 |
| `VULVATAR_HEAD_KP_SCALE` | SimCC 顔 kp σ 膨張の倍率 (既定 0.5) |
| `VULVATAR_HEAD_CULL_FACING` | 頭 far-side cull の facing 閾値 (既定 -0.15) |
| `VULVATAR_Q_HEAD` | 頭ジョイントのプロセスノイズ q (既定 0.15 = trunk と同一) |
| `VULVATAR_HAND_SIGMA_SCALE` | 手ランドマーク σ の倍率 (既定 1.0) |
| `VULVATAR_FUSION_NO_WHOLD` / `VULVATAR_FUSION_WHOLD_SIGMA` | 未観測手首ホールドの無効化 / σ (m) (§3.8) |
| `VULVATAR_FUSION_SEEDGATE_SIGMA` | アームシードの定常ゲート閾値 (data-σ)。0 = 常時シード (旧動作) |
| `VULVATAR_FUSION_NO_DENSE` / `VULVATAR_DENSE_ARMS` / `VULVATAR_DENSE_NOHEAD` / `VULVATAR_DENSE_NONECK` | 密表面項の無効化 / 腕カプセル許可 / 頭・首の除外 |
| `VULVATAR_DENSE_NEFF` / `VULVATAR_DENSE_NEFF_HEAD` / `VULVATAR_DENSE_FRONT` / `VULVATAR_DENSE_FREEZE` | 実効点数・前方ゲート・形状凍結の上書き |
| `VULVATAR_HOLD_Q` / `VULVATAR_NO_UPRIGHT` / `VULVATAR_TRUNK_AXIS_SIGMA` | 未観測肢ホールド係数 / 胴軸事前の無効化・σ |
| `VULVATAR_FUSION_OBSDUMP=<frame>` | 観測とモデルの対応ダンプ |
| `VULVATAR_REPLAY_CPU` / `VULVATAR_REPLAY_NO_YOLOX` | リプレイの EP / YOLOX 無効化 (CPU と DirectML の SimCC 統計は小数 4 桁で一致) |

---

## 8. 実装状況と既知の課題

| 項目 | 状態 |
|---|---|
| 関節体モデル・推定器・共分散・predict | ✅ |
| 可視性層 (p_vis + 深度シルエット + 脚高さ規則 + state-driven crop) | ✅ |
| 密表面点を主観測に (楕円柱胴、部位別対応付け、前方ゲート、胴軸事前) | ✅ 胴・頭のみ。腕は未接続 |
| 顔 (canonical テンプレート、OriObs、深度裏付け) | ✅ |
| 手 (state-driven crop、handedness、裏付け、重複ロック) | ✅ 2D のみ |
| リターゲット・アプリ配線 | ✅ (v1 経路は削除済み) |
| align-to-color 廃止 (native 深度 + extrinsics) | ❌ 未着手 |
| AprilTag GT リグ / 実データ mm-deg ベンチ | ❌ 未着手 |
| 脚・床平面・学習姿勢事前 | △ 脚はモデル・観測にあるが床なし。姿勢事前はモデル relaxed pose のみ (キャリブレーション由来の q_neutral は 2026-09-13 に撤去済み) |

既知の課題:

- 腕は密表面に接続していない (2D + 深度リフト + 手クロップのみ)。接続すると腕シードのコスト比較に面項が混ざり、手を振る録画で手首が 0.5〜0.8 m 跳ぶ (seed wins 3 → 15)。腕モデル (円柱 1 本) と対応付けの改良が前提。
  - 2026-09-15 実測 (seed 比較の面項除外を入れ、12 録画ベンチ): `VULVATAR_DENSE_ARMS=1` で snap 合計 244→179 だが胴 yaw |err| 合計 19.0→32.7 (s1789242856 は yaw +27°→+56° の盆地に入る — 腕が肩域の点を引き取る自由度が胴 yaw と鎖骨開きの代替説明を作る)。観測あり帧だけ見ると snap 72→89 で純増。**腕カプセルの表面行を腕チェーン param のみにマスク**する案 (ARM_CHAIN_DEPTH と違い既存の 2D/3D 証拠は削らない) は更に悪い (|err| 145、snap 243): 肩穴が固定されるため腕がソケット内で歪み、対応付けが入れ替わって胴の点集合自体が崩れる。腕表面の接続は 2 段ソルブ (胴ステージ → 腕ステージ) と肘深度リフトの系統誤差 (~14 cm) の解決が前提。
- 手を素早く顔前で振る録画 (wave、5 フレーム間引き) では密表面項ありで胴 yaw std 3.6 → 13°。胸前を横切る手が前方ゲート内 (< 3 cm) で胴の点を奪う/隠すため。デスク配信では起きない。
- 推定器の崩壊モード (手上げ時に 2D 項が root をカメラ側へ引く) は密表面項で大きく減った (s1787219804: 胴 yaw std 24 → 13) が消えてはいない。
- 頭 yaw 振幅: mesh conf < 0.2 の区間は方位ソースが無い。
- namaste 持続カバー中の偽お辞儀: 手に覆われた SimCC 鼻/目が手の表面に捏造される。per-kp ゲートでは解けず据置、次の一手は耳ベース頭アンカー。
- 未観測腕のプロセスノイズ (q_joint 2.0) が緩く、腕の観測が消えた/戻った時の往復が残る。
- 肘の深度リフトに約 14 cm の系統誤差 (Cauchy で無視されている)。
- 参照 (胸部深度勾配) と `ShoulderYawObs` は同じ物理信号なので、参照だけでは姿勢の正しさを証明できない。独立指標は 2D 再投影誤差と合成目視。
- align-to-color 廃止の初手として「リプレイが meta.jsonl の実 intrinsics を読む」改善が有効 (現在は名目 D435 値で固定、AGENTS.md の前提)。ただし bench 数値の連続性が切れるので opt-in フラグ (`--real-intrinsics`) で。
