# Tracking v2 — 観測融合型ボディフィッタ (現行仕様)

本番経路は `FusionProvider` (`src/tracking/fusion/provider.rs`) の一本のみ。
本書は **As-Is の仕様** を書く。変更の経緯・計測記録は git log と `diagnostics/` に置き、ここには残さない。

関連: [architecture.md](architecture.md), [threading-model.md](threading-model.md),
[calibration-ux.md](calibration-ux.md)。運用手順 (ライブ計測・録画・リプレイ) は `CLAUDE.md`。

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
  ├─ 深度リフト / 肩 yaw 観測 / 表面点 (§3.6)
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
- SimCC 顔 kp (0..4) は mesh 重心がある間 σ ×(1 + 2·conf 連続補間)。頭の depth_at 窓は予測頭 ±0.20 m。

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
- `ShoulderYawObs`: 胸部ストリップの列中央値深度の LS 勾配 (手クロップ画素のみ除外、胸が塞がる姿勢では肩上面バンド) を 1-DoF の胴 yaw 観測として注入。胴カプセルは自軸対称なので表面項は yaw 情報を持たない。
- 密点群項は実装済みだが `cloud_budget = 0` で無効 (z-buffer 可視性付き対応付けに作り直すまで)。

---

## 4. 関節体モデル (`fusion/model.rs`)

- VRM T-pose 基準、約 120 DoF: root 6DoF、脊椎 3・首・頭・鎖骨 2・肩 2・肘 2 (屈曲 + 回内)・手首 2・指 15×2・股 2・膝 2・足首 2。
- 形状 β = 骨長 + 分節カプセル半径。セッション内で推定し収束後は強い事前で固定 (`shape_frozen`)。
- 各分節はカプセル、頭は楕円体。サイト (`head_top`, `head_center` など) は関節に固定したオフセット点。解析ヤコビアン。
- 関節限界はソフト境界 (hinge)、姿勢事前は neutral 周りの弱いガウス。

---

## 5. 推定器 (`fusion/estimator.rs`, `fusion/seed.rs`)

```
E = Σ ρ_C(‖π(J(x)) − u‖²/σ²)      2D 再投影 (body / face 重心 / hand)、Cauchy
  + Σ ρ_C(‖J(x) − p‖²/σ²)          深度リフト 3D 点
  + Σ ρ_C(d(p, surface)²/σ²)       表面点 (gate 0.20 m × GNC)
  + OriObs / ShoulderYawObs         SO(3) / 1-DoF 方位観測
  + hinge(q) + ‖q − q_neutral‖²    関節限界・姿勢事前
  + ‖x_t − f(x_{t−1})‖²_Q          時間事前 (q_joint 2.0 rad²/s、頭は trunk に含め q 0.15)
  + ‖β − β̂‖²                       形状定常
```

- sparse LM、warm start = `predict(t)`、GNC ブートストラップ。共分散は H⁻¹ の対角ブロック → 関節ごとの周辺 σ (`joint_world_sigma`) と観測到達度 (`joint_data_sigma`、< 0.5 で「追跡中」)。
- `predict(t)` は任意時刻の状態 + 共分散 (レンダースレッドが 60 Hz で呼ぶ)。1€ フィルタは持たない。
- 健全性: スパース主要関節の 2D 残差中央値 (`med_sparse_2d_px`) が `lost_rms_px` (40 px) を 2 フレーム連続で超えたら lost。頭/肩の 3D アンカーが追従中 (`n_kp3d ≥ 4` かつ `mean_3d_m < 0.12`) なら lost にしない。lost 時も直前深度を引き継ぐ。
- 再シード: 解析的アームシード候補 (`update_with_arm_seeds`) は現解のコストの 0.95 倍未満で勝った時だけ採用 (僅差採用は手首瞬間移動になる)。
- root は 1:1 metric、recenter 無し (`root_recenter_horizon_s = None`)。

---

## 6. リターゲット (`src/avatar/retarget.rs`)

- 入力 `TrackingRigPose { t, 関節回転 (親相対), root, 手関節角, blendshape, チャンネル別 σ }`。
- rest offset を掛けた回転コピー (world-delta)。σ がしきい値を超えた関節は idle へ smoothstep、root はアンカー。
- 表情・視線・スプリング・クロスは従来どおり。

---

## 7. 計測基盤

| ツール | 用途 |
|---|---|
| `diagnose_fusion_replay <dir> [out] [--render N] [--avatar]` | 録画を本番プロバイダで無人リプレイ。`frames.csv` (毎フレームのコスト内訳・σ・関節位置)、summary (胴 yaw std / 肩深度参照との誤差 / 手首・膝ジャンプと snap / data-σ duty / root jump / seed wins / 再捕捉)。GPU なら 400 フレーム約 24 秒 |
| `VULVATAR_REPLAY_VISDUMP=1` | `kps.csv`: 133 関節 × 全フレームの SimCC 統計、p_vis、シルエット距離、crop、crop ヒント、各段階のスコア |
| `diagnostics/visibility/` | `analyze_vis.py` (較正・混同行列)、`bench_compare.py <runsA> <runsB>` (セッション別 + 合計、crop 遷移数、顔可視率)、`overlay_vis.py` (判定オーバーレイ)、`ablate.ps1` (1 セッションのアブレーション行列) |
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
| `VULVATAR_FUSION_OBSDUMP=<frame>` | 観測とモデルの対応ダンプ |
| `VULVATAR_REPLAY_CPU` / `VULVATAR_REPLAY_NO_YOLOX` | リプレイの EP / YOLOX 無効化 (CPU と DirectML の SimCC 統計は小数 4 桁で一致) |

---

## 8. 実装状況と既知の課題

| 項目 | 状態 |
|---|---|
| 関節体モデル・推定器・共分散・predict | ✅ |
| 可視性層 (p_vis + 深度シルエット + 脚高さ規則 + state-driven crop) | ✅ |
| 顔 (canonical テンプレート、OriObs、深度裏付け) | ✅ |
| 手 (state-driven crop、handedness、裏付け、重複ロック) | ✅ 2D のみ |
| リターゲット・アプリ配線 | ✅ (v1 経路は削除済み) |
| 密点群項 | ⚠️ 既定 OFF |
| align-to-color 廃止 (native 深度 + extrinsics) | ❌ 未着手 |
| AprilTag GT リグ / 実データ mm-deg ベンチ | ❌ 未着手 |
| 脚・床平面・学習姿勢事前 | △ 脚はモデル・観測にあるが床/事前なし |

既知の課題:

- 推定器の崩壊モード (手上げ時に 2D 項が root をカメラ側へ引く): 可視性層では解けない。観測可能性の問題として推定器側で扱う。
- 頭 yaw 振幅: mesh conf < 0.2 の区間は方位ソースが無い。
- namaste 持続カバー中の偽お辞儀: 手に覆われた SimCC 鼻/目が手の表面に捏造される。per-kp ゲートでは解けず据置、次の一手は耳ベース頭アンカー。
- 未観測腕のプロセスノイズ (q_joint 2.0) が緩く、腕の観測が消えた/戻った時の往復が残る。
- 肘の深度リフトに約 14 cm の系統誤差 (Cauchy で無視されている)。
- 参照 (胸部深度勾配) と `ShoulderYawObs` は同じ物理信号なので、参照だけでは姿勢の正しさを証明できない。独立指標は 2D 再投影誤差と合成目視。
