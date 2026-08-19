# Tracking v2 — 観測融合型ボディフィッタ (根本リライト設計)

Status: **design (2026-08-19)** — 実装未着手。
本書は現行パイプライン (`src/tracking/skeleton_from_depth.rs` + `src/avatar/pose_solver.rs`) を
**置き換える**設計であり、既存層への追加ではない。

関連: [tracking-retargeting.md](tracking-retargeting.md) (`TrackingRigPose` 契約は維持),
[onnx-tracking-pipeline.md](onnx-tracking-pipeline.md) (v1 の記録),
[threading-model.md](threading-model.md)。

---

## 0. なぜ書き直すのか (v1 の構造的欠陥)

2026-08-19 の精度監査で確定した事実 (計測: `diagnostics/validation_gt/summary.md`,
`diagnostics/depth/w2_palms_summary.log`, `diagnostics/live_head_auto2.jsonl`):

| 症状 (実測) | 構造的原因 |
|---|---|
| 胴ツイスト 30° → 追跡段階で 0° | 胴の向き = 肩 2 点の 7×7 median 深度差。点群は面として使われない |
| 首曲げ std 36°、face yaw 78°/frame ジャンプ | 頭の向きに深度が入っていない。body 経路は捨てたはずの SimCC-z、mesh 経路は 2D 比 (±63° 飽和) |
| 前方リーチ gain 0.09 | Z 軸 1€ cutoff 0.10–0.15 Hz (単眼時代の前提)、観測肘を IK で必ず上書き |
| 手 15 Hz 点滅・0.5 m テレポート | 深度必須の存在判定 → 二値の有無 → 各層が増幅。不確かさが無いので「弱く見えている」を表現できない |
| 前屈・側屈が出ない | 背骨の情報源が 3 点、UpperChest = Neck 同一点 |
| 修正が収束しない | 決定論ゲート/ラッチ/ホールド/EMA が 20 段以上、各層が閾値を持ち、閾値が次の症状を作る |
| 精度が改善サイクルに入らない | 実データで mm/deg を測る計器が無い (合成往復 gain と GT 無し自己整合のみ) |

共通の根: **「観測は信用できない」を前提に、観測 → 判定 → 補完 を層で積んだ**こと。
v2 は逆に **「全観測を不確かさ付きで一つの推定器に入れ、判定はコスト関数に吸収させる」**。

---

## 1. 目標と非目標

### 目標
- **寄り** (顔+肩、顔+手のクローズアップ): 頭 6DoF、視線/表情、指の 3D 関節角、前腕の回内回外を、
  フレーム内で観測できる限り**フィルム品質**で。手が画面下に消える (デスク配信) を正常状態として扱う。
- **引き** (全身): 胴の 3 分節屈曲・骨盤・脚・足接地・メートル並進。近寄る/離れる/横移動が 1:1。
- 上記を**モード切替なしで連続に**。フレーミングは推定器から見た「観測可能性」の変化にすぎない。
- 各出力チャンネルに**校正された不確かさ** (共分散由来) を付け、アバター側は不確かさで駆動/休止をブレンドする。
- 実データ精度指標 (MPJPE / 関節角 RMS / 頭 6DoF 誤差 / ジッタ / レイテンシ) を継続計測する。

### 非目標
- 単眼 (D435 なし) 動作。v2 は D435 前提 (v1 の RTMW3D-only フォールバックは廃止)。
- 多人数。

---

## 2. アーキテクチャ概観

```
D435 ─color 1280×720@30─┐
      └depth  848×480@30 ┤ (native、align しない。extrinsics で相互投影)
                          ▼
              ┌────────── Perception (観測生成、GPU) ──────────┐
              │ body wholebody 2D (+σ)  ← 予測 bbox で crop     │
              │ face landmarks (+σ)     ← 予測頭位置で crop     │
              │ hand landmarks ×2 (+σ)  ← 予測手首で crop       │
              │ depth point cloud (native frame)               │
              └───────────────┬────────────────────────────────┘
                              ▼   Observation{t, kind, values, Σ}
              ┌────────── Estimator (CPU, 関節体 MAP フィット) ─┐
              │ 状態 x = [root 6DoF, 関節角 q, 形状 β]           │
              │ 残差: 2D 再投影 / 点群→カプセル / 顔剛体 /       │
              │       関節限界 / 姿勢事前 / 定速度 / 骨長定常     │
              │ 解: sparse LM, 2–3 フレーム窓, 共分散 = H⁻¹       │
              │ predict(t): 任意時刻の状態 + 共分散              │
              └───────────────┬────────────────────────────────┘
                              ▼   TrackingRigPose{t, 関節回転, root, 表情, 不確かさ}
              ┌────────── Retarget (render thread) ────────────┐
              │ 関節角 → アバターボーン回転 (rest offset)        │
              │ 不確かさ → 駆動/idle ブレンド                     │
              └────────────────────────────────────────────────┘
```

v1 との最大の差: **「関節位置の集合 (SourceSkeleton)」を運ばない**。運ぶのは
関節体モデルのパラメータと共分散。SourceSkeleton / pose_solver の direction-match は消える。

---

## 3. センサ前段 (`src/tracking/capture/`)

- color **1280×720@30** (寄りの顔・指の画素数確保。1920×1080 は D435 の露光/帯域とのトレードで後述ベンチで決める)、
  depth **848×480@30 native**。**align-to-color を廃止**: 深度は自フレームのまま点群化し、
  color↔depth extrinsics で 2D キーポイントの光線を深度フレームへ、点群を color へ相互投影する。
  align のリサンプルは近距離のエッジで深度を混ぜ、寄りの顔輪郭・指で致命的。
- librealsense フィルタ順序を推奨通りに: disparity 変換 → spatial → temporal → 逆変換。**hole-filling は使わない**
  (穴は「観測なし」であって捏造しない)。
- 各フレームに HW timestamp、露光、深度スケールを添付。color/depth の同期ゲートは v1 のまま。
- D435 のノイズモデル (距離依存 σ_z ≈ 0.001·z² 級 + エッジ) をセンサ層で**画素ごとの σ**として出す。
  これが以降の全深度残差の重みになる。

---

## 4. Perception (観測生成、`src/tracking/perception/`)

各観測は `Observation { t, source, values, cov }` として推定器へ渡す。**判定はしない**。

| 観測源 | モデル | 入力 | 出力 (+不確かさ) | 起動条件 |
|---|---|---|---|---|
| Body 2D | RTMW3D wholebody 133 (現行、SimCC) | 予測 bbox の 288×384 crop | 2D 点 + **SimCC 分布からの σ²** (峰の鋭さ/エントロピー)。z/score も渡す (使うかは推定器) | 常時 |
| Face | FaceMesh 478 (attention/iris 付き) | 予測頭位置の 256 crop (color 原寸から) | 478 点 2D + relative-z、blendshape 用 | 頭が > ~60 px |
| Hands | 手専用 2.5D ランドマーク (21 点、RTMPose-hand 級 or MediaPipe Hands) | 予測手首の 224 crop | 21 点 2D + 相対 z + handedness 尤度 | 手が > ~40 px かつフレーム内 |
| Depth | 点群 (native) | — | 画素ごと (x,y,z,σ) | 常時 |
| Person bbox | YOLOX (CPU、初期化/ロスト時のみ) | 全体 | bbox | 追跡ロスト時 |

設計要点:
- **crop は推定器の予測から切る** (state-driven cropping)。YOLOX は初期捕捉とロスト復帰のみ。
  bbox の feedback loop 対策 (v1 の self-track ヒステリシス) は、予測 + 共分散マージンで crop するので不要。
- 寄り/引きの切替はここに現れる: 引きでは face/hand crop が画素不足で走らず body 2D のみ、
  寄りでは body 2D が上半身しか見ずに face/hand crop がフル解像度で走る。**同じ推定器が両方を消費する**。
- 133 点のうち手ブロック 42 点は、hand crop が走っているフレームでは crop 側で置換 (高分解能)。
- 全 ONNX セッションは GPU 逐次実行 (`GpuExclusiveGuard` 継承、Arc ドライバ TDR 対策)。
  予算 (30 fps 換算): body ~38 ms は超過 → body を 15 Hz、face/hand を 30 Hz というレート分離が可能
  (推定器が非同期観測を時刻で扱うため、レートを揃える必要が無い)。

---

## 5. 関節体モデル (`src/tracking/body_model/`)

### 状態
- root: 位置 (m, カメラ座標) + 回転 (SO(3))
- 関節角 q (回転ベクトル / オイラー、関節ごとに DoF 制限):
  骨盤・脊椎 ×3 (下部/中部/胸) ・首・頭・鎖骨 ×2・肩 ×2・肘 ×2 (屈曲+回内)・手首 ×2・
  指 5×3 ×2 (屈曲 + 基節外転)・股 ×2・膝 ×2・足首 ×2 → 約 90 DoF
- 形状 β: 骨長 (約 30) + 各分節のカプセル半径。**セッション内で推定し、収束後は強い事前で固定**。

### 幾何
- 各分節はカプセル (中心線 + 半径)。頭は楕円体、胴は 3 分節の楕円柱。
  **点群残差はモデルの表面**に対して評価する (v1 が肩「点」に深度を割り当てて生じた「肩表面 vs 関節中心」の系統差を除去)。
- 顔: 頭フレームに固定した正準顔ランドマーク集合 (FaceMesh の canonical 478)。個人差はセッション内で
  スケール + 小変形をフィット。

### 関節限界と事前
- 関節ごとの可動域 (ソフト境界、ヒンジロス)。
- 姿勢事前は弱いガウス (neutral 周り)。VPoser 級の学習事前は Phase 4 の選択肢 (脚が見えないときの補完品質を上げる)。

---

## 6. 推定器 (`src/tracking/estimator/`)

### 目的関数 (フレーム t、状態 x_t)
```
E = Σ_k ρ( ‖π(J_k(x)) − u_k‖²_{Σ_k} )            2D 再投影 (body/face/hand)
  + Σ_p ρ( d(p, surface(x))² / σ_p² )                点群 → 分節表面 (対応付け: 予測状態のカプセルへ最近傍)
  + ρ( ‖T_head(x)·L_canon − L_obs‖² )              顔剛体 (2D 再投影 + 顔領域の深度)
  + Σ_j hinge(q_j)                                 関節限界
  + ‖q − q_neutral‖²_{Σ_prior}                      姿勢事前
  + ‖x_t − f(x_{t−1}, dt)‖²_{Q}                     定速度ダイナミクス (プロセスノイズ Q)
  + ‖β − β̂‖²_{Σ_β}                                  形状定常 (収束後)
```
- ρ は Geman-McClure/Cauchy 級のロバスト核。**遮蔽・背景混入・穴は外れ値として自然に落ちる**
  (v1 の person-band / occlusion-gap / 正面ペア捏造 / 5× reach 棄却 は全部これに吸収される)。
- 解法: sparse Levenberg–Marquardt、2–3 フレームのスライディング窓、warm start = 予測。90 DoF で 1–3 ms (CPU)。
- **共分散**: 収束点のガウス・ニュートン近似 H⁻¹ の対角ブロック → 各関節の周辺分散 → 出力の不確かさ。
  観測が無い関節は Q によって分散が増大し、出力側で idle にフェードする。**HandHold / ArmEngageGate /
  mailbox hold / Schmitt confidence gate は不要**。
- **L/R 対応付け**: 手ブロック・手 crop の左右割当は、両割当のコストを評価して低い方 + 前フレーム割当への
  ヒステリシス項。投影平面だけでなく深度と前腕運動学が同時に効く (v1 `arm_z` の未実装 TODO を解消)。
- **予測 API**: `predict(t) -> (x, Σ)`。レンダースレッドは 60 Hz でこれを呼ぶ (30 Hz 観測の間を
  ダイナミクスで補間、+1 フレーム先読みでレイテンシ補償)。1€ フィルタは持たない —
  滑らかさは Q (プロセスノイズ) の設計であり、応答性との折衷は物理量として調整する。

### 観測可能性と寄り/引き
- 引き: 全関節に 2D 残差 + 全身の点群残差。脚・足首まで駆動、床接地は点群の床平面推定を事前として付加。
- 寄り (デスク): 手首以下・骨盤以下は 2D 残差が無い → 姿勢事前 + ダイナミクスで neutral へ収束し分散が増大 →
  アバターは腕をリラックス位置に。**「hidden-hands で暴れない」は推定器の性質として出る**。
  一方、顔は 478 点 + 顔領域点群で 6DoF が過剰決定 → 頭の向きは深度で決まり、SimCC-z 依存が消える。
- 寄りで胸元だけ見えている場合、胴 3 分節のうち上部のみが点群で拘束される。下部は事前 → 自然。

### スケール・neutral
- 骨長は最初の数秒 (動きのある窓) で β として推定、収束後固定。T ポーズ校正は任意 (精度向上のみ)。
- 「neutral 姿勢」は状態の事前平均。ユーザー校正 UI (`calibration-ux.md`) は q_neutral の設定として残る。
- root は 1:1 metric。v1 の「10 秒 neutral 吸収」は無い。

---

## 7. リターゲット (`src/avatar/retarget/`)

- 入力: `TrackingRigPose { t, per-joint rotation (親相対), root pose, hand joint angles, blendshapes, per-channel σ }`
- 関節角 → アバターボーン: rest-pose offset を掛けた回転コピー (VRM の humanoid rest 前提)。
  骨長差は関節角空間では問題にならない (v1 の direction-match/reach IK が必要だった理由が消える)。
  接触 (合掌) の見た目保証だけ、手先位置の軽い IK 補正を σ が小さいときに限り適用。
- 不確かさ駆動: σ_j がしきい値を超えた関節は idle ポーズへ smoothstep。フェード時間は σ の増加速度 (Q) が決める。
- 表情・視線・スプリング・クロスは現行のまま。

---

## 8. 計測基盤 (**Phase 0、実装より先**)

1. **実データ GT リグ**: 頭 (ヘッドバンド) と両手首にリジッド固定した AprilTag キューブ。同じ D435 で
   タグ 6DoF を検出 → 頭 6DoF・手首 3D の GT (≈ 2 mm / 1°)。プロトコル: 寄り 3 種 (正面/3/4/デスク) ×
   引き 2 種 (立ち/座り) × 動作セット (ツイスト、前屈、リーチ、合掌、指、頭 3 軸)。
   `diagnostics/gt/<session>/` に color/depth/tag pose を記録。
2. **指標**: MPJPE / PA-MPJPE (可観測関節)、関節角 RMS、頭 6DoF 誤差、手指 21 点誤差、
   ジッタ (静止時 σ)、レイテンシ (タグ運動の位相差)、可用性 (σ しきい値以下の duty)。
3. **合成ベンチ**: `validate_gt` を D435 ノイズモデル付きレンダに拡張 (穴・エッジ・距離依存 σ)。
4. **回帰ゲート**: 各 Phase は前 Phase を desk/wide 両セットで上回るまでマージしない。

---

## 9. 実装フェーズ (各段が完成状態。中間フォールバック層は作らない)

| Phase | 内容 | 受け入れ |
|---|---|---|
| 0 | GT リグ + 記録/採点ツール、D435 native 点群 + extrinsics 投影、観測 σ 生成 | v1 を GT で採点し基準値を確定 |
| 1 | 関節体モデル + 推定器 (root/脊椎/首/頭/肩/肘/手首) を body 2D + 点群で。`predict(t)` と σ 出力。リターゲット新経路 | 胴ツイスト gain > 0.9、頭 6DoF 誤差 < 5°、静止ジッタ < 0.5°、desk セットで hidden-hands 暴れ 0 |
| 2 | 顔: state-driven crop + FaceMesh + 顔剛体/深度残差 + 表情 | 頭 6DoF < 2°/5 mm、寄りでのフレーム間ジャンプ < 1° |
| 3 | 手: state-driven crop + 手ランドマーク + 指関節角 + L/R 対応付け | 手首 3D < 15 mm、指 21 点 < 8 mm、swap 0、寄り合掌の貫通なし |
| 4 | 引き: 脚・足・床平面・(任意) 学習姿勢事前 | 全身 MPJPE < 30 mm、接地スライド < 10 mm |
| 5 | v1 撤去: `skeleton_from_depth.rs`, `hand_hold.rs`, `arm_z.rs`, `auto_neutral.rs`, pose_solver の位置ベース経路 | テスト・診断バイナリを v2 API へ移行 |

Phase 1 完了時点で **v2 が本番経路に切り替わる** (v1 は比較用にビルド可能に残すだけ)。

---

## 10. 計算予算 (Intel Arc / DirectML 前提)

| 段 | 見積 | 備考 |
|---|---|---|
| RTMW3D crop | ~38 ms GPU | 15 Hz 運用可 (推定器が非同期観測を扱う) |
| FaceMesh crop | ~5 ms | 30 Hz |
| Hand crop ×2 | ~2×5 ms | 30 Hz、見えているときのみ |
| 点群化 + 対応付け | ~2 ms CPU | native 848×480、間引きなし |
| LM 推定 | 1–3 ms CPU | 90 DoF sparse |
| predict(60 Hz) | < 0.1 ms | |

GPU は逐次 (Arc TDR 対策)。合計 ≤ 60 ms/GPU フレームで body 15 Hz + face/hand 30 Hz が成立。

---

## 11. 主要な設計判断 (確定)

- **位置ではなく関節角を運ぶ**: リターゲットの体型不変性、IK 上書きの根絶。
- **align-to-color 廃止**: 寄りのエッジ品質のため。
- **不確かさは共分散から**: 手決めの confidence 減衰・Schmitt・hold を全廃。
- **1€ 廃止、ダイナミクス + 予測**: 平滑と応答性はプロセスノイズで物理的に調整。
- **判定層ゼロ**: 遮蔽/穴/背景/L-R はロバスト核 + コスト比較 + 事前で扱う。新しい症状が出たら
  「ゲートを足す」のではなく「どの残差・事前・σ が誤っているか」を GT で測って直す。

## 12. リスク

- 推定器の局所解 (腕の折り畳み、深度対称性): 予測 warm start + 2D 残差の強さで通常回避。
  ロスト時のみ複数初期値からの再収束 (Phase 1 で実測)。
- 学習姿勢事前を入れる場合の ONNX 化コスト (Phase 4 で判断)。
- D435 の寄り (30–40 cm) は最小距離近傍で穴が増える → 顔は 2D 478 点が支配、深度は補助として設計済み。
