# AGENTS.md

## Build

```bash
cargo build          # dev (opt-level 2 for vulvatar + egui crates)
cargo build --release
cargo run            # needs RUST_LOG=vulvatar=info for log output
```

`shaderc-sys` requires a pre-built native shaderc library on the system.
If CMake is outdated or unavailable, the from-source fallback will fail.
The dev build caches shaderc at opt-level 0; changing `[profile.dev.package."*"]`
will invalidate that cache and trigger a rebuild.

## Git rules

- **`git reset` は使うな。** ステージング解除であっても、ユーザーが別作業で追加した変更を巻き込むリスクがある。コミット対象を絞りたい場合は `git add <file>` で必要なファイルだけをステージせよ。`git reset HEAD` も禁止。
- `git checkout -- <file>` や `git restore` など working tree を上書きする操作も、ユーザーに確認なしで実行しない。
- コミット時は `git add -A` ではなく、変更したファイルを明示的に `git add <file>` で指定する。

## Validation / test data

- `validation_images/` は **推論入力データ専用** のディレクトリ (中身は `validation_images/README.md` の合成素材セット)。検出・診断・可視化など、コードを動かして得たテスト出力をここに書き出してはいけない。
- 診断系バイナリ (`diagnose_*` など) やアドホック検証スクリプトの出力は、`.gitignore` 済みの `diagnostics/` 配下に書く。バイナリのデフォルト出力先が入力ファイルの隣になる場合は、必ず明示的に `diagnostics/...` を渡して実行する。
- 新しい診断ツールを追加する際も、デフォルト出力先を `diagnostics/` 側にするか、`validation_images/` 配下に書き込もうとしたらエラーにする。
- ポーズ品質のベンチ: 合成 GT は `cargo run --bin validate_gt` (既知ポーズをレンダ→追跡→復元、coupling ゲイン算出、`diagnostics/validation_gt/summary.md`)、実録画は `diagnose_fusion_replay` (下の Tracking 節)。
- **デスク配信エンベロープが主戦場** (頭+肩のみ・カメラ斜め・手は常時デスク下 = ユーザーの本番運用)。腕・顔まわりの変更は正面系リプレイ (wave/palms/namaste) に加えて必ずデスク系録画 (`diagnostics/sessions/<id>` / `diagnostics/depth/desk_*`) でも `diagnose_fusion_replay` を回すこと。デスク向け指標: 手首ジャンプ/snap 数、data-σ duty (非観測の手が誤駆動されていないか)、胴 yaw std、root ジャンプ。

## Cloth / garment

- **Auto-cloth**: skirt-classified primitives get a GPU-backed `ClothAsset`
  derived at load (`src/simulation/auto_cloth.rs` — classifier mirrors
  Phase-1: skirt/bottom/pants name or ≥40% skirt-bone weights, minus the
  upper-body name list). Runtime-only slots (no `source_path`, never in
  project saves); project `.vvtcloth` overlays supersede them per-primitive.
  Opt-out: `settings.json` `auto_cloth: false` / Cloth パネルのチェックボックス。
  CPU ベンド拘束の反転バグ (T07) は 2026-09-13 に修正済み — 3点 edge-angle
  hinge として翼頂点のみを補正するモデル (`cloth_solver/constraints.rs`、
  方向・収束・pinned・退化のテスト8件)。GPU へ bend を移植する際はこの
  参照実装を使うこと (旧実装は禁止)。**weld×selfcol 崩壊も 2026-09-13
  に修正済み**: 位置weld のコピー群を保つ拘束が無く、自己衝突がコピーを
  押し分けて全面スパイク崩壊する (CPU/GPU 共通) → 群内一致拘束
  (`intra_weld_group_constraints`, rest=0) + 近接重複epsilon
  (`SELF_COL_COINCIDENT_EPS_M` = 0.5mm, CPU/GPU ミラー) で解決。
  実機検証: `diagnostics/cloth_gpu_selfcol_glued/` (正常) vs
  `cloth_gpu_every_frame/` (修正前の崩壊)。auto-cloth の位置weld は層を
  区別しない限界あり (`WeldGroups` コメント、Phase C 課題)。
- **スカート干渉の実測切分け (2026-09-16)**: `diagnose_skirt_fit`
  (pose sweep で pin skew・collider カバレッジ・体の帯外突出、既定出力
  `diagnostics/skirt_fit_*/`) と `diagnose_cloth` の `AUTO_CLOTH=1` +
  `SKIRT_POSE=sit|lean|sit_lean|desk` (描画 A/B、`CLOTH_OFF=1` が authored
  リグ参照、`diagnostics/skirt_*/`)。確定済み: (a) Yumeka のスカート骨は
  全部 Hips 子なので単一ノード深ピン帯 (40%) と各頂点 LBS はどのポーズでも
  一致 (skew 0) — multi-bone pin 化はこのリグでは無効果。(b) VRC 由来
  collider は cloth に対し過太 (太腿 74mm vs 実測 ~53mm) で、座り pose の
  t=0 に自由粒子を弾き飛ばす (33ms で 93mm、前面パネルが腿の後ろへ散る)。
  (c) 腿頂点 (y≈0.83) はウエストバンド (0.789) より上 — 深く屈曲した座り
  では帯領域が腿に貫通するのは authored リグでも同時 (`CLOTH_OFF` 参照)。
  (d) 実験フラグ `VULVATAR_AUTO_CONFORMAL_COLLIDERS=1` で cloth 用 collider
  を body メッシュ適合半径に再計測 (pelvis/shin 追加、スカート静止内包絡で
  cap、station-nearest 半径推定、冪等) — 密着は直るが sway のたびに hip
  capsule がプリーツを押し開ける (rest で裂け目、`skirt_rest_capped/` vs
  健全な `skirt_rest_vrc/`) ので既定 OFF。恒久対応は SDF ベースのスムーズ
  押し出し + GPU bend 制約 (Phase C 系)。ピン帯深さは
  `VULVATAR_AUTO_PIN_FRACTION`、selfcol は `VULVATAR_AUTO_NO_SELFCOL` /
  `VULVATAR_AUTO_SELFCOL_RADIUS` で A/B 可能。
- **cloth body-SDF 接触 (2026-09-16 既定 ON)**: `ClothSimState::
  sdf_contact` (既定 `AUTO_CLOTH_SDF_CONTACT_M` = 0.004 m、
  `VULVATAR_AUTO_SDF_CONTACT=<m>` で調整・`=0` で capsule のみに戻す) が
  立つと、自由粒子を body SDF の `sdf_contact` 等値面へ勾配に沿って投影する
  スムーズ接触段階が capsule 段階の後に走る (CPU `collision::collide` と
  GPU `cloth_collide_cs` のミラー — sentinel/NaN セマンティクス含む)。
  発動時は attach が humanoid-bound capsule のうち**上半身** (spine/chest/
  腕/頭) をマスクする (hard radial push が先にプリーツを裂くため)。腿/腰
  カプセルは保持 — 全 humanoid をマスクすると前パネルが腿間に落ち込み、
  両腿の SDF 勾配の谷で引き裂かれる
  (`diagnostics/skirt_tent_rest_sdfdefault`)。既定 ON の理由: Spine カプセル
  (r90) が live デスク姿勢でスカート前面より ~1.5cm 突出し前パネルを常時
  ドーム化していた (「ちんちん」、計測は `skirt-front-dome` メモ /
  `diagnostics/skirt_tent_*`)。アプリは `collider_enabled` 全 true で cloth
  に渡すが、`diagnose_cloth` の A/B は下肢フィルタで上半身押しを見ていな
  かった — `COLLIDERS_ALL=1` でアプリと同じ構成になる (bin に追加済み)。
  SDF field は
  spring と同じ splat 資源 (`self.sdf_slots`, avatar-root 空間) を共有し、
  idle フレームは最後の field を再利用する (splat plan は pose 変化時のみ —
  計画し直すと sentinel 再フィルで壊れる)。settle の入力ハッシュには
  `sdf_contact` を追加済み (field は pose の純関数なので transforms ハッシュ
  が wake をカバー)。A/B: `diagnostics/skirt_rest_sdf` (rest sway 裂けなし・
  密着 — VRC 参照 `skirt_rest_vrc` の浮きと conformal `skirt_rest_capped`
  の裂け両方を解消)、`skirt_sit_sdf` / `skirt_desk_sdf` (座りは lap 前面開放
  が残る = 腿頂点がバンド上方的幾何限界、bend/摩擦は未実装)、
  `skirt_tent_rest_sdfv2` / `skirt_tent_desk_sdfv2` (既定 ON 構成の再確認)。
  **制約**: field は spring 有効時しか splat されない (app 側ゲート)。
- **GPU bend 制約 (2026-09-16 実装)**: `auto_cloth` が welded edge ごとに
  edge-angle hinge を 2 本 (`edge_angle_bend_constraints` — 両端 hinge、
  翼は反対頂点の weld 代表、`from_asset` が rest 角を計算、既定 stiffness
  0.9 / `VULVATAR_AUTO_BEND_STIFFNESS` / `VULVATAR_AUTO_NO_BEND=1`) 生成し、
  GPU は `cloth_bend_{update,accumulate,apply}_cs` 3 カーネルで
  `constraints.rs` の T09 参照をミラー (hinge 不動・free 翼のみ inv_mass
  配分・under-relaxed Jacobi Δx/(n+1))。距離制約の反復ループ内で apply の
  後に毎イテレーション dispatch。attach data は wing CSR を同梱
  (`ClothGpuAttachData.bend*`)。`VULVATAR_AUTO_NO_SELFCOL` 系ノブと併用可。
  **ランタイム A/B 実施済み (2026-09-16, `diagnostics/skirt_*_bend/`)**:
  rest sway はプリーツが鮮明なまま崩壊なし (bend 単体で rest 安定)、
  sit/desk は「片側への流脱」が解消され両側構造を保持
  (`skirt_desk_sdfbend` = SDF 接触との併用が本命構成)。lap 正面の開放のみ
  幾何限界として残る (腿頂点 > ウエストバンド + 摩擦/lap テント未実装)。
  テスト: bend 生成 3 件追加 (`bend_generation_tests`)。
- **ウエスト cloth チャーン = GPU/FPS 戦役の根因 (2026-09-16)**: ライブで cloth が
  一度も settle せず (`quiet_frames` 0 固定、5-6mm/step)、GPU cloth 全チェーンが恒時
  操業してレンダースレッドを 34Hz まで圧迫していた (`render_cpu_ms` EMA 29ms、bench_render
  単体は 3ms — レンダラ本体は健常で GPU キュー待ち)。オフライン A/B (`diagnose_cloth`
  AUTO_CLOTH=1 + SKIRT_POSE=desk + COLLIDERS_ALL=1 + VULVATAR_CLOTH_GPU=1 +
  CLOTH_RENDER_EVERY=1 + CLOTH_CHURN_CSV、p95 mm/step) で切分けた結果:
  SDF 接触 / bend 単独 / self-col / torso・arm・thigh collider は無罪、conformal
  collider で半減、**根本は (a) attach 時に `wind_response 0.35` の定数風が常時 ON
  (デモ用残骸 → 既定 0 に修正、GUI スライダーで opt-in)、(b) damping 0.035 + iter 8
  の限界サイクル (p95 5.3mm → damping 0.15 + iter 32 で 0.22mm、レンダ A/B で
  drape 同等)**。さらに settle fingerprint は f32 bit 一致契約だったため tracking
  微動で毎フレーム wake していた → `cloth_gpu_inputs_hash` は 0.5mm グリッド量子化
  に改訂 (`settle_quant`、テスト `cloth_gpu_inputs_hash_is_quantized`)。残課題:
  p95 ~0.22mm は quiet 閾値 100µm に未達 — ライブで `debug_gui.json` の
  `cloth.settle.max_delta_mm` を観測し、`SETTLE_SLEEP_EPS` 引き上げを判断する。
  A/B ノブ: `VULVATAR_AUTO_DAMPING` / `VULVATAR_AUTO_ITERATIONS`、diagnose_cloth に
  `COLLIDERS_NONE` / `COLLIDERS_NO_THIGH` / `WIND_ON` を追加。
- クリアランス (アンチ貫通アンカー) は `src/asset/clearance.rs` の Phase 1/2/3。
  Phase 3 は containment スロットに clearance-mode アンカーを入れる cross-region
  (ジャケット裾↔スカート、スカート↔下着)。アンカー実装を変えたら
  `VVT_CACHE_VERSION` を上げること (v17 = 放射シルエット拡張)。
- **Settle-sleep (idle z-fight 対策, 2026-09-15)**: spring チェーンと cloth
  (GPU/CPU 両バックエンド) は「quiet (< 100 µm/step) が 5 ステップ続いた
  AND 全入力が最後のステップと同一」ならステップ全体をスキップし、頂点
  ストリームを bit-stable に保つ (ソルバーの残留振動 ~10 µm でも、~1 mm
  離れた 2 面の深度順位を反転させて点滅するため)。spring の wake キーは
  `simulation/spring.rs`: chain root world 位置 / 関数パラメータの scene
  gravity / dt / tuning / scene collider ハッシュ / 関節位置での body-SDF
  サンプル (許容 1 mm — 上流にない体部位の手の接近と SDF garment レイヤー
  を検知するため)。**`sleeping` はラッチではなく quiet ストリークから毎
  ステップ再評価** (初版はラッチで、ドライバ停止時に揺れ途中で凍るバグが
  あった)。GPU cloth は `ClothState::settle` が app 側ゲート
  (`app/render.rs` `update_cloth_settle_gate`、入力は f32 全ビットの
  FNV フィンガープリント) でスナップショットの `substeps` を 0 に強制し、
  既存の frozen-frame 契約 (dispatch/pin 書き込み/version bump 全スキップ)
  で `cloth_pos_ssbo` を bit-stable 維持。quiet の給源は 1 フレーム遅延の
  位置 readback (`fold_cloth_readback_settle`)。CPU cloth は
  `cloth_solver::step_cloth_*` 内で同ゲート。観測:
  `debug_gui.json` の `scene.cloth.*.settle` (sleeping / quiet_frames /
  max_delta_mm / suppressed_this_frame)。回帰テスト:
  `tests/settle_verify.rs` (公開 API 経由の統合テスト — `cargo test --lib`
  が tracking 側 WIP で壊れている間も実行可能な実行版を兼ねる) ほか
  `simulation/tests.rs` / `cloth_solver/tests.rs` / `app/render.rs` の各
  テストモジュール。

## Tracking (fusion estimator)

- 本番プロバイダは `FusionProvider` (`src/tracking/fusion/provider.rs`) の一本のみ。ファクトリは `create_pose_provider` (`src/tracking/provider.rs`、`inference` feature 経由)。現行仕様は `docs/tracking-v2-design.md` (As-Is のみ、経緯は書かない)。
- **姿勢検出器は YOLO26-pose** (`src/tracking/detector/yolo26.rs`、body-17 を COCO-Wholebody 133 の他ブロック score 0 で満たす)。重みソース `yolo26{n,s}-pose.pt` はリポジトリ直下だが**実行には ONNX エクスポートが要る** (`models/` は ignore 済みでエクスポート物は commit されない)。新環境での手順:
  ```bash
  python -m venv "$TEMP/yolo_export_venv"
  "$TEMP/yolo_export_venv/Scripts/pip" install ultralytics onnx onnxslim
  "$TEMP/yolo_export_venv/Scripts/python" -c "from ultralytics import YOLO; YOLO('yolo26n-pose.pt').export(format='onnx', imgsz=480, opset=17, simplify=True)"
  mv yolo26n-pose.onnx models/yolo26n-pose_480.onnx   # ファイル名の _480 が入力サイズの契約
  ```
  無い場合 tracking は「YOLO26-pose model not found」でブロッキングエラーになる (自動化: dev.ps1 Setup グループ「export yolo26-pose ONNX」— ultralytics venv を $TEMP に作って n/s 両方を `_480` 契約でエクスポート)。実測 (2026-09-16, 無競合): 単体ベンチ (diagnose_yolo26_pose, s1789360037 261f, DirectML) 中央値 5.1ms / p95 19ms、fusion リプレイの provider 全体 ~20-30ms = 実効 30Hz 以上 (RTMW3D 時は検出 ~14ms・全体 ~30ms・尾 390ms)。4録画 (デスク+正面3) の品質は YOLO11n と互角だが session 毎に優劣が分かれる (namaste 被覆ストレスは 26 が劣位 std 30° vs 15° — 既知の崩壊症例、ライブwatch項目)。
- **hand モデルバックエンド (2026-09-18 RTMPose two-stage 一本化 — MediaPipe hand は削除済み, 558744e)**: hand は `fusion/hands.rs` の `HandBackend` (RTMPose 専用)。**既定は rtmpose**、`VULVATAR_HAND_BACKEND` env は廃止 (6 録画 A/B ゲート合格を受けて MediaPipe hand-landmarker コードと fallback は 558744e で削除 — 参照実装は git history)。モデル契約は 3 本 (`models/*.onnx` は ignore 済み、再現は再学習; export 無しは hand チェーンが起動しないブロッキングエラー): (1) `models/rtmpose-m-hand_256.onnx` — SimCC `(1,21,512)`×2、ImageNet 正規化 RGB。**SimCC 座標は bins 正規化後に crop size を掛ける** (model_size 割算のままだと 21 点が crop 原点に潰れる)。(2) `models/rtmpose-hand-presence_64.onnx` — MP ラベル蒸留の hand/背景 64px 分類器 (**hands.csv の cx,cy は crop 左上座標**の罠あり)。学習は **v5** (`scratchpad/make_hand_presence_v5_data.py` → datasets/hand_presence_v5): v3 (palm 峰值中心窓 + `datasets/frontal_labels/` 目視ラベル) に**配置拡張** (中心 ±25px・positive 窓 40-90px) を足し、**negative も全てランタイム窓スケール (40-90px) に統一**したもの。v4 まで nohands の random negative は 60-140px 窓で、ランタイムの 48px クエリではシャツ折り目がより手らしく見えるという mismatch が nohands FP の一因だった (rt37 85/48 → rt38 41/51 → rt42 31/30)。v3 までは訓練窓の**正確な配置に過適合**しており、同一の手でも 20px 離れると 1.00↔0.00 で振れ、窓を大きくすると calibration cliff で落ちる (clasp R: 48px 窓 0.994 / 67px 窓 0.001)。**nohands は negative 専用** — MP の「ロック」は袖の幻覚で、v3 はそれに 5.0 重みを付けて教えていた。val pos-acc 0.999。再学習したら必ず desk A/B。(3) `models/rtmpose-hand-palm_256.onnx` — 全フレーム letterbox 256 → 16×16 手首ヒートマップ。**ターゲットは必ず両手首** (`datasets/hand_palm_v2`、`scratchpad/make_hand_palm_data_v2.py`): v1 の単一手首ターゲットは clasp フレームで相方の手を**陽に background 訓練**していた (R の cell が 0.11 — clasp R 5/782 の根本原因、2026-09-18 fix)。mining ラウンドは正面付きデータでは**禁止** (天井照明 negative と持ち上げた正面手が同一 cell で衝突、namaste recall 崩壊) — `train_hand_palm.py` は `PALM_DATA=hand_palm` の時しか mining しない。
  **runtime パイプライン (2026-09-18 確定)**: presence は **presence net が唯一の権威** — provider が全候補を再スコアする。窓スケールは **base 96 = 48px 窓に固定** (v5 net は配置ロバストなのでどの中心でも読める)。palm 候補は **peak + SimCC デコードの 2 中心で max** (chin は peak 0.89/decode 0.00、clasp R は逆 — どちらか片方の中心は必ず外れるポーズがある)。他候補はデコード手首中心 + crop size 由来窓 (従来通り)。SimCC sharpness は crop size 依存で誤窓デコードと**逆相関**のため排他。幾何ゲート (掌長/指関節幅比 [0.15,0.9]) は estimate 内に残る。palm 候補は候補チェーン末尾に最大 2 つ (top-1 峰值 × 0.30/0.50×frame幅)。**峰值→スロット割当は det wrist ピクセル (score≥0.5) をアンカーにし、無信頼時のみ FK 射影にフォールバック** — FK は未追跡スロットで相方の手に向くことがあり、さらに hand==1 の filter は当初**反転していた** (自分のアンカーより相方近い峰を採る — clasp R 枯渇の第二原因)。深度ゲート palm ティア: 計測アンカー 0.25m / (net≥0.70 && FK 0.45m) / (net≥0.50 && FK 0.25m) / 前腕レイ (**t≤2.5 前腕長新に物理上限** — 無限 t だとレイ上の 5m 背景を通す、namaste 4.9m jump 実測)。publish は **全ソース共通 0.5**、獲得 streak は**証拠条件付き** (presence≥0.93 → 2 フレーム、未満 → 3 フレーム): 公開スコア分布は真の手 95-97% が ≥0.93 / fold FP は 28% (median 0.85) できれいに分離 — 平坦 streak 3 は nohands FP を消す代わりに chin duty 0.90→0.82・namaste L 0.97→0.53 を出し、平坦 2 は nohands 2× を残した。soft hold 帯は 0.20-0.50×5 フレーム (desk 実測では no-op — dip は hard drop)。det wrist を第 3 の窓中心にするのは**禁止** (chin/namaste dip に効果ゼロで nohands R 24→45 に悪化、実測)。chirality は 2D 幾何推定で net≥0.85 の palm 候補は hard veto を回避 (chin 拳は MP 学習頭と逆極)。
  **A/B 実績 (rt42 = 既定 flip の根拠, ab_mp_* 比)**: **clasp L 780 / R 781、duty 1.00/1.00、snaps 0/0 = MP 完全同等 (修正前 R 5)**。chin L 436 duty 0.90 (MP 0.88 超)、R 0 = MP 同等。palms L 17 duty 0.89 (MP 2/0.26)、namaste L 11 / R 8 duty 0.97/0.75 dupes 0 (MP 6/8 0.66/0.72)、wave ≈MP (3/8 0.26/0.38 vs 4/9 0.31/0.40)、nohands 31/30 (MP 20/31 — 同等圏)。**既知残課題**: chin snaps 14 vs MP 4 — 2D 位置は完全安定で、snaps の正体は**手/背後の顔面の深度サンプル flip** (深度側の残課題)。solve 原価: 候補全 Fail フレームで 50-70ms — ライブ 30Hz で要観測。
- **顔 = MediaPipe ランタイム削除済み (2026-09-22, 最後の MP 消費者)**: `face_mediapipe.rs` (FaceMeshV2 478 + BlendshapeV2 52) と `VULVATAR_FACE_BACKEND` は削除。唯一の顔経路は RTMPose-face sidecar (下の節)。幾何ヘルパー (`FaceBbox` / `derive_face_bbox`) は `tracking/face_bbox.rs` に抽出、GPU budget の `facemesh_cpu_ep` ノブと `FACEMESH_EP_CPU` も同時廃止。MP 戻しは git history (84406f6 まで) のみ。残る開始項目: (5) head pose 定数校正 → (6) イベントレベル A/B (意図的瞬き・口開閉の較正録画 — 既存録画には含まれない)。
  **オフライン由来 (runtime が消えても models/ の入力はこれに依存する)**: `models/mp_canonical478.npy` (MP 478 canonical メッシュ) と `models/mp_wflw98_idx.json` (WFLW98→MP478 対応) は `scratchpad/build_face98_mapping.py` が生成 — WFLW test 149 枚 → similarity-ICP 整列 → 距離カーネル重み付き投票 → 全局貪欲割当で **98/98・衝突ゼロ** (in-sample dist_rel median 0.26%、holdout 3.4%、iris anchor 正解)。blendshape MLP (`rtmpose-face-blendshape_98.onnx`) は `scratchpad/make_face_distill_data.py` が MP FaceMeshV2+BlendshapeV2 (PINTO 410/390 — **dev.ps1 配布対象外、再蒸留時は手動取得**) で作った教師ラベル `datasets/face_distill/` から学習。MP FaceMesh conf ヘッドは写真ドメインで死んでおり (`Identity_1` は常時 ~0 の死んだ logit、真値は `Identity_2`) — MP 系ヘッドをofflineで再利用するときの注意。
- **RTMPose-face sidecar は runtime 既定 (2026-09-18〜) — 稼働には dev.ps1 provisioning が要る**: 既定チェーンは body 検出の顔点 → 256 zero-pad crop → `scripts/face98_service.py` (LiteRT, `models/rtm_face_fp16.tflite` — litert-community/RTMPose-Face-WFLW-LiteRT の Apache-2 ready-made、Rust 側は tflite を読めないため Python sidecar 経由) → 98 WFLW → 幾何式表情 + 蒸留 MLP (`models/rtmpose-face-blendshape_98.onnx`, 任意 — 無ければ幾何式のみで warn)。**dev.ps1 が面倒を見る**: setup/run エントリは `Install-Models` 経由で (a) tflite を HF から DL、(b) `tools\face98-venv` (ai-edge-litert + numpy) を冪等構築し、run エントリは `VULVATAR_FACE_SIDECAR_PYTHON` をこの venv python に設定する。env 未設定時は PATH の裸 `python` を spawn する — litert が無いと sidecar は即死し、**毎フレーム respawn + error ログが 30fps で流れる** (fail_run ≥3 でも respawn 継続) ので、生 python で動かす場合は litert 導入済みか確認。インストーラ (.iss) は tflite/npy/json/script を同梱済みだが **sidecar 用 Python 同梱は未解決** — python 無しの導入環境では expressions 無効で起動する。YOLO26-pose ONNX エクスポートも dev.ps1 Setup グループの「export yolo26-pose ONNX」で実行可能 (下記 Tracking 節の手順を自動化)。
- **主観測は密深度表面点**: シルエット内部をサンプリングした点群 (~1,300 点 @640×480, stride 8) をカプセルモデル表面までの距離で拘束 (Cauchy×GNC、AABB 事前棄却、trunk/head 優先)。2D キーポイントは表面だけでは決まらない自由度 (左右の判別・表面に沿う位置・手) の補助に降格している。腕は既定で表面主張に参加しない (`VULVATAR_DENSE_ARMS` で有効化)、脚はシルエット高さと膝追跡の条件付き、手は常に除外。
- オフライン検証: `cargo run --bin diagnose_fusion_replay -- <dir> [out_dir] [--render N] [--avatar]`
  (`*_color.png` + `*_depth_mm.npy` ペアのディレクトリを食う = `diagnostics/depth/*_replay` と sequence recorder の `diagnostics/sessions/<id>`。出力先の既定は `diagnostics/fusion/<dirname>`)。
  summary に 胴 yaw std / 肩深度参照との差 / メトリック関節残差 / 手首ジャンプ・snap / data-σ duty / root ジャンプ / seed wins・再捕捉・cov failures / hand crops、
  `frames.csv` に毎フレームのコスト内訳・σ・形状。`VULVATAR_REPLAY_VISDUMP=1` でキーポイント単位の詳細ダンプ。
  リプレイは**名目 D435 intrinsics (解像度別) と名目 30fps クロック**で動く (録画の実 intrinsics・タイムスタンプは読まない)。結果はその前提で読むこと。
- アブレーション: `VULVATAR_FUSION_NO_DENSE` (密表面)、`NO_3D`、`NO_SURF` (キーポイント直下深度の sparse 表面項)、`NO_BURNIN`、`NO_CHESTYAW`、`NO_ORI`。
  `VULVATAR_FUSION_OBSDUMP=<frame>` で観測とモデルの対応ダンプ (`999_999` = 最初のフレーム)、`VULVATAR_FUSION_KEEP_CLOUD=1` でオーバーレイに表面点を描く。
  旧ゲート群 (border cull / reach / leg gate / coherence) は `VULVATAR_FUSION_OLDGATES` を付けた時のみ有効。完全な一覧は `docs/tracking-v2-design.md` §7。
- 新しい症状に場当たり的なゲートを足さない。どの残差・事前・σ が誤っているかをベンチで測ってから直す。
- ライブ: `VULVATAR_AUTOSTART_TRACKING=1` で起動時にカメラ開始 (realsense2.dll を PATH に)。
  `debug_state.json` の `rig` ブロックに quality / 主要ボーン σ・data_sigma / root / `diag`
  (solve_ms, est_ms, iters, cost, n2d, n3d, med_2d_px, lost_events, seed_wins, cov_failures, hand_crops, face68_learned, mesh_learned, shape_frozen)。

## Live debugging (実機計測 — アプリを止めない)

ユーザーが「おかしい」と言ったら、**アプリからカメラを奪わずに** live debug
channel で計測する。アプリがカメラを掴んでいる間は pyrealsense2 等で別プロセスから
開いてもフレームは来ない (open は通るが `wait_for_frames` がタイムアウトする)。

- 有効化: `C:\ProgramData\VulVATAR\debug.on` (空ファイル、約2秒で反映、再ビルド不要)。
- `debug_state.json` — 推論フレーム毎 (tracking worker)。`kp` (COCO 17 の 2D+score)、
  `kp_mcp` (両手ブロックの MCP 4点)、`torso`/`arm` の各関節 `{p, c, d}`
  (`d` = サンプラーが実際に返したカメラ空間深度 m — 「特徴点に正しい深度が付いたか」の一次証拠)、
  `root` / `root_is_hip` / `metric` (anchor_cam_m, mpsu, ref_span_m)、`face`、`mesh_c`、`face_dbg`。
- `debug_avatar.json` — ソルバー後のアバター主要関節ワールド座標 (`seq` で新フレーム検出)。
  ユーザーが見ているものの数値化はこちら (rest 判定は Hips y ≈ 0.845 等)。
- `debug_gui.json` — GUI フレーム毎。`paused` / `frame_count` / `sim_substeps` に加え
  `scene` ブロック = **今表示しているもの** (アバター毎の source ファイル・primitive 数・
  cloth スロット・カメラ/出力設定・fade opacity) と render ヘルスカウンタ。
- `debug_render.json` — レンダーフレーム毎 (render thread)。**描画の課題**の一次情報:
  `kind` (`ok` / `error` / `gpu_exclusive_skip`)、`input` (描画依頼の内容) と `result`
  (`RenderStats`・ピクセル有無・export pool) の対比、累積 error / pixel 無しカウンタ。
  「ビューポートが古いまま」が render error なのか exclusive skip なのかパイプ未到達なのかは
  この `counters` で決まる (ログには出ない)。
- `debug_depth.bin` — 毎秒1回、フル解像度アライン済み深度 (`VDBD` 32B ヘッダ + u16 mm)。
  任意キーポイント直下の生深度ピクセルの監査用。`debug_camera.bin` は 640px 幅 RGBA (`VDBG`)。
- 計測手順: Python ポーラで `debug_state.json`/`debug_avatar.json` を 5-10ms 間隔で読み
  (`frame`/`seq` でデデュープ)、jsonl に貯めて統計 (中央値・sd・フレーム間ジャンプ)。
  ユーザーに「20秒ポーズをキープ」と依頼してから回す。スクリプトは `scratchpad/` に書く。

アプリ稼働中の本 target の `cargo build --features realsense` は **通常は成功する**。
かつては realsense-sys の build.rs が毎ビルド再実行され、`target\debug\deps\realsense2.dll`
へのコピーが稼働プロセスのロックで失敗してビルドが落ちていたが、vendor patch
(docs/realsense-build.md「The vendored realsense-sys patch」参照) で build.rs の再実行は
「build.rs 変更・SDK ヘッダ/DLL 更新・pkg-config 環境変数変更」時に限られ、DLL コピーも
そのときしか走らない。2026-09 実測では、稼働プロセスは `deps\realsense2.dll` をロック
していなかった (exe と同じディレクトリに DLL はなく、PATH 上の SDK コピーをロードすると
推定)。診断バイナリを本 target と完全に分離して回したいときは別 target でビルドする:

```powershell
$env:CARGO_TARGET_DIR = "$PWD\target-test"
$env:SHADERC_LIB_DIR  = "$PWD\target\debug\build\shaderc-sys-<hash>\out\lib"  # 本targetのキャッシュ流用 (無いと CMake 非互換で from-source が死ぬ)
# + docs/realsense-build.md の3環境変数
cargo build --features realsense --bin diagnose_fusion_replay
```

アプリ本体の exe リンクも稼働中に通る (cargo が rename 置換するため稼働プロセスは
旧イメージのまま実行継続)。ただし新バイナリが反映されるのは次回起動からなので、
稼働中のアプリに変更を反映したいときは再起動してもらう。

## 実機不具合の調べ方 (最優先)

**ライブパイプラインの不具合をコード読解だけで診断するな。** 症状 (「体が暴れる」「棒立ちで固まる」) からコードを読んで因果の物語を組み立てると、もっともらしいが間違った結論に達する。特に `reference_span_m` のような載荷スカラーを未計測のまま変更すると、症状が別の症状に化けるだけで前に進まない。**まず録る。**

### 1. アプリ起動中なら — ライブデバッグチャネル

フラグファイル `%ProgramData%\VulVATAR\debug.on` を作ると 2 秒以内に有効化 (リビルド不要)。同ディレクトリに毎フレーム上書きで出る:

| ファイル | 書き手 | 中身 |
|---|---|---|
| `debug_gui.json` | GUI スレッド (**全ゲートの手前**) | `paused` / `avatars_loaded` / `tracking_enabled` / `frame_count` / `seq` / `sim_substeps` (0 = そのフレームは固定ステップの積算が足らず spring solver 未実行。1フレームだけの髪めり込みはまずこれと突き合わせる) / `scene` (何が表示されているか — `Application::scene_debug_snapshot`: アバター毎の source ファイル・primitive/頂点数・cloth スロット・アニメ状態、カメラ/出力設定、fade opacity、GPU budget、render ヘルスカウンタ) |
| `debug_state.json` | トラッキングワーカー | 2D キーポイント + source 関節 (位置・信頼度) + face pose |
| `debug_avatar.json` | `run_frame` 内 | ソルブ後のアバター world 関節 + head 軸 |
| `debug_render.json` | レンダースレッド (RenderFrame 処理毎) | `kind` (`ok` / `error` / `gpu_exclusive_skip`) + `input` (描画を依頼した内容: instances / mesh prims / cloth deforms / body SDF plan 数 / 出力 extent / export mode) + `result` (`RenderStats` instances・meshes・materials・cloth instances・export pool スロット、ピクセル有無、handoff fallback) + `counters` (累積 render errors / GPU exclusive skip / pixel 無し結果 / render_cpu_ms EMA)。`input` と `result.stats` の食い違いで「依頼したのに描かれていない」をローカライズする。render errors はここを見る (ログだけでは外部ポーラが読めない) |
| `debug_camera.bin` | トラッキングワーカー | カメラ RGBA (32 byte ヘッダ `VDBG`) |

「アバターが動かない」の切り分けは `debug_gui.json` の 2 値で決まる。`seq` は毎 GUI フレーム、`frame_count` は**非 pause フレームのみ**進む:

- `seq` 進む・`frame_count` 止まる → 一時停止中 (Space が `TogglePause` に**修飾キーなし**でバインド)
- 両方進む・`avatars_loaded: 0` → アバター未ロード
- 両方進む・`avatars_loaded: 1` → 原因は `run_frame` より下流

これらは値であって解釈ではない。推測する前に読め。

### 2. キャプチャ → 無人リプレイ (品質改善の本線)

人間がカメラの前に座るのは**一度だけ**にする。録ったら以降は無人で何度でも回す。

**リプレイ用フルレート録画 (sequence recorder)** — アプリ稼働中にフラグファイル `%ProgramData%\VulVATAR\record.on` を作る (2 秒ポーリング、リビルド不要)。ファイルの中身にフレーム数を書ける (空なら既定 600 frame、RAM 1.2 GB 上限)。**フラグは録画開始時に消費される (削除。削除失敗時は mtime/len 同一性で記憶)** なので再録画には再作成が要る — 1 プロセスで何度でも録画可能で、放置フラグが連続録画を引き起こすこともない。フレームは RAM にギャップレス確保され、終了後に別スレッドで `diagnostics/sessions/s<unix>/` へ `f#####_color.png` + `f#####_depth_mm.npy` + `meta.jsonl` (実 intrinsics・タイムスタンプ) として吐かれる。

```powershell
cargo run --release --bin diagnose_fusion_replay -- diagnostics\sessions\s<unix>   # 温度状態を継続して本番プロバイダに流す
```

**published skeleton の録画 (session recorder)** — 環境変数で起動する:

```powershell
$env:VULVATAR_RECORD="1"; $env:VULVATAR_RECORD_RAW="1"
cargo run                       # 再現させてトラッキング停止 (生フレーム収集は既定 900 frame = 30 秒で打ち切り、アプリは続行)
```

`diagnostics/session_<unix>/` に出る:

- `pose.jsonl` — published `SourceSkeleton` の全フレーム時系列。各関節に provenance タグ (`o` = `O`: 深度実測 / `E`: 深度が穴で骨長レイ外挿) と `metric` (`reference_span_m` / `mpsu` / アンカー)
- `frame_NNNNNN_color.bmp` + `_depth_mm.npy` — 生フレーム。BMP (非圧縮) なのは dev プロファイルで PNG encode が 30fps に追いつかず静かに間引かれるため (実測: `cargo test -- --ignored --nocapture frame_write_throughput`)。`diagnose_fusion_replay` は PNG ペアしか読まないので、BMP セッションは `analyze_session` 用
- `manifest.jsonl` — **実 intrinsics とデバイスタイムスタンプ**

```powershell
cargo run --bin analyze_session -- diagnostics\session_<unix>                   # summary.md に判定
```

`analyze_session` は「飛んだ関節は測ったのか、でっち上げたのか」を両端の provenance で分類する。`E` 側に偏れば骨長 (=`reference_span_m`) が疑い、`O` 両端に偏れば深度サンプリング側。L/R ブロック反転とグローバルスケール異常も別枠で出す。

**環境変数**: `VULVATAR_RECORD_JUMP` (ジャンプ検出閾値、既定 0.35 source unit)、`VULVATAR_RECORD_RAW_FRAMES` (フレーム上限、既定 900)、`VULVATAR_RECORD_RAW=N` (N フレームおき)。

**注意点**:
- ディスク 1.5 MB/frame (30 秒で約 1.4 GB)
- drop が出たら `session_record` が warn を出す。**その録画は使うな**

## Architecture

- GUI thread: eframe/egui — `src/gui/mod.rs` (`GuiApp::update`)
- Render thread: Vulkan via vulkano — `src/renderer/mod.rs` (`VulkanRenderer::render`)
- Communication: command queue `sync_channel(2)` + result mailbox (`ResultMailbox`) in `src/app/render_thread.rs`
- Output: shared-memory writer on a worker thread — `src/output/`

See `docs/architecture.md` and `docs/threading-model.md`.

## Profiling

See `docs/profiling.md` for instrumentation recipes and known bottlenecks.

Profile logs go in `profile/` (gitignored):
```bash
RUST_LOG=vulvatar=info cargo run 2> profile/run_$(date +%Y%m%d_%H%M%S).log
```
