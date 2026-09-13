# 設計契約とレビュー項目の状態 — 2026-09-13 実装パス

[quality-improvement-plan.md](quality-improvement-plan.md) 第9節のレビュー項目
(R1〜R8) と第3節の設計契約について、コードに反映した契約・採用した設計判断・
各項目の状態を記録する。計画書が計画と完了条件を管理するのに対し、本書は
「現行コードが何を保証し、何を保証しないか」を管理する。

状態は計画書第10節の分類に従う: **修正済み** (コード変更+テスト済み) /
**部分対応** (契約を限定、恒久対策は残す) / **反証済み** / **未再現・未実施**。
静的レビューと単体テストのみで実機 GPU・カメラ計測は行っていない。

---

## 1. 境界ごとの契約 (実装に対応付けた形)

計画書第3節の契約表を、実在する型・関数に対応付ける。監査結果が既知のものだけ
記載する (未監査の境界は「未監査」と明記)。

### 1.1 推定と配信 (Phase B の中核区分)

| 区分 | 実装上の所在 | 現在の契約 |
|---|---|---|
| センサ観測 | `tracking::SourceSkeleton` (provenance タグ `O`/`E`) | 深度実測か骨長外挿かがタグで区別される。分析系はこのタグを正として扱う |
| 推定状態 | `FusionProvider` の rig (関節回転+σ) | 観測と事前の融合結果。σ=推定不確かさの意味を持つ |
| 配信用姿勢 | `retarget::apply_rig_pose` + hold/fade (`app/render.rs` `TRACKING_HOLD_WINDOW`) | 未観測肢の保持・復帰は配信側の選択であり、推定の確信ではない。hold 中は confidence を線形減衰させ、推定値の再利用を「確信」として偽装しない |

**未監査**: 休止ブレンド (rest 判定) と data-σ の意味分離、crop/可視性
フィードバックの回復経路。Phase A〜B の計測を要する (計画書通り)。

### 1.2 物理と描画 (今回の修正で確立した契約)

| 契約 | 内容 | 実装箇所 |
|---|---|---|
| substep 実行頻度 (R1) | GPU cloth の1 substep = CPU `step_cloth` 1回と同じ処理列 (verlet → XPBD 反復 → 自己衝突 → カプセル衝突)。substep 数は CPU と同じ `SimulationClock::advance` 値を使い、`clamp(0,8)`。**0 substep = 物理凍結フレーム**: GPU は verlet・拘束・衝突・ピン書き込みの一切を dispatch せず、SSBO は前フレーム状態を保つ | `compute_prepass.rs` 記録半分の substep ループ、準備半分の FROZEN-FRAME CONTRACT |
| 法線の再計算時点 | 法線は substep ループの**後**に一度だけ再計算する (最終 substep の法線だけが描画に使われるため)。CPU が毎 substep 計算するのは最後の1回のみが有効であり、GPU はそれを1回に圧縮した同等実装 | `record_compute_prepass_planned` の S3.1 ブロック |
| containment 参照面の時刻 (R3) | containment 子は親の**前フレーム最終 VBO** (`containment_prev_vbo`) を読む。履歴バッファは全 transform dispatch の後の `copy_buffer` で毎フレーム公開され、dispatch 順序に依存しないことがデータ構造で保証される。初回フレームは未初期化で、シェーダの sanity band / NaN ガードにより clamp は不活性 | `TransformGpuData.containment_prev_vbo`、`materialize_parent_vbo(as_containment=true)`、記録半分末尾の copy |
| clearance 参照面の時刻 | clearance 親は同一フレームの freshly-skinned VBO を読む (Kahn 順序がグラフ辺で保証)。 | `prepare_compute_prepass` の依存グラフ |
| 描画側補正の上限と監査 (R2) | clearance/containment の描画補正は物理状態へ書き戻されないため、1 anchor あたりの適用変位を `MAX_RENDER_CORRECTION_M = 0.25 m` に制限。適用量の合計は `out_v.position.w` に telemetry として排出され (頂点シェーダは .xyz のみ読む)、`VULVATAR_VBO_AUDIT=1` で final VBO が host staging に copy され、frame fence 後に `VboAuditEntry` (頂点数/NaN数/補正max/p95/max半径) として読み戻され `debug_vbo_audit.json` に流れる。監査対象は clearance/containment anchor を持つ primitive のみ | `transform_cs` GLSL、`TransformGpuData.audit_staging`、`read_vbo_audit`、`dump_vbo_audit` |
| GPU cloth readback の配送 (R4) | readback 行は `instance_id + mesh_id + primitive_id` の三つ組が一致する `ClothState` にのみ適用される。instance 未記録の行は適用しない。owner は `ensure_cloth_gpu_slot` が毎フレーム刻印する | `ClothReadback.instance_id`、`cloth_readback_matches`、`apply_cloth_readback` |
| bend 拘束のモデル (R7) | 3点 edge-angle hinge (p0 における2辺のなす角)。2三角形の dihedral では**ない**。補正は自由な翼頂点 (p1,p2) のみを動かし、翼の移動は角度を 1/|辺| の線形レートで変える。ヒンジは動かさない (距離拘束が修復する)。pinned 翼が2枚なら不動点。共線・零長は不活性 | `cloth_solver/constraints.rs` `project_bend_constraints` モデル契約コメント |
| auto-cloth weld (R6) | 位置 (0.05 mm 量子化) による weld は物理粒子を統合しない — round-robin 距離拘束で保持する。シーム分割頂点は群内で同一位置を保つ**保証ではなく傾向**であり、同一位置の別レイヤーは1群に統合される。層の真の分離は Phase C の粒子/描画頂点分離設計で扱う | `auto_cloth.rs` `WeldGroups` / `round_robin_weld_constraints` コメント |
| DQS の入力前提 (R8) | `transform_cs` の DQS はスキニング行列が**剛体** (直交3x3、det=+1) であることを前提とする。非一様 scale は回転抽出で無視され (LBS とは乖離)、mirror は保存されない。実素材 Yumeka の全スキニング行列は剛体であることをテストで確認済み | `pipeline.rs` `dqs_contract_tests` |

### 1.4 観測された規約の割れ (要監査、今回は変更しない)

`Mat4` の格納規約が呼び出し元によって異なる: `gpu_pin_targets` /
`apply_pin_targets` は `m[col][row]` (数学行列の転置格納、翻訳は `m[3][0..3]`)
で扱う一方、`app/render.rs` の `sensor_camera_tests::mul` と
`build_view_matrix` 系は `m[row][col]` で扱う。それぞれの閉じた範囲では
自己無矛盾だが、同じ型に二つの規約が併存している。R8 の DQS ミラーテストは
`m[col][row]` 側 (GPU が見るメモリ配置) に統一して書いた。
次にこの型を触る者は規約の統一を最先の課題とすべき。

---

## 2. レビュー項目ごとの状態と証拠

| 項目 | 状態 | 実施内容 | 証拠 |
|---|---|---|---|
| R1 GPU衝突の実行頻度 | **修正済み+実機実行確認** | self-collision・capsule collision を substep ループ内へ移動 (CPU の step 順と同一)。0-substep は GPU dispatch 全面停止に。オフスクリーン GPU smoke (`diagnose_cloth`, `VULVATAR_CLOTH_GPU=1 VULVATAR_CLOTH_SELFCOL=1`) で新 dispatch 順序が実機 Vulkan でエラーなく実行されることを確認 (証拠: `diagnostics/cloth_vbo_audit/`) | `compute_prepass.rs`。substep 数別の貫通深さ比較 (0/1/2/5) は未実施 — E3/E7 実験として継続 |
| R2 最終頂点と物理状態の分離 | **部分対応 (検出経路実装+初回実測済み)** | 描画補正に 0.25 m/anchor の上限を導入し、補正量を `position.w` telemetry として排出。`VULVATAR_VBO_AUDIT=1` で final VBO の統計 (NaN数・補正 max/p95・最大頂点半径) を `debug_vbo_audit.json` と smoke 出力で取得できる。**初回実測 (Yumeka rest + sway, オフスクリーン)**: prim 5 (9,372頂点) corr max 185 mm / p95 26.5 mm、prim 12 スカート (2,460頂点) max 53 mm / p95 30 mm、NaN 0、clamp 飽和なし — これが退行評価のベースライン。段階別 (布直後/clearance直後/containment直後) の**分離** capture は未実装で、現行の監査は最終 VBO の合計のみ | `transform_cs` GLSL、`read_vbo_audit`、`diagnostics/cloth_vbo_audit/cloth_simulation_summary.md`。E7 実験の実施が恒久対策判断の条件 |
| R3 containment 参照面の時刻 | **修正済み+実機実行確認** | スロットに `containment_prev_vbo` 履歴バッファを追加し、全 dispatch 後の `copy_buffer` で前フレーム最終状態を公開。子の descriptor set は履歴バッファを束縛。shape key に copies を追加し、collide/selfcol セットのハッシュ漏れ (潜在バグ) も修正。実機 smoke で copy が実行時 usage 契約 (`TRANSFER_SRC`/`TRANSFER_DST`) 違反なく動くことを確認 — **usage 指定漏れを一つ検出・修正した (単体テストでは検出不可能だった)** | `transform_cache.rs`、`compute_prepass.rs`、`frame_plan.rs`。相互依存ケースの実機比較 (E5/E6) は未実施 |
| R4 readback 配送 | **修正済み** | `ClothReadback` に `instance_id` を追加、スロットに owner 刻印、三つ組一致のみ適用。同一アセット複数 instance で transform_cache が `(mesh_id, primitive_id)` 共有の問題は残存 (renderer 自体の instance 分離は Phase C 課題として記録) | `cloth_cache.rs`、`app/render.rs` `cloth_readback_matches` + 単体テスト4件 |
| R5 衣服プローブの測定対象 | **修正済み** | プローブを physics 前/後の2ステージ化 (`pre_physics`/`post_physics`)、ハードコードされた `circle.056` を廃止し cloth 対象 primitive を ID で直接測定。各 primitive に clearance/containment 親 ID を同梱 (§10 の対応表要件)。`measured` フィールドに測定範囲の限定 (GPU DQS/SSBO/clearance/最終VBO は見えない) を明記。`sim_substeps`/`fixed_dt` を同梱し凍結フレームを区別可能に。最終 VBO 側は R2 の `VULVATAR_VBO_AUDIT` 経路で補う | `debug_channel.rs` `dump_costume_probe`、`app/render.rs`、`dump_vbo_audit` |
| R6 auto-cloth weld | **修正済み (実機で解決検証済み)** | weld を純関数化し5試験追加。**実機で R6 予測を検出→修正**: 位置weldのみでは群のコピーを保持する拘束が皆無 (拘束は常に異なる群間) で、コピーが自己衝突の一致epsilon (0.5mm) を超えて離れると全面スパイク崩壊 — **CPU も GPU も同一挙動** (`diagnostics/cloth_gpu_every_frame/` vs `cloth_gpu_noselfcol/`)。修正は二層: (a) 近接重複対 (< 0.5mm) を自己衝突から除外する epsilon 契約 (`SELF_COL_COINCIDENT_EPS_M` = GLSL `COINCIDENT_EPS_SQ` とミラー、単体テスト付き)、(b) **群内一致拘束** `intra_weld_group_constraints` — 各群のコピー対すべてに rest=0 の距離拘束を追加し、ドリフトを再接着すると同時に connected_pairs/CSR 経由で自己衝突からも除外 (XPBD の 1nm ガードにより一致中はゼロコスト)。修正後、GPU cloth + selfcol + weld が **CPU 収束状態と同等の正常スカート**を描画 (`diagnostics/cloth_gpu_selfcol_glued/`)。拘束数は +6,742 (9,200→15,942)、GPU 比例コスト許容範囲 | `auto_cloth.rs` テスト6件、`collision.rs`、`pipeline.rs` selfcol shader、`diagnose_cloth` (CLOTH_RENDER_EVERY/CLOTH_SUBSTEPS/CLOTH_SELFCOL_RADIUS knob、エッジ長統計 print) |
| R7 曲げ拘束 | **修正済み** | 計画書の指示通り方向・収束テストを先に作成し、旧実装が5件すべて失敗することを実証 (逆方向、pinned 翼でヒンジ発散)。その後ヒンジ移動を廃止した翼のみの線形化補正に再実装。8件全合格。GPU に曲げ段階は無いため将来の移植ではこの参照実装を使うこと (旧実装は禁止) | `cloth_solver/tests.rs`、`constraints.rs` |
| R8 DQS 入力前提 | **反証済み (実素材は剛体)** | GLSL 数式の CPU ミラーを4試験で固定: 剛体1関節で DQS≡行列変換、剛体ブレンドは距離保存、非一様 scale で DQS≠LBS (違反検知のアンカー)、Yumeka の全スキニング行列が orthonormal・det=+1 | `pipeline.rs` `dqs_contract_tests`。他アセットでの確認と mirror 行列の監査は残る |

## 3. 採用しなかった案と理由

- **containment 辺の依存グラフへの追加** (R3): clearance 辺との 2-cycle で
  cycle fallback が双方の拘束を無効化するため計画書が明示的に禁止。代わりに
  履歴バッファ+後置 copy を採用した。
- **ヒンジ頂点の運動量補償の保持** (R7): 翼 pinned 時に単独で誤差を増幅する
  (テストで実証) ため廃止。2次の位置ドリフトは距離拘束が毎 substep 修復する。
- **R2 の恒久対策 (補正の物理統合 / 中間バッファ readback)**: GPU 実測なしに
  どちらが正しいか判断できないため、上限 clamp + telemetry のみ先に導入。
  E7 実験の結果で決定する。
- **transform_cache の instance 分離** (R4 の残課題): renderer のスロット構造
  全体の再編を伴うため、本次パスでは readback 配送の三つ組一致で誤配のみ
  防ぐ。同一アセット複数 instance のGPU状態分離は Phase C の設計監査へ。

## 4. 未実施の試験と残る制約

- **実機 GPU 実験の部分実施**: オフスクリーン GPU smoke
  (`diagnose_cloth`, Yumeka v1.0.3) を複数構成で実行し、段階分離実験
  (E系) の最初の切り分けを完了した:
  - 新 dispatch 順序・履歴copy・audit copy が実機 Vulkan でエラーなく
    動作。この実行で (a) `transformed_vbo` への `TRANSFER_SRC` usage
    指定漏れ、(b) containment 履歴 push の欠落 (編集ミス) を検出・修正 —
    GPU 実経路試験が単体テストを補完する実例。
  - レシピを weld+round-robin 化 (`CLOTH_RENDER_EVERY` /
    `CLOTH_SUBSTEPS` knob 追加) し confetti を解消。以後の smoke は
    物理的正しさの判定に使える。
  - **E3/E8 相当の切り分け結果**: GPU cloth + 衝突 + weld (selfcol なし)
    = 正常スカート (CPU 収束状態と同等)。GPU cloth + selfcol = 全面
    スパイク崩壊 (CPU も同一 — ソルバー共有の意味論問題)。→ 群内一致
    拘束 + 一致epsilon の導入で **GPU cloth + selfcol + weld が正常
    スカートに復活** (`diagnostics/cloth_gpu_selfcol_glued/`)。
    自己衝突半径の感度も測定: エッジ中央値 24.9 mm に対し排除距離
    2·radius = 2 cm は設計プリーツ間隔 (5-10 mm) を超え、それ単独でも
    層を押し分ける — `CLOTH_SELFCOL_RADIUS` で再実験可能に。
  - **未実施**: substep 数別の貫通深さ定量比較 (0/1/2/5 — 環境変数は
    用意済み)、デスク系録画でのリプレイ検証、実アプリ (auto-cloth
    attach 経路) での修正効果のライブ確認、自己衝突半径の
    メッシュスケール自動導出 (現状 auto-cloth は 1.2 cm 固定 —
    エッジ中央値基準の導出が望ましい)。

### E4/E5 分離 + clearance 生成監査の結果 (同日・第2ラウンド)

`diagnose_cloth` にアンカー剥離knob (`VULVATAR_STRIP_CLEARANCE` /
`VULVATAR_STRIP_CONTAINMENT`) と CPU アンカー評価
(`audit_clearance_anchors_with`: base pose と生成姿勢の両方で deficit
を評価) を追加し、90フレーム連続描画で検証した:

| 構成 | prim 5 (セーター) 補正 | 判定 |
|---|---|---|
| 両anchorあり | max 181 mm / p95 26.5 mm | 胸に浮遊破片あり |
| clearance 除去 | **0.0 mm** | 胸の破片消失 — **prim 5 の補正は 100% clearance 由来** |

アンカー生成時点での評価 (生成姿勢 = ベース姿勢、両者同値) で:

- prim 5: min_clearance 2 mm に対し deficit **81.8 mm** → 実効
  clearance **-79.8 mm** (child が親法線の裏側 80 mm)。生成コードは
  raw_clearance が負でもアンカーを作るため、fold/谷間/体軸側の頂点と
  の誤対応が「毎フレーム 80 mm の一気押し出し」として顕在化する
  (= 報告画像の胸の浮遊破片・裂け)。
- prim 12 (スカート): radial-silhouette 拡張の min_clearance が
  **60 mm** (clamp 0.06 に到達) — ±2 y-bin / ±3 angle-bin の広域窓が
  腰・胸の太い帯の r_max を腰の細いビンに伝播し、密着する
  ウエストバンドに 60 mm の浮きターゲットを与える。**「腰回りの
  メッシュがめちゃくちゃ」の直接原因**。

実装した修正 (両方 VVT_CACHE_VERSION で無効化される):

1. **v18**: `raw_clearance < -0.02` (20 mm 以上の裏側対応) の
   アンカーを生成しない (`clearance.rs` Phase 1)。
2. **v19**: radial-silhouette の clamp を 0.06 → **0.02** (20 mm)。
   胸の shape-to-fit (≈15 mm) は温存し、腰の 60 mm ターゲットを除去。

**検証結果 (90フレーム、GPU、selfcolあり)**: 胸の浮遊破片消失、
セーター無傷、スカートのプリーツと腰の密着回復。
- prim 5 corr max **180 → 9.7 mm**、p95 **26.5 → 1.5 mm**
- prim 12 corr max **53 → 24.3 mm**、p95 **36 → 17.5 mm**
- NaN 0、90フレーム通して安定。証拠: `diagnostics/cloth_verify_clean/`
  (frame 030 / 090)。

生成側の生データで確定した幾何: worst anchor の parent は**体の左側面**
(bn = 純 -x)、child は**胸の前面** (z = +0.09) — normal 互換 (dot ≥ 0.1)
を通過する誤対応が生成姿勢の評価では raw ≥ -5 mm でも、実行ポーズでは
-81.8 mm になり、shader 防御 (deep-negative skip) がこれを無効化する。

### お腹露出 (セーター裾の乗り上がり) — 同日・第3ラウンド

ユーザー指摘「お腹丸見え」の再現と修正。 clearances 修正前後の両実行で
裾の上昇が発生 (clearance とは無関係) を確認後、ボーン加重ヒストグラム
診断 (`audit_sweater_bones`) で確定:

- **セーター (prim 5) の裾頂点の支配ウェイトが `Breast_1_L/R`
  (spring 駆動チェーン) に塗られている** (裾バンド 596/593 — 全体でも
  Chest を上回る最大加重)。sway/spring の動きで裾が胸チェーンに
  引きずられ、腹部が露出する。**アセットのウェイト塗り問題**。
- 修正: `demote_distant_breast_weights` (clearance.rs, import 時後処理)
  — breast ノードから 1.5×チェーン長より遠い頂点の breast ウェイトを
  最近傍の非 breast 祖先 (Chest) へ降格。Yumeka で **5,376 weights**
  降格 (Breast_1/2/3 → Chest、gate 90-174 mm)。v21 キャッシュ無効化。
- 検証: 90フレーム通してニットが腹部を覆い、露出解消
  (`diagnostics/cloth_belly_fixed/belly_fixed_timeline.png`)。
  frame 90 に裾トリム下の小さな白いスリバー残存 (軽微 — Phase C の
  粒子/描画頂点分離またはウェイト再塗りの対象)。
- 残課題: breast 揺れ表現は breast 近傍ウェイトのみで成立するため
  影響は限定的だが、リアクティブな胸揺れの変化を実機で確認すること。

```powershell
$env:PKG_CONFIG_PATH="$PWD\build-support\pkgconfig"
$env:LIBCLANG_PATH="C:\Program Files\LLVM\bin"
$env:RUST_LOG="info"
$env:VULVATAR_CLOTH_GPU="1"; $env:VULVATAR_CLOTH_SELFCOL="1"
$env:VULVATAR_VBO_AUDIT="1"; $env:CLOTH_RENDER_EVERY="1"
$env:CLOTH_PIN_BAND="0.10"
$env:CLOTH_SUBSTEPS="4"
cargo run --bin diagnose_cloth -- sample_data/YUMEKA_v1.0.3/FBX/Yumeka_v1.0.3.fbx diagnostics/cloth_final_front 90
# 背面: $env:CAM_YAW="180" を追加して diagnostics/cloth_final_back へ
```

### メッシュ暴れ・ケツ見え・シルエット崩壊の分離実験 — 同日・第4ラウンド

ユーザー指摘「ケツが見える / FBXと違うシルエット / メッシュが暴れる」を
背面カメラ (`CAM_YAW=180`) と隔離knob (`SPRINGS_OFF` / `WIND_SCALE` /
`COLLIDERS_OFF` / `CLOTH_OFF`) で切り分けた:

| 構成 | スカート状態 |
|---|---|
| 全ON (cloth + springs + wind + colliders) | 側面で裾が折れ上がり、腰/尻が露出 |
| springs OFF (cloth+wind+colliders) | 同様に崩壊 — springs は無関係 |
| springs OFF + wind 0 | 同様に崩壊 — wind も無関係 |
| springs OFF + wind 0 + colliders OFF | 同様に崩壊 — colliders も無関係 |
| **cloth OFF (純スキニング, springs ON)** | **同様に崩壊 — Skirt_* スプリングチェーンの暴れがスキン経路でも再現** |
| **springs OFF + cloth OFF** | **完全正常 (FBXシルエット)** |
| **cloth + pin band 10 cm** (springs ON, wind 0.5, colliders ON) | **正常 — プリーツAライン維持、お尻カバー** |

結論: (a) 純スキニング経路の暴れは Skirt_* スプリングチェーンの
過剰スイング (spring tuning または authored DynamicBone 値の問題 —
ライブでは SpringTuning の sway を絞ることで緩和可能);
(b) cloth 経路の暴れは **ゼロ曲げ剛性 (距離拘束のみ) の布が腰の sway
で側面に座屈する現象**で、深いピン帯 (自由長の短縮) が実用対処。
恒久対処は曲げ拘束のGPU展開 (Phase D、CPU 参照モデルは R7 で修正済み
— auto-cloth は bend を生成していない) と body-SDF (並行リファクタで
実装中の `body_sdf` / SdfField)。

`auto_cloth.rs` のピン帯を garment 高さ比例 (40%, clamp 3.5-12 cm) に
変更済み — ライブ auto-cloth に反映される。

最終検証: `diagnostics/final_verification.png` — 正面・背面とも t0 と
t90 で FBXシルエット (プリーツAライン) を維持、お尻カバー回復、暴れ
なし、telemetry 安定 (prim 5 corr max 9.7 mm、prim 12 corr max
24.2 mm、NaN 0)。waistband の薄色スリバーとフラットシェード継ぎ目の
髪の毛ラインが残る (Phase C の粒子/描画頂点分離またはアセット側縫い目
ウェルドの対象)。


合格基準: prim 5 corr max < 30 mm (radial 温存分)、prim 12 corr max
< 40 mm (radial clamp 20 mm + margin)、NaN 0、胸の浮遊破片なし、
腰のスカート密着の回復。`(v18)` 再生成ログ
`clearance: generated ... skin anchors` の出現も確認 (v19 への
version 昇格で自動的にキャッシュは無効化される)。
- **ライブ計測**: `debug_avatar_extra.json` の新形式は実機での再確認が要る。
  旧形式 (`skirt_bbox`) を消費する外部ポーラは新形式
  (`cpu_lbs_prim_bbox.prims[].pre_physics/post_physics`、親ID付き) への更新が必要。
  `VULVATAR_VBO_AUDIT=1` はプロセス起動時に評価される (途中変更不可 —
  audit copy がキャッシュ済みコマンドバッファの一部のため)。実行コストは
  anchor 持ち primitive ごとに VBO 1コピー/フレーム。
- **映像品質・遅延・長時間運転** (Phase E)、退行3件の因果レポート (Phase A の
  録画再取得を含む): 未着手。計画書の進捗チェックリストを参照。
- コミットは行っていない。作業ツリーには本パス以前の未コミット変更が含まれる。
