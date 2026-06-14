# Ray-IK: 2D 再投影拘束による深度ソルブ

ステータス: 実装済み (2026-06-12) — `src/tracking/ray_ik.rs` (ソルバー核) +
`src/tracking/rtmw3d/arm_ray_ik.rs` (パイプライン統合)。
`reconstruct_arm_z` は削除済み。Phase 7.5–7.6 の DAv2 証拠項統合と
wrist Stage 2–4 の退役は未着手 (移行計画 3–4)。

2026-06-13 追記 (肩ひねり試行の教訓): 肩を自由変数化して span
投影短縮から Δz を解く案は試行→リバート。(a) 腕セグメントの解剖学
参照長が非人間体型で肩を引きずる (腕ソルブと肩 Δz 推定は別問題)、
(b) span 短縮の「要求」は sqrt 増幅で推定ジッタが偽ひねりに化ける
(旧 ±45° yaw-lock と同物理 — 正解は「Δz 証拠が提案、span は上限」)。
肩ひねりの本道は移行計画 3 (DAv2 bz の IK 証拠化)。ソルバーには
そのための per-segment `nz_weight` / `one_sided`、per-observation
`depth_weight` を実装済み。z0 アンカーは胴体長ベース (ひねり不変)。
評価は `validate_gt` (GT 自己整合レンダリング) を使うこと —
旧 validate_pipeline は往復一致のみで twist に盲目。summary.md には
neutral 比デルタの coupling ゲイン表が自動出力される。

同日、移行計画 3 の第一スライス実装: `arm_ray_ik` が joint の
`metric_depth_m` (DAv2) を肘の深度証拠として消費 (肩 DAv2 基準との
相対デルタ + 0.8 m 妥当性ゲート。手首は手の甲サンプルが背景深度を
拾う既知問題のため不使用 — Phase 7.5 が手首 z を再注入しないのと
同じ理由)。with-depth パイプラインの Phase 7.6 後に
`solve_arm_depth_contact_only` を配線: 7.6 のメトリック chain 置換は
実測ベースで優秀だが両手接触の深度コヒーレンスを切断するため、
接触ゲート発火時のみ DAv2 証拠+接触リンク付きで再融合する。
常時再ソルブは写真ベンチを mean 2.4°→3.6° に劣化させた (実測) ので
contact-only に限定。

## 動機

RTMW3D の 2D (nx, ny) は高精度だが、SimCC nz は「カメラへ向かう奥行き」を
ほぼ識別できない。現行パイプラインはこの欠損 Z を **逐次ヒューリスティックの
積層** で補っている:

1. `rtmw3d/arm_z.rs::reconstruct_arm_z` — 骨長不変条件 `|dz| = sqrt(L² − xy²)`
   で前方深度を注入
2. `rtmw3d/wrist.rs` Stage 2–4 — 前腕長 sanity check / temporal hold /
   「+0.2 カメラ方向」合成
3. `rtmw3d_with_depth.rs` Phase 7.5–7.6 — DAv2 メトリック深度による
   shoulder/elbow z 注入と arm chain 再配置

この構造には根本的な欠陥が 2 つある。

**(a) 基準骨長が投影空間の running max で汚染される。**
`reconstruct_arm_z` の基準長 L は「セッション中の投影 xy 長の生 running max」
(`forearm_xy_max` / `upper_xy_max` / `span_max`)。手がレンズ近くを通ると透視
投影で投影長が解剖学的限界を超えて膨らみ (実測 2 倍)、max は二度と戻らない。
以後、画面内に平行な普通の腕が常に foreshortened と誤判定され、毎フレーム
偽の前方 dz が注入される — 「2D アノテーションは正しいのにアバターが歪む」
症状の最有力原因。`repair_edge_clamped_arms` は同じ理由でバンドクランプ
(0.8–1.35×) を導入済みだが、`reconstruct_arm_z` 側は生 max のまま。

**(b) 深度証拠の融合点が分散している。**
nz・骨長不変・DAv2 メトリック・時間的継続性が別々の関数で順番に上書きし合い、
どの証拠がどれだけ効いたか追跡できず、一段が汚れると後段全部が汚染される。

## 設計: 観測レイ上の深度パラメータ化

### 核となる洞察

2D キーポイントは観測としてほぼ正しい。透視カメラモデルでは、各キーポイント
(nx, ny) は **カメラ原点からのレイ** を定義し、関節の 3D 位置はそのレイ上の
1 自由度 (深度 t) に拘束される:

```
P_i = t_i · d_i
d_i = normalize([ (nx_i − 0.5) · 2·tan(hfov/2),
                  (ny_i − 0.5) · 2·tan(vfov/2),
                  1 ])                         (カメラ空間, メートル)
```

これにより:

- **2D 再投影一致が残差ではなく構成上の恒等式になる。** 解いたポーズを
  再投影すれば必ず観測 2D に一致する — ビューポートのアノテーションと
  アバターの構図が原理的に乖離しない。
- **最適化変数は深度スカラー t ∈ R^N のみ** (体幹+四肢で N ≈ 14)。
  Gauss-Newton 数イテレーションで 60fps に余裕で収まる。
- **「レンズに近い手は大きく投影される」が構造的に説明される。** 近接は
  小さな t で吸収され、骨長推定を汚さない。欠陥 (a) の根本解決。

### コスト関数

```
min_{t}  Σ_seg    w_len  · (‖t_i d_i − t_j d_j‖ − L_ij)²      … 骨長
       + Σ_anchor w_dav2 · (t_i − t̂_i)²                        … DAv2 メトリック深度
       + Σ_seg    w_nz   · ((t_i − t_j)·ẑ − Δz_nz)²            … nz (decisive のみ)
       + Σ_joint  w_temp · (t_i − t_i^prev)²                    … 時間平滑
       + Σ_elbow  w_lim  · penalty(hyperextension)              … 関節可動域
       + Σ_wrist  w_fwd  · softplus(t_wrist − t_elbow)          … 前方事前分布 (弱)
```

各項の意味と既存レイヤーとの対応:

| コスト項 | 置き換える既存機構 | 重み方針 |
|---|---|---|
| 骨長 (メトリック、バンドクランプ) | `reconstruct_arm_z` の不変条件 | 強。L はキャリブ由来 (下記) |
| DAv2 深度証拠 | Phase 7.5–7.6 の注入群 | 中。サンプル信頼度で変調 |
| nz decisive-backward | `reconstruct_segment_z` の符号規則 | 中。後方が決定的なときのみ |
| 時間平滑 | wrist temporal hold (Stage 3) | 弱。occlusion 時は conf 低下で相対的に支配 |
| 前方事前分布 | Stage 4 の「+0.2 カメラ方向」合成 | 最弱。他の証拠が無いときの tie-break |
| 可動域 penalty | (新規) | 肘の過伸展・背面折りを抑止 |

### 骨長 L の推定 (メトリック空間)

- 絶対スケール: `PoseCalibration.shoulder_span_m` (キャリブ済み) または
  `ASSUMED_SHOULDER_SPAN_M = 0.40`。
- セグメント比: 既存の解剖学プリオール (upper 0.76 / forearm 0.62 × span)。
- セッション内精緻化: **メトリック空間で** EMA 更新し、解剖学バンド
  (プリオール × 0.8–1.35) にクランプ。レイ・パラメータ化により近接時の
  投影膨張は t に吸収されるので、ここが汚染されることは構造上ない。

### 絶対深度 (torso anchor)

- depth 有効時: DAv2 メトリックサンプル (`resolve_origin_metric`) が
  肩/腰 anchor の t̂ を与える。
- RTMW3D 単独時: 透視関係 `Z₀ = S_real / s_proj` (S = shoulder span,
  s_proj = 投影スパンの角度幅) から胴体深度を推定。仮定 FOV は既存の
  `ASSUMED_VERTICAL_FOV_DEG = 60.0` を共有。

### 出力: source 空間への変換

解いたメトリック関節 P_i を、hip-mid 原点・一様スケールで source 空間へ:

```
source_i = flip · k · (P_i − P_origin),   k = 2 / (2·tan(vfov/2)·Z₀)
```

k は「胴体深度における画面高さ」を [-1, 1] に写す係数なので、胴体深度に
ある関節の x/y は従来の `to_source` 出力と一致し、レンズに近い関節だけが
透視補正される (Phase 7.6 `replace_arm_chains_from_metric` が部分的に
やっていたことの一般化)。`pose_solver` の direction-match はそのまま消費
できる。

### スコープ

- 対象関節: shoulders / elbows / wrists / hips / knees / ankles
  (+ neck/head は弱い時間項のみ)。
- 指は対象外: 手首の t が決まったら指チェーンは既存どおり手首デルタで
  平行移動 (`shift_hand_chain`)。指の屈曲は手首ローカルでの相対問題で、
  `constrain_fingers` + palm-plane の既存機構が担当。
- 顔カスケード・表情は無関係 (変更なし)。

### 縮退時の挙動

- キーポイント欠損 (画面外): レイが無い関節は骨長項+時間項+held direction
  prior で外挿 — `repair_edge_clamped_arms` の OOF 検出器は維持し、その
  「last visible direction」を事前分布項として注入する。
- 全身が低信頼: 時間項が支配し前フレームへ収束 (= 今の temporal hold と
  同じ挙動が自然に出る)。

## 両手接触 (合掌) の扱い

direction-match は被写体の骨の **方向** を再現するが、合掌は **位置** の
制約である。アバターの体型 (肩幅 : 腕長 比) が人間と違う限り、方向を
完全再現しても手首位置は一致しない。さらに ray-IK は左右の腕を独立に
解くため、両手首の深度が揃う保証もない。2 層で対応する (2026-06-12 実装):

1. **トラッキング側** (`arm_ray_ik`): 両手首が 2D で近接し、かつ左右の
   ハンドキーポイント群の投影サイズが類似する (= 同深度の証拠; サイズが
   違うときは単に前後をすれ違っているだけなので接着してはならない) とき、
   LeftHand–RightHand 間に「等深度なら 2D 距離が含意する横距離」を長さと
   する接触セグメントを追加し、深度を整合させる。
2. **ソルバー側** (`pose_solver::compute_arm_contact_ik`): 両手首の距離が
   肩幅の 0.25× 以下で full、0.45× で 0 になる smoothstep 重みで、4 本の
   腕ボーンの方向を 2-bone 位置 IK の方向へブレンド。手首ターゲットは
   **単一のアフィン写像** (肩中点アンカー + half-armspan 比スケール +
   肩ライン整合回転) で写す — 写像が 1 つなら一致点は一致点に写るので
   接触が保存される (per-side スケールはこれを壊す)。肘の swivel は
   トラッキングされた肘を pole に使う。開いたポーズでは重み 0 で純粋な
   direction-match のまま (方向パリティ検証メトリクスもそこでは有効なまま)。

## 移行計画

1. `src/tracking/ray_ik.rs` 新設 — ソルバー核 + 単体テスト
   (arm_z の既存テスト相当: 完全 foreshortened 前腕 / 平面内の腕 /
   decisive backward、追加: 近接レンズで骨長推定が膨張しないこと)。
2. `rtmw3d/mod.rs` — `reconstruct_arm_z` 呼び出しを ray-IK に置換。
3. `rtmw3d_with_depth.rs` — Phase 7.5–7.6 の注入群を DAv2 証拠項の供給に
   置換。
4. `rtmw3d/wrist.rs` — Stage 4 合成を削除 (前方事前分布が代替)。
   Stage 2–3 は信頼度重み+時間項に吸収後、削除。
5. `validate_pipeline` の fs_err / angle メトリクスで前後比較
   (`streamer_display_webcam_capture` セット必須)。

## 診断ログ

```bash
RUST_LOG=vulvatar=info,vulvatar_lib::tracking::rtmw3d=debug \
  cargo run 2> profile/rayik_$(date +%Y%m%d_%H%M%S).log
```

- info (`arm_ray_ik`): `ray-ik diag fN: z0=…m span_src=… elbow_z=[…] wrist_z=[…]`
  (60 フレームごと)
- debug (`arm_ray_ik`): 解いた関節ごとの dz とカメラ深度 (per-frame)
- debug (`arm_z`): running max の成長イベント (5% 閾値、edge repair の
  バンドクランプが読む値) と edge-exit extension の engage/release
- debug (`wrist`): wrist temporal hold のリジェクト/保持/合成

旧 `reconstruct_arm_z` の注入ログ (`inject/60f`) は本体ごと削除済み。
膨張バグの確認ポイントだった「`fa_max` がプリオール比 1.35× を超えたまま
戻らない」は、ray-IK では構造的に無害 (max は edge repair のバンド内
微調整にしか使われない)。
