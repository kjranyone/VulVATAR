# TODO — トラッキング品質の残り本丸

2026-09-14/15 の調査セッション (a89a5c5 時点) で確定した事実を土台に、
優先度順に残課題を整理する。判断基準・ベンチの回し方は
[docs/tracking-v2-design.md](docs/tracking-v2-design.md) §7 と AGENTS.md に従う。

**2026-09-15 進捗**: P0 の 1./2. を実施し、腕密表面は**否決** (下記)。
副次で、snap の最大要因だった「観測ゼロ帧の速度残留」を修正
(12 録画 snap 244→109、胴 yaw |err| 不変、seed wins 20→16 —
`scratchpad/bench_p0.sh` + `agg_p0.py`、結果は `diagnostics/fusion/{base,p1,p1_da,p2_mask,p3}`)。
ローカルに残っている録画は `diagnostics/sessions` の 12 本のみ
(29 本セットの残り 17 は `diagnostics/depth/*_replay` 由来で既に削除済み) ので、
TODO 内の「29 録画」基準は当面この 12 録画サブセットで代用する。

**2026-09-15 夜 追加 (P0 続き)**: 2 段ソルブを実装・ベンチ (下記 P0 更新)。
肘深度リフトの監査ツールも追加 (`VULVATAR_REPLAY_ELBOW_DUMP=1` +
`VULVATAR_FUSION_KEEP_CLOUD=1` → `elbow.csv`、`scratchpad/agg_elbow.py`)。
**既定動作は完全不変** (s1789219959 の出力が p3 と全行一致することを検証済)。

## 前提: 実測で確定している構造

- 「手が暴れる」の実体は 2 種類:
  1. **観測ドロップアウト中の root 速度残留** (修正済 2026-09-15)。
     ジャンク検出 1 枚が root 速度を 3 m/s まで跳ねさせ、その後の観測ゼロ帧で
     `predict` の減衰と速度再摂取がほぼ相殺され、0.1–0.2 m/帧の暴走が
     ドロップアウト全体続く (s1789279985: 203 snap 中 172)。
  2. **腕鎖の跳び** (観測あり帧、残 ~50 snap / 12 録画)。指ローカルの churn は
     元々小さい (mean ~1.6°/frame, p95 ~9°)。
- 悪条件セッションでは**全クロップ由来**の手ランドマークが 19–33 px rms
  ノイズを持ち、σ はその 1/10–1/20 を申告している (`hands.csv` 由来別計測)。
- **手のノイズ抑制と胴の精度は腕鎖経由で結合**しており、σ・マスク・hold
  のどの介入も単独では両立しない (下の失敗テーブル参照)。
- 手首 >15 cm の動きのフレームでは胴 yaw が中央値 5.2° (max 18.3°) 動く。
- **肘深度リフトの ~14 cm はリフトの誤差ではない** (2026-09-15 監査、12,007 観測):
  肘ピクセル直下の生深度は観測 (push 0.035) と整合し、**既定ソルブのモデル腕が
  センサ表面より ~7.6 cm 深い均衡に留まる**のが実態 (全 12 録画で中央値 −0.05〜−0.10)。
  腕を密表面に乗せると肘残差は norm 中央 ~3 cm まで縮み、push 補正は不要
  (腕が面上の帧では観測はモデル肘と +0.8 cm)。

### 失敗テーブル (再試行前に読むこと)

| 介入 | 結果 | 再現コマンド |
|---|---|---|
| 手 σ 全域 ×2/×4/×8 | 胴盆地反転 (−23.7°→+59.5° 等) | `VULVATAR_HAND_SIGMA_SCALE` |
| 指のみ σ ×2/×4 | 指は静穏化、胴 +4.3°→−64° | `VULVATAR_FINGER_SIGMA_SCALE` |
| hold 活性化ヒステリシス (2f) | snap 悪化 (R 4→6) | revert 済み、コメントに記録 |
| hold σ ランプイン | 29 録画 snap 184→193 | revert 済み |
| q_finger 締め付け | churn 悪化 (9→25°/f) | `VULVATAR_Q_FINGER` |
| 腕鎖マスク | 手首 0.19→0.03 m だが胴 −13.5°/+23° | `VULVATAR_ARM_CHAIN_DEPTH=1` |
| 単段 DENSE_ARMS (シード修正込み) | snap 179 だが |err| 32.7 (胴盆地代替) | `VULVATAR_DENSE_ARMS=1` |
| 2 段ソルブ・胴ステージに腕観測を Cauchy 付きで残す | 胴 yaw sd 1–4°→29–108° (凍結腕の遅れが鎖ヤコビアンで胴を引く) | `VULVATAR_FUSION_TWOSTAGE=1` (修正前) |
| 〃 腕観測を完全排除 + 肩 2D も排除 | yaw sd 11° (肩ピクセルが胴証拠) | 〃 途中版 |
| 〃 肩より下だけ排除 + 肢カプセルを対応付けから除外 | yaw sd 3.6° ✓ (s1789242856) — 採用 | `VULVATAR_FUSION_TWOSTAGE=1` |

---

## P0 — 手首 snap の根本解: 腕の観測強化

**Why**: 上の失敗テーブルはすべて「腕の観測がランドマーク 21 点のみで弱い」
ことの帰結。腕が弱いから手ノイズが腕を動かし、腕が胴を引っ張る。
設計書 §8 の既知課題「腕は密表面に未接続」がまさにこれ。

**2026-09-15 実測 (12 録画, base = snap 244 / |err| 19.0° / seed 20)**:

1. ✅ 表面項をシード評価から除外 — 実装済 (設計書 §5)。DENSE_ARMS と併せても
   seed wins 18 に収まる (汚染解消)。単独では中立 (snap 257, |err| 19.6)。
2. ❌ 単段での `VULVATAR_DENSE_ARMS=1` — 上の失敗テーブル参照。
3. ❌ 腕カプセルの表面行を腕チェーン param にマスク — |err| 145 で大悪化、revert 済み。

**2026-09-15 夜: 2 段ソルブ (実装済・既定 OFF、`VULVATAR_FUSION_TWOSTAGE=1`)**:

胴ステージ (腕チェーン (肩ボール含む) を予測位置に凍結、肩より下の観渑と
wrist-hold を残差から除外、肢カプセルを表面対応付けから除外 = core-only)
→ 腕ステージ (root・胴・形状を凍結、腕のみ)。シードは胴ステージ結果を
ベースに腕ステージのみ再実行。(再) 捕捉帧 (temporal prior 無し) は
単一段のまま — 再捕捉の誤 torso hint を弾くのは胴ステージが排除する
腕観測だから (s1789279985: 壁 2.9 m に root が張り付き 5 回再捕捉)。

12 録画ベンチ (p3 基準 = snap 109 / |err| 19.6 / seed 16):

| 案 | snap | |err| | seed | 備考 |
|---|---|---|---|---|
| p3 (既定) | 109 | 19.6 | 16 | — |
| ts2 = TWOSTAGE 単独 | 158 | 30.4 | 12 | wave 改善 (13→8) だが机下未観測手首が跳ぶ (観測ノイズを腕がフル追従) |
| ts2_da = TWOSTAGE+DENSE_ARMS | **100** | 26.0 | 16 | 机系改善 (s1789246071 2→0, s1789311387 21→8)、ジャンクセッションで yaw err −16.5・再捕捉 5 |
| ts3 / ts3_da (再捕捉帧も単一段に戻した版) | 157 / 155 | 25.8 / 31.8 | 15 / 13 | ジャンクセッションの結果が反転 (50→108) — 同セッションはナイフエッジ |

**ts2_da は 11 の良観測セッションに限れば基準を満たす** (snap 50 vs p3 48、
|err| 9.5 vs 14.9)。残る障害は s1789279985 (ドロップアウト~40% のジャンク録画):
DENSE_ARMS 下で壁クラウド (ncl>1600) に root が飛び (1.8→3.2 m)、以降 yaw が
±179° 反転を繰り返す。単独 TWOSTAGE では同帧は暴走しない → 腕の点主張が壁を
「説明」してしまう。run 間で結果が入れ替わる (50↔108 snap) 不安定帯なので、
次はここを対象化してから (Junk バースト時の ncloud ゲート or 肢主張の一時停止)、
改めて 12 録画で判定すること。

**受け入れ基準 (12 録画版、変更なし)**: snap 合計 < 80、胴 yaw sd と |err| 合計は
p3 基準 (19.6) から悪化させない、seed wins は +10 以内。

**2026-09-16: ジャンククラウドゲートで P0 受け入れ基準 達成 (既定構成)**。

上の壁ロックの連鎖 (ジャンクバースト 150–198 で腕が ±2 m 引きずられ → 状態崩壊 →
209 で壁クラウド 1,657 点に root が飛び恒久ロック) の対策として、**前帧の sparse
残差中央値が lost しべル (`lost_rms_px` = 40 px) を超えていたら次帧の密クラウドを
落とす**ゲートを `update_with_seeds` に追加 (密クラウドの供給源 = 人物マスクは
sparse kp と同じ検出パイプラインなので、sparse が崩れている時はマスクも信用しない。
hard reset ではなく時間事前でポーズを保持)。既定 ON、`VULVATAR_FUSION_NO_JUNKCLOUD=1`
で無効化。契約テスト `junk_cloud_gate_drops_cloud_after_lost_level_sparse_frame` 付き。

隔離 worktree (3450653 + ゲートのみ、他エージェント WIP 排除) での 12 録画ベンチ:

| 構成 | snap | |err| | seed | reacq |
|---|---|---|---|---|
| p3 (旧基準) | 109 | 19.6 | 16 | 1 |
| **既定 + ジャンククラウドゲート** | **75** ✓ | **19.6** ✓ | **16** ✓ | 1 ✓ |
| TWOSTAGE+DENSE_ARMS+ゲート | 82 | 30.8 | 14 | 2 |

良観測 11 セッションは p3 と**全桁同一** (ゲートはジャンク帧のみ発火)、
s1789279985 のみ 61→27 snap・yaw sd 79.5→47.8・root max 2.04→0.92 m 改善。
**→ P0 受け入れ基準を既定構成で達成。** 残る実態: 良観測セッションの観測あり帧
腕鎖 snap ~48 (P0 本丸「腕の観測強化」の未解決部分 — 2 段ソルブ+DENSE_ARMS は
機構として有効だがジャンクセッションの yaw err (−16.1) で総合では未達、実験継続)。
検証時の注意: 他エージェントが provider 周りを編集中の working tree でビルドすると
観測経路が変わってベンチが汚染される (実測: 同一セッションで snap 1→11)。
隔離 worktree + `CARGO_TARGET_DIR=target-test` で測ること。

## P1 — 頭の残課題 (横顔)

**2026-09-16: a./b. 実施・コミット済み** (12 録画ベンチ YOLO11 基準 108/15.6/35 に対し
108/15.7/35 で全良観測セッション全指標中立、s1789279985 の頭/胴 yaw sd 85.5→19.4
・head yaw sd max 93→18.7 改善。validate_gt (Yumeka) の roll 往復利得は合成レンダ
では σ が cap に届かず変化なし — cap の効果は実機 tilt でのみ発現):

- ~~**頭 tilt 中の過小応答**~~ → 顔 kp σ に cap (既定 8 px、`VULVATAR_FACE_KP_MAX_PX`)。
  tilt 中は SimCC の spread × score-inflate が ~17 px まで膨張して eye line が消える
  のを防ぐ (信頼時 σ ~3-4 px は cap に触れない)。
- ~~**遠側 eye/ear の連続的 σ 膨張**~~ → far-side cull の二値閾値を facing 比例の
  σ 膨張 (facing 0→−0.15 で ×1→×4、以下は従来通り棄却) に置換。ナイフエッジ
  (0.25 だけで roll +40°→+14°) を構造的に除去。`VULVATAR_HEAD_CULL_FACING` は
  棄却下限のまま。
- ~~**安静時 roll バイアス**~~ → 2026-09-16 夜に機構解明 (下記) + roll 専用 ori 観測
  を実装。**デスク本番環境では mesh チャネル自体が不発火 (conf ~0 / 推論なし) なので
  バイアス −4〜−6° は残る** — 根本修正は顔サイト幾何の per-user 較正 (下記残課題)。

  **機構 (2026-09-16 監査、ピクセル GT 付き)**:
  - 生フレームの目視 GT (vision reader を合成画像で較正 ±0°、4 セッション):
    真の eye-line 傾きは +8/−1.5/0/+3° とセッション毎に違い、**est は常に
    −1〜−9° (平均 −5°) 負側にずれ**。mesh チャネル roll (ランドマーク 33/263 の
    画像傾き) は GT ±3° 以内で正確。
  - est−mesh チャネル差は 3 セッションで **−6.1〜−6.5° でほぼ一定** (HEADREF 拡張
    `VULVATAR_REPLAY_HEADREF`、`body_eye_tilt` 追加、`diagnostics/fusion/headref4_p1c`)。
  - ablation (`VULVATAR_ABL_NOHEADKP/NOEYES/NOEARS/NONOSE/NOMESH`、
    `diagnostics/fusion/ablp1c_*`): **eye 2D kp 単独が原因** — NOEYES で両検証
    セッションとも GT まで回復 (+1.2→+5.7 vs GT+8、−5.8→+6.5 vs GT+3)。
    nose/ears/mesh attach/3D lift/dense は中立。
  - 機構: OBSDUMP (frame dump) で model eye サイト射影の傾き +8.4° vs 観測 kp 傾き
    +5.0°@est roll +1.2° — 頭 yaw ~29° × pitch 誤差 (~+12° vs mesh チャネル) の
    **射影交絡項 (+7° 相当) を roll DOF が吸収**。モデル顔サイト幾何 (eye y+0.09/
    ear y+0.07/nose y+0.055) とユーザー解剖 (実測 ear-eye 落ち ~0.8cm vs モデル 2cm)
    のズレが残差を生み、斜め視点で roll/pitch に分配される。
  - **実装 (既定 OFF、`VULVATAR_FUSION_ROLLORI=1` で有効)**:
    `HeadOriTracker::estimate_roll_ori` (head_ori.rs) — mesh チャネルの
    eye-line tilt と現在予測の roll 差だけを forward 軸回転の OriObs として注入
    (yaw/pitch 残差は厳密に 0、契約テスト付き)。full ori 不発火帧のみ、conf ≥ 0.2・
    手遮蔽 cooldown 準拠、|Δ| ≤ 0.15 rad (トリム用)。`VULVATAR_ROLLORI_SIGMA` σ 上書き、
    `VULVATAR_ROLLORI_DUMP` 発火ログ。
  - **なぜ既定 OFF か (2026-09-16 実測)**: ① mesh チャネルの利用可能性がボトルネック
    — s1789246660 で候補 10/782 帧 (conf≥0.2 + cooldown)、s1789234881 は手が顔前に
    ある 595/600 帧、s1789246212/s1789303569 は mesh 推論自体なし。② 発火帧でも
    補正は eye kp + 事前に負けて roll 軌跡が全桁不変 (σ 0.005 まで確認)。③ その上で
    ソルバ搅乱で snap +3〜8 (12 録画 116/15.6/36 vs 基準 108/15.6/35、|err| は同一)。
    正面・良照明・手が離れた場面 (mesh が常時走る) でのみ意味があるため opt-in。
  - **残課題 (次の根本修正)**: 顔サイト (l/r_eye, l/r_ear, nose) のローカルオフセット
    を per-user 学習 (FaceFit の scale+offset と同じ EMA 系) — pitch/roll バイアスの
    both を消す。デスク環境の roll は eye kp 由来のままなので、これが本命。

## P2 — 指角度観測の較正 (`VULVATAR_FUSION_ANG=1`)

実装済み・既定 OFF。勝てなかった理由は (a) σ 無較正、(b) thumb の 3 点組が
モデルと共面でない。手元の指 churn が小さいので優先度は低いが、P0 で腕が
強化された後に胴反転耐性が変わる可能性があるので再評価する。
較正手順: 静止手の録画で `VULVATAR_REPLAY_HAND_DUMP` → 観測角度の
フレーム間分布から σ を実測。

## P3 — 計測・基盤

- ~~**録画のフレーム落ち**~~: 2026-09-15 時点で `diagnostics/depth/*_replay`
  (17 本) はディスク節約で削除済み。29 録画ベンチは再現不能になっており、
  手元の 12 録画 (`diagnostics/sessions`) が基準セット。次に録る時に
  フルセットを保管するか、ベンチ基準を 12 録画に再設定すること。
  フレーム落ち自体 (900 指定で 782、744→546) は未調査のまま。
- **リプレイの実 intrinsics 読み取り** (`--real-intrinsics`): meta.jsonl に
  実測 intrinsics がある。bench 数値の連続性が切れるので opt-in で。
- 手クロップ由来ダンプ (`hands.csv`) と finger churn (`fingers.csv`) は
  `diagnose_fusion_replay` に常設済み。P0 の検証に使うこと。
- ベンチランナー: `scratchpad/bench_p0.sh <name> [ENV=VAL...]` +
  `scratchpad/agg_p0.py` (12 録画を回して snap/yaw_sd/err/seed 集計)。

## P4 — 設計書 §8 の既知課題 (参考)

- ~~肘深度リフトの系統誤差 ~14 cm~~ → 解消 (2026-09-15 監査、上記「前提」参照)。
- 脚・床平面・学習姿勢事前 (△)。
- AprilTag GT リグ (❌) — 商用品質の独立検証に必要。
