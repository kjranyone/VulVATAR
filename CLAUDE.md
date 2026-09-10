# CLAUDE.md

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

- `validation_images/` は **推論入力データ専用** のディレクトリ。検出・診断・可視化など、コードを動かして得たテスト出力をここに書き出してはいけない (`README.md` 末尾にある "synthetic validation assets" の宣言を汚染しない)。
- 診断系バイナリ (`diagnose_fusion_replay` など) やアドホック検証スクリプトの出力は、`.gitignore` 済みの `diagnostics/` 配下に書く。バイナリのデフォルト出力先が入力ファイルの隣 (`<stem>_<suffix>/`) になっている場合は、必ず明示的に `diagnostics/...` を渡して実行する。
- 新しい診断ツールを追加する際も、デフォルト出力先を `diagnostics/` 側にするか、`validation_images/` 配下に書き込もうとしたらエラーにする。
- ポーズ品質のベンチ: 合成 GT は `cargo run --bin validate_gt` (既知ポーズをレンダ→追跡→復元、coupling ゲイン算出、`diagnostics/validation_gt/summary.md`)、実録画は `diagnose_fusion_replay` (下の Tracking v2 節)。
- **デスク配信エンベロープが主戦場** (頭+肩のみ・カメラ斜め・手は常時デスク下 = ユーザーの本番運用)。腕・顔まわりの変更は正面系リプレイ (wave/palms/namaste) に加えて必ずデスク系録画 (`diagnostics/sessions/<id>` / `diagnostics/depth/desk_*`) でも `diagnose_fusion_replay` を回すこと。デスク向け指標: 手首ジャンプ/snap 数、data-σ duty (非観測の手が誤駆動されていないか)、胴 yaw std、root ジャンプ。

## Tracking v2 (fusion estimator) — 2026-08-19 以降の本番経路

- 本番プロバイダは `src/tracking/fusion/provider.rs` (`FusionProvider`) の一本のみ
  (v1 位置ベース経路は 2026-08-20 に全削除)。設計と実装状況は `docs/tracking-v2-design.md` (§13)。
- オフライン検証: `cargo run --features realsense --bin diagnose_fusion_replay -- <dir> [out_dir] [--render N]`
  (`diagnostics/depth/{wave,palms_front,namaste}_replay`, `diagnostics/sessions/<id>` を食う)。
  summary に 胴 yaw std / 肩深度参照との差 / メトリック関節残差 / 手首ジャンプ / 再捕捉回数、
  `frames.csv` に毎フレームのコスト内訳・σ・形状。`VULVATAR_FUSION_OBSDUMP=<frame>` で観測とモデルの
  対応ダンプ、`VULVATAR_FUSION_NO_{SURF,3D,BURNIN,REACH}=1` でアブレーション、
  `VULVATAR_FUSION_KEEP_CLOUD=1` でオーバーレイに表面点を描く。
- ライブ: `VULVATAR_AUTOSTART_TRACKING=1` で起動時にカメラ開始 (realsense2.dll を PATH に)。
  `debug_state.json` の `rig` ブロックに quality / 主要ボーン σ・data_sigma / root / diag
  (est_ms, lost_events, seed_wins, n2d, med_2d_px, hand_crops, learned face counts)。
- 新しい症状に v1 流のゲートを足さない。どの残差・事前・σ が誤っているかをベンチで測ってから直す
  (密点群項は未使用のため削除済み — 表面点はスパースなキーポイント直下深度で拘束)。

## Live debugging (実機計測 — アプリを止めない)

ユーザーが「おかしい」と言ったら、**アプリからカメラを奪わずに** live debug
channel で計測する。アプリがカメラを掴んでいる間は pyrealsense2 等で別プロセスから
開いてもフレームは来ない (open は通るが `wait_for_frames` がタイムアウトする)。

- 有効化: `C:\ProgramData\VulVATAR\debug.on` (空ファイル)。約2秒以内に反映、再ビルド不要。
- `debug_state.json` — 推論フレーム毎 (tracking worker)。`kp` (COCO 17 の 2D+score)、
  `kp_mcp` (両手ブロックの MCP 4点)、`torso`/`arm` の各関節 `{p, c, d}`
  (`d` = サンプラーが実際に返したカメラ空間深度 m — 「特徴点に正しい深度が付いたか」の一次証拠)、
  `root` / `root_is_hip` / `metric` (anchor_cam_m, mpsu, ref_span_m)、`face`。
- `debug_avatar.json` — ソルバー後のアバター主要関節ワールド座標 (`seq` で新フレーム検出)。
  ユーザーが見ているものの数値化はこちら (rest 判定は Hips y ≈ 0.845 等)。
- `debug_depth.bin` — 毎秒1回、フル解像度アライン済み深度 (`VDBD` 32B ヘッダ + u16 mm)。
  任意キーポイント直下の生深度ピクセルの監査用。`debug_camera.bin` は 320px RGBA (`VDBG`)。
- 計測手順: Python ポーラで `debug_state.json`/`debug_avatar.json` を 5-10ms 間隔で読み
  (`frame`/`seq` でデデュープ)、jsonl に貯めて統計 (中央値・sd・フレーム間ジャンプ)。
  ユーザーに「20秒ポーズをキープ」と依頼してから回す。スクリプトは `scratchpad/` に書く。

アプリ稼働中は本 target の `cargo build --features realsense` が **必ず失敗する**
(realsense-sys の build.rs が `target\debug\deps\realsense2.dll` へコピーを試み、
稼働プロセスがロック中)。診断バイナリは別 target でビルドする:

```powershell
$env:CARGO_TARGET_DIR = "$PWD\target-test"
$env:SHADERC_LIB_DIR  = "$PWD\target\debug\build\shaderc-sys-<hash>\out\lib"  # 本targetのキャッシュ流用 (無いと CMake 非互換で from-source が死ぬ)
# + docs/realsense-build.md の3環境変数
cargo build --features realsense --bin diagnose_fusion_replay
```

アプリ本体の再ビルドだけはユーザーにアプリを閉じてもらう必要がある。

- リプレイベンチ (`diagnose_fusion_replay`) はファイル名の
  フレーム番号から実キャプチャ時刻を復元して dt 正規化推定器に供給する
  (ダンプは5フレーム間引きが通例 — 名目 30fps 扱いだと時間系ゲートが実機の5倍厳しく見える)。

## 実機不具合の調べ方 (最優先)

**ライブパイプラインの不具合をコード読解だけで診断するな。** 症状 (「体が暴れる」「棒立ちで固まる」) からコードを読んで因果の物語を組み立てると、もっともらしいが間違った結論に達する。特に `reference_span_m` のような載荷スカラーを未計測のまま変更すると、症状が別の症状に化けるだけで前に進まない。**まず録る。**

### 1. アプリ起動中なら — ライブデバッグチャネル

フラグファイル `%ProgramData%\VulVATAR\debug.on` を作ると 2 秒以内に有効化 (リビルド不要)。同ディレクトリに毎フレーム上書きで出る:

| ファイル | 書き手 | 中身 |
|---|---|---|
| `debug_gui.json` | GUI スレッド (**全ゲートの手前**) | `paused` / `avatars_loaded` / `tracking_enabled` / `frame_count` / `seq` |
| `debug_state.json` | トラッキングワーカー | 2D キーポイント + source 関節 (位置・信頼度) + face pose |
| `debug_avatar.json` | `run_frame` 内 | ソルブ後のアバター world 関節 + head 軸 |
| `debug_camera.bin` | トラッキングワーカー | カメラ RGBA (32 byte ヘッダ `VDBG`) |
| `debug_tuning.json` | **読み込み** | `arm_reach_ik` / `contact_ik` / `idle_arm_apose` / `joint_confidence_threshold` をリビルドなしで A/B |

「アバターが動かない」の切り分けは `debug_gui.json` の 2 値で決まる。`seq` は毎 GUI フレーム、`frame_count` は**非 pause フレームのみ**進む:

- `seq` 進む・`frame_count` 止まる → 一時停止中 (Space が `TogglePause` に**修飾キーなし**でバインド)
- 両方進む・`avatars_loaded: 0` → アバター未ロード
- 両方進む・`avatars_loaded: 1` → 原因は `run_frame` より下流

これらは値であって解釈ではない。推測する前に読め。

### 2. キャプチャ → 無人リプレイ (品質改善の本線)

人間がカメラの前に座るのは**一度だけ**にする。録ったら以降は無人で何度でも回す。

```powershell
$env:VULVATAR_RECORD="1"; $env:VULVATAR_RECORD_RAW="1"
cargo run                       # 再現させてトラッキング停止 (生フレーム収集は既定 900 frame = 30 秒で打ち切り、アプリは続行)
```

`diagnostics/session_<unix>/` に出る:

- `pose.jsonl` — published `SourceSkeleton` の全フレーム時系列。各関節に **`JointOrigin`** (`O`=深度実測 / `E`=深度が穴で骨長レイ外挿 / `S`=合成) と `reference_span_m` / `mpsu` / アンカー
- `frame_NNNNNN_color.bmp` + `_depth_mm.npy` — 生フレーム
- `manifest.jsonl` — **実 intrinsics とデバイスタイムスタンプ**

以降はカメラ不要:

```powershell
cargo run --release --bin diagnose_video_replay -- diagnostics\session_<unix>   # 温度状態を継続して本番プロバイダに流す
cargo run --bin analyze_session -- diagnostics\session_<unix>                   # summary.md に判定
```

`analyze_session` は「飛んだ関節は測ったのか、でっち上げたのか」を両端の provenance で分類する。`E` 側に偏れば骨長 (=`reference_span_m`) が疑い、`O` 両端に偏れば深度サンプリング側。L/R ブロック反転とグローバルスケール異常も別枠で出す。

**環境変数**: `VULVATAR_RECORD_JUMP` (ジャンプ検出閾値、既定 0.35 source unit)、`VULVATAR_RECORD_RAW_FRAMES` (フレーム上限、既定 900)、`VULVATAR_RECORD_RAW=N` (N フレームおき)。

**注意点**:
- 生フレームは **BMP (非圧縮)**。dev プロファイルは依存クレートが opt-level 0 で、PNG は 90.8 ms/frame = 11 fps 上限となり 30 fps に追いつかず**静かに間引かれた録画**になる。BMP は 3.6 ms/frame。間引かれた録画はフレーム間隔が実機と違うので温度フィルタ由来の不具合を再現できない。実測は `cargo test -- --ignored --nocapture frame_write_throughput`
- ディスク 1.5 MB/frame (30 秒で約 1.4 GB)
- drop が出たら `session_record` が warn を出す。**その録画は使うな**
- `manifest.jsonl` が無いディレクトリを replay すると、公称 1280×720 の intrinsics (`fx=924`) と名目クロックにフォールバックする。640×480 のキャプチャでは焦点距離が約 2 倍ずれる。replay は必ず「N with recorded intrinsics + device timestamps」を確認してから読む

## Architecture

- GUI thread: eframe/egui — `src/gui/mod.rs` (`GuiApp::update`)
- Render thread: Vulkan via vulkano — `src/renderer/mod.rs` (`VulkanRenderer::render`)
- Communication: `sync_channel(2)` in `src/app/render_thread.rs`
- Output: shared-memory writer on a worker thread — `src/output/`

See `docs/architecture.md` and `docs/threading-model.md`.

## Profiling

See `docs/profiling.md` for instrumentation recipes and known bottlenecks.

Profile logs go in `profile/` (gitignored):
```bash
RUST_LOG=vulvatar=info cargo run 2> profile/run_$(date +%Y%m%d_%H%M%S).log
```
