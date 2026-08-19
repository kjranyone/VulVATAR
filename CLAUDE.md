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
- 診断系バイナリ (`analyze_depth_provider` など) やアドホック検証スクリプトの出力は、`.gitignore` 済みの `diagnostics/` 配下に書く。バイナリのデフォルト出力先が入力ファイルの隣 (`<stem>_<suffix>/`) になっている場合は、必ず明示的に `diagnostics/...` を渡して実行する。
- 新しい診断ツールを追加する際も、デフォルト出力先を `diagnostics/` 側にするか、`validation_images/` 配下に書き込もうとしたらエラーにする。
- ポーズ品質のベンチは 2 本立て: `validate_pipeline` は「アバター vs ソース」の往復一致のみ (トラッキング誤差と骨軸 twist に盲目 — 0.0° でも実機破綻はあり得る)。真値比較は `cargo run --bin validate_gt` — 既知ポーズのアバターをレンダ→追跡→復元し、GT/SRC/REC 3 列で誤差を検出起因とソルバー起因に分解、neutral 比の coupling ゲインも自動算出する (`diagnostics/validation_gt/summary.md`)。
- **デスク配信エンベロープが主戦場** (頭+肩のみ・カメラ斜め・手は常時デスク下 = ユーザーの本番運用)。腕・顔まわりの変更は正面系リプレイ (wave/palms/namaste) に加えて必ずデスク系録画でも回すこと。録画は `scripts/depth_capture.py` かヘッドレス캡チャ (アプリのカメラ停止が必要) で `frame_NNNN_color.png + _depth_mm.npy` を `diagnostics/depth/desk_*/` に取り、`diagnose_depth_replay <dir>` の temporal summary で見る。デスク向け指標: hand presence duty / toggle 数 (点滅=hold 欠陥)、snaps>0.3/frame (腕スナップ)、yaw std (胸暴れ)、anchor_z jmax (root 暴れ)。

## Tracking v2 (fusion estimator) — 2026-08-19 以降の本番経路

- 本番プロバイダは `src/tracking/fusion/provider.rs` (`FusionProvider`)。v1 (`skeleton_from_depth` +
  `pose_solver` の位置ベース経路) は `VULVATAR_TRACKING_V1=1` で A/B 用に選択可。設計と実装状況は
  `docs/tracking-v2-design.md` (§13)。
- オフライン検証: `cargo run --features realsense --bin diagnose_fusion_replay -- <dir> [out_dir] [--render N]`
  (`diagnostics/depth/{wave,palms_front,namaste}_replay`, `diagnostics/sessions/<id>` を食う)。
  summary に 胴 yaw std / 肩深度参照との差 / メトリック関節残差 / 手首ジャンプ / 再捕捉回数、
  `frames.csv` に毎フレームのコスト内訳・σ・形状。`VULVATAR_FUSION_OBSDUMP=<frame>` で観測とモデルの
  対応ダンプ、`VULVATAR_FUSION_NO_{CLOUD,SURF,3D,BURNIN,REACH}=1` でアブレーション、
  `VULVATAR_FUSION_KEEP_CLOUD=1` でオーバーレイに点群/表面点を描く。
- ライブ: `VULVATAR_AUTOSTART_TRACKING=1` で起動時にカメラ開始 (realsense2.dll を PATH に)。
  `debug_state.json` の `rig` ブロックに quality / 主要ボーン σ・data_sigma / root / diag
  (est_ms, lost_events, seed_wins, n2d, med_2d_px, hand_crops, learned face counts)。
- 新しい症状に v1 流のゲートを足さない。どの残差・事前・σ が誤っているかをベンチで測ってから直す
  (密点群項は `cloud_budget=0` で既定 OFF — 可視性付き対応付けに作り直すまで)。

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
cargo build --features realsense --bin diagnose_depth_replay
```

アプリ本体の再ビルドだけはユーザーにアプリを閉じてもらう必要がある。

- リプレイベンチ (`diagnose_depth_replay` / `diagnose_video_replay`) はファイル名の
  フレーム番号から実キャプチャ時刻を復元して dt 正規化推定器に供給する
  (ダンプは5フレーム間引きが通例 — 名目 30fps 扱いだと時間系ゲートが実機の5倍厳しく見える)。

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
