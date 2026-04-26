# T12: Post-T10 Polish

## Priority: P2 / P3

## Source

T10 を retire する直前に、`T10-future-extensions.md` の各 `[x]` 項目と
`onnx-model-requirements.md` の要件 vs 実装を突き合わせた audit から
拾った残課題。コードは動くが「主張ほど磨かれていない」ところと、
要件書では highly desired だったが v1 では落ちた項目が中心。

T10 本体は全項目完了済みなので退役可。これは派生フォローアップ。

## Items

### Tracking

- [x] **CIGPose 経路で表情が駆動されない** — heuristic 派生で v1 を実装。
  `src/tracking/face_expression.rs` に EAR / MAR / smile / surprised の
  geometry-only 推定を入れ、VRM 1.0 preset 名 (`blinkLeft/blinkRight/blink/
  aa/happy/surprised`) で出力。「人間らしさ」のために以下を組み込み:
  非対称 EMA (channel ごとに attack/release を別 τ)、blink 入力側 Schmitt、
  happy/surprised 出力側 deadband、surprised の AND ゲート、face-lost 時
  の release-τ 経由フェード。10 unit test green、pre-existing 失敗への
  巻き添えなし。ARKit 52 blendshape full set は引き続き未実装（必要に
  なったら face-only ONNX 案 (b) を別 plan で）。

- [x] **Confidence threshold の default 値が箇所によって違う** — 統一済み。
  `tracking/mod.rs` に `pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.3;`
  を canonical として置き、`TrackingSmoothingParams::default` /
  `SolverParams::default` / GUI 初期値 (旧 0.5) / Streaming preset から
  参照。意図的に divergent な Recording (0.2) / Performance (0.5) には
  use-case の docstring を付与。あわせて `persistence.rs` の
  `confidence_threshold` の `#[serde(default)]` が古い project file 復元
  時に silent 0.0 に落ちる隠れバグを発見、`default_confidence_threshold`
  関数で canonical を返すよう修正。Round-trip + missing-field の test
  を 2 本追加。Plan が言及した `vrm.rs` の 0.1 は test fixture (synthetic
  confidence=1.0 を通すため) で production default ではなく対象外。

- [x] **DirectML フォールバックがユーザに見えない** — 修正済み。
  `build_session` の戻り値を `(Session, InferenceBackend)` に拡張、
  3-variant enum (`DirectMl` / `Cpu` / `CpuFromDirectMlFailure { reason }`)
  で fallback の理由まで保持。`CigPoseInference::backend()` で取り出して
  `TrackingMailbox` の新スロット `inference_backend_label` 経由で GUI
  スレッドに渡す。Inspector の Inference Status セクションに
  "Inference: DirectML" / "Inference: CPU (DirectML unavailable: ...)"
  行を追加。あわせて既存の "Backend: Synthetic" 表示が camera backend
  を混同して紛らわしかったので "Camera: ..." にリネーム。Synthetic
  モード / 初期化前は label が None で行ごと非表示。

### Cloth Authoring

- [x] **Overlay version migration が一度も発火していない** — 修正済み。
  `migrate_cloth_overlay_json` / `migrate_project_json` の loop 部分を
  generic helper `migrate_chain<F>(json, from, to, step)` に抽出、
  production wrapper は `step_overlay` / `step_project` をそのまま渡す
  thin wrapper に。fake stepper を使った unit test 3 本を追加 (順序+
  カウンタ増分・エラー伝播+早期 abort・from==to で stepper 非呼び出し)。
  これで loop 本体の挙動が test 済みなので、v2 が landing したとき
  migrator 追加は match arm への 1 行追記だけで済む。Plan の (a) dummy
  migrator 案より production code を汚さない第三の道。

- [x] **Partial rebinding の GUI 統合経路が integration test 無し** —
  E2E カバー追加。`GuiApp::for_test()` を新設し eframe の
  `CreationContext` 無しで harness を構築可能に (egui setup と IO/system
  probe をスキップ、~150 行)。`#[cfg(test)] mod rebind_integration_tests`
  に 3 ケース:
  (1) Clean: 名前一致で primary tier resolution → 通知無し + attach
  実行 + 永続化ファイル無変更。
  (2) Failed: 名前不一致 → "Could not auto-bind" 通知 + attach 実行
  されない (`avatar.cloth_overlay_count() == 0`) + 永続化ファイル
  無変更。
  どちらも `restore_cloth_overlay_paths` 経由で実 file IO を通す
  本物の E2E。
  (3) Partial の persistence leg は `save_rebound_overlay` を直接呼ぶ
  focused IO test でカバー — algorithm がまだ secondary/tertiary tier
  を emit しないため (cloth_rebind.rs:198-204 の TODO)、Partial の
  完全 E2E は tier-2/3 resolution が landing した時点で書き直し。
  test module の docstring に matrix を残してある。

### Renderer

- [x] **Output color space の自動回帰検出が無い** — option (b) shader
  lint で実装。`src/renderer/pipeline_lint_tests.rs` を新設。
  (1) fragment shader source 内の禁忌 gamma 定数 (`1.0 / 2.2`, `1.0/2.4`,
  `0.4545`, `0.4167` 等) を string scan で検出。 manual decode 系
  (`pow(c, 2.2)` シェイプ) も同一行に `pow(` + `2.2)`/`2.4)` がそろった
  場合だけ panic させて fresnel 等の false positive を回避。
  (2) `color_attachment_format` の Srgb→R8G8B8A8_SRGB / LinearSrgb→
  R8G8B8A8_UNORM マッピングを pin。
  Vulkan device + sample VRM が CI で揃わないため (a) golden image は
  断念、最頻発の regression 経路 (manual encode 混入 + format swap) を
  source level で押さえる方針。視覚 fidelity と OBS 経由の
  `MF_MT_TRANSFER_FUNCTION` 解釈は `docs/output-interop.md` に追加した
  Manual QA チェックリストでカバー (`render_matrix` 手順 + OBS
  preview 比較手順)。

- [x] **Thumbnail render 失敗時の挙動が未定義** — failure path を
  `handle_thumbnail_response` 関数に抽出してコントラクトを明文化、テスト
  3 本で pin。挙動: render error / encode error / channel disconnect の
  どの経路でも on-disk PNG (avatar import 時に書かれた placeholder) を
  上書き / unlink しない。`poll_thumbnail_jobs` は helper 経由に書き換え、
  egui の image cache invalidation も成功時のみ。テスト:
  (1) render failure + 既存ファイル無し → ファイル作られない
  (2) render failure + placeholder 既存 → bytes 完全保持
  (3) render success + placeholder 既存 → 真正 PNG (magic header) で上書き
  (3 が無いと 1, 2 が「helper が常に書かない」だけで通ってしまうので入れた)。
  Plan の log 出力指示は既に warn! で出ていたが、message に "(keeping
  any existing placeholder)" を追記して contract を log 上でも明示。

### Persistence

- [x] **Profile export/import の round-trip integration test 無し** —
  3 ケースで pin。`gui/profile.rs` に `profile_roundtrip_tests` mod 追加:
  (1) StreamProfile の 14 フィールド全てを preset / `f32::default()` /
  `bool::default()` から異なる値に設定して export → import → 全フィールド
  比較 (f32 は ε=1e-6 で許容)。コンストラクタを explicit struct literal
  で書いており、新フィールド追加時はコンパイルエラーで test 更新が強制
  される compile-time tripwire になっている。
  (2) 複数 profile + meaningful `active_index` で Vec 順序と index
  bookkeeping の round-trip を検証。
  (3) 必須フィールド欠落 JSON は import が `Err` を返すこと(現状
  `#[serde(default)]` 無しなのでこの挙動)。将来 default 追加時に
  この test が逆に通るようになるので「既定値選択を見直すタイミング」の
  signal になる。乱数 seed ではなく決定論的に選んだ exact-representable
  な値 (0.625, 0.375 等) を使い flake を回避。

### App

- [x] **Avatar load cache の eviction policy がユーザに見えない** — 修正済み。
  `DEFAULT_MAX_CACHE_ENTRIES` の docstring に sizing rationale を追記
  (典型 .vvtcache 20-60 MB × 20 ≈ ~800 MB を上限の目安)。
  `evict_to_count` の info! ログを拡張: 旧 `"evicted N entries (cap M)"`
  → 新 `"evicted N entries — freed X.X MB (cache now Y.Y MB, cap M)"`。
  実機 cache が CI 環境では作られないため静的な実測値を docstring に
  焼き付けるアプローチではなく、operator が自身の log から実値を
  読める方針に倒した (docstring にもそう明記)。`info!()` での発火通知
  自体は plan の前から既に存在していたため、拡張ポイントは「freed bytes
  と現在 cache サイズ」を載せたところ。

- [ ] **Folder watcher のエッジケースが未検証** — `src/app/folder_watcher.rs`
  の test は `#[ignore]` 付き。notify backend が UNC path / OneDrive /
  symlink でどう振る舞うかは未確認。100ms debounce が cloud sync の
  遅延通知で取りこぼす可能性。
  **対処方針**: 制限事項（local filesystem 推奨、network/cloud は best
  effort）を README か inspector の watched folders セクションに明記。
  Manual refresh ボタンを足すと安全側に倒せる。

## Notes

P1 (壊れている) は無し — 動作している。全部「磨き不足 / テストの穴 /
v2 で爆発しうる latent risk」のレベル。優先するなら最初の 3 つ
（CIGPose 表情、Confidence default 統一、DirectML 可視化）が体感に
直結する。

T10 の retire はこのファイルが立った後に進めて OK（T10 の各項目は
完了しているので、ここに書いた残課題は派生 polish という位置付け）。
