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

- [ ] **Overlay version migration が一度も発火していない** — `src/persistence.rs:283-292`
  の `step_overlay` の match arm は空（v1 しか released されていない）。
  テストはエラー経路 (newer rejection) と no-op 経路しか叩いていないので、
  v2 を切る時に migrator が初発火 → デグレに気付きにくい。
  **対処方針**: dummy v0 fixture と dummy v0→v1 migrator を一時的に
  作って unit test で実経路を通す。発火確認後 dummy は削除。
  もしくは `migrate_cloth_overlay_json` を直接呼ぶテストを追加。

- [ ] **Partial rebinding の GUI 統合経路が integration test 無し** —
  `src/asset/cloth_rebind.rs` の 8 unit test は algorithm レベルのみ。
  `src/gui/mod.rs:986-1022` の Failed → attach しない / Partial →
  `last_rebound_with` 永続化 / 通知発火、の路は手動確認のみ。
  **対処方針**: avatar reimport で Failed になる overlay を仕込んだ
  fixture を作り、ロード後の状態（attached overlays / persistence /
  notification queue）を assert する integration test を 1 本足す。

### Renderer

- [ ] **Output color space の自動回帰検出が無い** — `cargo run --bin
  render_matrix` は手動 smoke テスト。shader regression で linear/sRGB
  write が壊れても CI では気付かない。下流（OBS 等）が `MF_MT_TRANSFER_FUNCTION`
  を見て解釈しているかも実測されていない。
  **対処方針**: (a) `render_matrix` の出力を golden image と比較する
  test を 1 ケース追加（4 セルのうち代表 1 つ）、もしくは (b) shader
  GLSL の lint で manual sRGB encode の不在を検査。OBS 連携検証は
  別途手動 QA のチェックリストに足す。

- [ ] **Thumbnail render 失敗時の挙動が未定義** — `finalize_avatar_load`
  が `RenderCommand::RenderThumbnail` を発行するが、render thread 側で
  失敗した時の挙動（placeholder 残留 / 古いサムネのキャッシュ / 無音 fail）
  がコード上で保証されていない。
  **対処方針**: render thread 側の error path を log で出す。失敗時は
  placeholder を保持する保証を入れる（既存 PNG が残っているなら上書き
  しない）。VRM パース成功 + thumbnail render 失敗のテスト fixture を 1 つ。

### Persistence

- [ ] **Profile export/import の round-trip integration test 無し** —
  `ProfileLibrary::export_to_file` / `import_from_file` 単体は動く想定
  だが、「現在の GUI state を export → 別状態にして import → 全フィールド
  一致」の通し検証が無い。fields が増えた時に silent default が混入しても
  気付けない。
  **対処方針**: integration test 1 本。GUI state の主要フィールドを
  乱数 seed で埋めて round-trip → assert_eq。

### App

- [ ] **Avatar load cache の eviction policy がユーザに見えない** —
  `src/asset/cache.rs:40` の `DEFAULT_MAX_CACHE_ENTRIES = 20` を超えると
  silent eviction。Cap の根拠（典型 cache サイズ × 20 = 何 GB?）が
  docstring に無いので、後から見直す材料が無い。
  **対処方針**: 代表的な VRM での cache file size を実測し docstring に
  「~X MB × 20 = ~Y GB」と書く。`evict_to_count` の発火時に件数を info!
  ログに出す。

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
