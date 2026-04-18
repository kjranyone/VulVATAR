# T11: GUI ↔ Pipeline State Synchronization

## Priority: P1

## Status

- **Phase A: COMPLETED (2026-04-19)** — reconcile pattern + App-side mutators で drift bug class を構造的に消去。詳細は "Implementation summary (Phase A)"
- **Phase C: COMPLETED (2026-04-19)** — pipeline-bound shadow フィールド (output_sink_index / lipsync.enabled / lipsync.mic_device_index) を `OutputGuiState` / `LipSyncGuiState` から削除し、Application を単一 source of truth に。reconcile_app_with_gui 自体を廃止
- **Phase D: COMPLETED (2026-04-19)** — `Application::requested_sink` / `requested_lipsync_enabled` フィールド追加。setter は requested を更新してから active の試行 (失敗しても requested 保持)。`to_project_state` は requested を保存するため MF 起動失敗で fallback しても autosave で saved 値が上書きされない。combo は requested 表示 + 不一致時の警告ラベル
- **Phase B: 部分完了 (2026-04-19)**
  - B-1: COMPLETED — tracking camera resolution / framerate combo 変更で webcam worker が in-place restart
  - B-2: COMPLETED — `output_resolution_index` → renderer output target は元から wired。helper `output_resolution_for_index` に集約
  - B-3: COMPLETED — `output_framerate_index` → `OutputRouter::set_target_fps` で publish 側 throttle (旧 frame は drop)
  - B-4: COMPLETED — `output_has_alpha` → 共有メモリ header offset 28 の `flags` u32 (bit 0 = `FRAME_FLAG_PRESERVE_ALPHA`) 経由で DLL に伝達。RGB32 path はソース alpha 保持/破棄を切替
  - **B-5: DEFERRED** — `output_color_space_index` (sRGB / Linear sRGB)。GUI → app への配線は trivial だが下流の renderer が `output_export.rs` で `color_space: "srgb"` をハードコードしており、render target format selection / MTOON shader 出力 gamma 分岐 / MF sample の `MF_MT_VIDEO_PRIMARIES` `MF_MT_TRANSFER_FUNCTION` attribute まで含めた renderer color management 改修が必要。T11 scope (drift fix) を超えるため別タスク化

## Source

- 2026-04-19 セッションで発覚した MF Virtual Camera が出力されない事象の根本原因
- `src/gui/mod.rs` `apply_project_state` / `apply_profile`
- `src/gui/inspector.rs` 各 combo box / checkbox の onchange
- `src/app/mod.rs` `set_output_sink`, `start_tracking_with_params`, `stop_tracking`

## Problem

GUI 状態 (`OutputGuiState`, `TrackingGuiState`, `LipSyncGuiState`) が App 側のランタイム状態 (OutputRouter / tracking_worker / lipsync_processor / mf_virtual_camera) を **shadow** しているが、両者の同期経路に欠落がある。

### 具体例 (本バグ)

`apply_project_state` (project load) は `state.output.output_sink_index = state.output_sink_index` のように **GUI 側 shadow に代入するだけ** で `app.set_output_sink()` を呼ばない。結果:

- GUI 上は "Virtual Camera" が選択されているように表示される
- 実際の `OutputRouter` は default の `SharedMemory` のまま
- `mf_virtual_camera` は `None` のまま
- combo box の change 検出 (`prev != now`) は起動時に既に "Virtual Camera" 表示なので発火しない
- ユーザは詰む

## 全 GuiState 設定の棚卸し

### A. Drift バグあり (load 時に App へ反映されない)

| 設定 | 必要な mutator | 現状 |
|---|---|---|
| `output_sink_index` | `app.set_output_sink()` | ❌ apply_* が生代入のみ |
| `lipsync.enabled` | processor start/stop | ❌ apply_* が生代入のみ |
| `lipsync.mic_device_index` | enabled 中なら restart | ❌ combo onchange も dirty 立てるだけ |
| `tracking.toggle_tracking` | tracking_worker start/stop | ❌ apply_* が生代入のみ |

### B. 機能未実装 (combo onchange で何も起こらない)

| 設定 | 想定挙動 | 現状 |
|---|---|---|
| `output_resolution_index` | render target / output 解像度切替 | ❌ App で消費されない |
| `output_framerate_index` | output cadence | ❌ App で消費されない |
| `output_has_alpha` | RGBA/RGB 切替 | ❌ App で消費されない |
| `output_color_space_index` | sRGB / Linear | ❌ App で消費されない |
| `tracking.camera_resolution_index` | webcam 再起動 | ⚠️ 「Start Camera」ボタン押下時のみ消費 |
| `tracking.camera_framerate_index` | 同上 | ⚠️ 同上 |
| `camera_index` (webcam) | tracking 再起動 | ⚠️ 「Start Camera」ボタン押下時のみ消費 |

### C. Per-frame input (drift なし、毎フレーム GuiState→RenderFrameInput で再構築)

`rendering.*` 全部、`transform.*`、`camera_orbit.*`、`tracking.{smoothing_strength, confidence_threshold, hand_tracking_enabled, face_tracking_enabled, tracking_mirror}`、`lipsync.{volume_threshold, smoothing}`

→ 同期不要、現状のまま放置で OK

### D. UI-only (App 側に反映先がそもそもない)

`show_camera_wipe`, `show_detection_annotations`, viewport / panel 開閉状態

→ 同期不要、現状のまま放置で OK

## Scope

本タスクは **A のみ** を対象とする。

- **A** (drift 修正) — 必須。production bug を構造的に消す
- **B** (機能実装) — 別タスク。現状でも "値を保存できる" 以上の機能が無いので壊れていないように見えるが、ユーザは選択しても効かないという認知不一致を抱える。修正するなら App 側に該当機能を新規実装する必要があり scope が大きく異なる
- **C** (single source of truth、GuiState shadow 廃止) — 別タスク。A の上に乗る大型 refactor。GUI rendering 側全 panel を書き換える必要があり、A の bug fix と切り離してリスクを下げる

## Design (Phase A) — Revised after review

### 原則

GUI shadow → App ランタイム反映は **`reconcile_*` 関数群に集約**。各 reconcile 関数は 1 サブシステム (output / lipsync) の責任を持ち、idempotent で、失敗時は GUI shadow を runtime 実態へ rollback して通知する。

### Sub-system reconciliation 設計

3 つの runtime サブシステムを別 reconcile に分け、まとめて `reconcile_app_with_gui()` から呼ぶ:

- `reconcile_output_runtime()` — OutputRouter sink + MF camera lifetime
- `reconcile_lipsync_runtime()` — lipsync processor lifetime (audio capture)
- *(tracking は reconcile しない — 後述 Finding 2)*

### 1. App 側 mutator の追加

`src/app/mod.rs` に以下を追加:

```rust
/// Bring the OutputRouter and MF virtual camera (Windows only) in line with
/// the requested sink. Idempotent. Returns Err if the requested sink is
/// VirtualCamera and MFCreateVirtualCamera::Start fails — caller can roll
/// back GUI shadow and notify.
///
/// Critically, if a previous call switched the OutputRouter to VirtualCamera
/// but `mf_virtual_camera.start()` failed, a subsequent call still tries
/// to start the camera (the OutputRouter swap is no-op but the MF lifetime
/// path is retried).
pub fn ensure_output_sink_runtime(&mut self, sink: FrameSink) -> Result<(), String>;

/// Start / restart / stop the lipsync processor based on the requested state.
/// Idempotent over (enabled, mic_device_index).
pub fn set_lipsync_enabled(&mut self, enabled: bool, mic_device_index: usize) -> Result<(), String>;

/// Run lipsync inference for one frame. Reads the latest audio, applies
/// viseme weights to the active avatar, and returns the current rms volume
/// for the volume meter. No-op (returns None) when lipsync is not active.
///
/// Called every frame from `GuiApp::update`, so lipsync output is not gated
/// on whether the lip sync inspector panel is visible (the previous design
/// ran inference inside `draw_lipsync` and would freeze if the panel was
/// collapsed).
pub fn step_lipsync(&mut self, smoothing: f32, volume_threshold: f32, dt: f32) -> Option<f32>;
```

`lipsync_processor` の所有権は GuiApp から `Application` へ移動。Application は内部に `lipsync_active_mic: usize` を保持して `set_lipsync_enabled` 冪等性を担保。

**Note:** `OutputRouter::active_sink()` は既存 (`src/output/mod.rs:306`)。新規追加不要。

### 2. `reconcile_*` 関数群

`src/gui/mod.rs`:

```rust
impl GuiApp {
    /// 全サブシステム reconcile。combo onchange / project load の末尾で呼ぶ。
    pub fn reconcile_app_with_gui(&mut self) {
        self.reconcile_output_runtime();
        self.reconcile_lipsync_runtime();
        // Tracking は reconcile しない (Finding 2)
    }

    fn reconcile_output_runtime(&mut self) {
        let want_sink = match self.output.output_sink_index {
            0 => FrameSink::VirtualCamera,
            1 => FrameSink::SharedTexture,
            2 => FrameSink::SharedMemory,
            _ => FrameSink::ImageSequence,
        };
        if let Err(e) = self.app.ensure_output_sink_runtime(want_sink.clone()) {
            // VirtualCamera 起動失敗時の rollback。GUI shadow を SharedMemory
            // に戻し、router もそちらへ切替。次回 reconcile では SharedMemory
            // と一致しているので no-op。
            self.push_notification(format!("Virtual Camera unavailable: {e}"));
            self.output.output_sink_index = 2; // Shared Memory
            let _ = self.app.ensure_output_sink_runtime(FrameSink::SharedMemory);
        }
    }

    fn reconcile_lipsync_runtime(&mut self) {
        if let Err(e) = self
            .app
            .set_lipsync_enabled(self.lipsync.enabled, self.lipsync.mic_device_index)
        {
            warn!("reconcile: lipsync apply failed: {e}");
            self.lipsync.enabled = false;
            self.push_notification(format!("Lip sync failed: {e}"));
        }
    }
}
```

### 3. 呼び出し箇所の整理 (revised)

| 場所 | 変更後 |
|---|---|
| `apply_project_state` / `apply_profile` (gui/mod.rs) | 末尾で `self.reconcile_app_with_gui()` |
| inspector.rs sink combo (~1071) | shadow 更新後 `reconcile_app_with_gui()` |
| inspector.rs lipsync enable / mic (~742-781) | shadow 更新後 `reconcile_app_with_gui()` |
| inspector.rs **Start/Stop Camera ボタン** (~526-553) | **直接** `app.start_tracking_with_params` / `app.stop_tracking` を呼ぶ。reconcile を経由しない (Finding 2) |
| **`GuiApp::update` (per-frame)** | `self.app.step_lipsync(self.lipsync.smoothing, self.lipsync.volume_threshold, dt)` を `run_frame` の前に呼ぶ |
| inspector.rs `draw_lipsync` 内の inference loop | **削除**。volume meter / viseme 表示のみ残す |

### 4. lipsync_processor の所有権移動 + 処理ループ移動

- Owner: `GuiApp.lipsync_processor` → `Application.lipsync_processor`
- Per-frame inference: `draw_lipsync` 内 → `Application::step_lipsync` (GuiApp::update から呼出)
- GUI は volume meter (`state.lipsync.current_volume`) と viseme 表示のみ

## Implementation steps (revised)

1. `Application::ensure_output_sink_runtime(sink) -> Result<(), String>` 実装。`set_output_sink` は廃止 (内容を吸収)
2. `Application::step_lipsync(smoothing, threshold, dt) -> Option<f32>` 実装
3. `Application::set_lipsync_enabled(enabled, mic) -> Result<(), String>` 実装 (冪等、mic 変更時 restart)
4. `Application::is_tracking_running()` getter 追加 (現状コード `worker.is_running()` を露出)
5. `lipsync_processor` を `Application` へ移動、`GuiApp` から削除
6. `GuiApp::reconcile_output_runtime` / `reconcile_lipsync_runtime` / `reconcile_app_with_gui` 実装
7. `apply_project_state` / `apply_profile` 末尾に `self.reconcile_app_with_gui()` 追加
8. inspector.rs sink combo / lipsync enable+mic onchange を reconcile 経由に書換
9. inspector.rs Start/Stop Camera **直接呼出のまま維持** (reconcile 経由にしない)
10. inspector.rs `draw_lipsync` の per-frame inference を削除
11. `GuiApp::update` で `run_frame` の前に `self.app.step_lipsync(...)` を呼出
12. `cargo check --features lipsync` / `--no-default-features` で warning 0
13. 手動テスト

## Review findings (2026-04-19) と対策

設計レビューで指摘された 5 件と対応:

1. **High — `active_sink()` だけだと MF camera 起動失敗後 retry 不能**: `set_output_sink` は router を切り替えてから MF を起動するので、MF 失敗時 `output.sink == VirtualCamera && mf_virtual_camera == None` が永続。次回 reconcile は `active_sink() == want_sink` で no-op → 復旧不能。
   - 対策: `ensure_output_sink_runtime()` を新設し、router swap と MF lifetime を独立に idempotent 管理。MF 失敗は Result で上げ、reconcile 側で GUI rollback + 通知。
2. **High — `tracking.toggle_tracking` を worker lifecycle に流用するのは危険**: GUI 上 `Tracking Enabled` は推論 gate、`Start Camera` ボタンが webcam worker 起動と意味が分かれている。一緒くたにすると project load で意図せず webcam が開く。
   - 対策: tracking は reconcile しない。Start/Stop Camera ボタンは従来どおり `app.start_tracking_with_params` / `stop_tracking` を直接呼ぶ。`toggle_tracking` は推論 gate のまま。
3. **High — lipsync processing loop の置き場所が未定義**: 現状 `draw_lipsync` 内で `process_frame` を呼んでおり、panel 折りたたみで止まる。所有権移動だけだと改善しない。
   - 対策: `Application::step_lipsync(smoothing, threshold, dt)` を `GuiApp::update` から毎フレーム呼ぶ。GUI は volume meter / viseme 表示のみ。
4. **Medium — reconcile が error を握り潰す**: 元案の `let _ = self.app.set_lipsync_enabled(...)` だと失敗時に GUI shadow だけ enabled のまま残る (今回のバグと同型)。
   - 対策: 各 reconcile で `Result` を受け、失敗時は `push_notification` + GUI shadow を runtime 実態へ rollback。
5. **Medium — `set_lipsync_enabled` 冪等性に必要な runtime state**: load path から繰り返し呼ばれる前提なので、現在の (enabled, mic) を App 側で記憶しないと毎回 restart になる。
   - 対策: `Application` に `lipsync_active_mic: usize` を持つ。`lipsync_processor.is_some()` を enabled flag に流用。差分があれば restart、なければ no-op。

(別途: `OutputRouter::active_sink()` は既存のため Implementation step から除外)

## Risks / Tradeoffs

- **Reconcile コール頻度**: `reconcile_app_with_gui()` は冪等なので combo onchange 毎に呼んでもよい。差分検出で no-op 化されるためコスト低
- **lipsync 処理の per-frame 化**: 従来は panel 表示中のみ動いていたのを常時実行に変更 → CPU コスト微増。lipsync 機能の正しい挙動だがユーザに見えない場所で audio capture が動くことになる点は意識
- **MF camera 起動失敗時の rollback**: VirtualCamera 選択 → MF start fail → SharedMemory に rollback、という挙動。ユーザに通知はするが、user が再度 VirtualCamera を選び直せば再 retry される (DLL 再登録などで治るケースに備える)

## Acceptance Criteria

- Project に Virtual Camera sink + lipsync enabled を保存して再起動 → GUI を一切操作せずに MF camera が自動 register、`Local\VulVATAR_VirtualCamera` の magic が `0x5643_4D46` になる
- 上記で MF camera 起動が失敗するケースでも、GUI shadow は SharedMemory に rollback され、push notification でユーザに通知される
- `tracking.toggle_tracking` が true の project を load しても webcam worker は **起動しない** (Start Camera ボタンを押すまで)
- lipsync inspector panel を折りたたんでも viseme が継続する (`step_lipsync` が GuiApp::update で呼ばれるため)
- `cargo check` (default / `--features lipsync` / `--no-default-features`) warning 0 / error 0

## Implementation summary (Phase A)

### 変更ファイル

- `src/output/frame_sink.rs` — `FrameSink` に `PartialEq, Eq` derive、`to_gui_index()` / `from_gui_index()` ヘルパ追加
- `src/output/mod.rs` — `OutputRouter::active_sink()` getter (既存) を露出
- `src/app/mod.rs` — `set_output_sink` を `ensure_output_sink_runtime() -> Result` に置換 (router swap と MF lifetime を独立 idempotent に)、`set_lipsync_enabled() -> Result`、`step_lipsync() -> Option<f32>` 新設、`lipsync_processor` フィールドを App へ移動 (private)
- `src/gui/mod.rs` — `reconcile_app_with_gui()` / `reconcile_output_runtime` / `reconcile_lipsync_runtime` 実装 (失敗時 push_notification + GUI shadow rollback)、`apply_project_state` / `apply_profile` 末尾で reconcile、`GuiApp::update` で `step_lipsync` を per-frame 呼出、`camera_resolution_for_index` / `camera_fps_for_index` 共通化
- `src/gui/inspector.rs` — sink combo / lipsync enable+mic onchange を reconcile 経由に統一、`draw_lipsync` から per-frame inference loop 削除、Start/Stop Camera は直接呼出のまま維持

### Phase A 完了条件達成

| Criterion | 状況 |
|---|---|
| Virtual Camera + lipsync enabled の project を再起動 → 自動 register | 実装済 (要手動テスト) |
| MF 起動失敗時の rollback + 通知 | 実装済 |
| `toggle_tracking=true` の load でも webcam 自動起動しない | 実装済 (reconcile から tracking 削除) |
| lipsync panel 折りたたみで viseme 継続 | 実装済 (`step_lipsync` を GuiApp::update に移動) |
| cargo check warning 0 / error 0 | 達成 (3 feature combo 全部) |

### 既知の限界 (Phase D で対応予定)

| 項目 | 詳細 |
|---|---|
| Rollback の project 永続化 | `reconcile_output_runtime` が VirtualCamera 失敗時に GUI shadow を SharedMemory に書き換える。autosave / 手動 save 時にユーザの真の選択 (VirtualCamera) が失われる |
| Pause 中の lipsync 停止 | `step_lipsync` が `if !self.paused` ブロック内なので pause で audio capture も止まる。旧コード (`draw_lipsync` 内実行) は panel 表示中のみだが pause 影響なし |
| dead control 4 個未配線 | output resolution/fps/alpha/color_space は GUI で選べるが pipeline で消費されない (Phase B 対象) |

---

## Implementation summary (Phase C)

### 変更ファイル

- `src/app/mod.rs` — `lipsync_active_mic` を `lipsync_mic_device_index` に rename し、`set_lipsync_enabled` が常に mic を保存するよう変更 (disable 時も次回 enable 用に preserve)。`is_lipsync_enabled() -> bool` と `lipsync_mic_device_index() -> usize` getter を追加 (両者 feature 無効時にも利用可能な stub 付き)
- `src/gui/mod.rs` — `OutputGuiState.output_sink_index` 削除、`LipSyncGuiState.enabled` / `mic_device_index` 削除。`reconcile_app_with_gui` / `reconcile_output_runtime` / `reconcile_lipsync_runtime` を全削除し、`apply_pipeline_bound_settings(sink, lipsync_enabled, mic)` ヘルパに集約。`apply_project_state` / `apply_profile` / `to_project_state` を Application getter / mutator 経由に書換
- `src/gui/inspector.rs` — sink combo / lipsync mic combo / lipsync enable checkbox を Application 由来 local 変数 (`active_idx = state.app.output.active_sink().to_gui_index()` 等) に bind し、変更検出で App mutator 呼出。失敗は push_notification のみ (rollback 不要 — 次フレームで App state を再 read するので combo は実態を表示)
- `src/gui/status_bar.rs` — sink label を `state.app.output.active_sink().to_gui_index()` から取得

### Phase C 完了条件達成

| Criterion | 状況 |
|---|---|
| `OutputGuiState.output_sink_index` 削除 | ✅ |
| `LipSyncGuiState.enabled` / `mic_device_index` 削除 | ✅ |
| 全 GUI 表示が App getter から派生 | ✅ (sink label, mic dropdown, enabled checkbox) |
| `reconcile_*` 関数廃止 | ✅ (Phase A の reconcile は shadow 同期のために存在していたので不要に) |
| Phase A の acceptance criteria 維持 | ✅ (project load → MF camera 自動 register、MF 失敗時 notify、tracking 自動起動なし、lipsync inference per-frame) |
| cargo check warning 0 / error 0 | ✅ (3 feature combo 全部) |

### Phase C による副次的改善

- 「load 経路で reconcile を呼び忘れる」事故が**構造的に発生不可能**に。shadow が無いので新規 GUI フィールド追加時にも load path で代入し忘れることが無い (代入できる shadow が無い)
- Phase A の reconcile の MF rollback ロジックが消えた → コードシンプル化。rollback の project 永続化問題 (Phase D 対象) は残存するが、影響範囲が単純化
- 失敗時 GUI 表示の整合: 例えば VirtualCamera 選択中に MF 起動失敗 → App は前 sink のまま → 次フレームで combo が前 sink を表示。ユーザは「失敗した」ことを通知 + 視覚で同時に認識できる

### Phase C で削除した API

- `GuiApp::reconcile_app_with_gui()` → 廃止
- `GuiApp::reconcile_output_runtime()` → 廃止
- `GuiApp::reconcile_lipsync_runtime()` → 廃止

→ 後方互換が必要な場合は復活させるが、現状内部 API なので問題なし

---

## Implementation summary (Phase B + D)

### Phase B 変更ファイル

- `src/gui/mod.rs` — helper 関数 `output_resolution_for_index` / `output_fps_for_index` 追加 (camera 系と同じ pattern)。GuiApp::update で `app.output_extent` / `app.output.set_target_fps` / `app.output_preserve_alpha` を per-frame sync
- `src/gui/inspector.rs` — tracking resolution/framerate combo 変更時に `app.is_tracking_running()` ならば `app.start_tracking_with_params` で in-place restart (B-1)
- `src/output/mod.rs` — `OutputRouter` に `forward_min_interval: Duration` / `last_forward_at: Option<Instant>` 追加。`set_target_fps(fps)` で interval 更新。`publish` 内で前フレームから interval 未満なら drop (B-3)
- `src/output/frame_sink.rs` — `Win32NamedSharedMemorySink::write_frame` が `OutputFrame.alpha_mode` を見て shared memory header offset 28 の `flags` u32 に `SHMEM_FLAG_PRESERVE_ALPHA` (bit 0) を書く (B-4)
- `src/app/mod.rs` — `output_preserve_alpha: bool` フィールド追加。`process_render_result` で `OutputFrame.alpha_mode` を `Premultiplied` / `Opaque` に設定 (B-4)。`is_tracking_running()` getter を再追加 (B-1)
- `vulvatar-mf-camera/src/shared_memory.rs` — `FRAME_FLAG_PRESERVE_ALPHA` 公開定数。`Header.flags: u32` 追加、`FrameView.flags` 公開 (B-4)
- `vulvatar-mf-camera/src/media_stream.rs` — `build_sample_rgb32_from_rgba` に `preserve_alpha: bool` 引数。true ならソース alpha バイトを保持、false なら従来通り 0xFF 強制 (B-4)

### Phase D 変更ファイル

- `src/app/mod.rs` — `requested_sink: FrameSink` / `requested_lipsync_enabled: bool` フィールド追加。`requested_sink()` / `requested_lipsync_enabled()` getter、`set_requested_sink()` / `set_requested_lipsync()` setter (まず requested 更新 → active 試行、失敗しても requested 保持)
- `src/gui/mod.rs` — `to_project_state` が `requested_*` getter 経由で読むよう変更 (active ではない)。`apply_pipeline_bound_settings` も `set_requested_*` 経由に
- `src/gui/inspector.rs` — sink combo / lipsync enable+mic combo は `requested_*` を表示し `set_requested_*` で書込。requested ≠ active のときは黄色で「(running on X — requested Y unavailable)」を表示

### Phase B + D 完了条件達成

| Criterion | 状況 |
|---|---|
| Tracking resolution/framerate 変更が webcam 起動中に即時反映 | ✅ (B-1) |
| Output resolution が renderer 出力サイズに反映 | ✅ (B-2 — 元から wired を確認) |
| Output framerate combo で publish が throttle される | ✅ (B-3) |
| `output_has_alpha` 切替で RGB32 出力の alpha チャンネル保持/破棄が切り替わる | ✅ (B-4 — flag 経由) |
| MF init 失敗時 autosave で saved 値が上書きされない | ✅ (D — to_project_state が requested を保存) |
| requested ≠ active の状態が GUI で見えて分かる | ✅ (D — 黄色警告ラベル) |
| cargo check warning 0 / error 0 | ✅ (3 feature combo 全部) |

---

## Phase B-5 (deferred to separate task)

### なぜ deferred か

GUI → App への色空間プリファレンス引き渡し自体は数行で済むが、**意味のある実装にするには renderer 全体の color management 改修が必要**:

1. `src/renderer/output_export.rs` 行 211, 240, 259, 499 で `color_space: "srgb".to_string()` をハードコード — output frame に常に "srgb" を貼る
2. Render target の format が固定 (B8G8R8A8 系)。Linear sRGB を真に出力するには render target の gamma 変換セマンティクスを変える必要
3. MTOON shader が sRGB-encoded output を前提。Linear 出力モードを足すには shader 側に gamma 分岐
4. DLL 側 MF sample の `MF_MT_VIDEO_PRIMARIES` / `MF_MT_TRANSFER_FUNCTION` attribute 設定が現状無い (clients が解釈するかは別問題)

これらは render pipeline / shader 側の実装で、GUI ↔ pipeline drift 修正の T11 scope 外。別 task (T12 想定: "renderer color management") として切り出すのが妥当。

---

## Phase C: Single source of truth (GuiState shadow 廃止)

### 目的

GUI state と App state の二重管理を解消し、App を唯一の真実とする。「load 経路で reconcile を呼び忘れる」事故が構造的に起き得なくなる。

### 設計方針案

- `OutputGuiState` / `LipSyncGuiState` の pipeline-bound フィールドを削除
- 各 panel の combo box は `app.output.active_sink()` 等の getter から値を読み、`set_*` mutator で書き戻す
- egui の `selectable_value(&mut field, ...)` パターンが使えないので、`local_idx = app.output.active_sink().to_gui_index()` で局所変数を作って bind し、変化を検出して `app.ensure_output_sink_runtime()` を呼ぶ pattern (Phase A で reconcile_app_with_gui がやってる pattern を panel 側に展開)

### リスク

- 全 panel に波及するので diff 規模が大きい
- 「per-frame input」だった rendering / transform フィールドも巻き込むかは要判断 (drift しないので必要性が低い)

### 推奨

T11 Phase A で「reconcile pattern」が確立しユーザに動作確認されたら Phase C に進む。Phase B が先 (機能不全の解消が優先)。

---

## Phase D: Session-active vs requested separation (Implemented)

実装としては「saved/session 分離」ではなく **"requested vs active" 分離** に落ち着いた:

- `Application::requested_sink: FrameSink` — ユーザが望んだ sink (project に save される)
- `Application::output.active_sink()` — 現セッションで実際に running
- 両者は失敗時に乖離。`requested_sink` は失敗しても保持される

`to_project_state` が `requested_sink()` を呼ぶことで「runtime の実態」ではなく「ユーザの意図」を save する。MF 起動失敗で active が SharedMemory にフォールバックしても、project ファイルには VirtualCamera が残る。

同じパターンを lipsync.enabled に適用 (`requested_lipsync_enabled`)。mic は `set_lipsync_enabled` が常に保存するため別フィールド不要。

GUI 上は **requested を表示**。requested ≠ active のときは Inspector に黄色の警告ラベル (`(running on X — requested Y unavailable)` / `(requested, but audio device couldn't open)`) を出すことで「希望は通ってないが意図は記憶されている」ことをユーザに伝える。

互換性: project ファイル schema 変更なし (output_sink_index フィールドは同じ意味、ただし保存される値が現セッション active から requested に変わっただけ)。

---

## Out of scope for T11 (永続的に T11 の対象外)

- combo box dropdown を多解像度カメラ実体と動的同期させる (Phase B が片付いた後の独立タスク)
- GUI architecture の wholesale refactor (egui → 別 framework 等)
