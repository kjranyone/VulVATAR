# T11: GUI ↔ Pipeline State Synchronization

## Priority: P1

## Status

**Phase A: COMPLETED (2026-04-19)** — drift bug class structurally eliminated. See "Implementation summary" below.

Phase B / C / D は未着手。既知の限界 (rollback の project 永続化、pause 中の lipsync 停止挙動、dead control の未配線) は当該 Phase で扱う。

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

## Phase B: Wire dead controls to pipeline

### 目的

現状 GUI で選べるが pipeline へ流れていない設定 4 件を実装する。

### 対象

| 設定 | 現状 | 実装内容 |
|---|---|---|
| `output.output_resolution_index` | `GuiApp::update` で `app.output_extent = Some([w,h])` に書いているが renderer 側で消費されていない疑い | renderer の output target 解像度に反映 + DLL 側 SUPPORTED_FORMATS との整合 |
| `output.output_framerate_index` | 全く消費されない | sample timestamp cadence + frame skipping (output worker 側) |
| `output.output_has_alpha` | 全く消費されない | RGBA / RGB の切替。DLL 側で `MFVideoFormat_ARGB32` 追加または既存 RGB32 と切替 |
| `output.output_color_space_index` | 全く消費されない | render target の sRGB / Linear 切替。MTOON shader の output gamma 対応 |
| `tracking.camera_resolution_index` | 「Start Camera」押下時のみ消費 | combo onchange で `app.start_tracking_with_params` を再呼出 (camera 起動中のみ) |
| `tracking.camera_framerate_index` | 同上 | 同上 |

### 設計方針

各設定について App 側に対応する `set_*` mutator を追加し、reconcile_app_with_gui() に対応する reconcile_* 関数を追加。Phase A で確立したパターンをそのまま適用。

注意: `output_color_space_index` の Linear sRGB 化は MTOON shader 出力との整合確認が必要。

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

## Phase D: Session-active vs saved-state separation

### 目的

Phase A の rollback (VirtualCamera 失敗時の SharedMemory フォールバック) が autosave で project に永続化される問題を解決する。

### 設計方針案

`OutputGuiState.output_sink_index` を 2 系統に分割:

- `saved_sink_index: usize` — project にシリアライズされる、ユーザの意図
- `active_sink_index: usize` — 現セッションでの実態 (rollback で書換可)

`to_project_state` は `saved_sink_index` を使用、reconcile は `active_sink_index` を更新。combo onchange は両方を更新 (ユーザ意図の更新 + 反映)。`apply_project_state` は両方を `saved_sink_index` から初期化 → reconcile で active 側のみ rollback され saved 側は維持。

### 適用範囲

同様の rollback がある全ての shadow value:
- `output_sink_index` (Phase A で対象)
- `lipsync.enabled` (Phase A の reconcile_lipsync_runtime も failed 時 false に書き換え)
- 将来 Phase B で増える可能性のある settings

### リスク

- Project state schema 変更 (saved_* 追加)。既存 project ファイルとの互換性管理
- combo box の displayed value がどちらかを混乱なく扱う UI 設計が必要

---

## Out of scope for T11 (永続的に T11 の対象外)

- combo box dropdown を多解像度カメラ実体と動的同期させる (Phase B が片付いた後の独立タスク)
- GUI architecture の wholesale refactor (egui → 別 framework 等)
