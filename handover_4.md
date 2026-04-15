# VulVATAR Handover — 2026-04-15 (Session 4)

## ビルド状態

- `cargo check`: エラー 0（警告 1 — `unused_imports`）
- `cargo check --features virtual-camera`: エラー 0
- `cargo test`: 52 passed / 0 failed

## 今セッションでやったこと

1. **T03: 仮想カメラネイティブ実装**
   - `windows = "0.58"` crate 追加（feature `virtual-camera` で optional）
   - `src/output/virtual_camera_native.rs` — MediaFoundation `MFStartup` 初期化
   - `CreateFileMappingW` / `MapViewOfFile` で名前付き共有メモリに RGBA フレーム書き込み
   - VVC0 ヘッダー (magic + width + height + timestamp + frame_index + pixel_data)
   - フレームサイズバリデーション + エラーハンドリング
   - テスト 5 件追加

2. **T10: Multiple Avatars 対応**
   - `src/app/mod.rs` — `run_frame()` を `self.avatars` 全イテレーションに変更
   - `build_frame_input_multi(&[AvatarInstance])` 追加（全アバターの RenderAvatarInstance を生成）
   - `build_frame_input` / `build_frame_input_placeholder` を削除して統合
   - レンダラーは既に `Vec<RenderAvatarInstance>` をサポートしているので、そのまま動作

3. **T10: Drag-and-drop VRM Import**
   - `src/gui/mod.rs` — `eframe::App::update` で `ctx.input(|i| i.raw.dropped_files)` 処理
   - `.vrm` 拡張子のファイルを自動検出して `load_avatar_from_drop()` で読み込み
   - `add_avatar()` 経由でマルチアバターとして追加

4. **T10: Cloth LoD Authoring**
   - `src/simulation/cloth.rs` — `ClothLoDLevel` (Full/Half/Quarter/Custom) + `ClothLoDConfig`
   - 距離ベース自動選択: `ClothLoDConfig::select_for_distance(presets, camera_distance)`
   - LOD レベルごとの solver iterations 自動調整
   - `serde` Serialize/Deserialize 対応

5. **T10: Model Library 拡張**
   - `src/app/avatar_library.rs` — `AvatarLibraryEntry` に category/vrm_title/vrm_author/vrm_version/thumbnail_path 追加
   - `AvatarLibrary::categories()` / `by_category()` — カテゴリ別フィルタ
   - `export_catalog()` / `import_catalog()` — JSON カタログ共有

6. **T10: Image Background + Ground Alignment**
   - `src/renderer/frame_input.rs` — `RenderFrameInput` に `background_image_path` / `show_ground_grid` 追加
   - `src/app/mod.rs` — `Application` に `viewport_background` / `ground_grid_visible` フィールド
   - `snap_avatar_to_ground()` — Y座標を0にスナップ
   - `set_background_image()` / `set_ground_grid_visible()` — GUI からの制御

## 未完了項目

| タスク | 未完了 | 理由 |
|---|---|---|
| T03 | DirectShow フィルタとしての実際の仮想カメラデバイス登録 | カスタム DirectShow フィルタ DLL の作成が必要 |
| T10 | Thumbnail generation | オフスクリーンレンダリングパスが必要 |
| T10 | Folder watching | `notify` crate 統合が必要 |
| T10 | Multiple cloth overlays per avatar | 大規模機能 |
| T10 | Offline bake / cache generation | 大規模機能 |
| T10 | Webcam hand/lower-body tracking, GPU inference | 追跡スタックの拡張 |

## 主なファイルの現状

- `src/output/virtual_camera_native.rs` — MediaFoundation + CreateFileMappingW による共有メモリフレームプッシュ
- `src/app/mod.rs` — `run_frame` マルチアバター対応 + `build_frame_input_multi` + ground/background API
- `src/gui/mod.rs` — drag-and-drop VRM + wind_presets
- `src/gui/profile.rs` — ProfileLibrary export/import + WindPresetLibrary + WindPreset::sample_at
- `src/simulation/cloth.rs` — ClothLoDLevel + ClothLoDConfig + distance-based LOD selection
- `src/app/avatar_library.rs` — categories + VRM metadata + catalog export/import
- `src/renderer/frame_input.rs` — background_image_path + show_ground_grid
- `src/renderer/gpu_handle.rs` — Win32 KMT handle export
- `src/renderer/pipeline.rs` — MatCap fragment shader
- `Cargo.toml` — `ash`, `windows` (optional), `virtual-camera` feature
