# VulVATAR

Windows 専用の VRM 1.0 アバター モーションキャプチャ・アプリケーション。
**Intel RealSense D435** デプスカメラを唯一のキャプチャバックエンドとして演奏者の全身（顔・手・体）をトラッキングし、`Vulkano` (Vulkan) でレンダリングしたアバターを Windows 11 Media Foundation 仮想カメラ経由で OBS Studio / Meet / Teams / Zoom 等へ送出します。

実装されている主な機能:

- VRM 1.0 アセット読み込みと `MToon-like` スキンドレンダリング
- スプリングボーンによるセカンダリモーション
- 布シミュレーション（CPU XPBD / GPU 計算パス）
- **RealSense D435** によるデプス付き全身ポーズトラッキング（RTMW3D + YOLOX + MediaPipe FaceMesh/Blendshapes, ONNX Runtime / DirectML）
- リップシンク（マイク入力）
- Media Foundation 仮想カメラ (`vulvatar-mf-camera`) 経由の映像出力

システム設計の全体は [docs/architecture.md](docs/architecture.md) を参照してください。

> **注意**: このプロジェクトは Windows 専用です（MF 仮想カメラ、Vulkan on D3D12、DirectML 推論）。キャプチャは D435 専用ビルドで、`realsense` 機能は `default` 機能に含まれます（D435 以外のカメラには対応しません）。

## Prerequisites

- **Vulkan SDK**: [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) をインストールしてください。shaderc ライブラリが同梱されており、ビルドに必要です。
  - インストール後、環境変数 `VULKAN_SDK` が設定されていることを確認してください（インストーラが自動設定します）。
  - 新しいターミナルを開いてから `cargo build` を実行してください。
- **Intel RealSense SDK 2.0**: キャプチャバックエンド（`librealsense2`）のリンクに必要です。Windows インストーラで導入します。
- **LLVM / libclang** と **pkg-config**: `realsense-rust` の `buildtime-bindgen` が FFI バインディングを再生成するのに使います（`winget install LLVM.LLVM` / `winget install bloodrock.pkg-config-lite`）。

RealSense SDK のパスや `PKG_CONFIG_PATH` / `LIBCLANG_PATH` の設定、`dev.ps1` の `build (realsense)` / `run (realsense)` メニューなど、ネイティブビルドの手順は [docs/realsense-build.md](docs/realsense-build.md) にまとめてあります。

## Current Direction

- target `VRM 1.0 only`（`VRM 0.x` はベストエフォートのローダ shim のみ、互換性保証なし）
- キャプチャは **D435 専用**。デプスをトラッキングへのメトリック入力として扱い、ボーンを直接支配させない
- アセット / シミュレーション / レンダラの各層を分離し、アバターポーズをレイヤー間の主契約にする
- トラッキングと出力を独立したサブシステムとして扱う
- スプリングボーンと布を別々のシミュレーション問題として扱う

## Current Modules

- `src/app/`: フレーム orchestration
- `src/asset/`: VRM/glTF アセット読み込み
- `src/avatar/`: 実行時アバター状態
- `src/editor/`: 布オーサリング等のプロジェクトローカル編集
- `src/simulation/`: スプリングボーンと布のシミュレーション
- `src/renderer/`: Vulkano レンダリング
- `src/tracking/`: RealSense D435 デプスキャプチャ、ポーズ推定、リターゲット
- `src/lipsync/`: マイク入力からのリップシンク
- `src/output/`: フレーム出力シンク（MF 仮想カメラへ共有メモリ送信）
- `src/gui/`: eframe/egui ベースの GUI
- `vulvatar-mf-camera/`: Windows 11 用 Media Foundation 仮想カメラ DLL（別クレート）

## 関連ドキュメント

- [docs/architecture.md](docs/architecture.md) — システム設計
- [docs/realsense-build.md](docs/realsense-build.md) — RealSense ネイティブビルド手順
- [docs/mf-virtual-camera.md](docs/mf-virtual-camera.md) — Media Foundation 仮想カメラ
- [docs/threading-model.md](docs/threading-model.md) — GUI / レンダラ / ワーカーのスレッド構成
- [docs/profiling.md](docs/profiling.md) — 計測レシピ

## License

VulVATAR is licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for the full license text.

If you distribute a built binary, provide the corresponding source code
for that binary as required by GPL-3.0.

## Third-party components

VulVATAR uses third-party libraries and model assets under their own
licenses. In particular:

- RTMW3D-x whole-body 3D pose estimation model (Apache-2.0), trained
  by OpenMMLab / mmpose and re-distributed as ONNX by Soykaf:
  - https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d
  - https://huggingface.co/Soykaf/RTMW3D-x
- YOLOX-m Human-Art tuned person detector (Apache-2.0), exported as
  ONNX by OpenMMLab and bundled in mmpose's `rtmposev1` ONNX SDK:
  - https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- MediaPipe FaceMeshV2 (478 face landmarks) and BlendshapeV2 (52 ARKit
  blendshape coefficients) by Google AI Edge (Apache-2.0), packaged
  as ONNX in PINTO_model_zoo:
  - https://github.com/PINTO0309/PINTO_model_zoo/tree/main/410_FaceMeshV2
  - https://github.com/PINTO0309/PINTO_model_zoo/tree/main/390_BlendShapeV2
- ONNX Runtime: MIT
  - https://github.com/microsoft/onnxruntime
- Rust crate dependencies: see each crate's license metadata.
