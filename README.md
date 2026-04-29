# VulVATAR

POC-first Rust project for rendering `VRM 1.0` avatars with `Vulkano`, with room for:

- MToon-like shading
- spring bone secondary motion
- cloth simulation
- webcam-based motion tracking
- OBS-routable output

This repository is intentionally design-first. The current code is only a scaffold for validating module boundaries.

See [architecture.md](/C:/lib/github/kjranyone/VulVATAR/docs/architecture.md) for the current system design.

## Prerequisites

- **Vulkan SDK**: [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) をインストールしてください。shaderc ライブラリが同梱されており、ビルドに必要です。
  - インストール後、環境変数 `VULKAN_SDK` が設定されていることを確認してください（インストーラが自動設定します）。
  - 新しいターミナルを開いてから `cargo build` を実行してください。

## Current Direction

- target `VRM 1.0 only`
- keep asset, simulation, and renderer layers separate
- keep tracking and output as separate subsystems
- use final avatar pose as the main cross-layer contract
- treat spring bones and cloth as different simulation problems
- treat webcam tracking as rig input, not direct bone ownership
- aim for `unlit/simple-lit skinned rendering` before exact MToon parity

## Current Modules

- `src/app/`: frame orchestration
- `src/asset/`: VRM asset loading
- `src/avatar/`: runtime avatar state
- `src/simulation/`: spring and cloth simulation
- `src/renderer/`: Vulkano rendering
- `src/tracking/`: webcam tracking and retargeting
- `src/output/`: OBS and frame output sinks

## Next Design Focus

1. define concrete Rust structs for avatar, spring, and cloth runtime state
2. lock the VRM 1.0 loader boundary
3. design the first MToon-like material parameter set
4. choose the first cloth target region for the POC
5. define the first tracking target set and output sink contract

## License

VulVATAR is licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for the full license text.

If you distribute a built binary, provide the corresponding source code
for that binary as required by GPL-3.0.

## Third-party components

VulVATAR uses third-party libraries and model assets under their own
licenses. In particular:

- MediaPipe Pose / Hand / Face Landmarker models (Apache-2.0), distributed
  by Google AI Edge and re-exported as ONNX by the OpenCV community:
  - https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
  - https://huggingface.co/opencv/pose_estimation_mediapipe
  - https://huggingface.co/opencv/palm_detection_mediapipe
  - https://huggingface.co/opencv/handpose_estimation_mediapipe
- ONNX Runtime: MIT
  - https://github.com/microsoft/onnxruntime
- Rust crate dependencies: see each crate's license metadata.
