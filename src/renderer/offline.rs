//! Offline (headless) avatar rendering helpers for benches and
//! validation binaries: pose an [`AvatarInstance`], render one frame
//! through the real Vulkan pipeline with a webcam-style framing, and
//! read the pixels back on the CPU. Extracted from `validate_gt` so the
//! replay benches can produce camera|avatar composites with the same
//! code path.

use std::sync::Arc;

use crate::app::ViewportCamera;
use crate::asset::{AvatarAsset, Mat4, Transform};
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use crate::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderColorSpace, RenderCullMode, RenderDebugFlags, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use crate::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use crate::renderer::VulkanRenderer;

/// Default bench framing: chest-height target, close enough that the
/// torso fills the frame the way a desk webcam sees a streamer.
pub fn bench_camera() -> ViewportCamera {
    ViewportCamera {
        distance: 1.6,
        pan: [0.0, 0.95],
        ..ViewportCamera::default()
    }
}

/// Build a one-avatar frame input with the given camera.
pub fn build_frame_input(
    avatar: &AvatarInstance,
    extent: [u32; 2],
    camera: &ViewportCamera,
) -> RenderFrameInput {
    let mesh_instances: Vec<RenderMeshInstance> = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.primitives.iter().map(|prim| {
                let material_asset =
                    avatar.asset.materials.iter().find(|m| m.id == prim.material_id);
                let mut material_binding = material_asset
                    .map(MaterialUploadRequest::from_asset_material)
                    .unwrap_or_else(MaterialUploadRequest::default_material);
                material_binding.mode = MaterialShaderMode::ToonLike;
                let alpha_mode = match material_binding.alpha_mode {
                    crate::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
                    crate::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                    crate::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
                };
                let cull_mode = if material_binding.double_sided {
                    RenderCullMode::DoubleSided
                } else {
                    RenderCullMode::BackFace
                };
                RenderMeshInstance {
                    mesh_id: mesh.id,
                    primitive_id: prim.id,
                    material_binding,
                    bounds: prim.bounds,
                    alpha_mode,
                    cull_mode,
                    outline: Default::default(),
                    primitive_data: Some(Arc::clone(prim)),
                    morph_weights: Vec::new(),
                }
            })
        })
        .collect();

    let (view, eye_pos) = build_view_matrix(camera);
    let projection = build_projection_matrix(
        camera.fov_deg,
        extent[0] as f32 / extent[1].max(1) as f32,
        0.1,
        1000.0,
    );

    RenderFrameInput {
        camera: CameraState {
            view,
            projection,
            position_ws: eye_pos,
            viewport_extent: extent,
        },
        lighting: LightingState::default(),
        instances: vec![RenderAvatarInstance {
            instance_id: avatar.id,
            world_transform: avatar.world_transform.clone(),
            mesh_instances,
            skinning_matrices: avatar.pose.skinning_matrices.clone(),
            cloth_deforms: Vec::new(),
            debug_flags: RenderDebugFlags::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent,
            color_space: RenderColorSpace::Srgb,
            alpha_mode: RenderOutputAlpha::Opaque,
            export_mode: RenderExportMode::CpuReadback,
            msaa: crate::renderer::frame_input::MsaaMode::Off,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.35, 0.35, 0.38],
        transparent_background: false,
        avatar_opacity: 1.0,
        bloom: Default::default(),
        generative_background: Default::default(),
        background_tracking: Default::default(),
        time_seconds: 0.0,
    }
}

/// Render one frame of `avatar` and return its RGBA pixels. Renders the
/// frame twice so the pipelined CPU readback harvests THIS frame's pixels.
pub fn render_avatar(
    renderer: &mut VulkanRenderer,
    avatar: &AvatarInstance,
    extent: [u32; 2],
    camera: &ViewportCamera,
) -> Result<Vec<u8>, String> {
    let frame_input = build_frame_input(avatar, extent, camera);
    let _warm = renderer
        .render(&frame_input)
        .map_err(|e| format!("warm render: {e}"))?;
    let result = renderer
        .render(&frame_input)
        .map_err(|e| format!("render: {e}"))?;
    let pixels = result
        .exported_frame
        .as_ref()
        .and_then(|f| f.cpu_pixel_data())
        .ok_or_else(|| "no pixel data".to_string())?;
    Ok((*pixels).clone())
}

/// Instance with explicit local transforms, fully posed for rendering.
pub fn make_instance(asset: &Arc<AvatarAsset>, locals: Vec<Transform>) -> AvatarInstance {
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(asset));
    avatar.build_base_pose();
    avatar.pose.local_transforms = locals;
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();
    avatar
}

pub fn build_view_matrix(cam: &ViewportCamera) -> (Mat4, [f32; 3]) {
    let yaw = cam.yaw_deg.to_radians();
    let pitch = cam.pitch_deg.to_radians();
    let (sy, cy) = (yaw.sin(), yaw.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let wx = cam.pan[0] * cy;
    let wy = cam.pan[1];
    let wz = cam.pan[0] * (-sy);
    let eye_x = cam.distance * cp * sy + wx;
    let eye_y = cam.distance * sp + wy;
    let eye_z = cam.distance * cp * cy + wz;
    let target = [wx, wy, wz];
    let fwd = [target[0] - eye_x, target[1] - eye_y, target[2] - eye_z];
    let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt().max(1e-6);
    let f = [fwd[0] / len, fwd[1] / len, fwd[2] / len];
    let world_up = [0.0f32, 1.0, 0.0];
    let r = [
        f[1] * world_up[2] - f[2] * world_up[1],
        f[2] * world_up[0] - f[0] * world_up[2],
        f[0] * world_up[1] - f[1] * world_up[0],
    ];
    let rlen = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-6);
    let r = [r[0] / rlen, r[1] / rlen, r[2] / rlen];
    let u = [
        r[1] * f[2] - r[2] * f[1],
        r[2] * f[0] - r[0] * f[2],
        r[0] * f[1] - r[1] * f[0],
    ];
    (
        [
            [r[0], r[1], r[2], -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z)],
            [u[0], u[1], u[2], -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z)],
            [-f[0], -f[1], -f[2], f[0] * eye_x + f[1] * eye_y + f[2] * eye_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [eye_x, eye_y, eye_z],
    )
}

pub fn build_projection_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_deg.to_radians() * 0.5).tan();
    let a = far / (near - far);
    let b = far * near / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0],
    ]
}
