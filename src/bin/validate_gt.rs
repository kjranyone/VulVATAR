//! Ground-truth round-trip validation (self-consistency rendering).
//!
//! `validate_pipeline` scores avatar-vs-SOURCE parity, which is blind
//! to (a) tracking errors (the source itself being wrong) and (b)
//! twist around bone axes (direction vectors don't change under
//! twist). This harness closes both holes by inverting the setup:
//! pose the avatar with KNOWN joint rotations, render it with a known
//! camera, run the full tracking+solve pipeline on the render, and
//! compare the recovered pose against the ground truth with
//! position-based coupling metrics that DO see twist and contact:
//!
//! * shoulder-line yaw delta vs neutral  — trunk twist coupling
//! * inter-wrist distance               — hands-contact preservation
//! * wrist forward (z) / height (y)     — arm elevation coupling
//! * head world-rotation geodesic delta — head coupling
//!
//! Caveat: RTMW3D is trained on photos; detection on a stylized
//! render is noisier than on a webcam frame. The metrics here are
//! therefore COARSE coupling gains (does a 30° twist come back as
//! ~30°, ~0°, or inverted?), not sub-degree accuracy scores.
//!
//! Output: diagnostics/validation_gt/summary.md + per-pose composite
//! renders (GT | recovered).
//!
//! Usage: cargo run --bin validate_gt [-- --filter NAME]

use std::path::PathBuf;
use std::sync::Arc;

use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::{AvatarAsset, HumanoidBone, NodeId, SkeletonAsset, Transform};
use vulvatar_lib::avatar::pose_solver::{solve_avatar_pose, PoseSolverState, SolverParams};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::math_utils::{
    quat_conjugate, quat_mul, quat_normalize, quat_rotate_vec3, vec3_add, vec3_length,
    vec3_normalize, vec3_scale, vec3_sub,
};
use vulvatar_lib::renderer::frame_input::{
    CameraState, LightingState, OutputTargetRequest, RenderAlphaMode, RenderAvatarInstance,
    RenderColorSpace, RenderCullMode, RenderDebugFlags, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::tracking::provider::create_pose_provider;

const RENDER_EXTENT: [u32; 2] = [1024, 1024];
/// AliciaSolid, NOT AvatarSample_A: the tracker reads Alicia's
/// silhouette far more reliably (neutral shoulder-yaw bias −1.8° vs
/// −44.9°, head-yaw→shoulder leak −5.4° vs +45°, right-twist coupling
/// gain 1.26 vs ~0.3). With AvatarSample_A the harness mostly
/// measures detection mush on its huge sleeves instead of pipeline
/// behaviour. Override with VULVATAR_VRM for rig A/B.
const VRM_PATH_DEFAULT: &str = "sample_data/AliciaSolid.vrm";
const OUT_DIR: &str = "diagnostics/validation_gt";

type Quat = [f32; 4];
type Vec3 = [f32; 3];

// -----------------------------------------------------------------------
// Pose suite
// -----------------------------------------------------------------------

/// One ground-truth posing operation, applied sequentially with an FK
/// refresh between ops (so `AimAt` sees the effect of earlier ops).
enum PoseOp {
    /// Pre-multiply the bone's world rotation by an axis-angle delta.
    RotateWorld(HumanoidBone, Vec3, f32),
    /// Rotate the bone so its bone→tip world direction points at a
    /// target expressed relative to the shoulder midpoint, in units of
    /// the avatar's full arm length (rig-agnostic).
    AimAt {
        bone: HumanoidBone,
        tip: HumanoidBone,
        /// [x right, y up, z toward camera] × arm_length, origin at
        /// the shoulder midpoint.
        target_rel: Vec3,
    },
}

/// Natural standing base: the sample rig rests in a T-pose, which no
/// webcam subject ever holds (and which parks the wrists at the frame
/// edge where detection degrades). Every pose starts from arms
/// hanging slightly outward instead.
fn base_ops() -> Vec<PoseOp> {
    use HumanoidBone::*;
    use PoseOp::*;
    vec![
        AimAt { bone: LeftUpperArm, tip: LeftLowerArm, target_rel: [0.45, -0.5, 0.05] },
        AimAt { bone: LeftLowerArm, tip: LeftHand, target_rel: [0.55, -1.0, 0.1] },
        AimAt { bone: RightUpperArm, tip: RightLowerArm, target_rel: [-0.45, -0.5, 0.05] },
        AimAt { bone: RightLowerArm, tip: RightHand, target_rel: [-0.55, -1.0, 0.1] },
    ]
}

fn pose_suite() -> Vec<(&'static str, Vec<PoseOp>)> {
    use HumanoidBone::*;
    use PoseOp::*;
    let y: Vec3 = [0.0, 1.0, 0.0];
    let z: Vec3 = [0.0, 0.0, 1.0];
    let with_base = |extra: Vec<PoseOp>| -> Vec<PoseOp> {
        let mut ops = base_ops();
        ops.extend(extra);
        ops
    };
    vec![
        ("neutral", with_base(vec![])),
        ("twist_left_30", with_base(vec![RotateWorld(UpperChest, y, 30.0)])),
        ("twist_right_30", with_base(vec![RotateWorld(UpperChest, y, -30.0)])),
        ("head_yaw_30", with_base(vec![RotateWorld(Head, y, 30.0)])),
        ("shrug_left_20", with_base(vec![RotateWorld(LeftShoulder, z, 20.0)])),
        (
            "arms_forward",
            with_base(vec![
                AimAt { bone: LeftUpperArm, tip: LeftLowerArm, target_rel: [0.15, -0.1, 0.6] },
                AimAt { bone: LeftLowerArm, tip: LeftHand, target_rel: [0.15, -0.1, 1.2] },
                AimAt { bone: RightUpperArm, tip: RightLowerArm, target_rel: [-0.15, -0.1, 0.6] },
                AimAt { bone: RightLowerArm, tip: RightHand, target_rel: [-0.15, -0.1, 1.2] },
            ]),
        ),
        (
            "palms_meet",
            with_base(vec![
                AimAt { bone: LeftUpperArm, tip: LeftLowerArm, target_rel: [0.0, -0.25, 0.55] },
                AimAt { bone: LeftLowerArm, tip: LeftHand, target_rel: [0.0, -0.15, 0.7] },
                AimAt { bone: RightUpperArm, tip: RightLowerArm, target_rel: [0.0, -0.25, 0.55] },
                AimAt { bone: RightLowerArm, tip: RightHand, target_rel: [0.0, -0.15, 0.7] },
            ]),
        ),
        (
            // Fingertips touching at the midline in front of the chest
            // (elbows out, forearms angled inward to meet at x≈0). Lower
            // than palms_meet so detection has a better chance; the
            // L/R-swap-at-the-midline crossing repro.
            "fingertips_touch",
            with_base(vec![
                AimAt { bone: LeftUpperArm, tip: LeftLowerArm, target_rel: [0.28, -0.35, 0.30] },
                AimAt { bone: LeftLowerArm, tip: LeftHand, target_rel: [0.0, -0.35, 0.55] },
                AimAt { bone: RightUpperArm, tip: RightLowerArm, target_rel: [-0.28, -0.35, 0.30] },
                AimAt { bone: RightLowerArm, tip: RightHand, target_rel: [0.0, -0.35, 0.55] },
            ]),
        ),
    ]
}

// -----------------------------------------------------------------------
// FK + posing
// -----------------------------------------------------------------------

/// World (rotation, position) per node from local transforms.
fn fk(skeleton: &SkeletonAsset, locals: &[Transform]) -> Vec<(Quat, Vec3)> {
    let mut world: Vec<(Quat, Vec3)> = vec![([0.0, 0.0, 0.0, 1.0], [0.0; 3]); skeleton.nodes.len()];
    let mut stack: Vec<NodeId> = skeleton.root_nodes.clone();
    while let Some(NodeId(idx)) = stack.pop() {
        let i = idx as usize;
        if i >= skeleton.nodes.len() || i >= locals.len() {
            continue;
        }
        let (p_rot, p_pos) = skeleton.nodes[i]
            .parent
            .map(|NodeId(p)| world[p as usize])
            .unwrap_or(([0.0, 0.0, 0.0, 1.0], [0.0; 3]));
        let rot = quat_normalize(&quat_mul(&p_rot, &locals[i].rotation));
        let pos = vec3_add(&p_pos, &quat_rotate_vec3(&p_rot, &locals[i].translation));
        world[i] = (rot, pos);
        stack.extend(skeleton.nodes[i].children.iter().copied());
    }
    world
}

fn axis_angle(axis: &Vec3, deg: f32) -> Quat {
    let a = vec3_normalize(axis);
    let h = deg.to_radians() * 0.5;
    let s = h.sin();
    [a[0] * s, a[1] * s, a[2] * s, h.cos()]
}

/// Shortest-arc quaternion from one direction to another.
fn quat_between(from: &Vec3, to: &Vec3) -> Quat {
    let f = vec3_normalize(from);
    let t = vec3_normalize(to);
    let cross = [
        f[1] * t[2] - f[2] * t[1],
        f[2] * t[0] - f[0] * t[2],
        f[0] * t[1] - f[1] * t[0],
    ];
    let dot = f[0] * t[0] + f[1] * t[1] + f[2] * t[2];
    if dot < -0.9999 {
        // Opposite: rotate 180° around any perpendicular.
        let perp = if f[0].abs() < 0.9 {
            vec3_normalize(&[0.0, -f[2], f[1]])
        } else {
            vec3_normalize(&[-f[2], 0.0, f[0]])
        };
        return [perp[0], perp[1], perp[2], 0.0];
    }
    let w = 1.0 + dot;
    quat_normalize(&[cross[0], cross[1], cross[2], w])
}

/// Pre-multiply `bone`'s world rotation by `delta_world`, expressed as
/// a local-rotation update so child bones follow through FK.
fn apply_world_delta(
    skeleton: &SkeletonAsset,
    humanoid: &std::collections::HashMap<HumanoidBone, NodeId>,
    locals: &mut [Transform],
    bone: HumanoidBone,
    delta_world: Quat,
) {
    let Some(NodeId(idx)) = humanoid.get(&bone).copied() else {
        return;
    };
    let i = idx as usize;
    let world = fk(skeleton, locals);
    let p_rot = skeleton.nodes[i]
        .parent
        .map(|NodeId(p)| world[p as usize].0)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let local_delta = quat_mul(&quat_mul(&quat_conjugate(&p_rot), &delta_world), &p_rot);
    locals[i].rotation = quat_normalize(&quat_mul(&local_delta, &locals[i].rotation));
}

/// Apply the pose ops on top of the rest pose; returns the GT locals.
fn build_gt_pose(asset: &AvatarAsset, ops: &[PoseOp]) -> Vec<Transform> {
    let skeleton = &asset.skeleton;
    let humanoid = &asset.humanoid.as_ref().expect("humanoid map").bone_map;
    let mut locals: Vec<Transform> = skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();

    // Rig-agnostic frame: shoulder midpoint + arm length from rest FK.
    let rest = fk(skeleton, &locals);
    let pos_of = |b: HumanoidBone| -> Option<Vec3> {
        humanoid.get(&b).map(|NodeId(i)| rest[*i as usize].1)
    };
    let (sl, sr) = (
        pos_of(HumanoidBone::LeftUpperArm).expect("L shoulder"),
        pos_of(HumanoidBone::RightUpperArm).expect("R shoulder"),
    );
    let mid = vec3_scale(&vec3_add(&sl, &sr), 0.5);
    let arm_len = {
        let e = pos_of(HumanoidBone::LeftLowerArm).expect("L elbow");
        let w = pos_of(HumanoidBone::LeftHand).expect("L wrist");
        vec3_length(&vec3_sub(&e, &sl)) + vec3_length(&vec3_sub(&w, &e))
    };

    for op in ops {
        match op {
            PoseOp::RotateWorld(bone, axis, deg) => {
                apply_world_delta(skeleton, humanoid, &mut locals, *bone, axis_angle(axis, *deg));
            }
            PoseOp::AimAt { bone, tip, target_rel } => {
                let world = fk(skeleton, &locals);
                let (Some(NodeId(bi)), Some(NodeId(ti))) =
                    (humanoid.get(bone).copied(), humanoid.get(tip).copied())
                else {
                    continue;
                };
                let base = world[bi as usize].1;
                let tip_pos = world[ti as usize].1;
                let target = vec3_add(&mid, &vec3_scale(target_rel, arm_len));
                let cur = vec3_sub(&tip_pos, &base);
                let want = vec3_sub(&target, &base);
                if vec3_length(&cur) < 1e-5 || vec3_length(&want) < 1e-5 {
                    continue;
                }
                let delta = quat_between(&cur, &want);
                apply_world_delta(skeleton, humanoid, &mut locals, *bone, delta);
            }
        }
    }
    locals
}

// -----------------------------------------------------------------------
// Metrics
// -----------------------------------------------------------------------

/// All distance metrics are normalised by that skeleton's own
/// shoulder span, so ground truth (avatar units), detection (source
/// units) and recovery (avatar units) compare on one scale.
#[derive(Clone, Copy, Default)]
struct PoseMetrics {
    /// Shoulder-line yaw in degrees: atan2(z, x) of L−R shoulder.
    shoulder_yaw_deg: f32,
    /// |LeftHand − RightHand| / span.
    wrist_gap: f32,
    /// Mean wrist z (forward, toward camera) − shoulder mid z, / span.
    wrist_forward: f32,
    /// Mean wrist y − shoulder mid y, / span.
    wrist_height: f32,
    /// Left shoulder-joint height − mid, / span.
    left_shoulder_lift: f32,
    /// Head world-rotation geodesic angle vs rest, degrees (NaN for
    /// the source skeleton — it has no head rotation, only a face
    /// pose).
    head_rot_deg: f32,
}

/// Metrics computed directly from the DETECTED source skeleton —
/// isolates tracking error (GT→SRC) from solver error (SRC→REC).
fn metrics_source(sk: &vulvatar_lib::tracking::source_skeleton::SourceSkeleton) -> Option<PoseMetrics> {
    let p = |b: HumanoidBone| sk.joints.get(&b).map(|j| j.position);
    let sl = p(HumanoidBone::LeftUpperArm)?;
    let sr = p(HumanoidBone::RightUpperArm)?;
    let wl = p(HumanoidBone::LeftHand)?;
    let wr = p(HumanoidBone::RightHand)?;
    let span = vec3_length(&vec3_sub(&sl, &sr)).max(1e-4);
    let mid = vec3_scale(&vec3_add(&sl, &sr), 0.5);
    let line = vec3_sub(&sl, &sr);
    Some(PoseMetrics {
        shoulder_yaw_deg: line[2].atan2(line[0]).to_degrees(),
        wrist_gap: vec3_length(&vec3_sub(&wl, &wr)) / span,
        wrist_forward: ((wl[2] - mid[2]) + (wr[2] - mid[2])) * 0.5 / span,
        wrist_height: ((wl[1] - mid[1]) + (wr[1] - mid[1])) * 0.5 / span,
        left_shoulder_lift: (sl[1] - mid[1]) / span,
        head_rot_deg: f32::NAN,
    })
}

fn metrics(asset: &AvatarAsset, locals: &[Transform], rest_head: &Quat) -> PoseMetrics {
    let skeleton = &asset.skeleton;
    let humanoid = &asset.humanoid.as_ref().expect("humanoid").bone_map;
    let world = fk(skeleton, locals);
    let pos = |b: HumanoidBone| -> Vec3 {
        humanoid
            .get(&b)
            .map(|NodeId(i)| world[*i as usize].1)
            .unwrap_or([0.0; 3])
    };
    let sl = pos(HumanoidBone::LeftUpperArm);
    let sr = pos(HumanoidBone::RightUpperArm);
    let span = vec3_length(&vec3_sub(&sl, &sr)).max(1e-4);
    let mid = vec3_scale(&vec3_add(&sl, &sr), 0.5);
    let line = vec3_sub(&sl, &sr);
    let wl = pos(HumanoidBone::LeftHand);
    let wr = pos(HumanoidBone::RightHand);

    let head_rot = humanoid
        .get(&HumanoidBone::Head)
        .map(|NodeId(i)| world[*i as usize].0)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let d = quat_mul(&head_rot, &quat_conjugate(rest_head));
    let head_rot_deg = 2.0 * d[3].clamp(-1.0, 1.0).acos().to_degrees().min(360.0);
    let head_rot_deg = if head_rot_deg > 180.0 { 360.0 - head_rot_deg } else { head_rot_deg };

    PoseMetrics {
        shoulder_yaw_deg: line[2].atan2(line[0]).to_degrees(),
        wrist_gap: vec3_length(&vec3_sub(&wl, &wr)) / span,
        wrist_forward: ((wl[2] - mid[2]) + (wr[2] - mid[2])) * 0.5 / span,
        wrist_height: ((wl[1] - mid[1]) + (wr[1] - mid[1])) * 0.5 / span,
        left_shoulder_lift: (sl[1] - mid[1]) / span,
        head_rot_deg,
    }
}

// -----------------------------------------------------------------------
// Render + recover
// -----------------------------------------------------------------------

fn make_instance(asset: &Arc<AvatarAsset>, locals: Vec<Transform>) -> AvatarInstance {
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(asset));
    avatar.build_base_pose();
    avatar.pose.local_transforms = locals;
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();
    avatar
}

fn render_avatar(
    renderer: &mut VulkanRenderer,
    avatar: &AvatarInstance,
    extent: [u32; 2],
) -> Result<Vec<u8>, String> {
    let frame_input = build_frame_input(avatar, extent);
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

fn build_frame_input(avatar: &AvatarInstance, extent: [u32; 2]) -> RenderFrameInput {
    let mesh_instances: Vec<RenderMeshInstance> = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|mesh| {
            mesh.primitives.iter().map(|prim| {
                let material_asset = avatar.asset.materials.iter().find(|m| m.id == prim.material_id);
                let mut material_binding = material_asset
                    .map(MaterialUploadRequest::from_asset_material)
                    .unwrap_or_else(MaterialUploadRequest::default_material);
                material_binding.mode = MaterialShaderMode::ToonLike;
                let alpha_mode = match material_binding.alpha_mode {
                    vulvatar_lib::asset::AlphaMode::Opaque => RenderAlphaMode::Opaque,
                    vulvatar_lib::asset::AlphaMode::Mask(_) => RenderAlphaMode::Cutout,
                    vulvatar_lib::asset::AlphaMode::Blend => RenderAlphaMode::Blend,
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

    // Upper-body webcam-style framing: chest-height target, close
    // enough that the torso fills the frame the way a desk webcam
    // sees a streamer (the pipeline's dominant use case).
    let camera = ViewportCamera {
        distance: 1.6,
        pan: [0.0, 0.95],
        ..ViewportCamera::default()
    };
    let (view, eye_pos) = build_view_matrix(&camera);
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
            msaa: vulvatar_lib::renderer::frame_input::MsaaMode::Off,
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

fn build_view_matrix(cam: &ViewportCamera) -> (vulvatar_lib::asset::Mat4, [f32; 3]) {
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

fn build_projection_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> vulvatar_lib::asset::Mat4 {
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

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut filter: Option<String> = None;
    let mut depth_enabled = true;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--filter" => {
                filter = args.get(i + 1).cloned();
                i += 2;
            }
            // Provenance probe: shoulder z is touched only by raw
            // RTMW3D nz and by the depth pipeline's DAv2 bz
            // injection; A/B-ing this flag attributes spurious
            // shoulder Δz to one or the other.
            "--no-depth" => {
                depth_enabled = false;
                i += 1;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    let vrm = std::env::var("VULVATAR_VRM").unwrap_or_else(|_| VRM_PATH_DEFAULT.to_string());
    eprintln!("loading VRM {vrm}…");
    let asset = VrmAssetLoader::new()
        .load(&vrm)
        .map_err(|e| format!("load VRM: {e:?}"))?;

    eprintln!("loading pose provider (depth_enabled={depth_enabled})…");
    let config = vulvatar_lib::tracking::provider::TrackingPipelineConfig {
        depth_enabled,
        ..Default::default()
    };
    let mut infer =
        create_pose_provider("models", config).map_err(|e| format!("provider: {e}"))?;
    for w in infer.take_load_warnings() {
        eprintln!("  warning: {w}");
    }

    eprintln!("initializing Vulkan renderer…");
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let out_dir = PathBuf::from(OUT_DIR);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("mkdir: {e}"))?;

    // Rest head rotation for the head metric baseline.
    let rest_locals: Vec<Transform> =
        asset.skeleton.nodes.iter().map(|n| n.rest_local.clone()).collect();
    let rest_world = fk(&asset.skeleton, &rest_locals);
    let rest_head = asset
        .humanoid
        .as_ref()
        .and_then(|h| h.bone_map.get(&HumanoidBone::Head))
        .map(|NodeId(i)| rest_world[*i as usize].0)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);

    // Neutral render for the per-pose session warmup: live tracking
    // learns its session estimators (frontal shoulder span, torso
    // depth EMA) while the user sits naturally before moving. Feeding
    // each pose as an isolated single frame instead would deny the
    // pipeline exactly the calibration the live session always has —
    // and the anatomical floor then fabricates twist on narrow rigs.
    let neutral_rgba = {
        let locals = build_gt_pose(&asset, &pose_suite()[0].1);
        let avatar = make_instance(&asset, locals);
        render_avatar(&mut renderer, &avatar, RENDER_EXTENT)?
    };
    let neutral_rgb: Vec<u8> = neutral_rgba
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();

    let mut rows: Vec<(String, PoseMetrics, PoseMetrics, PoseMetrics)> = Vec::new();
    for (idx, (name, ops)) in pose_suite().into_iter().enumerate() {
        if let Some(f) = filter.as_deref() {
            if !name.contains(f) {
                continue;
            }
        }
        // GT pose + render.
        let gt_locals = build_gt_pose(&asset, &ops);
        let gt_metrics = metrics(&asset, &gt_locals, &rest_head);
        let gt_avatar = make_instance(&asset, gt_locals.clone());
        let rgba = render_avatar(&mut renderer, &gt_avatar, RENDER_EXTENT)?;

        // RGBA → RGB for the tracker.
        let rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();

        // Track + solve: reset, neutral warmup (session calibration),
        // then two passes on the pose frame (acquire + self-tracked
        // crop — same protocol as validate_pipeline).
        infer.reset_temporal_state();
        infer.set_calibration_mode_hint(Some(vulvatar_lib::tracking::CalibrationMode::FullBody));
        let base = (idx as u64) * 8;
        for k in 0..3 {
            let _ = infer.estimate_pose(&neutral_rgb, RENDER_EXTENT[0], RENDER_EXTENT[1], base + k);
        }
        let _ = infer.estimate_pose(&rgb, RENDER_EXTENT[0], RENDER_EXTENT[1], base + 3);
        let est = infer.estimate_pose(&rgb, RENDER_EXTENT[0], RENDER_EXTENT[1], base + 4);

        let mut rec_locals = rest_locals.clone();
        let mut state = PoseSolverState::default();
        let params = SolverParams {
            rotation_blend: 1.0,
            joint_confidence_threshold: 0.1,
            face_confidence_threshold: 0.1,
            ..Default::default()
        };
        solve_avatar_pose(
            &est.skeleton,
            &asset.skeleton,
            asset.humanoid.as_ref(),
            &mut rec_locals,
            &params,
            &mut state,
        );
        // Probe: raw source shoulder positions (z provenance triage).
        if std::env::var("VULVATAR_GT_PROBE").is_ok() {
            for b in [
                HumanoidBone::LeftUpperArm,
                HumanoidBone::RightUpperArm,
                HumanoidBone::LeftLowerArm,
                HumanoidBone::RightLowerArm,
                HumanoidBone::LeftHand,
                HumanoidBone::RightHand,
            ] {
                if let Some(j) = est.skeleton.joints.get(&b) {
                    eprintln!(
                        "  probe {b:?}: pos=[{:+.3},{:+.3},{:+.3}] conf={:.2}",
                        j.position[0], j.position[1], j.position[2], j.confidence
                    );
                }
            }
        }
        let src_metrics = metrics_source(&est.skeleton).unwrap_or_default();
        let rec_metrics = metrics(&asset, &rec_locals, &rest_head);

        // Side-by-side composite GT | recovered for visual triage.
        let rec_avatar = make_instance(&asset, rec_locals);
        if let Ok(rec_rgba) = render_avatar(&mut renderer, &rec_avatar, RENDER_EXTENT) {
            let _ = save_composite(&out_dir.join(format!("{name}.png")), &rgba, &rec_rgba);
        }

        eprintln!(
            "[{name}] yaw gt={:+.1}° src={:+.1}° rec={:+.1}° | gap gt={:.2} src={:.2} rec={:.2} | fwd gt={:+.2} src={:+.2} rec={:+.2}",
            gt_metrics.shoulder_yaw_deg,
            src_metrics.shoulder_yaw_deg,
            rec_metrics.shoulder_yaw_deg,
            gt_metrics.wrist_gap,
            src_metrics.wrist_gap,
            rec_metrics.wrist_gap,
            gt_metrics.wrist_forward,
            src_metrics.wrist_forward,
            rec_metrics.wrist_forward,
        );
        rows.push((name.to_string(), gt_metrics, src_metrics, rec_metrics));
    }

    write_summary(&out_dir.join("summary.md"), &rows)?;
    eprintln!("wrote: {}", out_dir.display());
    Ok(())
}

fn save_composite(path: &std::path::Path, left: &[u8], right: &[u8]) -> Result<(), String> {
    let (w, h) = (RENDER_EXTENT[0], RENDER_EXTENT[1]);
    let mut out = image::RgbaImage::new(w * 2, h);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            out.put_pixel(x, y, image::Rgba([left[i], left[i + 1], left[i + 2], 255]));
            out.put_pixel(w + x, y, image::Rgba([right[i], right[i + 1], right[i + 2], 255]));
        }
    }
    out.save(path).map_err(|e| format!("save composite: {e}"))
}

fn write_summary(
    path: &std::path::Path,
    rows: &[(String, PoseMetrics, PoseMetrics, PoseMetrics)],
) -> Result<(), String> {
    let mut md = String::new();
    md.push_str("# GT round-trip validation (self-consistency rendering)\n\n");
    md.push_str(
        "GT pose → render → track (SRC = detected source skeleton) → solve\n\
         (REC = recovered avatar). Distances are normalised by each\n\
         skeleton's own shoulder span. GT→SRC deviation = tracking error;\n\
         SRC→REC deviation = solver error. The tracker selfie-mirrors the\n\
         subject, so SRC/REC yaw-type metrics come back SIGN-FLIPPED\n\
         relative to GT; coupling means |Δ| matching vs the neutral row.\n\n",
    );
    md.push_str("| pose | shoulder_yaw gt/src/rec | wrist_gap gt/src/rec | wrist_fwd gt/src/rec | wrist_h gt/src/rec | L_lift gt/src/rec | head_rot gt/rec |\n");
    md.push_str("|---|---|---|---|---|---|---|\n");
    for (name, gt, src, rec) in rows {
        md.push_str(&format!(
            "| {name} | {:+.1}° / {:+.1}° / {:+.1}° | {:.2} / {:.2} / {:.2} | {:+.2} / {:+.2} / {:+.2} | {:+.2} / {:+.2} / {:+.2} | {:+.2} / {:+.2} / {:+.2} | {:.1}° / {:.1}° |\n",
            gt.shoulder_yaw_deg,
            src.shoulder_yaw_deg,
            rec.shoulder_yaw_deg,
            gt.wrist_gap,
            src.wrist_gap,
            rec.wrist_gap,
            gt.wrist_forward,
            src.wrist_forward,
            rec.wrist_forward,
            gt.wrist_height,
            src.wrist_height,
            rec.wrist_height,
            gt.left_shoulder_lift,
            src.left_shoulder_lift,
            rec.left_shoulder_lift,
            gt.head_rot_deg,
            rec.head_rot_deg,
        ));
    }

    // Coupling table: deltas vs the neutral row factor out both the
    // constant detection bias (the synthetic body-prior z misfiring
    // on stylized rigs) and the rig's rest geometry, leaving the
    // question that matters: does a GT change COME BACK, with which
    // sign and gain? gain = Δrec/Δgt (mirror makes yaw-type gains
    // legitimately negative; magnitude ≈ 1 is ideal coupling).
    if let Some((_, gt0, _, rec0)) = rows.iter().find(|(n, _, _, _)| n == "neutral") {
        md.push_str(
            "\n## Coupling vs neutral (Δrec/Δgt; |gain| ≈ 1 ideal, sign may mirror)\n\n\
             | pose | shoulder_yaw Δgt → Δrec (gain) | wrist_fwd Δgt → Δrec (gain) | wrist_gap Δgt → Δrec (gain) |\n\
             |---|---|---|---|\n",
        );
        let cell = |dgt: f32, drec: f32, unit: &str| -> String {
            if dgt.abs() < 1e-3 {
                format!("{drec:+.2}{unit} (n/a)")
            } else {
                format!("{dgt:+.2}{unit} → {drec:+.2}{unit} ({:+.2})", drec / dgt)
            }
        };
        for (name, gt, _, rec) in rows.iter().filter(|(n, _, _, _)| n != "neutral") {
            md.push_str(&format!(
                "| {name} | {} | {} | {} |\n",
                cell(
                    gt.shoulder_yaw_deg - gt0.shoulder_yaw_deg,
                    rec.shoulder_yaw_deg - rec0.shoulder_yaw_deg,
                    "°"
                ),
                cell(
                    gt.wrist_forward - gt0.wrist_forward,
                    rec.wrist_forward - rec0.wrist_forward,
                    ""
                ),
                cell(gt.wrist_gap - gt0.wrist_gap, rec.wrist_gap - rec0.wrist_gap, ""),
            ));
        }
    }
    std::fs::write(path, md).map_err(|e| format!("write summary: {e}"))
}
