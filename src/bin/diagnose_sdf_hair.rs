//! End-to-end verification for the body-surface distance field hair
//! collision (`simulation/sdf.rs` + `renderer/sdf_field.rs`).
//!
//! Loads an FBX/VRM avatar, turns the head sinusoidally for 60 frames,
//! and drives the real pipeline: the compute prepass splats the skinned
//! body primitive into the distance field, the field is read back
//! through `RenderResult::sdf_fields`, folded into the avatar, and the
//! spring solver resolves hair against it on the next frame — the same
//! one-frame-stale loop the live app runs.
//!
//! Outputs to `diagnostics/sdf_hair/`:
//! - `summary.md` — SDF occupancy stats, per-chain tail metrics, and
//!   whether any hair joint penetrates its radius band.
//! - `frame_*.png` — back-view renders at capture frames for eyeballing
//!   the shoulder drape (the spot where the removed heuristic chest
//!   capsule used to block hair above the shoulders).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use vulvatar_lib::renderer::frame_input::{
    BodySdfPlan, CameraState, LightingState, OutputTargetRequest, RenderAlphaMode,
    RenderAvatarInstance, RenderCullMode, RenderExportMode, RenderFrameInput, RenderMeshInstance,
    RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::MaterialShaderMode;
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::simulation::sdf::{SdfField, SdfGrid, SENTINEL};
use vulvatar_lib::simulation::spring::SpringTuning;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.3/FBX/Yumeka_v1.0.3.fbx".to_string());
    let output_dir = args
        .next()
        .unwrap_or_else(|| "diagnostics/sdf_hair".to_string());
    if output_dir.contains("validation_images") {
        return Err(
            "Diagnostics outputs must not be written to validation_images/ (see AGENTS.md)".into(),
        );
    }
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("create dir: {e}"))?;

    let loader = vulvatar_lib::asset::fbx::FbxAssetLoader::new();
    let asset = loader
        .load(&input_path)
        .map_err(|e| format!("load failed: {e}"))?;
    let mut avatar =
        vulvatar_lib::avatar::AvatarInstance::new(vulvatar_lib::avatar::AvatarInstanceId(1), asset);

    // Body primitive + grid, the same selection the app's frame input
    // builder performs.
    // Splat list: body + face/head surfaces + the garment layer, union
    // AABB for the grid — the same selection the app's frame input
    // builder performs.
    let garments = vulvatar_lib::asset::clearance::sdf_garment_splat_prims(&avatar.asset, 3);
    for (mesh_id, prim_id) in &garments {
        let name = avatar
            .asset
            .meshes
            .iter()
            .find(|m| m.id == *mesh_id)
            .map(|m| m.name.as_str())
            .unwrap_or("?");
        println!("garment pick '{}' (mesh {}, prim {})", name, mesh_id.0, prim_id.0);
    }
    let mut splat: Vec<(vulvatar_lib::asset::MeshId, vulvatar_lib::asset::PrimitiveId)> =
        Vec::new();
    let mut union = vulvatar_lib::asset::Aabb::empty();
    let splat_cap = 4 + garments.len();
    let mut push_prim = |mesh_id, primitive_id, bounds: &vulvatar_lib::asset::Aabb| {
        if splat.len() >= splat_cap
            || splat
                .iter()
                .any(|&(m, p)| m == mesh_id && p == primitive_id)
        {
            return;
        }
        splat.push((mesh_id, primitive_id));
        union.expand(bounds);
        println!(
            "splat prim ({}, {}) bounds y [{:.3} .. {:.3}]",
            mesh_id.0, primitive_id.0, bounds.min[1], bounds.max[1]
        );
    };
    if let Some((mesh_id, primitive_id)) =
        vulvatar_lib::asset::clearance::find_body_primitive(&avatar.asset)
    {
        for m in &avatar.asset.meshes {
            if m.id != mesh_id {
                continue;
            }
            for p in &m.primitives {
                if p.id == primitive_id {
                    push_prim(mesh_id, primitive_id, &p.bounds);
                }
            }
        }
    }
    for m in &avatar.asset.meshes {
        let mesh_hit =
            m.name.to_lowercase().contains("face") || m.name.to_lowercase().contains("head");
        for p in &m.primitives {
            let mat_hit = avatar
                .asset
                .materials
                .iter()
                .find(|mat| mat.id == p.material_id)
                .map(|mat| {
                    let n = mat.name.to_lowercase();
                    n.contains("face") || n.contains("head")
                })
                .unwrap_or(false);
            if (mesh_hit || mat_hit) && p.vertex_count > 100 {
                push_prim(m.id, p.id, &p.bounds);
            }
        }
    }
    for (mesh_id, primitive_id) in &garments {
        if let Some(bounds) = avatar
            .asset
            .meshes
            .iter()
            .find(|m| m.id == *mesh_id)
            .and_then(|m| m.primitives.iter().find(|p| p.id == *primitive_id))
            .map(|p| p.bounds)
        {
            push_prim(*mesh_id, *primitive_id, &bounds);
        }
    }
    if splat.is_empty() {
        return Err("no splattable primitives found".into());
    }
    // Bound the grid to the splatted surfaces' union AABB — strands
    // outside this grid are far from the body and correctly sample "no
    // collision".
    let grid = SdfGrid::for_aabb(&union);
    println!(
        "grid dims={:?} voxel={:.1} mm cells={} ({:.2} MB readback)",
        grid.dims,
        grid.voxel * 1000.0,
        grid.cell_count(),
        grid.cell_count() as f32 * 4.0 / 1e6
    );

    // Rest pose so bone-position measurements below are meaningful.
    for (i, node) in avatar.asset.skeleton.nodes.iter().enumerate() {
        avatar.pose.local_transforms[i] = node.rest_local.clone();
    }
    avatar.compute_global_pose();

    // Head node for the swing animation.
    let head_idx = avatar
        .asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::Head))
        .ok_or("no Head node")?;
    let head_rest = avatar.asset.skeleton.nodes[head_idx].rest_local.clone();

    let shoulder_y = avatar
        .asset
        .skeleton
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.humanoid_bone,
                Some(vulvatar_lib::asset::HumanoidBone::LeftUpperArm)
                    | Some(vulvatar_lib::asset::HumanoidBone::RightUpperArm)
            )
        })
        .map(|n| {
            avatar.pose.global_transforms
                [n.id.0 as usize]
            [3][1]
        })
        .fold(f32::MIN, f32::max);

    // Hair chains that opted into body-field collision.
    let hair_chains: Vec<(usize, String, f32)> = avatar
        .asset
        .spring_bones
        .iter()
        .enumerate()
        .filter(|(_, sb)| {
            sb.body_collision
                && avatar.asset.skeleton.nodes[sb.chain_root.0 as usize]
                    .name
                    .to_lowercase()
                    .contains("hair")
        })
        .map(|(i, sb)| {
            (
                i,
                avatar.asset.skeleton.nodes[sb.chain_root.0 as usize].name.clone(),
                sb.radius,
            )
        })
        .collect();
    println!(
        "hair chains with body collision: {} (shoulder_y = {:.4})",
        hair_chains.len(),
        shoulder_y
    );
    // Shell-budget evidence: the field band must cover
    // max(radius) + SDF_CONTACT_MARGIN + interpolation support; printing
    // the actual distribution grounds the SHELL_METRES / voxel policy.
    let mut radii: Vec<f32> = hair_chains.iter().map(|(_, _, r)| *r).collect();
    radii.sort_by(|a, b| a.total_cmp(b));
    if let (Some(min), Some(max)) = (radii.first(), radii.last()) {
        println!(
            "chain radius: min {:.1} mm / median {:.1} mm / max {:.1} mm (contact band = max + 4 mm margin)",
            min * 1000.0,
            radii[radii.len() / 2] * 1000.0,
            max * 1000.0
        );
    }
    for (_, name, r) in &hair_chains {
        println!("  chain {:?} radius {:.1} mm", name, r * 1000.0);
    }

    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let width = 900u32;
    let height = 1200u32;
    let dt = 1.0 / 60.0;
    let total_frames = 60;
    let capture_frames: HashSet<usize> = [0usize, 20, 40, 59].into_iter().collect();

    let tuning = SpringTuning::default();
    let mut report = String::new();
    report.push_str(&format!(
        "# Body-SDF hair collision run: {}

- splat prims: {:?}
- grid {:?} @ {:.1} mm
- shoulder_y {:.4}
- hair chains: {}

",
        input_path,
        splat.iter().map(|(m, p)| (m.0, p.0)).collect::<Vec<_>>(),
        grid.dims,
        grid.voxel * 1000.0,
        shoulder_y,
        hair_chains.len(),
    ));

    let mut sdf_ready_frames = 0u32;
    for frame in 0..=total_frames {
        // Head yaw swing: ±30° at 0.5 Hz, so strands must slide across
        // the face / shoulders every frame.
        if frame > 0 {
            let yaw = (frame as f32 * dt * std::f32::consts::PI).sin() * 0.52;
            let mut t = head_rest.clone();
            t.rotation = vulvatar_lib::math_utils::quat_mul(
                &head_rest.rotation,
                &[0.0, (yaw * 0.5).sin(), 0.0, (yaw * 0.5).cos()],
            );
            avatar.pose.local_transforms[head_idx] = t;
        }
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        // Step springs against the field folded in from LAST frame's
        // render (one frame stale, same as the live app).
        let sdf = avatar.body_sdf.clone();
        if sdf.is_some() {
            sdf_ready_frames += 1;
        }
        for _ in 0..2 {
            vulvatar_lib::simulation::spring::step_spring_bones(
                dt,
                &mut avatar,
                &[],
                &tuning,
                [0.0, -1.0, 0.0],
                1.0,
                sdf.as_ref(),
            );
        }

        let frame_input = build_frame_input(&avatar, splat.clone(), grid, width, height);
        // Pipelined readback: first call harvests the previous frame,
        // second call's result carries THIS frame's splat.
        let _ = renderer.render(&frame_input)?;
        let result = renderer.render(&frame_input)?;
        for entry in &result.sdf_fields {
            if entry.instance_id == avatar.id.0 {
                avatar.body_sdf = Some(SdfField::new(entry.grid, entry.data.clone()));
            }
        }

        // Metrics + captures.
        if let Some(field) = avatar.body_sdf.as_ref() {
            if frame % 20 == 0 || frame == total_frames {
                report.push_str(&format!("## frame {}\n\n", frame));
                for &(ci, ref name, radius) in &hair_chains {
                    let state = &avatar.secondary_motion.spring_states[ci];
                    if state.positions.len() < 2 {
                        continue;
                    }
                    // Closest approach of ANY joint in this chain, plus
                    // the tail's height — the old heuristic capsules
                    // blocked tails above the shoulder line, so tails
                    // hanging below shoulder_y is the acceptance signal.
                    let mut min_gap = f32::MAX;
                    let mut near_joints = 0u32;
                    for p in &state.positions {
                        let d = field.sample(*p);
                        if d < SENTINEL {
                            near_joints += 1;
                            min_gap = min_gap.min(d - radius);
                        }
                    }
                    let tail = state.positions[state.positions.len() - 1];
                    let gap_text = if min_gap >= f32::MAX {
                        "n/a".to_string()
                    } else {
                        format!("{:+.1}", min_gap * 1000.0)
                    };
                    report.push_str(&format!(
                        "- {} tail_y={:.4} near_joints={} min_gap={} mm{}\n",
                        name,
                        tail[1],
                        near_joints,
                        gap_text,
                        if min_gap < -0.002 {
                            "  <-- PENETRATING"
                        } else {
                            ""
                        },
                    ));
                }
                let splatted = field.data.iter().filter(|&&v| v < SENTINEL).count();
                report.push_str(&format!(
                    "\n- splatted cells: {} / {} ({:.1} %)\n\n",
                    splatted,
                    field.data.len(),
                    100.0 * splatted as f32 / field.data.len() as f32
                ));
            }
        }

        if capture_frames.contains(&frame) {
            let png = Path::new(&output_dir).join(format!("frame_{:03}.png", frame));
            if let Some(exported) = result.exported_frame.as_ref() {
                if let Some(pixels) = exported.cpu_pixel_data() {
                    save_png(&png, [width, height], &pixels)?;
                    println!("frame {:03} -> {}", frame, png.display());
                }
            }
        }
    }

    report.push_str(&format!(
        "\n- frames the solver ran WITH a field: {}\n",
        sdf_ready_frames
    ));

    // Lean sweep: the grid is rest-AABB-derived, so a forward-leaning
    // torso can walk the splatted surfaces out of it — joints sampling
    // SENTINEL silently lose body collision. `GRID_LEAN_PAD_M` must keep
    // the desk-envelope lean covered (measured live: head z p95 +0.10 /
    // max +0.17 m). Acceptance: no outside-grid joints through 15°.
    if let Some(spine_idx) = avatar.asset.skeleton.nodes.iter().position(|n| {
        n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::Spine)
    }) {
        let spine_rest = avatar.asset.skeleton.nodes[spine_idx].rest_local.clone();
        report.push_str("\n## Lean sweep (joints outside the grid = collision off)\n\n");
        report.push_str("| lean | head z (m) | joints outside grid |\n|---:|---:|---:|\n");
        for lean_deg in [0.0f32, 10.0, 20.0, 30.0] {
            // Sign the lean so the head moves +z (face forward): try
            // both rotations about the spine's local X and keep the
            // forward one.
            let head_z_for = |sign: f32, avatar: &mut vulvatar_lib::avatar::AvatarInstance| {
                for (i, node) in avatar.asset.skeleton.nodes.iter().enumerate() {
                    avatar.pose.local_transforms[i] = node.rest_local.clone();
                }
                let th = sign * lean_deg.to_radians();
                let q = [(th * 0.5).sin(), 0.0, 0.0, (th * 0.5).cos()];
                let mut t = spine_rest.clone();
                t.rotation = vulvatar_lib::math_utils::quat_mul(&q, &spine_rest.rotation);
                avatar.pose.local_transforms[spine_idx] = t;
                avatar.compute_global_pose();
                avatar.pose.global_transforms[head_idx][3][2]
            };
            let sign = if head_z_for(1.0, &mut avatar) >= head_z_for(-1.0, &mut avatar) {
                1.0
            } else {
                -1.0
            };
            let _ = head_z_for(sign, &mut avatar);
            avatar.build_skinning_matrices();

            // Settle the strands under the lean with a refreshing field
            // (same two-render harvest pattern as the swing loop).
            for _ in 0..30 {
                let sdf = avatar.body_sdf.clone();
                for _ in 0..2 {
                    vulvatar_lib::simulation::spring::step_spring_bones(
                        dt,
                        &mut avatar,
                        &[],
                        &tuning,
                        [0.0, -1.0, 0.0],
                        1.0,
                        sdf.as_ref(),
                    );
                }
                avatar.compute_global_pose();
                avatar.build_skinning_matrices();
                let fi = build_frame_input(&avatar, splat.clone(), grid, width, height);
                let _ = renderer.render(&fi)?;
                let result = renderer.render(&fi)?;
                for entry in &result.sdf_fields {
                    if entry.instance_id == avatar.id.0 {
                        avatar.body_sdf = Some(SdfField::new(entry.grid, entry.data.clone()));
                    }
                }
            }

            let mut outside = 0usize;
            let mut total = 0usize;
            if let Some(field) = avatar.body_sdf.as_ref() {
                for &(ci, _, _) in &hair_chains {
                    if let Some(state) = avatar.secondary_motion.spring_states.get(ci) {
                        for p in state.positions.iter().skip(1) {
                            total += 1;
                            if field.sample(*p) == SENTINEL {
                                outside += 1;
                            }
                        }
                    }
                }
            }
            let head_z = avatar.pose.global_transforms[head_idx][3][2];
            report.push_str(&format!(
                "| {:.0}° | {:+.3} | {} / {} |\n",
                lean_deg, head_z, outside, total
            ));
            println!(
                "lean {:.0}°: head_z {:+.3} m, outside-grid joints {}/{}",
                lean_deg, head_z, outside, total
            );
        }
    }

    // Garment-gap report (2026-09-15): the splat list is skin + face
    // only, so hair resolves against the BODY surface. Measure how far
    // the settled joints sit from each non-splatted garment surface —
    // a joint whose contact band (radius + 4 mm margin) exceeds that
    // distance renders INSIDE the garment. Also samples the body field
    // at each garment's skinned vertices to report how far the garment
    // itself floats off the skin (the shell thickness hair must clear).
    garment_gap_report(&avatar, &hair_chains, &mut report);

    let report_path = Path::new(&output_dir).join("summary.md");
    std::fs::write(&report_path, &report).map_err(|e| format!("write summary: {e}"))?;
    println!("summary: {}", report_path.display());
    Ok(())
}


fn build_frame_input(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    prims: Vec<(vulvatar_lib::asset::MeshId, vulvatar_lib::asset::PrimitiveId)>,
    grid: SdfGrid,
    width: u32,
    height: u32,
) -> RenderFrameInput {
    let mut mesh_instances = Vec::new();
    for mesh in &avatar.asset.meshes {
        for prim in &mesh.primitives {
            let material_asset = avatar
                .asset
                .materials
                .iter()
                .find(|m| m.id == prim.material_id);
            let mut material_binding = material_asset
                .map(vulvatar_lib::renderer::material::MaterialUploadRequest::from_asset_material)
                .unwrap_or_else(vulvatar_lib::renderer::material::MaterialUploadRequest::default_material);
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
            mesh_instances.push(RenderMeshInstance {
                mesh_id: mesh.id,
                primitive_id: prim.id,
                material_binding,
                bounds: prim.bounds,
                alpha_mode,
                cull_mode,
                outline: Default::default(),
                primitive_data: Some(Arc::clone(prim)),
                morph_weights: Vec::new(),
            });
        }
    }

    // Back three-quarter view of the head and shoulders, where the old
    // heuristic capsules used to block the hair.
    let camera = ViewportCamera {
        distance: 1.15,
        pan: [0.0, 1.05],
        yaw_deg: 160.0,
        pitch_deg: 5.0,
        fov_deg: 36.0,
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(
        camera.fov_deg,
        width as f32 / height as f32,
        0.05,
        20.0,
    );

    RenderFrameInput {
        camera: CameraState {
            view,
            projection,
            position_ws: eye_pos,
            viewport_extent: [width, height],
        },
        lighting: LightingState::default(),
        instances: vec![RenderAvatarInstance {
            instance_id: avatar.id,
            world_transform: avatar.world_transform.clone(),
            mesh_instances,
            skinning_matrices: avatar.pose.skinning_matrices.clone(),
            cloth_deforms: Vec::new(),
            body_sdf: Some(BodySdfPlan { prims, grid }),
            debug_flags: Default::default(),
        }],
        output_request: OutputTargetRequest {
            preview_enabled: false,
            output_enabled: true,
            extent: [width, height],
            color_space: vulvatar_lib::renderer::frame_input::RenderColorSpace::Srgb,
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

    let right = [cy, 0.0, -sy];
    let up = [-sy * sp, cp, -cy * sp];

    let wx = cam.pan[0] * right[0] + cam.pan[1] * up[0];
    let wy = cam.pan[0] * right[1] + cam.pan[1] * up[1];
    let wz = cam.pan[0] * right[2] + cam.pan[1] * up[2];

    let eye_x = cam.distance * cp * sy + wx;
    let eye_y = cam.distance * sp + wy;
    let eye_z = cam.distance * cp * cy + wz;

    let target = [wx, wy, wz];
    let fwd = [target[0] - eye_x, target[1] - eye_y, target[2] - eye_z];
    let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2])
        .sqrt()
        .max(1e-6);
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
            [
                r[0],
                r[1],
                r[2],
                -(r[0] * eye_x + r[1] * eye_y + r[2] * eye_z),
            ],
            [
                u[0],
                u[1],
                u[2],
                -(u[0] * eye_x + u[1] * eye_y + u[2] * eye_z),
            ],
            [
                -f[0],
                -f[1],
                -f[2],
                (f[0] * eye_x + f[1] * eye_y + f[2] * eye_z),
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [eye_x, eye_y, eye_z],
    )
}

fn build_projection_matrix(
    fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> vulvatar_lib::asset::Mat4 {
    let fov_rad = fov_deg.to_radians();
    let f = 1.0 / (fov_rad * 0.5).tan();
    let a = far / (near - far);
    let b = far * near / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, a, b],
        [0.0, 0.0, -1.0, 0.0],
    ]
}

fn save_png(path: &Path, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    let img: image::ImageBuffer<image::Rgba<u8>, _> =
        image::ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| format!("PNG buffer construction failed for {}", path.display()))?;
    img.save(path)
        .map_err(|e| format!("failed to save '{}': {e}", path.display()))
}

/// Settled-joint distance to every non-splatted garment surface, plus
/// each garment's own float off the skin (sampled from the body field).
/// `hair_chains` carries `(spring_bones index, chain name, radius)` —
/// the same list the min-gap loop above reports on.
fn garment_gap_report(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    hair_chains: &[(usize, String, f32)],
    report: &mut String,
) {
    use vulvatar_lib::asset::{MeshId, PrimitiveId};

    // Nodes belonging to a hair chain (root + joints) — used to exclude
    // the hair's own meshes, whose surfaces obviously contain the joints.
    let mut hair_nodes: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for &(ci, _, _) in hair_chains {
        if let Some(sb) = avatar.asset.spring_bones.get(ci) {
            hair_nodes.insert(sb.chain_root.0);
            for &n in &sb.joints {
                hair_nodes.insert(n.0);
            }
        }
    }

    let body_pid = vulvatar_lib::asset::clearance::find_body_primitive(&avatar.asset)
        .map(|(_, p)| p);

    // A prim is "hair" when a third of its vertices are majority-bound
    // to hair-chain nodes (Yumeka's meshes are all "Circle.xxx", so
    // names cannot make the call).
    let is_hair_prim = |vd: &vulvatar_lib::asset::VertexData| -> bool {
        if vd.positions.is_empty() {
            return false;
        }
        let mut hair_verts = 0usize;
        for (vi, joints) in vd.joint_indices.iter().enumerate() {
            let mut best_w = 0.0f32;
            let mut best_is_hair = false;
            for (slot, &ji) in joints.iter().enumerate() {
                let w = vd.joint_weights.get(vi).map_or(0.0, |w| w[slot]);
                if w > best_w {
                    best_w = w;
                    best_is_hair = hair_nodes.contains(&(ji as u64));
                }
            }
            if best_is_hair {
                hair_verts += 1;
            }
        }
        hair_verts as f32 / vd.positions.len() as f32 >= 0.3
    };

    // Splatted (skin + face/head) prims are not garments.
    let splat_pids: std::collections::HashSet<(MeshId, PrimitiveId)> = avatar
        .asset
        .meshes
        .iter()
        .flat_map(|m| {
            let mesh_hit = {
                let n = m.name.to_lowercase();
                n.contains("face") || n.contains("head")
            };
            m.primitives
                .iter()
                .filter(move |p| {
                    let mat_hit = avatar
                        .asset
                        .materials
                        .iter()
                        .find(|mat| mat.id == p.material_id)
                        .map(|mat| {
                            let n = mat.name.to_lowercase();
                            n.contains("face") || n.contains("head")
                        })
                        .unwrap_or(false);
                    mesh_hit || mat_hit
                })
                .map(move |p| (m.id, p.id))
        })
        .collect();

    // Settled joint positions per chain (skip the chain root — it rides
    // the head skeleton, not the solved strand).
    let settled: Vec<(usize, String, f32, Vec<[f32; 3]>)> = hair_chains
        .iter()
        .filter_map(|&(ci, ref name, radius)| {
            let positions: Vec<[f32; 3]> = avatar.secondary_motion.spring_states
                .get(ci)?
                .positions
                .iter()
                .skip(1)
                .copied()
                .collect();
            (!positions.is_empty()).then_some((ci, name.clone(), radius, positions))
        })
        .collect();

    let field = avatar.body_sdf.as_ref();
    report.push_str("\n## Garment gap (settled joints vs non-splatted garments)\n\n");

    for mesh in &avatar.asset.meshes {
        for prim in &mesh.primitives {
            let (Some(vd), Some(indices)) = (prim.vertices.as_ref(), prim.indices.as_ref())
            else {
                continue;
            };
            if indices.len() < 3 || Some(prim.id) == body_pid {
                continue;
            }
            if splat_pids.contains(&(mesh.id, prim.id)) || is_hair_prim(vd) {
                continue;
            }

            // Skinned triangles at the settled pose (weight-normalized
            // LBS, the clearance/auto-cloth CPU recipe).
            let world: Vec<[f32; 3]> = vd
                .positions
                .iter()
                .enumerate()
                .map(|(i, &pos)| lbs_position(avatar, vd, i, pos))
                .collect();
            let tri: Vec<[[f32; 3]; 3]> = indices
                .chunks_exact(3)
                .map(|t| [world[t[0] as usize], world[t[1] as usize], world[t[2] as usize]])
                .collect();
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for p in &world {
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }

            // Identification: hair-bound share plus the top bones by
            // accumulated weight mass, so the reader can tell a true
            // garment (jacket → arm/chest bones) from a hair shell the
            // 30% classifier missed.
            let mut bone_mass: std::collections::HashMap<u64, f32> =
                std::collections::HashMap::new();
            for (vi, joints) in vd.joint_indices.iter().enumerate() {
                for (slot, &ji) in joints.iter().enumerate() {
                    let w = vd.joint_weights.get(vi).map_or(0.0, |w| w[slot]);
                    if w > 1e-4 {
                        *bone_mass.entry(ji as u64).or_insert(0.0) += w;
                    }
                }
            }
            let hair_mass: f32 = bone_mass
                .iter()
                .filter(|(n, _)| hair_nodes.contains(n))
                .map(|(_, w)| *w)
                .sum();
            let total_mass: f32 = bone_mass.values().sum();
            let mut top: Vec<(f32, &u64)> = bone_mass.iter().map(|(n, w)| (*w, n)).collect();
            top.sort_by(|a, b| b.0.total_cmp(&a.0));
            let top_bones = top
                .iter()
                .take(3)
                .map(|(w, n)| {
                    format!(
                        "{}:{:.0}%",
                        avatar
                            .asset
                            .skeleton
                            .nodes
                            .get(**n as usize)
                            .map(|nd| nd.name.as_str())
                            .unwrap_or("?"),
                        100.0 * w / total_mass.max(1e-6)
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            let (p05, p50) = match field {
                Some(f) => (pct_off_skin(f, &world, 0.05), pct_off_skin(f, &world, 0.5)),
                None => (None, None),
            };
            let float_text = match (p05, p50) {
                (Some(a), Some(b)) => format!("float off skin p5 {:.0} / p50 {:.0} mm", a, b),
                _ => "float off skin n/a".to_string(),
            };
            report.push_str(&format!(
                "- garment '{}' ({} verts): {}; hair-bound mass {:.0}%; top bones {}\n",
                mesh.name,
                world.len(),
                float_text,
                100.0 * hair_mass / total_mass.max(1e-6),
                top_bones
            ));
            println!(
                "garment '{}' ({} verts): {}; hair-bound {:.0}%; top: {}",
                mesh.name,
                world.len(),
                float_text,
                100.0 * hair_mass / total_mass.max(1e-6),
                top_bones
            );


            // Per-chain closest joint distance to this garment's
            // triangles (AABB-culled brute force).
            let mut lines: Vec<String> = Vec::new();
            for (_, name, radius, positions) in &settled {
                let mut best = f32::MAX;
                for p in positions {
                    if p[0] < lo[0] - 0.15
                        || p[0] > hi[0] + 0.15
                        || p[1] < lo[1] - 0.15
                        || p[1] > hi[1] + 0.15
                        || p[2] < lo[2] - 0.15
                        || p[2] > hi[2] + 0.15
                    {
                        continue;
                    }
                    for t in &tri {
                        best = best.min(point_triangle_dist(*p, t));
                    }
                }
                if best.is_finite() {
                    // Mirror the solver's contact band: radius +
                    // SDF_CONTACT_MARGIN (spring.rs, 4 mm).
                    let band = radius + 0.004;
                    let verdict = if best < band {
                        format!("  <-- INSIDE by {:.0} mm", (band - best) * 1000.0)
                    } else {
                        String::new()
                    };
                    lines.push(format!(
                        "  - {:?} r {:.0} mm: joint-to-garment {:.0} mm{}\n",
                        name,
                        radius * 1000.0,
                        best * 1000.0,
                        verdict
                    ));
                }
            }
            if !lines.is_empty() {
                report.push_str(&format!(
                    "- joint distances vs '{}':\n{}",
                    mesh.name,
                    lines.join("")
                ));
            }
        }
    }
}

/// Percentile (mm) of the garment's skinned vertices' body-field
/// distance — how far the garment floats off the skin. `q` in 0..=1.
fn pct_off_skin(field: &SdfField, world: &[[f32; 3]], q: f32) -> Option<f32> {
    let mut gaps: Vec<f32> = world
        .iter()
        .map(|p| field.sample(*p))
        .filter(|d| *d < SENTINEL)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(|a, b| a.total_cmp(b));
    let idx = ((gaps.len() as f32 - 1.0) * q).round() as usize;
    Some(gaps[idx.min(gaps.len() - 1)] * 1000.0)
}

/// Weight-normalized 4-influence LBS against the instance's skinning
/// matrices (same recipe as `auto_cloth` / `clearance` CPU passes).
fn lbs_position(
    avatar: &vulvatar_lib::avatar::AvatarInstance,
    vd: &vulvatar_lib::asset::VertexData,
    i: usize,
    pos: [f32; 3],
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    let mut total_w = 0.0;
    if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
        for k in 0..4 {
            let w = vd.joint_weights[i][k];
            if w > 0.0001 {
                let j = vd.joint_indices[i][k] as usize;
                if let Some(sm) = avatar.pose.skinning_matrices.get(j) {
                    for c in 0..3 {
                        out[c] += w
                            * (sm[0][c] * pos[0] + sm[1][c] * pos[1] + sm[2][c] * pos[2]
                                + sm[3][c]);
                    }
                    total_w += w;
                }
            }
        }
    }
    if total_w > 0.001 {
        for c in 0..3 {
            out[c] /= total_w;
        }
    }
    out
}

/// Squared-free point-to-triangle distance (Ericson 5.1.5).
fn point_triangle_dist(p: [f32; 3], t: &[[f32; 3]; 3]) -> f32 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
    let d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];
    if d1 <= 0.0 && d2 <= 0.0 {
        return len(ap);
    }
    let bp = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
    let d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
    let d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];
    if d3 >= 0.0 && d4 <= d3 {
        return len(bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return len([ab[0] * v - ap[0], ab[1] * v - ap[1], ab[2] * v - ap[2]]);
    }
    let cp = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    let d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
    let d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
    if d6 >= 0.0 && d5 <= d6 {
        return len(cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return len([ac[0] * w - ap[0], ac[1] * w - ap[1], ac[2] * w - ap[2]]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        return len([
            bc[0] * w - bp[0],
            bc[1] * w - bp[1],
            bc[2] * w - bp[2],
        ]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    len([
        ab[0] * v + ac[0] * w - ap[0],
        ab[1] * v + ac[1] * w - ap[1],
        ab[2] * v + ac[2] * w - ap[2],
    ])
}

fn len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
