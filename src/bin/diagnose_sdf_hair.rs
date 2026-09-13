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
    // Splat list: body + face/head surfaces, union AABB for the grid —
    // the same selection the app's frame input builder performs.
    let mut splat: Vec<(vulvatar_lib::asset::MeshId, vulvatar_lib::asset::PrimitiveId)> =
        Vec::new();
    let mut union = vulvatar_lib::asset::Aabb::empty();
    let mut push_prim = |mesh_id, primitive_id, bounds: &vulvatar_lib::asset::Aabb| {
        if splat.len() >= 4
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
    let report_path = Path::new(&output_dir).join("summary.md");
    std::fs::write(&report_path, &report).map_err(|e| format!("write summary: {e}"))?;
    println!("summary: {}", report_path.display());
    Ok(())
}

fn save_png(
    path: &Path,
    extent: [u32; 2],
    rgba: &[u8],
) -> Result<(), String> {
    let expected = (extent[0] as usize) * (extent[1] as usize) * 4;
    if rgba.len() < expected {
        return Err(format!(
            "pixel buffer too small: {} < {}",
            rgba.len(),
            expected
        ));
    }
    let mut img = image::ImageBuffer::new(extent[0], extent[1]);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let i = ((y as usize) * extent[0] as usize + x as usize) * 4;
        *pixel = image::Rgba([rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]);
    }
    img.save(path)
        .map_err(|e| format!("save {}: {e}", path.display()))
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
    let eye = [0.55_f32, 1.35, 1.35];
    let target = [0.0_f32, 1.05, 0.0];
    let view = look_at(&eye, &target);
    let fov = 36.0_f32.to_radians();
    let aspect = width as f32 / height as f32;
    let proj = perspective(fov, aspect, 0.05, 20.0);

    RenderFrameInput {
        camera: CameraState {
            view,
            projection: proj,
            position_ws: eye,
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

fn look_at(eye: &[f32; 3], target: &[f32; 3]) -> [[f32; 4]; 4] {
    let up = [0.0_f32, 1.0, 0.0];
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let norm = |a: [f32; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let fwd = norm(sub(*target, *eye));
    let right = norm(cross(fwd, up));
    let up2 = cross(right, fwd);
    // Column-major view matrix.
    [
        [right[0], up2[0], -fwd[0], 0.0],
        [right[1], up2[1], -fwd[1], 0.0],
        [right[2], up2[2], -fwd[2], 0.0],
        [-dot(right, *eye), -dot(up2, *eye), dot(fwd, *eye), 1.0],
    ]
}

fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) * nf, -1.0],
        [0.0, 0.0, 2.0 * far * near * nf, 0.0],
    ]
}
