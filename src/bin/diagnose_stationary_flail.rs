//! Stationary-tracking-noise amplification probe for the "chest mesh
//! flails while the user sits still" report (Yumeka, 2026-09-15).
//!
//! Live measurement first (AGENTS.md): with no tracking input the avatar
//! is bit-frozen (all joints 0.00 mm/frame, rendered output diff ~0),
//! so the flail requires the tracking stream. A seated "stationary"
//! user still feeds the solver mm/deg-level jitter. This harness asks
//! the complementary question OFFLINE: how much do the spring chains
//! that skin the chest region (breast chains, side hair, twintales)
//! AMPLIFY that stationary noise — and does body-SDF contact add
//! chatter on top.
//!
//! Protocol: rest pose, settle with no input, then band-limited chest
//! rotation noise (sinusoid + white, amplitude in deg) plus 1 mm root
//! white noise — a stationary-user stand-in. Same pipeline as the live
//! app: skinning → splat render → one-frame-stale SDF → spring step.
//! garments=N selects the splat's garment layer (0 = the pre-2026-09-15
//! skin+face-only binary the user is running, 3 = current default).
//!
//! Outputs to `diagnostics/stationary_flail/`:
//! - `summary.md` — per-chain displacement std / peak-to-peak in the
//!   noise phase, ranked; control (no-noise) drift for comparison.
//! - `frames.csv` — applied noise + tracked joint positions per frame.

use std::collections::HashSet;
use std::path::Path;

use vulvatar_lib::renderer::frame_input::{
    BodySdfPlan, CameraState, LightingState, OutputTargetRequest, RenderAlphaMode,
    RenderAvatarInstance, RenderCullMode, RenderExportMode, RenderFrameInput, RenderMeshInstance,
    RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::MaterialShaderMode;
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::simulation::sdf::{SdfField, SdfGrid};
use vulvatar_lib::simulation::spring::SpringTuning;

fn main() -> Result<(), String> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx".to_string());
    let output_dir = args.next().unwrap_or_else(|| "diagnostics/stationary_flail".to_string());
    if output_dir.contains("validation_images") {
        return Err("diagnostics outputs must not target validation_images/ (AGENTS.md)".into());
    }
    let noise_deg: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(0.3);
    let garments: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let total_frames: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(240);
    let settle_frames: usize = 90;
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("create dir: {e}"))?;

    let loader = vulvatar_lib::asset::fbx::FbxAssetLoader::new();
    let asset = loader
        .load(&input_path)
        .map_err(|e| format!("load failed: {e}"))?;
    let mut avatar =
        vulvatar_lib::avatar::AvatarInstance::new(vulvatar_lib::avatar::AvatarInstanceId(1), asset);

    // Splat selection mirrors diagnose_sdf_hair: body + face/head
    // surfaces (+ garment layer per the `garments` cap).
    let garment_prims = vulvatar_lib::asset::clearance::sdf_garment_splat_prims(&avatar.asset, garments);
    let mut splat: Vec<(vulvatar_lib::asset::MeshId, vulvatar_lib::asset::PrimitiveId)> = Vec::new();
    let mut union = vulvatar_lib::asset::Aabb::empty();
    let mut push_prim = |mesh_id, primitive_id, bounds: &vulvatar_lib::asset::Aabb| {
        if splat.iter().any(|&(m, p)| m == mesh_id && p == primitive_id) {
            return;
        }
        splat.push((mesh_id, primitive_id));
        union.expand(bounds);
    };
    if let Some((mesh_id, primitive_id)) =
        vulvatar_lib::asset::clearance::find_body_primitive(&avatar.asset)
    {
        for m in &avatar.asset.meshes {
            if m.id == mesh_id {
                for p in &m.primitives {
                    if p.id == primitive_id {
                        push_prim(mesh_id, primitive_id, &p.bounds);
                    }
                }
            }
        }
    }
    for m in &avatar.asset.meshes {
        let mesh_hit = m.name.to_lowercase().contains("face")
            || m.name.to_lowercase().contains("head");
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
    for &(mesh_id, primitive_id) in &garment_prims {
        if let Some(bounds) = avatar
            .asset
            .meshes
            .iter()
            .find(|m| m.id == mesh_id)
            .and_then(|m| m.primitives.iter().find(|p| p.id == primitive_id))
            .map(|p| p.bounds)
        {
            push_prim(mesh_id, primitive_id, &bounds);
        }
    }
    if splat.is_empty() {
        return Err("no splattable primitives found".into());
    }
    let grid = SdfGrid::for_aabb(&union);

    for (i, node) in avatar.asset.skeleton.nodes.iter().enumerate() {
        avatar.pose.local_transforms[i] = node.rest_local.clone();
    }
    avatar.compute_global_pose();

    let chest_idx = avatar
        .asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::Chest))
        .ok_or("no Chest node")?;
    let chest_rest = avatar.asset.skeleton.nodes[chest_idx].rest_local.clone();
    let hips_idx = avatar
        .asset
        .skeleton
        .nodes
        .iter()
        .position(|n| n.humanoid_bone == Some(vulvatar_lib::asset::HumanoidBone::Hips))
        .ok_or("no Hips node")?;
    let hips_rest = avatar.asset.skeleton.nodes[hips_idx].rest_local.clone();

    // Inventory: every spring chain, its root name, radius, and whether
    // it resolves against the body SDF.
    let chains: Vec<(usize, String, f32, bool)> = avatar
        .asset
        .spring_bones
        .iter()
        .enumerate()
        .map(|(i, sb)| {
            (
                i,
                avatar.asset.skeleton.nodes[sb.chain_root.0 as usize].name.clone(),
                sb.radius,
                sb.body_collision,
            )
        })
        .collect();
    println!(
        "spring chains: {} (garments in splat: {})",
        chains.len(),
        garment_prims.len()
    );

    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    let width = 900u32;
    let height = 1200u32;
    let dt = 1.0 / 60.0;
    let tuning = SpringTuning::default();

    // Deterministic white noise (LCG) so runs are comparable.
    let mut rng: u32 = 0x5EED_1234;
    let mut white = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 16777216.0) - 0.5 // [-0.5, 0.5)
    };

    let mut rows: Vec<String> = Vec::new();
    let mut history: Vec<Vec<Vec<[f32; 3]>>> = Vec::new(); // frame -> chain -> joints
    let mut applied: Vec<(f32, f32)> = Vec::new(); // chest yaw/pitch deg

    let capture_frames: HashSet<usize> = [settle_frames - 1, total_frames - 1].into_iter().collect();

    for frame in 0..total_frames {
        // Stationary-user stand-in after the settle phase: 1.2 Hz
        // sinusoid + white noise on chest yaw/pitch (deg), 1 mm root
        // translation jitter.
        let t = frame as f32 * dt;
        let noise_on = frame >= settle_frames;
        let (ny, np) = if noise_on {
            let s = (std::f32::consts::TAU * 1.2 * t).sin();
            (
                noise_deg * (0.6 * s + 0.8 * white()),
                noise_deg * 0.5 * (0.6 * s.cos() + 0.8 * white()),
            )
        } else {
            (0.0, 0.0)
        };
        applied.push((ny, np));
        {
            let mut tr = chest_rest.clone();
            let (hy, hp) = (ny.to_radians() * 0.5, np.to_radians() * 0.5);
            // yaw (Y) then pitch (X) small-angle composition.
            let q = vulvatar_lib::math_utils::quat_mul(
                &[0.0, hy.sin(), 0.0, hy.cos()],
                &[hp.sin(), 0.0, 0.0, hp.cos()],
            );
            tr.rotation = vulvatar_lib::math_utils::quat_mul(&chest_rest.rotation, &q);
            avatar.pose.local_transforms[chest_idx] = tr;
        }
        {
            let mut tr = hips_rest.clone();
            // Root jitter scales with the noise knob so the 0.0 run is a
            // true zero-input control (self-excitation test).
            let root_mm = if noise_deg > 0.0 { 0.001 } else { 0.0 };
            if noise_on && root_mm > 0.0 {
                tr.translation[0] += root_mm * white();
                tr.translation[2] += root_mm * white();
            }
            avatar.pose.local_transforms[hips_idx] = tr;
        }
        avatar.compute_global_pose();
        avatar.build_skinning_matrices();

        let sdf = avatar.body_sdf.clone();
        vulvatar_lib::simulation::spring::step_spring_bones(
            dt,
            &mut avatar,
            &[],
            &tuning,
            [0.0, -1.0, 0.0],
            1.0,
            sdf.as_ref(),
        );

        let frame_input = build_frame_input(&avatar, splat.clone(), grid, width, height);
        let _ = renderer.render(&frame_input)?;
        let result = renderer.render(&frame_input)?;
        for entry in &result.sdf_fields {
            if entry.instance_id == avatar.id.0 {
                avatar.body_sdf = Some(SdfField::new(entry.grid, entry.data.clone()));
            }
        }

        let mut frame_chains = Vec::new();
        let mut row = format!(
            "{},{:.4},{:.4}",
            frame, applied[frame].0, applied[frame].1
        );
        for &(ci, ref _name, _r, _bc) in &chains {
            let pos: Vec<[f32; 3]> = avatar.secondary_motion.spring_states[ci]
                .positions
                .clone();
            for p in &pos {
                row.push_str(&format!(",{:.5},{:.5},{:.5}", p[0], p[1], p[2]));
            }
            frame_chains.push(pos);
        }
        history.push(frame_chains);
        rows.push(row);

        if capture_frames.contains(&frame) {
            let png = Path::new(&output_dir).join(format!(
                "frame_{:03}_n{}.png",
                frame,
                noise_deg * 10.0
            ));
            if let Some(exported) = result.exported_frame.as_ref() {
                if let Some(pixels) = exported.cpu_pixel_data() {
                    let img: image::ImageBuffer<image::Rgba<u8>, _> =
                        image::ImageBuffer::from_raw(width, height, pixels.to_vec())
                            .ok_or("png buf")?;
                    img.save(&png).map_err(|e| format!("save: {e}"))?;
                    println!("frame {frame} -> {}", png.display());
                }
            }
        }
    }

    // Per-chain displacement stats over the noise phase, measured from
    // the settle-end pose (drift-inclusive worst case) and frame-to-
    // frame (jitter). Ranked by p2p.
    let base = &history[settle_frames - 1];
    let mut report = String::new();
    let noise_deg_s = noise_deg;
    report.push_str(&format!(
        "# Stationary flail probe: {input_path}

- noise: chest yaw {noise_deg_s:.2} deg / pitch {:.2} deg (1.2 Hz + white), root 1 mm white
- garments in SDF splat: {garments} ({} picked)
- frames: {total_frames} @ 60 Hz, settle {settle_frames}
- spring chains: {}

| chain | radius mm | SDF | max joint p2p mm | tail p2p mm | max f->f mm |
|---|---|---|---|---|---|
",
        noise_deg * 0.5,
        garment_prims.len(),
        chains.len(),
    ));
    let mut ranked: Vec<(f32, String, f32, bool, f32, f32)> = Vec::new();
    for (k, &(_, ref name, radius, bc)) in chains.iter().enumerate() {
        let nj = base[k].len();
        if nj == 0 {
            continue;
        }
        let mut max_p2p = 0.0f32;
        let mut max_ff = 0.0f32;
        for j in 0..nj {
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for f in history.iter().take(total_frames).skip(settle_frames) {
                let p = f[k][j];
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
            let mut p2p = 0.0;
            for c in 0..3 {
                p2p += (hi[c] - lo[c]).powi(2);
            }
            max_p2p = max_p2p.max(p2p.sqrt());
            for f in 1..history.len() {
                let a = history[f - 1][k][j];
                let b = history[f][k][j];
                let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
                max_ff = max_ff.max(d.sqrt());
            }
        }
        let tail = base[k].last().copied().unwrap();
        let mut lo = tail;
        let mut hi = tail;
        for f in history.iter().skip(settle_frames) {
            let p = f[k][nj - 1];
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        let mut tp2p = 0.0;
        for c in 0..3 {
            tp2p += (hi[c] - lo[c]).powi(2);
        }
        ranked.push((max_p2p, name.clone(), radius, bc, tp2p.sqrt(), max_ff));
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (max_p2p, name, radius, bc, tp2p, max_ff) in &ranked {
        report.push_str(&format!(
            "| {name} | {:.0} | {} | {:.1} | {:.1} | {:.2} |\n",
            radius * 1000.0,
            if *bc { "Y" } else { "-" },
            max_p2p * 1000.0,
            tp2p * 1000.0,
            max_ff * 1000.0
        ));
    }
    // Input actuals for context.
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for &(y, _) in applied.iter().skip(settle_frames) {
        lo = lo.min(y);
        hi = hi.max(y);
    }
    let yaw_p2p = hi - lo;
    report.push_str(&format!(
        "\ninput chest yaw p2p over noise phase: {yaw_p2p:.3} deg\n"
    ));
    let summary_path = Path::new(&output_dir).join("summary.md");
    std::fs::write(&summary_path, &report).map_err(|e| format!("write summary: {e}"))?;
    // Full per-joint dump: frame, applied noise, then every chain's
    // joints as x/y/z triplets in chain order (see summary table).
    let mut csv = String::from("frame,chest_yaw_deg,chest_pitch_deg");
    for (_, name, _, _) in &chains {
        let nj = history.first().map(|f| f.len()).unwrap_or(0);
        let _ = nj;
        csv.push_str(&format!(",{name}..."));
    }
    csv.push('\n');
    for row in &rows {
        csv.push_str(row);
        csv.push('\n');
    }
    let csv_path = Path::new(&output_dir).join(format!("frames_n{}.csv", (noise_deg * 10.0) as i32));
    std::fs::write(&csv_path, csv).map_err(|e| format!("write csv: {e}"))?;
    println!("summary: {}", summary_path.display());
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
                primitive_data: Some(std::sync::Arc::clone(prim)),
                morph_weights: Vec::new(),
            });
        }
    }

    // Front three-quarter view so the chest region is visible.
    let camera = vulvatar_lib::app::ViewportCamera {
        distance: 1.15,
        pan: [0.0, 1.02],
        yaw_deg: 18.0,
        pitch_deg: 3.0,
        fov_deg: 36.0,
    };
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(camera.fov_deg, width as f32 / height as f32, 0.05, 20.0);

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

fn build_view_matrix(cam: &vulvatar_lib::app::ViewportCamera) -> (vulvatar_lib::asset::Mat4, [f32; 3]) {
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
    let fwd = [
        target[0] - eye_x,
        target[1] - eye_y,
        target[2] - eye_z,
    ];
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
            [-f[0], -f[1], -f[2], (f[0] * eye_x + f[1] * eye_y + f[2] * eye_z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [eye_x, eye_y, eye_z],
    )
}

fn build_projection_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> vulvatar_lib::asset::Mat4 {
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
