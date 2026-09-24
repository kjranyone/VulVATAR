//! Quantifies the body-SDF field's surface error in the cloth/hair
//! contact band (0-10 mm off the skin): for points at TRUE distance `t`
//! along the surface normal, `err = field.sample(P) − t`. Positive err
//! = the field OVERESTIMATES distance = collision resolves too close to
//! the skin (cloth sinks in, skin pokes through). This is the failure
//! mode the hair-chain A/B (3-20 mm band) cannot see and the desk-skirt
//! A/B renders too coarsely to judge.
//!
//! Runs the real pipeline (renderer splat → readback) at the rest pose,
//! then probes K random surface points × offsets. A/B knobs are the
//! live ones: VULVATAR_SDF_SPLAT_STRIDE / VULVATAR_SDF_MAX_CELLS.
//!
//! Output: `diagnostics/sdf_surface_error/summary.md`.

use std::path::Path;

use vulvatar_lib::renderer::frame_input::{
    BodySdfPlan, CameraState, LightingState, OutputTargetRequest, RenderAvatarInstance,
    RenderFrameInput, RenderMeshInstance,
};
use vulvatar_lib::renderer::material::MaterialUploadRequest;
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::simulation::sdf::{SdfField, SdfGrid, SENTINEL};

fn main() -> Result<(), String> {
    let input_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.3/FBX/Yumeka_v1.0.3.fbx".to_string());
    let output_dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "diagnostics/sdf_surface_error".to_string());
    if output_dir.contains("validation_images") {
        return Err(
            "Diagnostics outputs must not be written to validation_images/ (see AGENTS.md)".into(),
        );
    }
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("create dir: {e}"))?;

    let loader = vulvatar_lib::asset::fbx::FbxAssetLoader::new();
    let asset = loader.load(&input_path).map_err(|e| format!("load: {e}"))?;
    let mut avatar =
        vulvatar_lib::avatar::AvatarInstance::new(vulvatar_lib::avatar::AvatarInstanceId(1), asset);

    // Splat list: the same selection the app's frame input builder
    // performs (body + face/head + garment layer).
    let garments = vulvatar_lib::asset::clearance::sdf_garment_splat_prims(&avatar.asset, 3);
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
    let grid = SdfGrid::for_aabb(&union);
    println!(
        "grid dims={:?} voxel={:.1} mm cells={}",
        grid.dims,
        grid.voxel * 1000.0,
        grid.cell_count()
    );

    // Rest pose: skinned vertices reproduce the authored surface, so the
    // rest triangles below are the splat's input geometry.
    for (i, node) in avatar.asset.skeleton.nodes.iter().enumerate() {
        avatar.pose.local_transforms[i] = node.rest_local.clone();
    }
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    let mut renderer = VulkanRenderer::new();
    renderer.initialize();
    let frame_input = build_frame_input(&avatar, splat.clone(), grid, 320, 240);
    let _ = renderer.render(&frame_input)?;
    let result = renderer.render(&frame_input)?;
    let mut field: Option<SdfField> = None;
    for entry in &result.sdf_fields {
        if entry.instance_id == avatar.id.0 {
            field = Some(SdfField::new(entry.grid, entry.data.clone()));
        }
    }
    let field = field.ok_or("renderer returned no SDF field")?;

    // Surface samples from the BODY primitive (the skin the cloth/hair
    // resolve against).
    let (mesh_id, primitive_id) = vulvatar_lib::asset::clearance::find_body_primitive(&avatar.asset)
        .ok_or("no body primitive")?;
    let prim = avatar
        .asset
        .meshes
        .iter()
        .find(|m| m.id == mesh_id)
        .and_then(|m| m.primitives.iter().find(|p| p.id == primitive_id))
        .ok_or("body primitive missing")?;
    let vd = prim
        .vertices
        .as_ref()
        .ok_or("body primitive has no vertex data")?;
    let indices = prim.indices.as_ref().ok_or("body primitive has no indices")?;
    let tri_count = indices.len() / 3;
    println!(
        "body prim: {} tris, {} verts",
        tri_count,
        vd.positions.len()
    );

    // Sample points must live in the SAME space the splat read: the
    // SKINNED vertices. The renderer's transform_cs applies
    // avatar.pose.skinning_matrices (column-major) to the bind-pose
    // positions with per-vertex joint weights; at rest local transforms
    // that differs from the authored positions whenever the FBX bind
    // pose isn't the rest local pose (measured ~9.5 mm on Yumeka —
    // probing the raw rest positions reported that offset as fake field
    // error, identically for every setting).
    let skin = prim.skin.as_ref();
    let skinned: Vec<[f32; 3]> = vd
        .positions
        .iter()
        .enumerate()
        .map(|(vi, p)| {
            let Some(skin) = skin else {
                return *p;
            };
            let ji = vd.joint_indices.get(vi).copied().unwrap_or([0; 4]);
            let w = vd.joint_weights.get(vi).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
            let mut out = [0.0f32; 3];
            for k in 0..4 {
                if w[k] <= 0.0 {
                    continue;
                }
                let node = skin.joint_nodes.get(k).map(|n| n.0 as usize).unwrap_or(0);
                let Some(m) = avatar.pose.skinning_matrices.get(node) else {
                    continue;
                };
                // Column-major M × [x, y, z, 1].
                let x = p[0];
                let y = p[1];
                let z = p[2];
                out[0] += w[k]
                    * (m[0][0] * x + m[1][0] * y + m[2][0] * z + m[3][0]);
                out[1] += w[k]
                    * (m[0][1] * x + m[1][1] * y + m[2][1] * z + m[3][1]);
                out[2] += w[k]
                    * (m[0][2] * x + m[1][2] * y + m[2][2] * z + m[3][2]);
            }
            out
        })
        .collect();
    println!(
        "skinned[0] = [{:.3}, {:.3}, {:.3}]  rest[0] = [{:.3}, {:.3}, {:.3}]",
        skinned[0][0],
        skinned[0][1],
        skinned[0][2],
        vd.positions[0][0],
        vd.positions[0][1],
        vd.positions[0][2]
    );

    // Deterministic pseudo-random triangle picks across the surface.
    let samples_per_offset = 4000usize;
    let offsets_mm = [0.0f32, 2.0, 4.0, 6.0, 8.0, 12.0];
    let mut report = String::new();
    report.push_str(&format!(
        "# Body-SDF surface error: {}

- grid {:?} @ {:.1} mm ({} cells)
- stride: env VULVATAR_SDF_SPLAT_STRIDE (see renderer/sdf_field.rs)
- K = {} surface points × offsets {{0, 2, 4, 6, 8, 12}} mm along the
  interpolated vertex normal
- err = field.sample(P) − t; **positive = field overestimates = the
  collision resolves the cloth/hair too close to the skin**

| t (mm) | sentinels | err p50 (mm) | p95 (mm) | max (mm) | >1 mm | >2 mm | >4 mm |
|---|---|---|---|---|---|---|---|
",
        input_path,
        grid.dims,
        grid.voxel * 1000.0,
        grid.cell_count(),
        samples_per_offset,
    ));

    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for &t_mm in &offsets_mm {
        let t = t_mm / 1000.0;
        let mut errs: Vec<f32> = Vec::with_capacity(samples_per_offset);
        let mut sentinels = 0usize;
        for _ in 0..samples_per_offset {
            let tri = (rng() % tri_count as u64) as usize;
            let (u, v) = (
                ((rng() % 10_000) as f32 / 10_000.0),
                ((rng() % 10_000) as f32 / 10_000.0),
            );
            let (u, v) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };
            let ia = indices[tri * 3] as usize;
            let ib = indices[tri * 3 + 1] as usize;
            let ic = indices[tri * 3 + 2] as usize;
            let (pa, pb, pc) = (skinned[ia], skinned[ib], skinned[ic]);
            let w = |i: usize| 1.0 - u - v;
            let _ = w;
            let bary = |vp: &vulvatar_lib::asset::Vec3| {
                [
                    pa[0] * (1.0 - u - v) + pb[0] * u + pc[0] * v,
                    pa[1] * (1.0 - u - v) + pb[1] * u + pc[1] * v,
                    pa[2] * (1.0 - u - v) + pb[2] * u + pc[2] * v,
                ]
            };
            let pos = bary(&pa);
            // Interpolated shading normal, geometric fallback.
            let mut n = [
                vd.normals[ia][0] * (1.0 - u - v)
                    + vd.normals[ib][0] * u
                    + vd.normals[ic][0] * v,
                vd.normals[ia][1] * (1.0 - u - v)
                    + vd.normals[ib][1] * u
                    + vd.normals[ic][1] * v,
                vd.normals[ia][2] * (1.0 - u - v)
                    + vd.normals[ib][2] * u
                    + vd.normals[ic][2] * v,
            ];
            let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if nl < 1e-6 {
                let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
                n = [n[0] / l, n[1] / l, n[2] / l];
            } else {
                n = [n[0] / nl, n[1] / nl, n[2] / nl];
            }
            let p = [
                pos[0] + n[0] * t,
                pos[1] + n[1] * t,
                pos[2] + n[2] * t,
            ];
            let s = field.sample(p);
            if s >= SENTINEL {
                sentinels += 1;
            } else {
                errs.push(s - t);
            }
        }
        errs.sort_by(|a, b| a.total_cmp(b));
        let pick = |q: f32| -> f32 {
            let idx = ((errs.len() as f32 * q).round() as usize).min(errs.len().saturating_sub(1));
            errs.get(idx).copied().unwrap_or(f32::NAN) * 1000.0
        };
        let count = |thresh: f32| -> String {
            if errs.is_empty() {
                return "n/a".into();
            }
            let over = errs.iter().filter(|e| **e > thresh).count();
            format!("{:.1}%", over as f32 / errs.len() as f32 * 100.0)
        };
        report.push_str(&format!(
            "| {:.0} | {} | {:+.2} | {:+.2} | {:+.2} | {} | {} | {} |\n",
            t_mm,
            sentinels,
            pick(0.5),
            pick(0.95),
            errs.last().copied().unwrap_or(f32::NAN) * 1000.0,
            count(0.001),
            count(0.002),
            count(0.004),
        ));
        println!(
            "t={:>2} mm: sentinels {} p50 {:+.2} p95 {:+.2} max {:+.2} mm",
            t_mm,
            sentinels,
            pick(0.5),
            pick(0.95),
            errs.last().copied().unwrap_or(f32::NAN) * 1000.0
        );
    }

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
            let material_binding = material_asset
                .map(MaterialUploadRequest::from_asset_material)
                .unwrap_or_else(MaterialUploadRequest::default_material);
            mesh_instances.push(RenderMeshInstance {
                mesh_id: mesh.id,
                primitive_id: prim.id,
                material_binding,
                bounds: prim.bounds,
                alpha_mode: vulvatar_lib::renderer::frame_input::RenderAlphaMode::Opaque,
                cull_mode: vulvatar_lib::renderer::frame_input::RenderCullMode::BackFace,
                outline: Default::default(),
                primitive_data: Some(std::sync::Arc::clone(prim)),
                morph_weights: Vec::new(),
            });
        }
    }
    let camera = vulvatar_lib::app::ViewportCamera {
        distance: 1.15,
        pan: [0.0, 1.05],
        yaw_deg: 160.0,
        pitch_deg: 5.0,
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
            output_enabled: false,
            extent: [width, height],
            color_space: vulvatar_lib::renderer::frame_input::RenderColorSpace::Srgb,
            alpha_mode: vulvatar_lib::renderer::frame_input::RenderOutputAlpha::Opaque,
            export_mode: vulvatar_lib::renderer::frame_input::RenderExportMode::CpuReadback,
            msaa: vulvatar_lib::renderer::frame_input::MsaaMode::Off,
        },
        generative_background: Default::default(),
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.08, 0.08, 0.08],
        transparent_background: false,
        avatar_opacity: 1.0,
        bloom: Default::default(),
        background_tracking: Default::default(),
        time_seconds: 0.0,
    }
}

/// Column-major view matrix + eye position — mirrors diagnose_sdf_hair's
/// helpers (kept local so this bin stays standalone).
fn build_view_matrix(
    cam: &vulvatar_lib::app::ViewportCamera,
) -> ([[f32; 4]; 4], [f32; 3]) {
    let yaw = cam.yaw_deg.to_radians();
    let pitch = cam.pitch_deg.to_radians();
    let eye = [
        cam.pan[0] + cam.distance * pitch.cos() * yaw.sin(),
        cam.pan[1] + cam.distance * pitch.sin(),
        cam.distance * pitch.cos() * yaw.cos(),
    ];
    let target = [cam.pan[0], cam.pan[1], 0.0];
    let up = [0.0_f32, 1.0, 0.0];
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f32; 3]| {
        let l = dot(a, a).sqrt().max(1e-9);
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let fwd = norm(sub(target, eye));
    let right = norm(cross(fwd, up));
    let up2 = cross(right, fwd);
    let view = [
        [right[0], up2[0], -fwd[0], 0.0],
        [right[1], up2[1], -fwd[1], 0.0],
        [right[2], up2[2], -fwd[2], 0.0],
        [-dot(right, eye), -dot(up2, eye), dot(fwd, eye), 1.0],
    ];
    (view, eye)
}

fn build_projection_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_deg.to_radians() * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}
