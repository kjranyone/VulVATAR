use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::ImageBuffer;
use vulvatar_lib::app::ViewportCamera;
use vulvatar_lib::asset::fbx::FbxAssetLoader;
use vulvatar_lib::asset::vrm::VrmAssetLoader;
use vulvatar_lib::asset::{
    AvatarAsset, ClothAsset, ClothConstraintSet, ClothMappingMode, ClothMeshMapping,
    ClothOverlayId, ClothOverlayMetadata, ClothPin, ClothRegionTag, ClothRenderRegionBinding,
    ClothSimVertex, ClothSimulationMesh, ClothSolverParams, ClothStableRefSet, DistanceConstraint,
    MeshId, MeshRef, NodeRef, PrimitiveId, PrimitiveRef, VertexSubsetRef,
};
use vulvatar_lib::avatar::{AvatarInstance, AvatarInstanceId};
use vulvatar_lib::renderer::frame_input::{
    CameraState, ClothDeformSnapshot, LightingState, MsaaMode, OutputTargetRequest,
    RenderAlphaMode, RenderAvatarInstance, RenderColorSpace, RenderCullMode, RenderDebugFlags,
    RenderExportMode, RenderFrameInput, RenderMeshInstance, RenderOutputAlpha,
};
use vulvatar_lib::renderer::material::{MaterialShaderMode, MaterialUploadRequest};
use vulvatar_lib::renderer::VulkanRenderer;
use vulvatar_lib::simulation::cloth_gpu_boundary::ClothSolverBackend;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "sample_data/YUMEKA_v1.0.1/FBX/Yumeka_v1.0.fbx".to_string());
    let output_dir = args
        .next()
        .unwrap_or_else(|| "diagnostics/cloth".to_string());
    let total_frames: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(90);

    if output_dir.contains("validation_images") {
        return Err("Diagnostics outputs must not be written to validation_images/ (see CLAUDE.md)".into());
    }

    let output_dir_path = PathBuf::from(&output_dir);
    std::fs::create_dir_all(&output_dir_path)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    let input_path_buf = PathBuf::from(&input_path);
    let is_fbx = input_path_buf
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("fbx"))
        .unwrap_or(false);

    println!("Loading avatar from: {}", input_path);
    let asset: Arc<AvatarAsset> = if is_fbx {
        let loader = FbxAssetLoader::new();
        loader.load(&input_path).map_err(|e| format!("Failed to load FBX: {}", e))?
    } else {
        let loader = VrmAssetLoader::new();
        loader.load(&input_path).map_err(|e| format!("Failed to load VRM: {}", e))?
    };

    println!(
        "Avatar loaded: nodes: {}, meshes: {}, colliders: {}",
        asset.skeleton.nodes.len(),
        asset.meshes.len(),
        asset.colliders.len()
    );

    // Build base avatar instance to compute rest global transforms & skinning matrices
    let mut avatar = AvatarInstance::new(AvatarInstanceId(1), Arc::clone(&asset));
    avatar.build_base_pose();
    avatar.compute_global_pose();
    avatar.build_skinning_matrices();

    // 1. Build cloth asset targeting skirt (Circle.056)
    let pin_y_threshold: f32 = std::env::var("CLOTH_PIN_Y")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.755);

    let (cloth_asset, skirt_prim_id) = build_skirt_cloth_asset(&avatar, pin_y_threshold)?;
    println!(
        "ClothAsset created: {} particles, {} distance constraints, {} pin points",
        cloth_asset.simulation_mesh.vertices.len(),
        cloth_asset.constraints.distance_constraints.len(),
        cloth_asset.pins.len()
    );

    // Save cloth overlay to disk as .vvtcloth
    let overlay_save_path = output_dir_path.join("yumeka_skirt.vvtcloth");
    let overlay_file = vulvatar_lib::persistence::ClothOverlayFile {
        format_version: vulvatar_lib::persistence::OVERLAY_FORMAT_VERSION,
        created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
        last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
        overlay_name: "Yumeka Skirt Cloth".to_string(),
        target_avatar_path: Some(input_path.clone()),
        cloth_asset: Some(cloth_asset.clone()),
        last_rebound_with: None,
    };
    if let Ok(json) = serde_json::to_string_pretty(&overlay_file) {
        let _ = std::fs::write(&overlay_save_path, json);
        println!("Saved cloth overlay asset to: {}", overlay_save_path.display());
    }

    // 2. Attach Cloth to AvatarInstance
    let overlay_id = ClothOverlayId(1);
    let overlay_slot_idx = avatar.attach_cloth_overlay(overlay_id);
    avatar.init_cloth_overlay(overlay_slot_idx, &cloth_asset);

    // Also init primary cloth for direct snapshot feed
    avatar.init_cloth_sim(&cloth_asset);
    avatar.cloth_enabled = true;

    // Enable colliders for thighs (UpperLeg_L and UpperLeg_R)
    for (i, col) in avatar.asset.colliders.iter().enumerate() {
        let node_name = avatar.asset.skeleton.nodes.get(col.node.0 as usize)
            .map(|n| n.name.to_lowercase())
            .unwrap_or_default();
        if node_name.contains("upperleg") || node_name.contains("hips") {
            avatar.collider_enabled[i] = true;
            println!("Enabled collider #{}: {} on node '{}'", i, col.id.0, node_name);
        }
    }

    // Set cloth physics tuning
    if let Some(ref mut sim) = avatar.cloth_sim {
        sim.solver_iterations = 8;
        sim.damping = 0.035;
        sim.gravity_scale = 1.0;
        sim.collision_margin = 0.015; // 1.5 cm collider margin around legs
        sim.wind_direction = [0.3, 0.0, 0.15];
        sim.wind_response = 0.8; // visible flutter
    }

    // 3. Initialize Vulkan renderer
    let render_width = 1024u32;
    let render_height = 1024u32;
    println!("Initializing Vulkan renderer ({}x{})...", render_width, render_height);
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    println!("Starting simulation and offline render across {} frames...", total_frames);
    let dt = 1.0 / 60.0;
    let mut recorded_snapshots = Vec::new();

    // Keyframes to save images for
    let capture_frames: HashSet<usize> = [0, 10, 20, 30, 45, 60, 75, 90]
        .iter()
        .copied()
        .filter(|&f| f <= total_frames)
        .collect();

    // Record initial hips transform
    let hips_node_idx = avatar.asset.skeleton.nodes.iter().enumerate()
        .find(|(_, n)| n.name.to_lowercase() == "hips")
        .map(|(i, _)| i)
        .unwrap_or(0);
    let initial_hips_transform = avatar.pose.local_transforms[hips_node_idx].clone();

    for frame in 0..=total_frames {
        let t = frame as f32 * dt;

        // Apply natural idle swaying to hips:
        // sway side-to-side (±2.5 cm) and subtle roll (±2.5 deg)
        if frame > 0 {
            let sway_x = (t * 3.0).sin() * 0.025;
            let sway_roll = (t * 3.0).sin() * 0.04; // rad

            let mut cur_t = initial_hips_transform.clone();
            cur_t.translation[0] += sway_x;
            // Roll rotation around Z axis: [0, 0, sin(roll/2), cos(roll/2)]
            let q_roll = [0.0, 0.0, (sway_roll * 0.5).sin(), (sway_roll * 0.5).cos()];
            // Multiply rotation
            cur_t.rotation = vulvatar_lib::math_utils::quat_mul(&initial_hips_transform.rotation, &q_roll);

            avatar.pose.local_transforms[hips_node_idx] = cur_t;
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();

            // Step secondary motion (hair, accessories) with newly attached body colliders
            let spring_tuning = vulvatar_lib::simulation::spring::SpringTuning::default();
            vulvatar_lib::simulation::spring::step_spring_bones(
                dt,
                &mut avatar,
                &[],
                &spring_tuning,
                [0.0, -1.0, 0.0],
                1.0,
            );
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();

            // Apply gentle fluctuating breeze
            if let Some(ref mut sim) = avatar.cloth_sim {
                let gust = (t * 2.2).sin() * 0.4 + 0.6;
                sim.wind_direction = [0.35 * gust, (t * 1.8).cos() * 0.05, 0.2 * gust];
            }

            vulvatar_lib::simulation::cloth_solver::step_cloth(dt, &mut avatar, &[]);
        }

        // Check simulation health
        if let Some(ref cs) = avatar.cloth_state {
            let mut y_min = f32::INFINITY;
            let mut y_max = f32::NEG_INFINITY;
            let mut has_nan = false;
            for p in &cs.sim_positions {
                if p[0].is_nan() || p[1].is_nan() || p[2].is_nan() {
                    has_nan = true;
                    break;
                }
                y_min = y_min.min(p[1]);
                y_max = y_max.max(p[1]);
            }
            if has_nan {
                return Err(format!("Cloth particles produced NaN at frame {}", frame));
            }

            if frame % 15 == 0 || capture_frames.contains(&frame) {
                println!(
                    "Frame {:03}/{:03}: Cloth Y range [{:.3}, {:.3}] m, deform version {}",
                    frame, total_frames, y_min, y_max, cs.deform_output.version
                );
            }
        }

        // Render and capture image
        if capture_frames.contains(&frame) {
            let frame_input = build_render_frame_input(
                &avatar,
                skirt_prim_id,
                render_width,
                render_height,
                frame,
                dt,
            );

            // Double render invocation for pipelined CPU readback
            let _ = renderer.render(&frame_input)
                .map_err(|e| format!("Rendering warm-up failed at frame {}: {}", frame, e))?;
            let render_result = renderer
                .render(&frame_input)
                .map_err(|e| format!("Rendering failed at frame {}: {}", frame, e))?;

            if let Some(exported) = render_result.exported_frame.as_ref() {
                if let Some(pixels) = exported.cpu_pixel_data() {
                    let out_png = output_dir_path.join(format!("skirt_frame_{:03}.png", frame));
                    save_png(&out_png, [render_width, render_height], pixels.as_slice())?;
                    println!("  -> Saved render frame to {}", out_png.display());
                    recorded_snapshots.push((frame, out_png));
                }
            }
        }
    }

    // Write a summary report
    let report_path = output_dir_path.join("cloth_simulation_summary.md");
    let mut report = String::new();
    report.push_str("# Yumeka Skirt Cloth Simulation Verification Report\n\n");
    report.push_str(&format!("- Target Avatar: `{}`\n", input_path));
    report.push_str(&format!("- Target Skirt Mesh: Circle.056 (PrimitiveId({}))\n", skirt_prim_id.0));
    report.push_str(&format!("- Vertices: {}\n", cloth_asset.simulation_mesh.vertices.len()));
    report.push_str(&format!("- Distance Constraints: {}\n", cloth_asset.constraints.distance_constraints.len()));
    report.push_str(&format!("- Pinned Particles: {}\n", cloth_asset.pins.len()));
    report.push_str(&format!("- Total Simulated Frames: {} ({} s at 60 FPS)\n\n", total_frames, total_frames as f32 / 60.0));
    report.push_str("## Captured Keyframes\n\n");
    for (f, p) in &recorded_snapshots {
        report.push_str(&format!("### Frame {:03}\n\n", f));
        report.push_str(&format!("![Frame {:03}]({})\n\n", f, p.file_name().unwrap().to_string_lossy()));
    }
    std::fs::write(&report_path, report)
        .map_err(|e| format!("Failed to write report: {}", e))?;
    println!("\nWrote verification summary to {}", report_path.display());

    println!("\n=== Cloth simulation diagnostic completed successfully! ===");
    Ok(())
}

fn build_skirt_cloth_asset(
    avatar: &AvatarInstance,
    pin_y: f32,
) -> Result<(ClothAsset, PrimitiveId), String> {
    let asset = &avatar.asset;

    // Explicitly target Circle.056 (the actual skirt mesh)
    let mut skirt_prim: Option<(PrimitiveId, MeshId, String, usize)> = None;
    for mesh in &asset.meshes {
        if mesh.name.eq_ignore_ascii_case("circle.056") {
            for prim in &mesh.primitives {
                if let Some(ref vd) = prim.vertices {
                    skirt_prim = Some((prim.id, mesh.id, mesh.name.clone(), vd.positions.len()));
                    break;
                }
            }
        }
    }

    // Fallback search if name changed
    let (prim_id, mesh_id, mesh_name, vert_count) = skirt_prim.or_else(|| {
        for mesh in &asset.meshes {
            for prim in &mesh.primitives {
                if let Some(ref vd) = prim.vertices {
                    let b_min_y = prim.bounds.min[1];
                    let b_max_y = prim.bounds.max[1];
                    if b_min_y > 0.55 && b_max_y < 0.85 && vd.positions.len() >= 2000 && vd.positions.len() <= 3000 {
                        return Some((prim.id, mesh.id, mesh.name.clone(), vd.positions.len()));
                    }
                }
            }
        }
        None
    }).ok_or_else(|| "Could not locate Circle.056 skirt mesh in avatar".to_string())?;

    println!("Selected skirt primitive: {:?} on mesh '{}' ({} vertices)", prim_id, mesh_name, vert_count);

    let prim = asset
        .meshes
        .iter()
        .flat_map(|m| &m.primitives)
        .find(|p| p.id == prim_id)
        .unwrap();
    let vd = prim.vertices.as_ref().unwrap();
    let indices = prim.indices.as_ref().unwrap();

    // Look for Skirt_root or Hips node for pin binding
    let pin_node_idx = asset
        .skeleton
        .nodes
        .iter()
        .enumerate()
        .find(|(_, n)| {
            let l = n.name.to_lowercase();
            l == "skirt_root" || l.contains("skirt_root")
        })
        .or_else(|| {
            asset
                .skeleton
                .nodes
                .iter()
                .enumerate()
                .find(|(_, n)| n.name.to_lowercase() == "hips")
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let pin_node = &asset.skeleton.nodes[pin_node_idx];
    println!("Binding pins to bone: '{}' (node idx {})", pin_node.name, pin_node_idx);

    // Compute inverse of pin node's global transform for local pin offsets
    let m = &avatar.pose.global_transforms[pin_node_idx];
    let r0 = [m[0][0], m[1][0], m[2][0]];
    let r1 = [m[0][1], m[1][1], m[2][1]];
    let r2 = [m[0][2], m[1][2], m[2][2]];
    let t = [m[3][0], m[3][1], m[3][2]];

    let mut sim_vertices = Vec::with_capacity(vd.positions.len());
    let mut cloth_pins = Vec::new();

    for (i, &pos) in vd.positions.iter().enumerate() {
        // Transform raw local pos to bind-pose world pos using skinning matrices
        let mut world_pos = [0.0f32; 3];
        let mut total_w = 0.0f32;
        if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
            for k in 0..4 {
                let w = vd.joint_weights[i][k];
                if w > 0.0001 {
                    let j = vd.joint_indices[i][k] as usize;
                    if j < avatar.pose.skinning_matrices.len() {
                        let sm = &avatar.pose.skinning_matrices[j];
                        let tx = sm[0][0] * pos[0] + sm[1][0] * pos[1] + sm[2][0] * pos[2] + sm[3][0];
                        let ty = sm[0][1] * pos[0] + sm[1][1] * pos[1] + sm[2][1] * pos[2] + sm[3][1];
                        let tz = sm[0][2] * pos[0] + sm[1][2] * pos[1] + sm[2][2] * pos[2] + sm[3][2];
                        world_pos[0] += w * tx;
                        world_pos[1] += w * ty;
                        world_pos[2] += w * tz;
                        total_w += w;
                    }
                }
            }
        }

        if total_w > 0.001 {
            world_pos[0] /= total_w;
            world_pos[1] /= total_w;
            world_pos[2] /= total_w;
        } else {
            world_pos = pos;
        }

        let is_pin = world_pos[1] >= pin_y;
        let normal = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
        let uv = vd.uvs.get(i).copied().unwrap_or([0.0, 0.0]);

        if is_pin {
            let d = [world_pos[0] - t[0], world_pos[1] - t[1], world_pos[2] - t[2]];
            let ox = r0[0] * d[0] + r0[1] * d[1] + r0[2] * d[2];
            let oy = r1[0] * d[0] + r1[1] * d[1] + r1[2] * d[2];
            let oz = r2[0] * d[0] + r2[1] * d[1] + r2[2] * d[2];

            cloth_pins.push(ClothPin {
                sim_vertex_indices: vec![i as u32],
                binding_node: NodeRef {
                    id: pin_node.id,
                    name: pin_node.name.clone(),
                },
                offset: [ox, oy, oz],
            });
        }

        sim_vertices.push(ClothSimVertex {
            position: world_pos,
            normal,
            uv,
            pinned: is_pin,
        });
    }

    // Build unique distance constraints from mesh edges
    let mut edge_set: HashSet<(u32, u32)> = HashSet::new();
    let tri_count = indices.len() / 3;
    let mut tri_indices = Vec::with_capacity(indices.len());

    for t in 0..tri_count {
        let i0 = indices[t * 3];
        let i1 = indices[t * 3 + 1];
        let i2 = indices[t * 3 + 2];
        tri_indices.push(i0);
        tri_indices.push(i1);
        tri_indices.push(i2);

        for &(a, b) in &[(i0, i1), (i1, i2), (i2, i0)] {
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_set.insert(edge);
        }
    }

    let mut distance_constraints = Vec::with_capacity(edge_set.len());
    let mut rest_lengths = Vec::with_capacity(edge_set.len());

    for &(a, b) in &edge_set {
        let pa = sim_vertices[a as usize].position;
        let pb = sim_vertices[b as usize].position;
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        rest_lengths.push(dist);
        distance_constraints.push(DistanceConstraint {
            indices: [a, b],
            rest_length: dist,
            stiffness: 0.98,
        });
    }

    let sim_mesh = ClothSimulationMesh {
        vertices: sim_vertices,
        indices: tri_indices,
        rest_lengths,
        attachment_classes: Vec::new(),
        region_tags: vec![ClothRegionTag(1)],
    };

    let cloth_asset = ClothAsset {
        id: ClothOverlayId(1),
        target_avatar: asset.id,
        target_avatar_hash: asset.source_hash.clone(),
        metadata: ClothOverlayMetadata {
            name: "Yumeka Skirt".to_string(),
            format_version: 1,
            created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
            last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
        },
        stable_refs: ClothStableRefSet {
            node_refs: vec![NodeRef {
                id: pin_node.id,
                name: pin_node.name.clone(),
            }],
            mesh_refs: vec![MeshRef {
                id: mesh_id,
                name: mesh_name.clone(),
            }],
            primitive_refs: vec![PrimitiveRef {
                id: prim_id,
                name: "skirt_primitive".to_string(),
            }],
        },
        simulation_mesh: sim_mesh,
        render_bindings: vec![ClothRenderRegionBinding {
            primitive: PrimitiveRef {
                id: prim_id,
                name: "skirt_primitive".to_string(),
            },
            vertex_subset: VertexSubsetRef {
                offset: 0,
                count: vert_count as u32,
            },
            mapping_region: ClothRegionTag(1),
            mesh: Some(MeshRef {
                id: mesh_id,
                name: mesh_name.clone(),
            }),
        }],
        mesh_mapping: ClothMeshMapping {
            mapping_mode: ClothMappingMode::Nearest,
            entries: Vec::new(),
        },
        pins: cloth_pins,
        constraints: ClothConstraintSet {
            distance_constraints,
            bend_constraints: Vec::new(),
        },
        collision_bindings: Vec::new(),
        lods: Vec::new(),
        solver_params: ClothSolverParams {
            substeps: 4,
            iterations: 8,
            gravity_scale: 1.0,
            damping: 0.035,
            self_collision: false,
            collision_margin: 0.015,
            wind_response: 0.8,
        },
    };

    Ok((cloth_asset, prim_id))
}

fn build_render_frame_input(
    avatar: &AvatarInstance,
    skirt_prim_id: PrimitiveId,
    width: u32,
    height: u32,
    frame_idx: usize,
    dt: f32,
) -> RenderFrameInput {
    let mut mesh_instances = Vec::new();

    for mesh in &avatar.asset.meshes {
        for prim in &mesh.primitives {
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

    // Collect cloth deforms
    let mut cloth_deforms = Vec::new();
    if let Some(ref cs) = avatar.cloth_state {
        cloth_deforms.push(ClothDeformSnapshot {
            target_primitive_id: skirt_prim_id,
            target_mesh_id: cs.target_mesh_id,
            vertex_offset: cs.target_vertex_offset,
            vertex_count: cs.target_vertex_count,
            deformed_positions: cs.deform_output.deformed_positions.clone(),
            deformed_normals: cs.deform_output.deformed_normals.clone(),
            version: cs.deform_output.version,
            solver_backend: ClothSolverBackend::Cpu,
            gpu_control: None,
            gpu_attach: None,
        });
    }

    // Camera focused intimately on the skirt & thighs (pan Y: 0.68m, dist: 0.82m)
    let cam_yaw: f32 = std::env::var("CAM_YAW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(22.0);
    let cam_pitch: f32 = std::env::var("CAM_PITCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-5.0);
    let cam_dist: f32 = std::env::var("CAM_DIST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.85);
    let cam_pan_y: f32 = std::env::var("CAM_PAN_Y")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.68);

    let camera = ViewportCamera {
        distance: cam_dist,
        pan: [0.0, cam_pan_y],
        yaw_deg: cam_yaw,
        pitch_deg: cam_pitch,
        fov_deg: 36.0,
    };

    let extent = [width, height];
    let (view, eye_pos) = build_view_matrix(&camera);
    let projection = build_projection_matrix(
        camera.fov_deg,
        extent[0] as f32 / extent[1].max(1) as f32,
        0.05,
        20.0,
    );

    let avatar_instance = RenderAvatarInstance {
        instance_id: avatar.id,
        world_transform: avatar.world_transform.clone(),
        mesh_instances,
        skinning_matrices: avatar.pose.skinning_matrices.clone(),
        cloth_deforms,
        debug_flags: RenderDebugFlags::default(),
    };

    RenderFrameInput {
        camera: CameraState {
            view,
            projection,
            position_ws: eye_pos,
            viewport_extent: extent,
        },
        lighting: LightingState::default(),
        instances: vec![avatar_instance],
        output_request: OutputTargetRequest {
            preview_enabled: true,
            output_enabled: true,
            extent,
            color_space: RenderColorSpace::Srgb,
            alpha_mode: RenderOutputAlpha::Opaque,
            export_mode: RenderExportMode::CpuReadback,
            msaa: MsaaMode::Off,
        },
        background_image_path: None,
        show_ground_grid: false,
        background_color: [0.12, 0.12, 0.14],
        transparent_background: false,
        avatar_opacity: 1.0,
        bloom: Default::default(),
        generative_background: Default::default(),
        background_tracking: Default::default(),
        time_seconds: frame_idx as f32 * dt,
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

fn save_png(path: &Path, extent: [u32; 2], pixels: &[u8]) -> Result<(), String> {
    let img: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| format!("PNG buffer construction failed for {}", path.display()))?;
    img.save(path)
        .map_err(|e| format!("failed to save '{}': {e}", path.display()))
}
