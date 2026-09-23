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
    ClothSimVertex, ClothSimulationMesh, ClothSolverParams, ClothStableRefSet,
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
    let total_frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(90);

    if output_dir.contains("validation_images") {
        return Err(
            "Diagnostics outputs must not be written to validation_images/ (see CLAUDE.md)".into(),
        );
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
    let mut loaded_asset: Arc<AvatarAsset> = if is_fbx {
        let loader = FbxAssetLoader::new();
        loader
            .load(&input_path)
            .map_err(|e| format!("Failed to load FBX: {}", e))?
    } else {
        let loader = VrmAssetLoader::new();
        loader
            .load(&input_path)
            .map_err(|e| format!("Failed to load VRM: {}", e))?
    };

    // E4/E5 mechanism-isolation knobs: strip ONE anchor family at a
    // time so the per-primitive VBO audit attributes its corrections
    // to clearance vs containment. (`VULVATAR_STRIP_CLEARANCE=1` /
    // `VULVATAR_STRIP_CONTAINMENT=1`.)
    let strip = |a: &mut AvatarAsset, clearance: bool| {
        for mesh in a.meshes.iter_mut() {
            for prim in mesh.primitives.iter_mut() {
                // Primitives are Arc-shared with bake caches etc. —
                // clone-on-write only when stripping (diagnostic path).
                let prim = Arc::make_mut(prim);
                if clearance {
                    prim.skin_anchors = None;
                    prim.body_primitive_id = None;
                } else {
                    prim.containment_anchors = None;
                    prim.containment_primitive_id = None;
                }
            }
        }
    };
    // Freshly loaded — the Arc is uniquely owned here unless the loader
    // caches; fall back loudly rather than silently keeping anchors.
    match Arc::get_mut(&mut loaded_asset) {
        Some(a) => {
            if std::env::var_os("VULVATAR_STRIP_CLEARANCE").is_some() {
                strip(a, true);
                println!("STRIP_CLEARANCE: all clearance anchors removed");
            }
            if std::env::var_os("VULVATAR_STRIP_CONTAINMENT").is_some() {
                strip(a, false);
                println!("STRIP_CONTAINMENT: all containment anchors removed");
            }
        }
        None => {
            if std::env::var_os("VULVATAR_STRIP_CLEARANCE").is_some()
                || std::env::var_os("VULVATAR_STRIP_CONTAINMENT").is_some()
            {
                println!("WARN: asset Arc is shared; anchor stripping skipped");
            }
        }
    }
    let asset: Arc<AvatarAsset> = loaded_asset;

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

    // AUTO_CLOTH=1: attach through the app path (`attach_auto_cloth`) —
    // the auto classifier, the deep pin band, GPU backend, and the
    // cloth-target skin-anchor strip — instead of this bin's own legacy
    // pin recipe. Everything downstream (GPU dispatch, colliders, pin
    // targets) reads the same overlay slots, so the renderer flow is
    // unchanged. SKIRT_POSE=sit|lean|sit_lean then drives the skeleton
    // through the desk-envelope poses (ramped over frames 1..30).
    let auto_cloth = std::env::var_os("AUTO_CLOTH").is_some();
    let (cloth_asset_opt, skirt_prim_id) = if auto_cloth {
        let attached = vulvatar_lib::simulation::auto_cloth::attach_auto_cloth(&mut avatar);
        if attached == 0 {
            return Err("AUTO_CLOTH: no skirt classified".into());
        }
        // `attach_auto_cloth` sets its own constant wind (0.35) on every
        // slot AFTER the stilling block further down ran, so the A/B
        // below would silently run wind-driven flutter. Re-still here,
        // after the attach — `WIND_ON=1` opts back in for wind A/Bs.
        if std::env::var_os("WIND_ON").is_none() {
            for slot in avatar.cloth_overlays.iter_mut() {
                slot.sim.wind_response = 0.0;
                slot.sim.wind_direction = [0.0, 0.0, 0.0];
            }
        }
        let slot = avatar
            .cloth_overlays
            .iter()
            .max_by_key(|s| s.sim.particles.len())
            .unwrap();
        let pid = slot
            .state
            .target_primitive_id
            .ok_or("AUTO_CLOTH: slot has no target primitive")?;
        println!(
            "AUTO_CLOTH: {} garment(s), primary slot prim {:?} ({} particles, {} pins)",
            attached,
            pid.0,
            slot.sim.particles.len(),
            slot.sim.pin_targets.len()
        );
        (None, pid)
    } else {
        // 1. Build cloth asset targeting skirt (Circle.056)
        let pin_y_threshold: f32 = std::env::var("CLOTH_PIN_Y")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.755);
        // Pin-band depth knob: a deeper pinned band shortens the free cloth
        // length so hip sway translates the skirt instead of letting it
        // buckle/fold at the sides (distance-only XPBD has zero bending
        // stiffness).
        let pin_band_depth: f32 = std::env::var("CLOTH_PIN_BAND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.035);
        let cloth_damping: f32 = std::env::var("CLOTH_DAMPING")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.035);

        let (cloth_asset, skirt_prim_id) =
            build_skirt_cloth_asset(&avatar, pin_y_threshold, pin_band_depth, cloth_damping)?;
        (Some(cloth_asset), skirt_prim_id)
    };
    println!(
        "ClothAsset created: {} particles, {} distance constraints, {} pin points",
        cloth_asset_opt
            .as_ref()
            .map(|c| c.simulation_mesh.vertices.len())
            .unwrap_or_default(),
        cloth_asset_opt
            .as_ref()
            .map(|c| c.constraints.distance_constraints.len())
            .unwrap_or_default(),
        cloth_asset_opt.as_ref().map(|c| c.pins.len()).unwrap_or_default()
    );
    // SKIRT_POSE desk-envelope anchors: applied from the REST locals each
    // frame (overwriting the idle sway), ramped in over frames 1..30.
    let skirt_pose = std::env::var("SKIRT_POSE").unwrap_or_default();
    let rest_locals = avatar.pose.local_transforms.clone();
    if !skirt_pose.is_empty() {
        println!("SKIRT_POSE: driving '{skirt_pose}' (ramp over frames 1..30)");
    }

    // Save cloth overlay to disk as .vvtcloth (legacy recipe only —
    // AUTO_CLOTH slots are runtime-only and already attached).
    if let Some(cloth_asset) = &cloth_asset_opt {
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
            println!(
                "Saved cloth overlay asset to: {}",
                overlay_save_path.display()
            );
        }
    }

    // CLOTH_OFF=1: skip cloth entirely — the skirt renders as its
    // static skinned mesh (the ORIGINAL FBX silhouette reference the
    // user compares against).
    let cloth_off = std::env::var_os("CLOTH_OFF").is_some();

    // 2. Attach Cloth to AvatarInstance (legacy recipe only)
    if let Some(cloth_asset) = &cloth_asset_opt {
        let overlay_id = ClothOverlayId(1);
        let overlay_slot_idx = avatar.attach_cloth_overlay(overlay_id);
        avatar.init_cloth_overlay(overlay_slot_idx, cloth_asset);

        // Also init primary cloth for direct snapshot feed
        avatar.init_cloth_sim(cloth_asset);
    }
    avatar.cloth_enabled = !cloth_off && (cloth_asset_opt.is_some() || auto_cloth);
    if cloth_off {
        avatar.cloth_enabled = false;
        avatar.cloth_state = None;
        avatar.cloth_overlays.clear();
        println!("CLOTH_OFF: skirt renders as the static FBX skinned mesh");
    }

    // COLLIDERS_OFF=1: keep every body collider disabled — isolates
    // collider geometry as the cloth-push source.
    let colliders_off = std::env::var_os("COLLIDERS_OFF").is_some();
    // COLLIDERS_ALL=1: mirror the app path (`collider_enabled` starts
    // as vec![true]) — arms/spine/chest/head capsules included. The
    // default filter below only enables the lower-body set, so every
    // offline A/B up to now was blind to torso/arm collider pushes the
    // live app does apply. Subtract families for attribution:
    // COLLIDERS_NO_TORSO=1 (spine+chest), COLLIDERS_NO_ARMS=1.
    let colliders_all = std::env::var_os("COLLIDERS_ALL").is_some();
    let no_torso = std::env::var_os("COLLIDERS_NO_TORSO").is_some();
    let no_arms = std::env::var_os("COLLIDERS_NO_ARMS").is_some();
    // CLOTH_CHURN_CSV=<path>: per-capture-frame max/p95 particle
    // displacement (mm) since the previous capture — the offline
    // equivalent of debug_gui.json `cloth.settle.max_delta_mm`, for
    // attributing the continuous waist churn the live app shows
    // (bursts 20-80 mm/step, never quiet). Needs CLOTH_RENDER_EVERY=1
    // so every frame gets a readback fold.
    let churn_csv = std::env::var("CLOTH_CHURN_CSV").ok();
    let mut churn_prev: Vec<[f32; 3]> = Vec::new();
    // The auto path's SDF contact mask (upper-body humanoid capsules
    // off, thighs/hips kept) mirrors the app, where nothing re-enables
    // them afterwards — don't defeat it here either.
    let sdf_masked = avatar
        .cloth_overlays
        .iter()
        .any(|s| s.sim.sdf_contact > 0.0);
    // Enable colliders for thighs (UpperLeg_L and UpperLeg_R)
    for (i, col) in avatar.asset.colliders.iter().enumerate() {
        if sdf_masked {
            // SDF contact owns the upper-body mask; keep only the
            // lower-body set live, like the app.
            // COLLIDERS_NONE=1: leave every collider disabled (the SDF
            // contact field still applies) — isolates the capsule set
            // from pins/integration in the churn A/B.
            if std::env::var_os("COLLIDERS_NONE").is_some() {
                continue;
            }
            let node_name = avatar
                .asset
                .skeleton
                .nodes
                .get(col.node.0 as usize)
                .map(|n| n.name.to_lowercase())
                .unwrap_or_default();
            if node_name.contains("leg")
                || node_name.contains("thigh")
                || node_name.contains("hips")
            {
                // COLLIDERS_NO_THIGH=1: isolate the thigh-capsule
                // contribution to the waist churn (the VRC thighs measure
                // ~74 mm vs ~53 mm on the body).
                let no_thigh = std::env::var_os("COLLIDERS_NO_THIGH").is_some();
                if no_thigh
                    && (node_name.contains("thigh")
                        || node_name.contains("upperleg")
                        || (node_name.contains("leg") && !node_name.contains("hips")))
                {
                    continue;
                }
                avatar.collider_enabled[i] = true;
                println!(
                    "Enabled collider #{}: {} on node '{}' (SDF mask keeps lower body)",
                    i, col.id.0, node_name
                );
            }
            continue;
        }
        let node_name = avatar
            .asset
            .skeleton
            .nodes
            .get(col.node.0 as usize)
            .map(|n| n.name.to_lowercase())
            .unwrap_or_default();
        let lower_body = node_name.contains("upperleg")
            || node_name.contains("leg")
            || node_name.contains("thigh")
            || node_name.contains("hips");
        let torso = node_name.contains("spine") || node_name.contains("chest");
        let arm = node_name.contains("arm");
        let enable = if colliders_all {
            !(no_torso && torso) && !(no_arms && arm)
        } else {
            !colliders_off && lower_body
        };
        if enable {
            avatar.collider_enabled[i] = true;
            println!(
                "Enabled collider #{}: {} on node '{}'",
                i, col.id.0, node_name
            );
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
    if auto_cloth {
        // The auto path keeps its own solver params; just still the wind
        // so the A/B isolates pose + anchors.
        for slot in avatar.cloth_overlays.iter_mut() {
            slot.sim.wind_response = 0.0;
            slot.sim.wind_direction = [0.0, 0.0, 0.0];
        }
        if let Some(ref mut sim) = avatar.cloth_sim {
            sim.wind_response = 0.0;
            sim.wind_direction = [0.0, 0.0, 0.0];
        }
    }

    for (i, c) in vulvatar_lib::simulation::cloth::resolve_colliders(
        &avatar.asset.colliders,
        &avatar.pose.global_transforms,
        &avatar.collider_enabled,
    )
    .iter()
    .enumerate()
    {
        match c {
            vulvatar_lib::simulation::cloth::ResolvedCollider::Sphere { center, radius } => println!(
                "COLLIDER #{i} sphere center [{:.3},{:.3},{:.3}] r {:.3}",
                center[0], center[1], center[2], radius
            ),
            vulvatar_lib::simulation::cloth::ResolvedCollider::Capsule {
                center,
                radius,
                half_height,
                axis,
            } => println!(
                "COLLIDER #{i} capsule center [{:.3},{:.3},{:.3}] r {:.3} hh {:.3} axis [{:.2},{:.2},{:.2}]",
                center[0], center[1], center[2], radius, half_height, axis[0], axis[1], axis[2]
            ),
        }
    }

    // 3. Initialize Vulkan renderer
    let render_width = 1024u32;
    let render_height = 1024u32;
    println!(
        "Initializing Vulkan renderer ({}x{})...",
        render_width, render_height
    );
    let mut renderer = VulkanRenderer::new();
    renderer.initialize();

    println!(
        "Starting simulation and offline render across {} frames...",
        total_frames
    );
    let dt = 1.0 / 60.0;
    let mut recorded_snapshots = Vec::new();

    // Keyframes to save images for. `CLOTH_RENDER_EVERY=1` renders
    // EVERY frame — the GPU cloth solver only advances during render
    // (the CPU solver early-returns on the Gpu backend), so with the
    // default sparse capture the GPU state lags the skeleton by
    // multiple frames and the pinned band snaps ahead of an un-simulated
    // body, crumpling the surface. Per-frame render is the app's real
    // driving pattern and the only one that shows the GPU cloth's
    // converged shape.
    let render_every = std::env::var_os("CLOTH_RENDER_EVERY").is_some();
    let capture_frames: HashSet<usize> = if render_every {
        (0..=total_frames).collect()
    } else {
        [0, 10, 20, 30, 45, 60, 75, 90]
            .iter()
            .copied()
            .filter(|&f| f <= total_frames)
            .collect()
    };

    // Record initial hips transform
    let hips_node_idx = avatar
        .asset
        .skeleton
        .nodes
        .iter()
        .enumerate()
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
            cur_t.rotation =
                vulvatar_lib::math_utils::quat_mul(&initial_hips_transform.rotation, &q_roll);

            avatar.pose.local_transforms[hips_node_idx] = cur_t;
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();

            // SKIRT_POSE override: rebuild the locals from REST + the
            // desk-envelope ops (ramped over frames 1..30), replacing
            // the idle sway. Signs calibrated on this rig (probe in
            // diagnose_skirt_fit): spine −X = lean back, thighs −X =
            // thigh raise, knees +X = knee bend.
            if !skirt_pose.is_empty() {
                let ramp = (frame as f32 / 30.0).min(1.0);
                let mut locals = rest_locals.clone();
                apply_skirt_pose(
                    &asset.skeleton,
                    asset
                        .humanoid
                        .as_ref()
                        .expect("humanoid map")
                        .bone_map
                        .clone(),
                    &mut locals,
                    &skirt_pose,
                    ramp,
                );
                avatar.pose.local_transforms = locals;
                avatar.compute_global_pose();
                avatar.build_skinning_matrices();
            }

            // RIDE-UP diagnosis knob: skip spring stepping entirely to
            // test whether the skirt/hem deformation is spring-driven.
            // (The knob must wrap the SPRING step itself — an earlier
            // placement only wrapped wind + cloth and left springs
            // running, invalidating the first A/B.)
            let springs_off = std::env::var_os("SPRINGS_OFF").is_some();
            if !springs_off {
            // Step secondary motion (hair, accessories) with newly attached body colliders
            let spring_tuning = vulvatar_lib::simulation::spring::SpringTuning::default();
            vulvatar_lib::simulation::spring::step_spring_bones(
                dt,
                &mut avatar,
                &[],
                &spring_tuning,
                [0.0, -1.0, 0.0],
                1.0,
                None,
            );
            avatar.compute_global_pose();
            avatar.build_skinning_matrices();

            // Apply gentle fluctuating breeze. `WIND_SCALE` scales the
            // response for the cloth-stability experiment (0 = no wind).
            if let Some(ref mut sim) = avatar.cloth_sim {
                let gust = (t * 2.2).sin() * 0.4 + 0.6;
                let wind_scale = std::env::var("WIND_SCALE")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(1.0);
                let wind = [0.35 * gust, (t * 1.8).cos() * 0.05, 0.2 * gust];
                sim.wind_direction = [
                    wind[0] * wind_scale,
                    wind[1] * wind_scale,
                    wind[2] * wind_scale,
                ];
                sim.wind_response = 0.8 * wind_scale;
            }

            // Force-enable self-collision for the GPU smoke when asked —
            // the skirt asset ships with it off (matching the CPU default).
            if std::env::var_os("VULVATAR_CLOTH_SELFCOL").is_some() {
                if let Some(ref mut sim) = avatar.cloth_sim {
                    sim.self_collision = true;
                }
                for slot in avatar.cloth_overlays.iter_mut() {
                    slot.sim.self_collision = true;
                }
                // `CLOTH_SELFCOL_RADIUS` overrides the exclusion radius
                // for the R6 radius-sensitivity experiment. The default
                // 1 cm gives min_dist = 2 cm — LARGER than the pleat
                // spacing of this skirt, which makes self-collision blast
                // the designed folds apart (both solvers).
                if let Some(r) = std::env::var("CLOTH_SELFCOL_RADIUS")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                {
                    if let Some(ref mut sim) = avatar.cloth_sim {
                        sim.self_collision_radius = r;
                    }
                    for slot in avatar.cloth_overlays.iter_mut() {
                        slot.sim.self_collision_radius = r;
                    }
                    println!("Self-collision radius override: {} m", r);
                }
            }
            vulvatar_lib::simulation::cloth_solver::step_cloth(dt, &mut avatar, &[], None);
            } // !springs_off
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
                render_width,
                render_height,
                frame,
                dt,
            );

            // Double render invocation for pipelined CPU readback
            let _ = renderer
                .render(&frame_input)
                .map_err(|e| format!("Rendering warm-up failed at frame {}: {}", frame, e))?;
            let render_result = renderer
                .render(&frame_input)
                .map_err(|e| format!("Rendering failed at frame {}: {}", frame, e))?;

            // Fold the GPU readback into the overlay's ClothState so the
            // per-frame health check reads the LIVE solver state (GPU
            // backend: cs.sim_positions would otherwise stay frozen at
            // the attach-time rest pose).
            for entry in &render_result.cloth_readback {
                for slot in avatar.cloth_overlays.iter_mut() {
                    if slot.state.target_primitive_id == Some(entry.primitive_id) {
                        slot.state.sim_positions = entry.positions.clone();
                        slot.state.deform_output.deformed_positions = entry.positions.clone();
                        slot.state.deform_output.version = entry.version as u64;
                    }
                }
            }

            if let Some(path) = &churn_csv {
                for slot in avatar.cloth_overlays.iter() {
                    let pos = &slot.state.sim_positions;
                    if pos.is_empty() {
                        continue;
                    }
                    let mut deltas: Vec<f32> = if churn_prev.len() == pos.len() {
                        pos.iter()
                            .zip(churn_prev.iter())
                            .map(|(q, p)| {
                                ((q[0] - p[0]).powi(2)
                                    + (q[1] - p[1]).powi(2)
                                    + (q[2] - p[2]).powi(2))
                                .sqrt()
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let max_mm = deltas.last().map(|d| d * 1000.0).unwrap_or(0.0);
                    let p95_mm = deltas
                        .get((deltas.len() as f32 * 0.95) as usize)
                        .map(|d| d * 1000.0)
                        .unwrap_or(0.0);
                    if let Ok(mut f) =
                        std::fs::OpenOptions::new().create(true).append(true).open(path)
                    {
                        use std::io::Write as _;
                        let _ = writeln!(f, "{frame},{max_mm:.3},{p95_mm:.3}");
                    }
                    churn_prev = pos.clone();
                }
            }

            if frame == 0 {
                audit_sweater_bones(&avatar, PrimitiveId(5));
                audit_sweater_bones(&avatar, PrimitiveId(12));
            }

            // CPU anchor evaluation: locate WHICH clearance anchors
            // drive the large corrections the audit measures, and
            // compare base-pose vs generation-pose deficits.
            if std::env::var_os("VULVATAR_VBO_AUDIT").is_some() && frame % 10 == 0 {
                let gen_skinning = generation_pose_skinning(&asset);
                for pid in [PrimitiveId(5), PrimitiveId(12)] {
                    audit_clearance_anchors_with(
                        &avatar.pose.skinning_matrices,
                        "base pose",
                        &avatar,
                        pid,
                    );
                    audit_clearance_anchors_with(&gen_skinning, "generation pose", &avatar, pid);
                }
            }

            // R2 final-VBO audit (rows only when `VULVATAR_VBO_AUDIT=1`):
            // print the render-side correction telemetry so the smoke
            // doubles as the E-stage vertex-statistics probe.
            if !render_result.vbo_audit.is_empty() {
                for e in &render_result.vbo_audit {
                    println!(
                        "  VBO_AUDIT prim {} ({} verts): corr max {:.1} mm / p95 {:.1} mm, |pos| max {:.3} m, NaN {}",
                        e.primitive_id.0,
                        e.vertex_count,
                        e.max_correction_m * 1000.0,
                        e.p95_correction_m * 1000.0,
                        e.max_pos_len_m,
                        e.nan_count,
                    );
                }
            }

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
    report.push_str(&format!(
        "- Target Skirt Mesh: Circle.056 (PrimitiveId({}))\n",
        skirt_prim_id.0
    ));
    report.push_str(&format!(
        "- Target Skirt Mesh: Circle.056 (PrimitiveId({})){}\n",
        skirt_prim_id.0,
        if auto_cloth { " (auto-cloth app path)" } else { "" }
    ));
    if let Some(c) = cloth_asset_opt.as_ref() {
        report.push_str(&format!("- Vertices: {}\n", c.simulation_mesh.vertices.len()));
        report.push_str(&format!(
            "- Distance Constraints: {}\n",
            c.constraints.distance_constraints.len()
        ));
        report.push_str(&format!("- Pinned Particles: {}\n", c.pins.len()));
    } else {
        report.push_str("- Recipe: auto-cloth (params from `auto_cloth.rs`)\n");
    }
    report.push_str(&format!(
        "- Total Simulated Frames: {} ({} s at 60 FPS)\n\n",
        total_frames,
        total_frames as f32 / 60.0
    ));
    report.push_str("## Captured Keyframes\n\n");
    for (f, p) in &recorded_snapshots {
        report.push_str(&format!("### Frame {:03}\n\n", f));
        report.push_str(&format!(
            "![Frame {:03}]({})\n\n",
            f,
            p.file_name().unwrap().to_string_lossy()
        ));
    }
    std::fs::write(&report_path, report).map_err(|e| format!("Failed to write report: {}", e))?;
    println!("\nWrote verification summary to {}", report_path.display());

    println!("\n=== Cloth simulation diagnostic completed successfully! ===");
    Ok(())
}

/// CPU mirror of the shader's clearance branch (transform_cs binding 6):
/// for every anchor on `prim_id`, evaluate the clearance against the
/// parent primitive's CPU-LBS surface and report the worst offenders.
/// Locates WHICH anchors / parent vertices drive the ~180 mm pushes the
/// VBO audit measures.
/// RIDE-UP diagnosis: which bones drive the sweater mesh, and are they
/// spring-driven? Prints the dominant bones for the whole primitive and
/// for its lowest-y band (the hem) with spring-bone membership marks.
fn audit_sweater_bones(avatar: &AvatarInstance, prim_id: PrimitiveId) {
    use std::collections::HashMap;
    let asset = &avatar.asset;
    let Some(prim) = asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .find(|p| p.id == prim_id)
    else {
        return;
    };
    let Some(vd) = prim.vertices.as_ref() else {
        return;
    };
    // Spring chains list their joints per chain; every joint node is a
    // spring-driven node.
    let spring_nodes: std::collections::HashSet<usize> = asset
        .spring_bones
        .iter()
        .flat_map(|sb| sb.joints.iter().map(|j| j.0 as usize))
        .collect();
    let node_name = |i: usize| -> String {
        asset
            .skeleton
            .nodes
            .get(i)
            .map(|n| n.name.clone())
            .unwrap_or_else(|| format!("node{i}"))
    };

    // Full-mesh bone histogram + hem-band (lowest 15% by y) histogram.
    let mut ys: Vec<(f32, usize)> = vd
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| (p[1], i))
        .collect();
    ys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let hem_count = (vd.positions.len() as f32 * 0.15) as usize;
    let hem_set: std::collections::HashSet<usize> =
        ys.iter().take(hem_count).map(|(_, i)| *i).collect();

    let mut full: HashMap<usize, f32> = HashMap::new();
    let mut hem: HashMap<usize, f32> = HashMap::new();
    for (i, indices) in vd.joint_indices.iter().enumerate() {
        for (slot, &ji) in indices.iter().enumerate() {
            let w = vd.joint_weights.get(i).map(|w| w[slot]).unwrap_or(0.0);
            if w > 0.0001 && (ji as usize) < asset.skeleton.nodes.len() {
                *full.entry(ji as usize).or_default() += w;
                if hem_set.contains(&i) {
                    *hem.entry(ji as usize).or_default() += w;
                }
            }
        }
    }
    let dump = |m: &HashMap<usize, f32>, label: &str| {
        let mut v: Vec<(usize, f32)> = m.iter().map(|(k, w)| (*k, *w)).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("  BONES {label} prim {}:", prim_id.0);
        for (node, w) in v.iter().take(8) {
            println!(
                "    {:>6.2} {} {}",
                w,
                if spring_nodes.contains(node) { "[spring]" } else { "        " },
                node_name(*node)
            );
        }
    };
    dump(&full, "all");
    dump(&hem, "hem");
}

fn audit_clearance_anchors_with(
    skinning: &[vulvatar_lib::asset::Mat4],
    label: &str,
    avatar: &AvatarInstance,
    prim_id: PrimitiveId,
) {
    let asset = &avatar.asset;
    let Some(prim) = asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .find(|p| p.id == prim_id)
    else {
        return;
    };
    let Some(ref anchors) = prim.skin_anchors else {
        return;
    };
    let Some(parent_id) = prim.body_primitive_id else {
        return;
    };
    let Some(parent) = asset
        .meshes
        .iter()
        .flat_map(|m| m.primitives.iter())
        .find(|p| p.id == parent_id)
    else {
        return;
    };
    let lbs_positions = |vd: &vulvatar_lib::asset::VertexData| -> Vec<[f32; 3]> {
        vd.positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| {
                let mut w = [0.0f32; 3];
                let mut total = 0.0;
                if i < vd.joint_weights.len() && i < vd.joint_indices.len() {
                    for k in 0..4 {
                        let wt = vd.joint_weights[i][k];
                        if wt > 0.0001 {
                            let j = vd.joint_indices[i][k] as usize;
                            if let Some(m) = skinning.get(j) {
                                for c in 0..3 {
                                    w[c] += wt
                                        * (m[0][c] * pos[0]
                                            + m[1][c] * pos[1]
                                            + m[2][c] * pos[2]
                                            + m[3][c]);
                                }
                                total += wt;
                            }
                        }
                    }
                }
                if total > 0.001 {
                    for c in 0..3 {
                        w[c] /= total;
                    }
                }
                w
            })
            .collect()
    };
    let Some(parent_vd) = parent.vertices.as_ref() else {
        return;
    };
    let parent_lbs = lbs_positions(parent_vd);
    let child_lbs = prim.vertices.as_ref().map(|vd| lbs_positions(vd));
    let mut worst = 0.0f32;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    let mut worst_row: Option<(f32, u32, [f32; 3], f32, [f32; 3], [f32; 3], [f32; 3])> = None;
    // (deficit, vert, parent_pos, min_clearance, child_pos, parent_normal_raw, bn)
    for (vi, anc) in anchors.iter().enumerate() {
        if anc.body_vertex_idx == 0xFFFFFFFF || anc.weight <= 1e-4 {
            continue;
        }
        let Some(bp) = parent_lbs.get(anc.body_vertex_idx as usize) else {
            continue;
        };
        let Some(child_pos) = child_lbs.as_ref().and_then(|l| l.get(vi)).copied() else {
            continue;
        };
        let bn_raw = parent_vd
            .normals
            .get(anc.body_vertex_idx as usize)
            .copied()
            .unwrap_or([0.0, 1.0, 0.0]);
        let nlen = (bn_raw[0] * bn_raw[0] + bn_raw[1] * bn_raw[1] + bn_raw[2] * bn_raw[2])
            .sqrt();
        if nlen < 1e-4 {
            continue;
        }
        let bn = [bn_raw[0] / nlen, bn_raw[1] / nlen, bn_raw[2] / nlen];
        let clearance = (child_pos[0] - bp[0]) * bn[0]
            + (child_pos[1] - bp[1]) * bn[1]
            + (child_pos[2] - bp[2]) * bn[2];
        let deficit = (anc.min_clearance - clearance).max(0.0);
        sum += deficit;
        count += 1;
        let _ = bn_raw;
        if deficit > worst {
            worst = deficit;
            worst_row = Some((
                deficit,
                anc.body_vertex_idx,
                *bp,
                anc.min_clearance,
                child_pos,
                bn_raw,
                bn,
            ));
        }
    }
    let avg = if count > 0 { sum / count as f32 } else { 0.0 };
    match worst_row {
        Some((deficit, bvi, bp, mc, cp, bn_raw, bn)) => println!(
            "    [{}] prim {}: anchors {} worst {:.1} mm (vert {} parent at [{:.2},{:.2},{:.2}] child at [{:.2},{:.2},{:.2}] bn_raw [{:.2},{:.2},{:.2}] bn [{:.2},{:.2},{:.2}]) min_clearance {:.1} mm -> implied clearance {:.1} mm, avg {:.2} mm",
            label,
            prim_id.0,
            count,
            deficit * 1000.0,
            bvi,
            bp[0],
            bp[1],
            bp[2],
            cp[0],
            cp[1],
            cp[2],
            bn_raw[0],
            bn_raw[1],
            bn_raw[2],
            bn[0],
            bn[1],
            bn[2],
            mc * 1000.0,
            (mc - deficit) * 1000.0,
            avg * 1000.0
        ),
        None => println!("    [{}] prim {}: no active anchors", label, prim_id.0),
    }
}

/// Evaluate anchors under the GENERATION pose (rest_local globals) —
/// anchors built there must show ~zero deficit there. A large deficit
/// under the app's BASE pose but not under the generation pose proves
/// the anchor/garment mismatch is a pose-contract violation, not a
/// clearance-value bug.
fn generation_pose_skinning(asset: &vulvatar_lib::asset::AvatarAsset) -> Vec<vulvatar_lib::asset::Mat4> {
    let node_count = asset.skeleton.nodes.len();
    let locals: Vec<_> = asset
        .skeleton
        .nodes
        .iter()
        .map(|n| n.rest_local.clone())
        .collect();
    let mut globals = vec![vulvatar_lib::asset::identity_matrix(); node_count];
    vulvatar_lib::avatar::pose::compute_global_transforms(&asset.skeleton, &locals, &mut globals);
    let mut skinning = vec![vulvatar_lib::asset::identity_matrix(); node_count];
    vulvatar_lib::avatar::pose::build_skinning_matrices(&asset.skeleton, &globals, &mut skinning);
    skinning
}

fn build_skirt_cloth_asset(
    avatar: &AvatarInstance,
    pin_y: f32,
    pin_band_depth: f32,
    cloth_damping: f32,
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
    let (prim_id, mesh_id, mesh_name, vert_count) = skirt_prim
        .or_else(|| {
            for mesh in &asset.meshes {
                for prim in &mesh.primitives {
                    if let Some(ref vd) = prim.vertices {
                        let b_min_y = prim.bounds.min[1];
                        let b_max_y = prim.bounds.max[1];
                        if b_min_y > 0.55
                            && b_max_y < 0.85
                            && vd.positions.len() >= 2000
                            && vd.positions.len() <= 3000
                        {
                            return Some((prim.id, mesh.id, mesh.name.clone(), vd.positions.len()));
                        }
                    }
                }
            }
            None
        })
        .ok_or_else(|| "Could not locate Circle.056 skirt mesh in avatar".to_string())?;

    println!(
        "Selected skirt primitive: {:?} on mesh '{}' ({} vertices)",
        prim_id, mesh_name, vert_count
    );

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
    println!(
        "Binding pins to bone: '{}' (node idx {})",
        pin_node.name, pin_node_idx
    );

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
                        let tx =
                            sm[0][0] * pos[0] + sm[1][0] * pos[1] + sm[2][0] * pos[2] + sm[3][0];
                        let ty =
                            sm[0][1] * pos[0] + sm[1][1] * pos[1] + sm[2][1] * pos[2] + sm[3][1];
                        let tz =
                            sm[0][2] * pos[0] + sm[1][2] * pos[1] + sm[2][2] * pos[2] + sm[3][2];
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

        // Pin band: waistband threshold plus a deeper pinned band
        // (`pin_band_depth` below the threshold) — a longer pinned
        // region shortens the free cloth so hip sway translates the
        // skirt instead of letting the zero-bending-stiffness cloth
        // buckle at the sides.
        let is_pin = world_pos[1] >= pin_y - pin_band_depth;
        let normal = vd.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
        let uv = vd.uvs.get(i).copied().unwrap_or([0.0, 0.0]);

        if is_pin {
            let d = [
                world_pos[0] - t[0],
                world_pos[1] - t[1],
                world_pos[2] - t[2],
            ];
            let ox = r0[0] * d[0] + r0[1] * d[1] + r0[2] * d[2];
            let oy = r1[0] * d[0] + r1[1] * d[1] + r1[2] * d[2];
            let oz = r2[0] * d[0] + r2[1] * d[1] + r2[2] * d[2];

            cloth_pins.push(ClothPin {
                sim_vertex_indices: vec![i as u32],
                binding_node: NodeRef {
                    id: pin_node.id,
                    name: pin_node.name.clone(),
                    humanoid_bone: None,
                    parent_path: None,
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

    // Build unique distance constraints from WELDED positions. The
    // raw-index edge set gave the flat-shaded Circle.056 (2,460 verts /
    // 820 disjoint triangles) degree-2 particles and the cloth
    // free-fell as confetti on BOTH solvers — the same documented
    // limitation that drove the weld in `simulation::auto_cloth`.
    // Rendering keeps RAW indices; only the constraint topology is
    // welded, and each welded edge becomes one round-robin constraint
    // per duplicate pair (a full cross product overshoots the Jacobi
    // XPBD sum into an explosion on the first step).
    let weld = {
        let positions: Vec<[f32; 3]> =
            sim_vertices.iter().map(|v| v.position).collect();
        vulvatar_lib::simulation::auto_cloth::weld_by_quantized_position(&positions)
    };
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

        let (w0, w1, w2) = (
            weld.weld_of[i0 as usize],
            weld.weld_of[i1 as usize],
            weld.weld_of[i2 as usize],
        );
        for &(a, b) in &[(w0, w1), (w1, w2), (w2, w0)] {
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_set.insert(edge);
        }
    }

    let (mut distance_constraints, mut rest_lengths) =
        vulvatar_lib::simulation::auto_cloth::round_robin_weld_constraints(&weld, &edge_set, 0.98);
    // R6: hold weld-group copies coincident (see
    // `intra_weld_group_constraints`).
    let (intra, intra_rest) =
        vulvatar_lib::simulation::auto_cloth::intra_weld_group_constraints(&weld, 0.98);
    distance_constraints.extend(intra);
    rest_lengths.extend(intra_rest);
    {
        // Self-collision radius sanity basis: the exclusion distance
        // (2×radius) must stay BELOW the mesh's own in-plane spacing
        // (edge lengths), or designed folds read as penetrations.
        let mut sorted = rest_lengths.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let (min, median, max, avg) = (
            sorted.first().copied().unwrap_or(0.0),
            sorted[sorted.len() / 2],
            sorted.last().copied().unwrap_or(0.0),
            sorted.iter().sum::<f32>() / sorted.len().max(1) as f32,
        );
        println!(
            "Edge lengths: min {:.1} mm / median {:.1} mm / mean {:.1} mm / max {:.1} mm  (selfcol radius must keep 2·radius < median)",
            min * 1000.0,
            median * 1000.0,
            avg * 1000.0,
            max * 1000.0,
        );
    }
    println!(
        "Welded cloth topology: {} particles -> {} weld groups, {} constraints",
        sim_vertices.len(),
        weld.group_pos.len(),
        distance_constraints.len()
    );

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
                humanoid_bone: None,
                parent_path: None,
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
            damping: cloth_damping,
            self_collision: false,
            collision_margin: 0.015,
            wind_response: 0.8,
        },
    };

    Ok((cloth_asset, prim_id))
}

/// Per-particle pin world targets — diagnostic twin of the app-side
/// `gpu_pin_targets` (same math as the CPU solver's
/// `cloth_solver::collision::apply_pin_targets`), so the GPU backend
/// smoke can drive pins from the bin without reaching into private
/// app internals.
/// World-space avatar collision capsules for the GPU stage —
/// diagnostic twin of the app-side `gpu_colliders_for`.
fn gpu_colliders(avatar: &AvatarInstance) -> Vec<vulvatar_lib::renderer::frame_input::ClothGpuCollider> {
    vulvatar_lib::simulation::cloth::resolve_colliders(
        &avatar.asset.colliders,
        &avatar.pose.global_transforms,
        &avatar.collider_enabled,
    )
    .into_iter()
    .map(|c| {
        let (p0, p1, radius) = match c {
            vulvatar_lib::simulation::cloth::ResolvedCollider::Sphere { center, radius } => {
                (center, center, radius)
            }
            vulvatar_lib::simulation::cloth::ResolvedCollider::Capsule {
                center,
                radius,
                half_height,
                axis,
            } => {
                let a = vulvatar_lib::math_utils::vec3_sub(
                    &center,
                    &vulvatar_lib::math_utils::vec3_scale(&axis, half_height),
                );
                let b = vulvatar_lib::math_utils::vec3_add(
                    &center,
                    &vulvatar_lib::math_utils::vec3_scale(&axis, half_height),
                );
                (a, b, radius)
            }
        };
        vulvatar_lib::renderer::frame_input::ClothGpuCollider { p0, p1, radius }
    })
    .collect()
}

fn gpu_pin_targets(
    sim: &vulvatar_lib::simulation::cloth::ClothSimState,
    global_transforms: &[vulvatar_lib::asset::Mat4],
) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; sim.particles.len()];
    for pin in &sim.pin_targets {
        let Some(mat) = global_transforms.get(pin.node_index) else {
            continue;
        };
        let [ox, oy, oz] = pin.offset;
        let world = [
            mat[0][0] * ox + mat[1][0] * oy + mat[2][0] * oz + mat[3][0],
            mat[0][1] * ox + mat[1][1] * oy + mat[2][1] * oz + mat[3][1],
            mat[0][2] * ox + mat[1][2] * oy + mat[2][2] * oz + mat[3][2],
        ];
        for &pi in &pin.particle_indices {
            if pi < out.len() {
                out[pi] = world;
            }
        }
    }
    out
}

fn build_render_frame_input(
    avatar: &AvatarInstance,
    width: u32,
    height: u32,
    frame_idx: usize,
    dt: f32,
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

    // Collect cloth deforms from the primary ClothState and the
    // overlay slots (first-wins per primitive, matching the app's
    // `collect_cloth_deforms`). This bin attaches via
    // `attach_cloth_overlay`, so the overlay path is the live one —
    // until this collected both, the render never received any cloth
    // snapshot at all. The snapshot honors each ClothState's backend:
    // with `VULVATAR_CLOTH_GPU=1` the CPU solver early-returns
    // (frozen `deform_output`) and the renderer dispatches the cloth
    // compute pipelines — the rendered PNGs then show the GPU result,
    // which is exactly what this bin exists to verify.
    let mut cloth_deforms = Vec::new();
    let mut seen_targets: HashSet<vulvatar_lib::asset::PrimitiveId> = HashSet::new();
    let primary_iter = avatar
        .cloth_state
        .as_ref()
        .map(|cs| (cs, avatar.cloth_sim.as_ref()))
        .into_iter();
    let overlay_iter = avatar
        .cloth_overlays
        .iter()
        .filter(|s| s.enabled)
        .map(|s| (&s.state, Some(&s.sim)));
    for (cs, sim_opt) in primary_iter.chain(overlay_iter) {
        let Some(target_primitive_id) = cs.target_primitive_id else {
            continue;
        };
        if !seen_targets.insert(target_primitive_id) {
            continue;
        }
        let (gpu_control, gpu_attach) = if cs.solver_backend == ClothSolverBackend::Gpu {
            sim_opt
                .map(|sim| {
                    use vulvatar_lib::renderer::frame_input::{
                        ClothGpuAttachData, ClothGpuDispatchControl,
                    };
                    let wind = vulvatar_lib::math_utils::vec3_scale(
                        &sim.wind_direction,
                        sim.wind_response,
                    );
                    (
                        Some(ClothGpuDispatchControl {
                            dt,
                            // `CLOTH_SUBSTEPS` overrides the per-frame
                            // GPU substep count for the R1 frequency
                            // experiment (1/2/4 comparisons). Must match
                            // what the CPU loop would run for parity.
                            substeps: std::env::var("CLOTH_SUBSTEPS")
                                .ok()
                                .and_then(|v| v.parse::<u32>().ok())
                                .unwrap_or(1),
                            damping: sim.damping,
                            gravity: sim.gravity,
                            wind_force: wind,
                            solver_iterations: sim.solver_iterations as u32,
                            pin_positions: gpu_pin_targets(
                                sim,
                                &avatar.pose.global_transforms,
                            ),
                            collision_margin: sim.collision_margin,
                            colliders: gpu_colliders(avatar),
                            self_collision: sim.self_collision,
                            self_collision_radius: sim.self_collision_radius,
                            sdf_contact: sim.sdf_contact,
                        }),
                        Some({
                            // Bend-wing CSR (same recipe as the app's
                            // collect_cloth_deforms).
                            let mut bend_adj_offsets =
                                vec![0u32; sim.particles.len() + 1];
                            for bc in &sim.bend_constraints {
                                for wing in [bc.p1, bc.p2] {
                                    if wing < sim.particles.len() {
                                        bend_adj_offsets[wing + 1] += 1;
                                    }
                                }
                            }
                            for v in 1..bend_adj_offsets.len() {
                                bend_adj_offsets[v] += bend_adj_offsets[v - 1];
                            }
                            let mut cursor = bend_adj_offsets.clone();
                            let mut bend_adj_constraints: Vec<u32> =
                                vec![0; sim.bend_constraints.len() * 2];
                            for (ci, bc) in sim.bend_constraints.iter().enumerate() {
                                for wing in [bc.p1, bc.p2] {
                                    if wing < sim.particles.len() {
                                        bend_adj_constraints[cursor[wing] as usize] =
                                            ci as u32;
                                        cursor[wing] += 1;
                                    }
                                }
                            }
                            ClothGpuAttachData {
                                constraints: sim
                                    .distance_constraints
                                    .iter()
                                    .map(|c| {
                                        (c.a as u32, c.b as u32, c.rest_length, c.stiffness)
                                    })
                                    .collect(),
                                triangle_indices: sim.triangle_indices.clone(),
                                inv_masses: sim
                                    .particles
                                    .iter()
                                    .map(|p| p.inv_mass)
                                    .collect(),
                                pinned: sim.particles.iter().map(|p| p.pinned).collect(),
                                bend: sim
                                    .bend_constraints
                                    .iter()
                                    .map(|bc| {
                                        vulvatar_lib::renderer::pipeline::ClothBendGpu {
                                            p0: bc.p0 as u32,
                                            p1: bc.p1 as u32,
                                            p2: bc.p2 as u32,
                                            _pad: 0,
                                            rest_angle: bc.rest_angle,
                                            stiffness: bc.stiffness,
                                            _pad2: [0; 2],
                                        }
                                    })
                                    .collect(),
                                bend_adj_offsets,
                                bend_adj_constraints,
                            }
                        }),
                    )
                })
                .unwrap_or((None, None))
        } else {
            (None, None)
        };
        cloth_deforms.push(ClothDeformSnapshot {
            target_primitive_id,
            target_mesh_id: cs.target_mesh_id,
            vertex_offset: cs.target_vertex_offset,
            vertex_count: cs.target_vertex_count,
            deformed_positions: cs.deform_output.deformed_positions.clone(),
            deformed_normals: cs.deform_output.deformed_normals.clone(),
            version: cs.deform_output.version,
            solver_backend: cs.solver_backend,
            gpu_control,
            gpu_attach,
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

    // Body-SDF splat plan (body prim only): gives the cloth collide
    // kernel its smooth body-contact field offline. `SDF_ON=1` runs are
    // the whole point of the skirt-constraint A/B — without a slot the
    // GPU stage binds a dummy and stays inert.
    let body_sdf = avatar
        .asset
        .body_primitive_id
        .and_then(|pid| {
            avatar.asset.meshes.iter().find_map(|m| {
                m.primitives
                    .iter()
                    .find(|p| p.id == pid)
                    .map(|p| (m.id, p.id, p.bounds))
            })
        })
        .map(|(mesh_id, pid, bounds)| vulvatar_lib::renderer::frame_input::BodySdfPlan {
            prims: vec![(mesh_id, pid)],
            grid: vulvatar_lib::simulation::sdf::SdfGrid::for_aabb(&bounds),
        });

    let avatar_instance = RenderAvatarInstance {
        instance_id: avatar.id,
        world_transform: avatar.world_transform.clone(),
        mesh_instances,
        skinning_matrices: avatar.pose.skinning_matrices.clone(),
        cloth_deforms,
        body_sdf,
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
    let img: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::from_raw(extent[0], extent[1], pixels.to_vec())
            .ok_or_else(|| format!("PNG buffer construction failed for {}", path.display()))?;
    img.save(path)
        .map_err(|e| format!("failed to save '{}': {e}", path.display()))
}

// ---------------------------------------------------------------------------
// SKIRT_POSE desk-envelope driving (signs calibrated in diagnose_skirt_fit)
// ---------------------------------------------------------------------------

type Quat = [f32; 4];

fn pose_fk(
    skeleton: &vulvatar_lib::asset::SkeletonAsset,
    locals: &[vulvatar_lib::asset::Transform],
) -> Vec<(Quat, [f32; 3])> {
    let mut world: Vec<(Quat, [f32; 3])> =
        vec![([0.0, 0.0, 0.0, 1.0], [0.0; 3]); skeleton.nodes.len()];
    let mut stack: Vec<vulvatar_lib::asset::NodeId> = skeleton.root_nodes.clone();
    while let Some(idx) = stack.pop() {
        let i = idx.0 as usize;
        if i >= skeleton.nodes.len() || i >= locals.len() {
            continue;
        }
        let (p_rot, p_pos) = skeleton
            .nodes[i]
            .parent
            .map(|p| world[p.0 as usize])
            .unwrap_or(([0.0, 0.0, 0.0, 1.0], [0.0; 3]));
        let rot = vulvatar_lib::math_utils::quat_normalize(&vulvatar_lib::math_utils::quat_mul(
            &p_rot,
            &locals[i].rotation,
        ));
        let pos = vulvatar_lib::math_utils::vec3_add(
            &p_pos,
            &vulvatar_lib::math_utils::quat_rotate_vec3(&p_rot, &locals[i].translation),
        );
        world[i] = (rot, pos);
        stack.extend(skeleton.nodes[i].children.iter().copied());
    }
    world
}

fn pose_axis_angle(axis: &[f32; 3], deg: f32) -> Quat {
    let rad = deg.to_radians() * 0.5;
    let s = rad.sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, rad.cos()]
}

/// Pre-multiply `bone`'s world rotation by `delta_world` as a local-rotation
/// update (validate_gt recipe, incl. the Yumeka chest fallback).
fn pose_apply_world_delta(
    skeleton: &vulvatar_lib::asset::SkeletonAsset,
    humanoid: &std::collections::HashMap<
        vulvatar_lib::asset::HumanoidBone,
        vulvatar_lib::asset::NodeId,
    >,
    locals: &mut [vulvatar_lib::asset::Transform],
    bone: vulvatar_lib::asset::HumanoidBone,
    delta_world: &Quat,
) {
    use vulvatar_lib::asset::HumanoidBone;
    use vulvatar_lib::math_utils::{quat_conjugate, quat_mul, quat_normalize};
    let bone = match humanoid.get(&bone) {
        Some(_) => bone,
        None => {
            let candidates: &[HumanoidBone] = match bone {
                HumanoidBone::UpperChest => &[HumanoidBone::Chest, HumanoidBone::Spine],
                HumanoidBone::Chest => &[HumanoidBone::Spine],
                _ => &[],
            };
            match candidates.iter().copied().find(|b| humanoid.contains_key(b)) {
                Some(b) => b,
                None => return,
            }
        }
    };
    let Some(vulvatar_lib::asset::NodeId(idx)) = humanoid.get(&bone).copied() else {
        return;
    };
    let i = idx as usize;
    let world = pose_fk(skeleton, locals);
    let p_rot = skeleton
        .nodes[i]
        .parent
        .map(|p| world[p.0 as usize].0)
        .unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let local_delta = quat_mul(&quat_mul(&quat_conjugate(&p_rot), delta_world), &p_rot);
    locals[i].rotation = quat_normalize(&quat_mul(&local_delta, &locals[i].rotation));
}

/// `name`: `lean` (spine −15°), `sit` (thighs −70°, knees +80°),
/// `sit_lean` (both). `ramp` 0..1 scales all angles.
fn apply_skirt_pose(
    skeleton: &vulvatar_lib::asset::SkeletonAsset,
    humanoid: std::collections::HashMap<
        vulvatar_lib::asset::HumanoidBone,
        vulvatar_lib::asset::NodeId,
    >,
    locals: &mut [vulvatar_lib::asset::Transform],
    name: &str,
    ramp: f32,
) {
    use vulvatar_lib::asset::HumanoidBone;
    let mut ops: Vec<(HumanoidBone, [f32; 3], f32)> = Vec::new();
    if name.contains("desk") {
        // Production desk envelope: pelvis tips back, the spine
        // counter-rotates so the torso stays erect, thighs raise to the
        // seat. Exercises hips+spine+thighs together (the visible band
        // region), unlike the pure `sit`.
        ops.push((HumanoidBone::Hips, [1.0, 0.0, 0.0], -20.0));
        ops.push((HumanoidBone::Spine, [1.0, 0.0, 0.0], 20.0));
        for b in [HumanoidBone::LeftUpperLeg, HumanoidBone::RightUpperLeg] {
            ops.push((b, [1.0, 0.0, 0.0], -60.0));
        }
        for b in [HumanoidBone::LeftLowerLeg, HumanoidBone::RightLowerLeg] {
            ops.push((b, [1.0, 0.0, 0.0], 75.0));
        }
    }
    if name.contains("lean") && !name.contains("desk") {
        ops.push((HumanoidBone::Spine, [1.0, 0.0, 0.0], -15.0));
    }
    if name.contains("sit") && !name.contains("desk") {
        for b in [HumanoidBone::LeftUpperLeg, HumanoidBone::RightUpperLeg] {
            ops.push((b, [1.0, 0.0, 0.0], -70.0));
        }
        for b in [HumanoidBone::LeftLowerLeg, HumanoidBone::RightLowerLeg] {
            ops.push((b, [1.0, 0.0, 0.0], 80.0));
        }
    }
    for (bone, axis, deg) in ops {
        pose_apply_world_delta(
            skeleton,
            &humanoid,
            locals,
            bone,
            &pose_axis_angle(&axis, deg * ramp),
        );
    }
}
