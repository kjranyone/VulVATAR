use super::*;

/// Regression: a restored session can carry `tracking.enabled = true`
/// and `fade_on_tracking_loss = true` while no tracking worker is
/// running (the worker only starts from the Tracking Setup panel).
/// That state must NOT read as "person lost" — it faded the
/// freshly-loaded avatar to opacity 0 on startup, leaving a blank
/// viewport with no hint why (2026-06-12). The fade may only engage
/// once the worker is actually running.
#[test]
fn fade_keeps_avatar_opaque_until_tracking_worker_runs() {
    let mut app = Application::new();
    assert!(app.tracking_worker.is_none());
    let config = FrameConfig {
        toggles: RuntimeToggles {
            tracking_enabled: true,
            spring_enabled: false,
            cloth_enabled: false,
            collision_debug: false,
            skeleton_debug: false,
            mirror_view: false,
        },
        smoothing: crate::tracking::TrackingSmoothingParams::default(),
        material_mode_index: 0,
        hand_tracking_enabled: false,
        face_tracking_enabled: false,
        lower_body_tracking_enabled: false,
        root_translation_enabled: false,
        fade_on_tracking_loss: true,
        mouth_source: crate::tracking::MouthSource::Audio,
        spring_tuning: crate::simulation::spring::SpringTuning::default(),
        scene_gravity: crate::simulation::SceneGravity::default(),
        frame_dt: 1.0 / 60.0,
    };
    // 5 s of frames — far past the 0.6 s fade ramp, so any fade
    // toward 0 would be fully visible in the opacity by now.
    for _ in 0..300 {
        app.run_frame(&config);
    }
    assert_eq!(
        app.tracking_fade_opacity, 1.0,
        "avatar must stay opaque while the tracking worker is not running"
    );
}

fn mat4_mul_vec4(m: &crate::asset::Mat4, v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row] += m[row][col] * v[col];
        }
    }
    out
}

#[test]
fn projection_maps_near_plane_to_zero() {
    let proj = Application::build_projection_matrix(60.0, 16.0 / 9.0, 0.1, 1000.0);
    let point = [0.0, 0.0, -0.1, 1.0];
    let clip = mat4_mul_vec4(&proj, point);
    assert!(
        clip[3].abs() > 0.001,
        "w_clip should be nonzero, got {}",
        clip[3]
    );
    let ndc_z = clip[2] / clip[3];
    assert!(
        (-0.01..=0.01).contains(&ndc_z),
        "near plane should map to z_ndc=0, got z_ndc={}",
        ndc_z
    );
}

#[test]
fn projection_maps_far_plane_to_one() {
    let proj = Application::build_projection_matrix(60.0, 16.0 / 9.0, 0.1, 1000.0);
    let point = [0.0, 0.0, -1000.0, 1.0];
    let clip = mat4_mul_vec4(&proj, point);
    assert!(clip[3].abs() > 0.001, "w_clip should be nonzero");
    let ndc_z = clip[2] / clip[3];
    assert!(
        (0.99..=1.01).contains(&ndc_z),
        "far plane should map to z_ndc=1, got z_ndc={}",
        ndc_z
    );
}

#[test]
fn projection_avatar_in_front_of_camera_is_visible() {
    let cam = ViewportCamera::default();
    let (view, eye) = Application::build_view_matrix(&cam);
    let proj = Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0);

    let world_point = [0.0, 1.0, 0.0, 1.0];
    let view_point = mat4_mul_vec4(&view, world_point);
    let clip = mat4_mul_vec4(&proj, view_point);

    assert!(
        clip[3] > 0.0,
        "w_clip must be positive for visible geometry, got w={}",
        clip[3]
    );
    let ndc_z = clip[2] / clip[3];
    assert!(
            (0.0..=1.0).contains(&ndc_z),
            "avatar at origin should be in Vulkan clip range [0,1], got z_ndc={} (eye={:?}, view_pt={:?}, clip={:?})",
            ndc_z, eye, view_point, clip
        );
}

#[test]
fn view_matrix_looks_toward_origin() {
    let cam = ViewportCamera {
        yaw_deg: 0.0,
        pitch_deg: 0.0,
        distance: 5.0,
        pan: [0.0, 0.0],
        fov_deg: 60.0,
    };
    let (view, eye) = Application::build_view_matrix(&cam);
    assert!(
        eye[2] > 4.0,
        "camera at yaw=0 should be in +Z, got eye={:?}",
        eye
    );
    let origin_view = mat4_mul_vec4(&view, [0.0, 0.0, 0.0, 1.0]);
    assert!(
        origin_view[2] < 0.0,
        "origin should be at negative z in view space, got z={}",
        origin_view[2]
    );
}

fn make_test_avatar() -> AvatarInstance {
    use crate::asset::*;
    let skeleton = SkeletonAsset {
        nodes: vec![SkeletonNode {
            id: NodeId(0),
            name: "root".into(),
            parent: None,
            children: vec![],
            rest_local: Transform::default(),
            humanoid_bone: None,
        }],
        root_nodes: vec![NodeId(0)],
        inverse_bind_matrices: vec![identity_matrix()],
    };

    let verts = VertexData {
        positions: vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        uvs: vec![[0.0, 0.0]; 3],
        joint_indices: vec![[0, 0, 0, 0]; 3],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
    };

    let prim = MeshPrimitiveAsset {
        id: PrimitiveId(0),
        vertex_count: 3,
        index_count: 3,
        material_id: MaterialId(0),
        skin: Some(SkinBinding {
            joint_nodes: vec![NodeId(0)],
            inverse_bind_matrices: vec![identity_matrix()],
        }),
        bounds: Aabb {
            min: [-0.5, 0.0, 0.0],
            max: [0.5, 1.0, 0.0],
        },
        vertices: Some(verts),
        indices: Some(vec![0, 1, 2]),
        morph_targets: Vec::new(),
        skin_anchors: None,
        body_primitive_id: None,
    };

    let mesh = MeshAsset {
        id: MeshId(0),
        name: "test_mesh".into(),
        primitives: vec![Arc::new(prim)],
    };

    let asset = Arc::new(AvatarAsset {
        id: AvatarAssetId(0),
        source_path: std::path::PathBuf::from("test.vrm"),
        source_hash: AssetSourceHash([0u8; 32]),
        skeleton,
        meshes: vec![mesh],
        materials: vec![],
        humanoid: None,
        spring_bones: vec![],
        colliders: vec![],
        default_expressions: ExpressionAssetSet {
            expressions: vec![],
        },
        animation_clips: vec![],
        node_to_mesh: Default::default(),
        vrm_meta: Default::default(),
        root_aabb: crate::asset::Aabb::empty(),
        body_primitive_id: None,
        loaded_from_cache: false,
    });

    let mut inst = AvatarInstance::new(AvatarInstanceId(1), asset);
    inst.build_base_pose();
    inst.compute_global_pose();
    inst.build_skinning_matrices();
    inst
}

struct Ray {
    origin: [f32; 3],
    dir: [f32; 3],
}

struct RayHit {
    t: f32,
    point: [f32; 3],
}

fn ray_triangle_intersect(ray: &Ray, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> Option<f32> {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let h = [
        ray.dir[1] * e2[2] - ray.dir[2] * e2[1],
        ray.dir[2] * e2[0] - ray.dir[0] * e2[2],
        ray.dir[0] * e2[1] - ray.dir[1] * e2[0],
    ];
    let a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if a.abs() < 1e-8 {
        return None;
    }
    let f = 1.0 / a;
    let s = [
        ray.origin[0] - v0[0],
        ray.origin[1] - v0[1],
        ray.origin[2] - v0[2],
    ];
    let u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = [
        s[1] * e1[2] - s[2] * e1[1],
        s[2] * e1[0] - s[0] * e1[2],
        s[0] * e1[1] - s[1] * e1[0],
    ];
    let v = f * (ray.dir[0] * q[0] + ray.dir[1] * q[1] + ray.dir[2] * q[2]);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    if t > 1e-6 {
        Some(t)
    } else {
        None
    }
}

fn ray_cast_avatar(ray: &Ray, avatar: &AvatarInstance) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    let skin_mats = &avatar.pose.skinning_matrices;
    for mesh in &avatar.asset.meshes {
        for prim in &mesh.primitives {
            if let Some(ref vd) = prim.vertices {
                let indices = prim.indices.as_deref().unwrap_or(&[]);
                for tri in indices.chunks(3) {
                    if tri.len() < 3 {
                        continue;
                    }
                    let apply_skin = |idx: usize| -> [f32; 3] {
                        let p = vd.positions[idx];
                        let ji = vd.joint_indices[idx];
                        let jw = vd.joint_weights[idx];
                        let total = jw[0] + jw[1] + jw[2] + jw[3];
                        if total < 0.001 {
                            return p;
                        }
                        let mut out = [0.0f32; 3];
                        for j in 0..4 {
                            if jw[j] < 0.001 {
                                continue;
                            }
                            let mat = skin_mats
                                .get(ji[j] as usize)
                                .copied()
                                .unwrap_or_else(crate::asset::identity_matrix);
                            let w =
                                p[0] * mat[0][0] + p[1] * mat[1][0] + p[2] * mat[2][0] + mat[3][0];
                            let x =
                                p[0] * mat[0][1] + p[1] * mat[1][1] + p[2] * mat[2][1] + mat[3][1];
                            let y =
                                p[0] * mat[0][2] + p[1] * mat[1][2] + p[2] * mat[2][2] + mat[3][2];
                            out[0] += jw[j] * w;
                            out[1] += jw[j] * x;
                            out[2] += jw[j] * y;
                        }
                        out
                    };
                    let v0 = apply_skin(tri[0] as usize);
                    let v1 = apply_skin(tri[1] as usize);
                    let v2 = apply_skin(tri[2] as usize);
                    if let Some(t) = ray_triangle_intersect(ray, v0, v1, v2) {
                        if best.as_ref().is_none_or(|b| t < b.t) {
                            let pt = [
                                ray.origin[0] + t * ray.dir[0],
                                ray.origin[1] + t * ray.dir[1],
                                ray.origin[2] + t * ray.dir[2],
                            ];
                            best = Some(RayHit { t, point: pt });
                        }
                    }
                }
            }
        }
    }
    best
}

fn ndc_from_world(
    world: [f32; 3],
    view: &crate::asset::Mat4,
    proj: &crate::asset::Mat4,
) -> [f32; 3] {
    let v = mat4_mul_vec4(view, [world[0], world[1], world[2], 1.0]);
    let c = mat4_mul_vec4(proj, v);
    let w = c[3];
    if w.abs() < 1e-8 {
        return [f32::NAN; 3];
    }
    [c[0] / w, c[1] / w, c[2] / w]
}

#[test]
fn ray_cast_center_hits_test_avatar() {
    let avatar = make_test_avatar();
    let cam = ViewportCamera::default();
    let (view, eye) = Application::build_view_matrix(&cam);

    let target = [0.0, 0.5, 0.0];
    let dx = target[0] - eye[0];
    let dy = target[1] - eye[1];
    let dz = target[2] - eye[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let ray = Ray {
        origin: eye,
        dir: [dx / len, dy / len, dz / len],
    };

    let hit = ray_cast_avatar(&ray, &avatar).expect("center ray should hit the test triangle");
    assert!(hit.t > 0.0, "hit t should be positive, got {}", hit.t);

    let ndc = ndc_from_world(
        hit.point,
        &view,
        &Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0),
    );
    assert!(
        ndc[0] >= -1.0 && ndc[0] <= 1.0,
        "hit NDC x out of range: {}",
        ndc[0]
    );
    assert!(
        ndc[1] >= -1.0 && ndc[1] <= 1.0,
        "hit NDC y out of range: {}",
        ndc[1]
    );
    assert!(
        ndc[2] >= 0.0 && ndc[2] <= 1.0,
        "hit NDC z out of Vulkan range [0,1]: {}",
        ndc[2]
    );
}

#[test]
fn ray_cast_sweep_viewport_corners() {
    let avatar = make_test_avatar();
    let cam = ViewportCamera {
        distance: 3.0,
        ..ViewportCamera::default()
    };
    let (view, eye) = Application::build_view_matrix(&cam);
    let proj = Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0);

    let fov_rad = cam.fov_deg.to_radians();
    let half_fov = fov_rad * 0.5;
    let aspect = 16.0 / 9.0;
    let tan_h = half_fov.tan();
    let tan_w = tan_h * aspect;
    let fwd = [0.0 - eye[0], 0.0 - eye[1], 0.0 - eye[2]];
    let flen = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt();
    let fwd = [fwd[0] / flen, fwd[1] / flen, fwd[2] / flen];

    let right = [fwd[2], 0.0, -fwd[0]];
    let rlen = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    let right = [right[0] / rlen, right[1] / rlen, right[2] / rlen];
    let up = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ];

    let mut hits = 0usize;
    let mut visible_ndc = 0usize;
    for &(sx, sy) in &[
        (0.0f32, 0.0),
        (0.3, 0.0),
        (-0.3, 0.0),
        (0.0, 0.3),
        (0.0, -0.3),
    ] {
        let dx = fwd[0] + sx * tan_w * right[0] + sy * tan_h * up[0];
        let dy = fwd[1] + sx * tan_w * right[1] + sy * tan_h * up[1];
        let dz = fwd[2] + sx * tan_w * right[2] + sy * tan_h * up[2];
        let dl = (dx * dx + dy * dy + dz * dz).sqrt();
        let ray = Ray {
            origin: eye,
            dir: [dx / dl, dy / dl, dz / dl],
        };
        if let Some(hit) = ray_cast_avatar(&ray, &avatar) {
            hits += 1;
            let ndc = ndc_from_world(hit.point, &view, &proj);
            if ndc[0] >= -1.0
                && ndc[0] <= 1.0
                && ndc[1] >= -1.0
                && ndc[1] <= 1.0
                && ndc[2] >= 0.0
                && ndc[2] <= 1.0
            {
                visible_ndc += 1;
            }
        }
    }
    assert!(hits > 0, "at least one sweep ray should hit the avatar");
    assert!(
        visible_ndc > 0,
        "at least one hit should be in Vulkan NDC range, got {}/{} hits visible",
        visible_ndc,
        hits
    );
}

#[test]
fn ray_cast_aabb_frustum_culling() {
    let _avatar = make_test_avatar();
    let cam = ViewportCamera::default();
    let (view, _eye) = Application::build_view_matrix(&cam);
    let proj = Application::build_projection_matrix(cam.fov_deg, 16.0 / 9.0, 0.1, 1000.0);

    let mut inside_count = 0;
    let corners = [
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [-0.5, 1.0, 0.0],
        [0.5, 1.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [-0.5, 1.0, 0.0],
        [0.5, 1.0, 0.0],
    ];
    for &c in &corners {
        let ndc = ndc_from_world(c, &view, &proj);
        if ndc[0] >= -1.0
            && ndc[0] <= 1.0
            && ndc[1] >= -1.0
            && ndc[1] <= 1.0
            && ndc[2] >= 0.0
            && ndc[2] <= 1.0
        {
            inside_count += 1;
        }
    }
    assert!(
        inside_count >= 4,
        "AABB corners should be inside frustum: {}/8 corners visible",
        inside_count
    );
}
