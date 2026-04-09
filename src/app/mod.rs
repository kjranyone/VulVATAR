use crate::asset::vrm::VrmAssetLoader;
use crate::avatar::retargeting;
use crate::avatar::{AvatarInstance, AvatarInstanceId};
use crate::editor::EditorSession;
use crate::output::{FrameSink, OutputRouter};
use crate::renderer::frame_input::{
    CameraState, ClothDeformSnapshot, LightingState, OutlineSnapshot, OutputTargetRequest,
    RenderAlphaMode, RenderAvatarInstance, RenderCullMode, RenderExportMode, RenderFrameInput,
    RenderMeshInstance, RenderOutputAlpha,
};
use crate::renderer::material::MaterialUploadRequest;
use crate::renderer::VulkanRenderer;
use crate::simulation::{PhysicsWorld, SimulationClock};
use crate::tracking::TrackingSource;

pub struct Application {
    pub renderer: VulkanRenderer,
    pub physics: PhysicsWorld,
    pub tracking: TrackingSource,
    pub output: OutputRouter,
    pub sim_clock: SimulationClock,
    pub editor: EditorSession,
    pub avatar: Option<AvatarInstance>,
    pub next_avatar_instance_id: u64,
    pub running: bool,
}

impl Application {
    pub fn new() -> Self {
        Self {
            renderer: VulkanRenderer::new(),
            physics: PhysicsWorld::new(),
            tracking: TrackingSource::new(),
            output: OutputRouter::new(FrameSink::SharedMemory),
            sim_clock: SimulationClock::new(1.0 / 60.0, 8),
            editor: EditorSession::new(),
            avatar: None,
            next_avatar_instance_id: 1,
            running: false,
        }
    }

    pub fn bootstrap(&mut self) {
        self.renderer.initialize();

        let loader = VrmAssetLoader::new();
        let asset = match loader.load("assets/avatar.vrm") {
            Ok(a) => a,
            Err(e) => {
                eprintln!("bootstrap: failed to load avatar: {}", e);
                return;
            }
        };
        let instance_id = AvatarInstanceId(self.next_avatar_instance_id);
        self.next_avatar_instance_id += 1;

        self.physics.attach_avatar(&asset);
        self.avatar = Some(AvatarInstance::new(instance_id, asset));
        self.running = true;
    }

    pub fn run_frame(&mut self) {
        if !self.running {
            return;
        }

        let frame_dt: f32 = 1.0 / 60.0;

        // 1. input update
        self.input_update();

        // 2. read the latest completed tracking sample from the async tracking worker
        let tracking_sample = self.step_tracking();

        if let Some(avatar) = self.avatar.as_mut() {
            // 3. locomotion or gameplay update
            // 4. animation sampling
            // 5. base local pose build
            avatar.build_base_pose();

            // 6. merge tracking inputs
            if let Some(tracking_pose) = tracking_sample {
                avatar.apply_tracking(tracking_pose.clone());
                let humanoid = avatar.asset.humanoid.as_ref();
                retargeting::retarget_tracking_to_pose(
                    &tracking_pose,
                    &avatar.asset.skeleton,
                    humanoid,
                    &mut avatar.pose.local_transforms,
                );
            }

            // 7. compute driven global pose for the body before secondary motion
            avatar.compute_global_pose();

            // 8. fixed-timestep spring bone simulation against that pose
            // 9. recompute global pose after spring outputs are applied
            let substeps = self.sim_clock.advance(frame_dt);
            for _ in 0..substeps {
                self.physics.step_springs(self.sim_clock.fixed_dt(), avatar);
            }
            avatar.compute_global_pose();

            // 10. fixed-timestep cloth simulation against the resolved body motion for this frame
            // 11. recompute final global pose
            if avatar.cloth_enabled {
                for _ in 0..substeps {
                    self.physics.step_cloth(self.sim_clock.fixed_dt(), avatar);
                }
                avatar.compute_global_pose();
            }

            // 12. build skinning matrices
            avatar.build_skinning_matrices();

            // 13. upload pose buffers
            // 14. upload optional cloth deformation buffers

            // 15. build renderer-facing frame snapshot
            let frame_input = Self::build_frame_input(avatar);

            // 16. render
            let render_result = self.renderer.render(&frame_input);

            // 17. publish output frame to the async output sink
            if let Some(ref exported) = render_result.exported_frame {
                self.output.publish(exported.output_frame.clone());
            }
        }
    }

    fn build_frame_input(avatar: &AvatarInstance) -> RenderFrameInput {
        let mesh_instances: Vec<RenderMeshInstance> = avatar
            .asset
            .meshes
            .iter()
            .flat_map(|mesh| {
                mesh.primitives.iter().map(|prim| {
                    let material_asset = avatar
                        .asset
                        .materials
                        .iter()
                        .find(|m| m.id == prim.material_id);

                    let material_binding = material_asset
                        .map(MaterialUploadRequest::from_asset_material)
                        .unwrap_or_else(|| {
                            MaterialUploadRequest::from_asset_material(
                                &avatar.asset.materials.first().cloned().unwrap(),
                            )
                        });

                    let outline = OutlineSnapshot {
                        enabled: material_binding.outline_width > 0.0,
                        width: material_binding.outline_width,
                        color: material_binding.outline_color,
                    };

                    RenderMeshInstance {
                        mesh_id: mesh.id,
                        primitive_id: prim.id,
                        material_binding,
                        bounds: prim.bounds.clone(),
                        alpha_mode: RenderAlphaMode::Opaque,
                        cull_mode: RenderCullMode::BackFace,
                        outline,
                    }
                })
            })
            .collect();

        let cloth_deform = avatar.cloth_state.as_ref().map(|cs| ClothDeformSnapshot {
            deformed_positions: cs.deform_output.deformed_positions.clone(),
            deformed_normals: cs.deform_output.deformed_normals.clone(),
            version: cs.deform_output.version,
        });

        let render_instance = RenderAvatarInstance {
            instance_id: avatar.id,
            world_transform: avatar.world_transform.clone(),
            mesh_instances,
            skinning_matrices: avatar.pose.skinning_matrices.clone(),
            cloth_deform,
            debug_flags: Default::default(),
        };

        RenderFrameInput {
            camera: CameraState::default(),
            lighting: LightingState::default(),
            instances: vec![render_instance],
            output_request: OutputTargetRequest {
                preview_enabled: true,
                output_enabled: true,
                extent: [1920, 1080],
                color_space: crate::renderer::frame_input::RenderColorSpace::Srgb,
                alpha_mode: RenderOutputAlpha::Premultiplied,
                export_mode: RenderExportMode::GpuExport,
            },
        }
    }

    fn input_update(&mut self) {}

    fn step_tracking(&mut self) -> Option<crate::tracking::TrackingRigPose> {
        let frame = self.tracking.sample();
        println!(
            "tracking: frame {} body_conf={:.2} face_conf={:.2}",
            frame.capture_timestamp, frame.body_confidence, frame.face_confidence
        );
        self.tracking.mailbox().read()
    }
}
