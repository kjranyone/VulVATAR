use crate::asset::vrm::VrmAssetLoader;
use crate::avatar::{AvatarAsset, AvatarInstance};
use crate::output::{FrameSink, OutputRouter};
use crate::renderer::VulkanRenderer;
use crate::simulation::PhysicsWorld;
use crate::tracking::TrackingSource;

pub struct Application {
    renderer: VulkanRenderer,
    physics: PhysicsWorld,
    tracking: TrackingSource,
    output: OutputRouter,
    avatar: Option<AvatarInstance>,
}

impl Application {
    pub fn new() -> Self {
        Self {
            renderer: VulkanRenderer::new(),
            physics: PhysicsWorld::new(),
            tracking: TrackingSource::new(),
            output: OutputRouter::new(FrameSink::SharedMemory),
            avatar: None,
        }
    }

    pub fn bootstrap(&mut self) {
        self.renderer.initialize();

        let loader = VrmAssetLoader::new();
        let asset = AvatarAsset::from_vrm(loader.load("assets/avatar.vrm"));
        let avatar = AvatarInstance::new(asset);
        self.physics.attach_avatar(&avatar);
        self.avatar = Some(avatar);
    }

    pub fn run_frame(&mut self) {
        const FIXED_TIMESTEP_SECONDS: f32 = 1.0 / 60.0;

        if let Some(avatar) = self.avatar.as_mut() {
            let (tracking_frame, tracking_pose) = self.tracking.sample();
            println!(
                "tracking: frame {} body_conf={:.2} face_conf={:.2}",
                tracking_frame.frame_index,
                tracking_frame.body_confidence,
                tracking_frame.face_confidence
            );
            avatar.apply_tracking(tracking_pose);
            self.physics.step(FIXED_TIMESTEP_SECONDS, avatar);
            let frame = self.renderer.render(avatar);
            self.output.publish(&frame);
        }
    }
}
