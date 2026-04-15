use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::PersistentDescriptorSet;

use crate::renderer::pipeline::GpuVertex;

pub struct ForwardPassDraw {
    pub vertex_buffer: Subbuffer<[GpuVertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,
    pub skinning_set: Arc<PersistentDescriptorSet>,
    pub material_set: Arc<PersistentDescriptorSet>,
    pub outline_width: f32,
    pub outline_color: [f32; 3],
}
