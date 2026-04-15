use crate::asset::{AlphaMode, MaterialMode, MeshId, PrimitiveId, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::Sampler;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{GraphicsPipeline, Pipeline};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MaterialShaderMode {
    Unlit,
    SimpleLit,
    ToonLike,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MaterialDebugView {
    None,
    Uv,
    BaseTexture,
}

#[derive(Clone, Debug)]
pub struct MaterialUploadRequest {
    pub base_color: Vec4,
    pub mode: MaterialShaderMode,
    pub debug_view: MaterialDebugView,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    pub toon_ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_width: f32,
    pub outline_color: Vec3,
    pub textures: MaterialTextureSlots,
    pub mtoon_params: Option<crate::asset::MtoonStagedParams>,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialTextureSlots {
    pub base_color: Option<MaterialTextureBinding>,
    pub normal_map: Option<MaterialTextureBinding>,
    pub shade_ramp: Option<MaterialTextureBinding>,
    pub emissive: Option<MaterialTextureBinding>,
    pub matcap: Option<MaterialTextureBinding>,
}

#[derive(Clone, Debug)]
pub struct MaterialTextureBinding {
    pub uri: String,
    pub pixel_data: Option<std::sync::Arc<Vec<u8>>>,
    pub dimensions: (u32, u32),
}

impl MaterialUploadRequest {
    /// Returns a white opaque default material for use when no materials are available.
    pub fn default_material() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            mode: MaterialShaderMode::Unlit,
            debug_view: MaterialDebugView::None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            toon_ramp_threshold: 0.5,
            shadow_softness: 0.5,
            outline_width: 0.0,
            outline_color: [0.0, 0.0, 0.0],
            textures: MaterialTextureSlots::default(),
            mtoon_params: None,
        }
    }

    pub fn from_asset_material(asset: &crate::asset::MaterialAsset) -> Self {
        Self {
            base_color: asset.base_color,
            mode: match asset.base_mode {
                MaterialMode::Unlit => MaterialShaderMode::Unlit,
                MaterialMode::SimpleLit => MaterialShaderMode::SimpleLit,
                MaterialMode::ToonLike => MaterialShaderMode::ToonLike,
            },
            debug_view: MaterialDebugView::None,
            alpha_mode: asset.alpha_mode,
            double_sided: asset.double_sided,
            toon_ramp_threshold: asset.toon_params.ramp_threshold,
            shadow_softness: asset.toon_params.shadow_softness,
            outline_width: asset.toon_params.outline_width,
            outline_color: asset.toon_params.outline_color,
            textures: MaterialTextureSlots {
                base_color: asset.texture_bindings.base_color_texture.as_ref().map(|t| {
                    MaterialTextureBinding {
                        uri: t.uri.clone(),
                        pixel_data: t.pixel_data.clone(),
                        dimensions: t.dimensions,
                    }
                }),
                normal_map: asset.texture_bindings.normal_map_texture.as_ref().map(|t| {
                    MaterialTextureBinding {
                        uri: t.uri.clone(),
                        pixel_data: t.pixel_data.clone(),
                        dimensions: t.dimensions,
                    }
                }),
                shade_ramp: asset.texture_bindings.shade_ramp_texture.as_ref().map(|t| {
                    MaterialTextureBinding {
                        uri: t.uri.clone(),
                        pixel_data: t.pixel_data.clone(),
                        dimensions: t.dimensions,
                    }
                }),
                emissive: asset.texture_bindings.emissive_texture.as_ref().map(|t| {
                    MaterialTextureBinding {
                        uri: t.uri.clone(),
                        pixel_data: t.pixel_data.clone(),
                        dimensions: t.dimensions,
                    }
                }),
                matcap: asset.texture_bindings.matcap_texture.as_ref().map(|t| {
                    MaterialTextureBinding {
                        uri: t.uri.clone(),
                        pixel_data: t.pixel_data.clone(),
                        dimensions: t.dimensions,
                    }
                }),
            },
            mtoon_params: asset.mtoon_params.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU material uniform layout (matches shader `MaterialData`)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MaterialUniform {
    pub base_color: [f32; 4],
    pub alpha_cutoff: f32,
    pub alpha_mode: i32,
    pub shade_shift: f32,
    pub shade_toony: f32,
    pub shading_mode: i32,
    pub debug_view: i32,
    pub _pad_debug0: [i32; 2],
    pub shade_color: [f32; 4],
    pub emissive_color: [f32; 4],
    pub rim_color: [f32; 4],
    pub rim_fresnel_power: f32,
    pub rim_lift: f32,
    pub rim_lighting_mix: f32,
    pub uv_anim_scroll_x: f32,
    pub uv_anim_scroll_y: f32,
    pub uv_anim_rotation: f32,
    pub matcap_blend: f32,
}

// ---------------------------------------------------------------------------
// MaterialUploader -- creates GPU uniform buffers and descriptor sets
// ---------------------------------------------------------------------------

struct MaterialCacheEntry {
    buffer: Subbuffer<MaterialUniform>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    base_color_uri: String,
    shade_texture_uri: Option<String>,
    matcap_uri: Option<String>,
}

pub struct MaterialUploader {
    cache: HashMap<(MeshId, PrimitiveId), MaterialCacheEntry>,
}

impl MaterialUploader {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Upload material data to a GPU uniform buffer and return a descriptor set
    /// bound at set 2 of the given pipeline layout.
    ///
    /// The descriptor set contains:
    /// - binding 0: material uniform buffer
    /// - binding 1: combined image sampler (base color texture)
    pub fn upload_to_gpu(
        &mut self,
        cache_key: Option<(MeshId, PrimitiveId)>,
        request: &MaterialUploadRequest,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        pipeline: &Arc<GraphicsPipeline>,
        texture_view: Arc<ImageView>,
        shade_texture_view: Arc<ImageView>,
        sampler: Arc<Sampler>,
        matcap_texture_view: Option<Arc<ImageView>>,
    ) -> Result<Arc<PersistentDescriptorSet>, String> {
        let base_color_uri = request
            .textures
            .base_color
            .as_ref()
            .map(|b| b.uri.as_str())
            .unwrap_or("")
            .to_string();
        let shade_texture_uri = request.textures.shade_ramp.as_ref().map(|b| b.uri.clone());
        let matcap_uri = request.textures.matcap.as_ref().map(|b| b.uri.clone());

        let uniform_data = Self::compute_uniform(request);

        if let Some(key) = cache_key {
            if let Some(entry) = self.cache.get(&key) {
                if entry.base_color_uri == base_color_uri
                    && entry.shade_texture_uri == shade_texture_uri
                    && entry.matcap_uri == matcap_uri
                {
                    {
                        let mut guard = entry
                            .buffer
                            .write()
                            .map_err(|e| format!("failed to write material uniform buffer: {e}"))?;
                        *guard = uniform_data;
                    }
                    return Ok(entry.descriptor_set.clone());
                }
            }
        }

        let uniform_buffer = Buffer::from_data(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            uniform_data,
        )
        .map_err(|e| format!("failed to create material uniform buffer: {e}"))?;

        let layout = pipeline
            .layout()
            .set_layouts()
            .get(2)
            .ok_or_else(|| {
                "material: pipeline has no descriptor set at index 2 (material set)".to_string()
            })?
            .clone();
        let matcap_view = matcap_texture_view.unwrap_or_else(|| texture_view.clone());
        let descriptor_set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            layout,
            [
                WriteDescriptorSet::buffer(0, uniform_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(1, texture_view, sampler.clone()),
                WriteDescriptorSet::image_view_sampler(2, matcap_view, sampler.clone()),
                WriteDescriptorSet::image_view_sampler(3, shade_texture_view, sampler),
            ],
            [],
        )
        .map_err(|e| format!("failed to create material descriptor set: {e}"))?;

        if let Some(key) = cache_key {
            self.cache.insert(
                key,
                MaterialCacheEntry {
                    buffer: uniform_buffer,
                    descriptor_set: descriptor_set.clone(),
                    base_color_uri,
                    shade_texture_uri,
                    matcap_uri,
                },
            );
        }

        Ok(descriptor_set)
    }

    fn compute_uniform(request: &MaterialUploadRequest) -> MaterialUniform {
        let alpha_mode_int = match request.alpha_mode {
            AlphaMode::Opaque => 0i32,
            AlphaMode::Mask(_) => 1i32,
            AlphaMode::Blend => 2i32,
        };
        let alpha_cutoff = match request.alpha_mode {
            AlphaMode::Mask(cutoff) => cutoff,
            _ => 0.5,
        };

        let (shade_shift, shade_toony) = if let Some(ref mtoon) = request.mtoon_params {
            (mtoon.shade_shift, mtoon.shade_toony)
        } else {
            (0.0, 0.9)
        };

        let shading_mode = match request.mode {
            MaterialShaderMode::Unlit => 0i32,
            MaterialShaderMode::SimpleLit => 2i32,
            MaterialShaderMode::ToonLike => 1i32,
        };
        let debug_view = match request.debug_view {
            MaterialDebugView::None => 0i32,
            MaterialDebugView::Uv => 1i32,
            MaterialDebugView::BaseTexture => 2i32,
        };

        let (
            shade_color,
            emissive_color,
            rim_color,
            rim_fresnel_power,
            rim_lift,
            rim_lighting_mix,
            uv_scroll_x,
            uv_scroll_y,
            uv_rotation,
        ) = if let Some(ref mtoon) = request.mtoon_params {
            let emissive =
                if request.textures.emissive.is_some() || request.textures.shade_ramp.is_some() {
                    mtoon.emissive_color
                } else {
                    [
                        mtoon.emissive_color[0],
                        mtoon.emissive_color[1],
                        mtoon.emissive_color[2],
                        0.0,
                    ]
                };
            (
                mtoon.shade_color,
                emissive,
                mtoon.rim_color,
                mtoon.rim_fresnel_power,
                mtoon.rim_lift,
                mtoon.rim_lighting_mix,
                mtoon.uv_anim_scroll_x_speed,
                mtoon.uv_anim_scroll_y_speed,
                mtoon.uv_anim_rotation_speed,
            )
        } else {
            (
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        };

        // Disable the current matcap path until a proper MToon-compatible
        // implementation is added. The placeholder VRoid matcap textures
        // (e.g. `MatcapWarp`) badly over-brighten avatars when treated as a
        // direct color map.
        let matcap_blend = 0.0;

        MaterialUniform {
            base_color: request.base_color,
            alpha_cutoff,
            alpha_mode: alpha_mode_int,
            shade_shift,
            shade_toony,
            shading_mode,
            debug_view,
            _pad_debug0: [0, 0],
            shade_color,
            emissive_color,
            rim_color,
            rim_fresnel_power,
            rim_lift,
            rim_lighting_mix,
            uv_anim_scroll_x: uv_scroll_x,
            uv_anim_scroll_y: uv_scroll_y,
            uv_anim_rotation: uv_rotation,
            matcap_blend,
        }
    }
}
