use crate::asset::{MaterialMode, Vec3, Vec4};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MaterialShaderMode {
    Unlit,
    SimpleLit,
    ToonLike,
}

#[derive(Clone, Debug)]
pub struct MaterialUploadRequest {
    pub base_color: Vec4,
    pub mode: MaterialShaderMode,
    pub toon_ramp_threshold: f32,
    pub shadow_softness: f32,
    pub outline_width: f32,
    pub outline_color: Vec3,
    pub textures: MaterialTextureSlots,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialTextureSlots {
    pub base_color: Option<MaterialTextureBinding>,
    pub normal_map: Option<MaterialTextureBinding>,
    pub shade_ramp: Option<MaterialTextureBinding>,
    pub emissive: Option<MaterialTextureBinding>,
}

#[derive(Clone, Debug)]
pub struct MaterialTextureBinding {
    pub uri: String,
}

impl MaterialUploadRequest {
    pub fn from_asset_material(asset: &crate::asset::MaterialAsset) -> Self {
        Self {
            base_color: asset.base_color,
            mode: match asset.base_mode {
                MaterialMode::Unlit => MaterialShaderMode::Unlit,
                MaterialMode::SimpleLit => MaterialShaderMode::SimpleLit,
                MaterialMode::ToonLike => MaterialShaderMode::ToonLike,
            },
            toon_ramp_threshold: asset.toon_params.ramp_threshold,
            shadow_softness: asset.toon_params.shadow_softness,
            outline_width: asset.toon_params.outline_width,
            outline_color: asset.toon_params.outline_color,
            textures: MaterialTextureSlots {
                base_color: asset
                    .texture_bindings
                    .base_color_texture
                    .as_ref()
                    .map(|t| MaterialTextureBinding { uri: t.uri.clone() }),
                normal_map: asset
                    .texture_bindings
                    .normal_map_texture
                    .as_ref()
                    .map(|t| MaterialTextureBinding { uri: t.uri.clone() }),
                shade_ramp: None,
                emissive: None,
            },
        }
    }
}

pub struct MaterialUploader {
    upload_count: u64,
}

impl MaterialUploader {
    pub fn new() -> Self {
        Self { upload_count: 0 }
    }

    pub fn upload(&mut self, request: &MaterialUploadRequest) -> MaterialUploadHandle {
        let handle_id = self.upload_count;
        self.upload_count += 1;
        MaterialUploadHandle {
            handle_id,
            mode: request.mode.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaterialUploadHandle {
    pub handle_id: u64,
    pub mode: MaterialShaderMode,
}
