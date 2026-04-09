use crate::asset::Vec3;

#[derive(Clone, Debug, Default)]
pub struct MtoonStagedParams {
    pub shade_color: crate::asset::Vec4,
    pub shade_shift: f32,
    pub shade_toony: f32,
    pub lit_color: crate::asset::Vec4,
    pub gi_equalization: f32,
    pub matcap_texture: Option<MtoonTextureSlot>,
    pub rim_texture: Option<MtoonTextureSlot>,
    pub rim_color: crate::asset::Vec4,
    pub rim_lighting_mix: f32,
    pub rim_fresnel_power: f32,
    pub rim_lift: f32,
    pub emissive_texture: Option<MtoonTextureSlot>,
    pub emissive_color: crate::asset::Vec4,
    pub outline_width_mode: MtoonOutlineWidthMode,
    pub outline_color: Vec3,
    pub outline_width: f32,
    pub uv_anim_mask_texture: Option<MtoonTextureSlot>,
    pub uv_anim_scroll_x_speed: f32,
    pub uv_anim_scroll_y_speed: f32,
    pub uv_anim_rotation_speed: f32,
}

#[derive(Clone, Debug)]
pub struct MtoonTextureSlot {
    pub uri: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MtoonOutlineWidthMode {
    None,
    WorldCoordinates,
    ScreenCoordinates,
}

impl Default for MtoonOutlineWidthMode {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Debug, Default)]
pub struct MtoonCompatibilityStatus {
    pub base_color_supported: bool,
    pub texture_sampling_supported: bool,
    pub alpha_mode_supported: bool,
    pub culling_mode_supported: bool,
    pub toon_threshold_supported: bool,
    pub outline_supported: bool,
    pub outline_width_supported: bool,
    pub shade_controls_supported: bool,
    pub emissive_supported: bool,
    pub rim_supported: bool,
    pub uv_animation_supported: bool,
}

impl MtoonCompatibilityStatus {
    pub fn initial_stub() -> Self {
        Self {
            base_color_supported: true,
            texture_sampling_supported: false,
            alpha_mode_supported: false,
            culling_mode_supported: false,
            toon_threshold_supported: false,
            outline_supported: false,
            outline_width_supported: false,
            shade_controls_supported: false,
            emissive_supported: false,
            rim_supported: false,
            uv_animation_supported: false,
        }
    }
}
