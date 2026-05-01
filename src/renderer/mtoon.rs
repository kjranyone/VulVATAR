// Re-export MToon staged parameter types from the asset module.
// These are pure-data types with no GPU dependencies, so they live in `asset`.
// The re-exports are kept for backward compatibility even if nothing currently uses them.
#[allow(unused_imports)]
pub use crate::asset::{MtoonOutlineWidthMode, MtoonStagedParams, MtoonTextureSlot};

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
    pub matcap_supported: bool,
}

impl MtoonCompatibilityStatus {
    /// Returns the compatibility status matching the current POC renderer
    /// capabilities.  Features that the toon-like pipeline and outline pass
    /// already handle are marked as supported.
    pub fn initial_poc() -> Self {
        Self {
            base_color_supported: true,
            texture_sampling_supported: true,
            alpha_mode_supported: true,
            culling_mode_supported: true,
            toon_threshold_supported: true,
            outline_supported: true,
            outline_width_supported: true,
            shade_controls_supported: true,
            emissive_supported: true,
            rim_supported: true,
            uv_animation_supported: true,
            matcap_supported: true,
        }
    }
}
