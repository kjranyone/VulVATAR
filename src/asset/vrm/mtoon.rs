//! MToon material extension decoding.
//!
//! `VRMC_materials_mtoon` carries the toon-shader parameters that VRM
//! avatars rely on (shade colour, ramp toony-ness, parametric rim,
//! outline width / colour, UV-scroll animation, etc.) plus a small set
//! of texture-index fields that this module surfaces back to the
//! material builder so it can resolve them via the outer glTF document.
//!
//! This module is purely a JSON → params translator. It does *not*
//! reach into the glTF document itself — its return shape exposes
//! `MtoonTextureIndices` so the caller (in `mod.rs`) can resolve each
//! index into a `TextureBinding` after the fact, keeping the texture
//! decoder co-located with the rest of `build_materials`.

use crate::asset::{MtoonOutlineWidthMode, MtoonStagedParams, ToonMaterialParams};

/// Texture indices read out of the MToon JSON. The caller resolves
/// each `Some(index)` against the glTF document's texture array.
pub(super) struct MtoonTextureIndices {
    pub(super) shade_color_texture: Option<usize>,
    pub(super) emissive_texture: Option<usize>,
    pub(super) rim_texture: Option<usize>,
    pub(super) matcap_texture: Option<usize>,
    pub(super) uv_anim_mask_texture: Option<usize>,
}

/// Decode `VRMC_materials_mtoon` JSON into the engine's toon-material
/// param triple: `ToonMaterialParams` (legacy SimpleLit-derived knobs),
/// `MtoonStagedParams` (the full MToon set, with texture slots left
/// empty for the caller to fill), and the raw texture indices.
pub(super) fn parse_mtoon_params(
    mtoon: &serde_json::Value,
) -> (ToonMaterialParams, MtoonStagedParams, MtoonTextureIndices) {
    let get_f32 = |key: &str| mtoon.get(key).and_then(|v| v.as_f64()).map(|v| v as f32);
    let get_color3 = |key: &str| -> [f32; 3] {
        mtoon
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                [
                    arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                ]
            })
            .unwrap_or([0.0; 3])
    };
    let get_color3_as_4 = |key: &str| -> [f32; 4] {
        mtoon
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                [
                    arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    1.0f32,
                ]
            })
            .unwrap_or([0.0, 0.0, 0.0, 0.0])
    };

    let shade_color = get_color3_as_4("shadeColorFactor");
    let shade_shift = get_f32("shadingShiftFactor").unwrap_or(0.0);
    let shade_toony = get_f32("shadingToonyFactor").unwrap_or(0.9);
    let gi_equalization = get_f32("giEqualizationFactor").unwrap_or(0.9);

    let outline_width_mode_str = mtoon
        .get("outlineWidthMode")
        .and_then(|v| v.as_str())
        .unwrap_or("none");
    let outline_width_mode = match outline_width_mode_str {
        "worldCoordinates" => MtoonOutlineWidthMode::WorldCoordinates,
        "screenCoordinates" => MtoonOutlineWidthMode::ScreenCoordinates,
        _ => MtoonOutlineWidthMode::None,
    };
    let outline_width = get_f32("outlineWidthFactor").unwrap_or(0.0);
    let outline_color = get_color3("outlineColorFactor");

    let rim_color = get_color3_as_4("parametricRimColorFactor");
    let rim_fresnel_power = get_f32("parametricRimFresnelPowerFactor").unwrap_or(5.0);
    let rim_lift = get_f32("parametricRimLiftFactor").unwrap_or(0.0);
    let rim_lighting_mix = get_f32("rimLightingMixFactor").unwrap_or(1.0);

    let emissive_color = get_color3_as_4("emissiveFactor");

    let uv_anim_scroll_x = get_f32("uvAnimationScrollXSpeedFactor").unwrap_or(0.0);
    let uv_anim_scroll_y = get_f32("uvAnimationScrollYSpeedFactor").unwrap_or(0.0);
    let uv_anim_rotation = get_f32("uvAnimationRotationSpeedFactor").unwrap_or(0.0);

    let get_tex_index = |key: &str| {
        mtoon.get(key).and_then(|v| {
            v.as_u64()
                .or_else(|| v.get("index").and_then(|idx| idx.as_u64()))
                .map(|v| v as usize)
        })
    };

    let tex_indices = MtoonTextureIndices {
        shade_color_texture: get_tex_index("shadeColorTexture"),
        emissive_texture: get_tex_index("emissiveTexture"),
        rim_texture: get_tex_index("rimTexture"),
        matcap_texture: get_tex_index("matcapTexture"),
        uv_anim_mask_texture: get_tex_index("uvAnimMaskTexture"),
    };

    let toon_params = ToonMaterialParams {
        ramp_threshold: (1.0 - shade_toony).max(0.01),
        shadow_softness: (1.0 - shade_toony) * 0.5,
        outline_width,
        outline_color,
    };

    let staged = MtoonStagedParams {
        shade_color,
        shade_shift,
        shade_toony,
        lit_color: [1.0, 1.0, 1.0, 1.0],
        gi_equalization,
        matcap_texture: None,
        rim_texture: None,
        rim_color,
        rim_lighting_mix,
        rim_fresnel_power,
        rim_lift,
        emissive_texture: None,
        emissive_color,
        outline_width_mode,
        outline_color,
        outline_width,
        uv_anim_mask_texture: None,
        uv_anim_scroll_x_speed: uv_anim_scroll_x,
        uv_anim_scroll_y_speed: uv_anim_scroll_y,
        uv_anim_rotation_speed: uv_anim_rotation,
    };

    (toon_params, staged, tex_indices)
}
