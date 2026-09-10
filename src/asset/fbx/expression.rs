use std::collections::HashMap;

use crate::asset::{ExpressionAssetSet, ExpressionDef, ExpressionMorphBind};

/// Map raw FBX / VRChat blendshape names to standard VRM expression presets.
pub fn map_preset_expression(raw_name: &str) -> Option<&'static str> {
    let norm = raw_name.to_lowercase();
    let norm = norm.strip_prefix("blendshape.").unwrap_or(&norm);
    let norm = norm.strip_prefix("blendshape_").unwrap_or(norm);

    match norm {
        // Visemes
        "vrc.v_aa" | "v_aa" | "viseme_aa" | "mouth_a" | "a" | "aa" | "ah" => Some("aa"),
        "vrc.v_ih" | "v_ih" | "viseme_ih" | "mouth_i" | "i" | "ih" => Some("ih"),
        "vrc.v_ou" | "v_ou" | "viseme_ou" | "mouth_u" | "u" | "ou" | "oo" => Some("ou"),
        "vrc.v_e" | "v_e" | "viseme_e" | "mouth_e" | "e" | "ee" | "eh" => Some("ee"),
        "vrc.v_oh" | "v_oh" | "viseme_oh" | "mouth_o" | "o" | "oh" => Some("oh"),

        // Blinks
        "eye_close" | "eye_close_both" | "blink" | "vrc.blink" | "close_eyes" | "eyes_close"
        | "eye_blink" => Some("blink"),
        "eye_close_l" | "eye_close_left" | "blink_l" | "blink_left" | "vrc.blink_l"
        | "eye_blink_l" => Some("blinkLeft"),
        "eye_close_r" | "eye_close_right" | "blink_r" | "blink_right" | "vrc.blink_r"
        | "eye_blink_r" => Some("blinkRight"),

        // Emotions
        "happy" | "smile" | "joy" | "fun" => Some("happy"),
        "angry" | "anger" | "rage" => Some("angry"),
        "sad" | "sorrow" => Some("sad"),
        "relaxed" | "calm" => Some("relaxed"),
        "surprised" | "shock" => Some("surprised"),
        "neutral" => Some("neutral"),

        // Look
        "look_up" | "eye_up" | "lookup" => Some("lookUp"),
        "look_down" | "eye_down" | "lookdown" => Some("lookDown"),
        "look_left" | "eye_left" | "lookleft" => Some("lookLeft"),
        "look_right" | "eye_right" | "lookright" => Some("lookRight"),

        _ => None,
    }
}

/// Helper to build `ExpressionAssetSet` from collected morph bindings.
///
/// `morph_targets`: `(node_idx, morph_target_index, channel_name)`
pub fn build_expressions(
    all_morphs: &[(usize, usize, String)],
) -> ExpressionAssetSet {
    let mut expressions_map: HashMap<String, Vec<ExpressionMorphBind>> = HashMap::new();

    for &(node_idx, morph_target_index, ref name) in all_morphs {
        let bind = ExpressionMorphBind {
            node_index: node_idx,
            morph_target_index,
            weight: 1.0,
        };

        // 1. Register under exact blendshape name
        expressions_map
            .entry(name.clone())
            .or_default()
            .push(bind.clone());

        // 2. If it maps to a standard VRM preset, register under preset name as well
        if let Some(preset) = map_preset_expression(name) {
            expressions_map
                .entry(preset.to_string())
                .or_default()
                .push(bind);
        }
    }

    let mut expressions: Vec<ExpressionDef> = expressions_map
        .into_iter()
        .map(|(name, morph_binds)| ExpressionDef {
            name,
            weight: 0.0,
            morph_binds,
        })
        .collect();

    // Sort alphabetically so the list is stable and easy to navigate
    expressions.sort_by(|a, b| a.name.cmp(&b.name));

    ExpressionAssetSet { expressions }
}
