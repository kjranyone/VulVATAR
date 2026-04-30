//! VRM 1.x serde schema.
//!
//! Root extension keys: `VRMC_vrm` (humanoid + meta + expressions) and
//! `VRMC_springBone` (springs + colliders, parsed independently). The
//! 1.x JSON layout differs from 0.x in several load-bearing ways:
//!
//! * `humanoid.humanBones` is a map `{boneName: {node}}` instead of an
//!   array of `{bone, node}`.
//! * `expressions.preset` is a map with fixed camelCase keys; the *map
//!   key* is the expression name, so `Expression` itself has no `name`
//!   field.
//! * `expressions.custom` is a map `{name: Expression}`.
//! * `VRMC_springBone.springs[].joints[]` carries per-joint stiffness /
//!   drag / gravity in camelCase.
//! * Colliders may be sphere or capsule (0.x is sphere-only).

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct VrmExtension {
    pub humanoid: Humanoid,
    #[serde(default)]
    pub expressions: Option<Expressions>,
    #[serde(default, rename = "specVersion")]
    pub spec_version: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Meta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub authors: Option<Vec<String>>,
    #[serde(default, rename = "copyrightInformation")]
    pub copyright_information: Option<String>,
    #[serde(default, rename = "contactInformation")]
    pub contact_information: Option<String>,
    #[serde(default)]
    pub references: Option<Vec<String>>,
    #[serde(default, rename = "licenseUrl")]
    pub license_url: Option<String>,
    #[serde(default, rename = "thirdPartyLicenses")]
    pub third_party_licenses: Option<String>,
    /// VRM 1.0 stores the thumbnail as a glTF *image* index (not a
    /// texture index — this differs from 0.x).
    #[serde(default, rename = "thumbnailImage")]
    pub thumbnail_image: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Humanoid {
    #[serde(rename = "humanBones")]
    pub human_bones: HashMap<String, HumanBone>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HumanBone {
    pub node: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Expressions {
    #[serde(default)]
    pub preset: Option<Preset>,
    /// VRM 1.0: `custom` is a map {name: Expression}. The map key is the
    /// expression name; the `Expression` value has no `name` field.
    #[serde(default)]
    pub custom: Option<HashMap<String, Expression>>,
}

/// VRM 1.0 `preset` expression map. JSON keys are camelCase
/// (`blinkLeft`, `lookUp`, ...), so we keep snake_case Rust fields and
/// rely on `rename_all` to bridge the two. Each value is an expression
/// object; the *map key* is the name, so `Expression` has no `name`
/// field.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Preset {
    #[serde(default)]
    pub happy: Option<Expression>,
    #[serde(default)]
    pub angry: Option<Expression>,
    #[serde(default)]
    pub sad: Option<Expression>,
    #[serde(default)]
    pub relaxed: Option<Expression>,
    #[serde(default)]
    pub surprised: Option<Expression>,
    #[serde(default)]
    pub aa: Option<Expression>,
    #[serde(default)]
    pub ih: Option<Expression>,
    #[serde(default)]
    pub ou: Option<Expression>,
    #[serde(default)]
    pub ee: Option<Expression>,
    #[serde(default)]
    pub oh: Option<Expression>,
    #[serde(default)]
    pub blink: Option<Expression>,
    #[serde(default)]
    pub blink_left: Option<Expression>,
    #[serde(default)]
    pub blink_right: Option<Expression>,
    #[serde(default)]
    pub look_up: Option<Expression>,
    #[serde(default)]
    pub look_down: Option<Expression>,
    #[serde(default)]
    pub look_left: Option<Expression>,
    #[serde(default)]
    pub look_right: Option<Expression>,
    #[serde(default)]
    pub neutral: Option<Expression>,
}

impl Preset {
    /// Return `(canonical_name, expression)` pairs for every populated
    /// preset slot. The canonical name is the VRM 1.0 camelCase key
    /// (e.g. `"blinkLeft"`) and matches the identifiers produced by the
    /// tracking layer.
    pub fn iter_named(&self) -> Vec<(&'static str, &Expression)> {
        let mut out = Vec::new();
        macro_rules! push_if {
            ($field:ident, $name:literal) => {
                if let Some(ref e) = self.$field {
                    out.push(($name, e));
                }
            };
        }
        push_if!(happy, "happy");
        push_if!(angry, "angry");
        push_if!(sad, "sad");
        push_if!(relaxed, "relaxed");
        push_if!(surprised, "surprised");
        push_if!(aa, "aa");
        push_if!(ih, "ih");
        push_if!(ou, "ou");
        push_if!(ee, "ee");
        push_if!(oh, "oh");
        push_if!(blink, "blink");
        push_if!(blink_left, "blinkLeft");
        push_if!(blink_right, "blinkRight");
        push_if!(look_up, "lookUp");
        push_if!(look_down, "lookDown");
        push_if!(look_left, "lookLeft");
        push_if!(look_right, "lookRight");
        push_if!(neutral, "neutral");
        out
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Expression {
    #[serde(default, rename = "morphTargetBinds")]
    pub morph_target_binds: Vec<MorphTargetBind>,
    #[serde(default, rename = "isBinary")]
    pub is_binary: Option<bool>,
    #[serde(default, rename = "overrideBlink")]
    pub override_blink: Option<String>,
    #[serde(default, rename = "overrideLookAt")]
    pub override_look_at: Option<String>,
    #[serde(default, rename = "overrideMouth")]
    pub override_mouth: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MorphTargetBind {
    pub node: u32,
    pub index: u32,
    pub weight: f32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SpringBoneExtension {
    #[serde(default)]
    pub springs: Vec<Spring>,
    #[serde(default, rename = "colliderGroups")]
    pub collider_groups: Option<Vec<ColliderGroup>>,
    #[serde(default)]
    pub colliders: Option<Vec<Collider>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Spring {
    #[serde(default)]
    pub name: String,
    pub joints: Vec<Joint>,
    #[serde(default, rename = "colliderGroups")]
    pub collider_groups: Option<Vec<u32>>,
}

fn default_stiffness() -> f32 {
    1.0
}
fn default_gravity_power() -> f32 {
    0.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Joint {
    pub node: u32,
    #[serde(default, rename = "hitRadius")]
    pub hit_radius: Option<f32>,
    #[serde(default = "default_stiffness")]
    pub stiffness: f32,
    #[serde(default = "default_gravity_power", rename = "gravityPower")]
    pub gravity_power: f32,
    #[serde(default, rename = "gravityDir")]
    pub gravity_dir: Option<[f32; 3]>,
    #[serde(default, rename = "dragForce")]
    pub drag_force: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Collider {
    pub node: u32,
    #[serde(default)]
    pub shape: ColliderShape,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ColliderShape {
    #[serde(default)]
    pub sphere: Option<SphereShape>,
    #[serde(default)]
    pub capsule: Option<CapsuleShape>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SphereShape {
    #[serde(default)]
    pub radius: f32,
    #[serde(default)]
    pub offset: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CapsuleShape {
    #[serde(default)]
    pub radius: f32,
    #[serde(default)]
    pub offset: Option<[f32; 3]>,
    #[serde(default)]
    pub tail: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ColliderGroup {
    pub colliders: Vec<u32>,
}
