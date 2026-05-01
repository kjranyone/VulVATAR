//! VRM 0.x serde schema.
//!
//! Root extension key: `VRM`. The 0.x JSON layout is incompatible with
//! 1.x — `humanoid.humanBones` is an array of `{bone, node, ...}`,
//! `blendShapeMaster.blendShapeGroups` references meshes (not nodes)
//! with weights in `0..=100`, and `secondaryAnimation.boneGroups`
//! flattens spring parameters across each chain. The field is also
//! spelled `stiffiness` (sic) — preserved verbatim so files written by
//! Unity-era exporters continue to parse.
//!
//! Several fields here (`exporter_version`, `is_binary`, `comment`,
//! `center`, ...) are deserialised for schema completeness but never
//! read downstream. They document the on-disk shape; the
//! `#![allow(dead_code)]` below silences the lint at file scope rather
//! than per-field.

#![allow(dead_code)]

use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct VrmRoot {
    #[serde(default, rename = "exporterVersion")]
    pub exporter_version: Option<String>,
    #[serde(default, rename = "specVersion")]
    pub spec_version: Option<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
    #[serde(default)]
    pub humanoid: Option<Humanoid>,
    #[serde(default, rename = "blendShapeMaster")]
    pub blend_shape_master: Option<BlendShapeMaster>,
    #[serde(default, rename = "secondaryAnimation")]
    pub secondary_animation: Option<SecondaryAnimation>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Meta {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    /// VRM 0.x uses a single `author` string; multi-author files put
    /// names as a comma-separated list. The loader splits on commas.
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default, rename = "contactInformation")]
    pub contact_information: Option<String>,
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default, rename = "licenseName")]
    pub license_name: Option<String>,
    #[serde(default, rename = "otherLicenseUrl")]
    pub other_license_url: Option<String>,
    /// VRM 0.x stores the thumbnail as a glTF *texture* index. Some
    /// exporters emit `-1` for "no thumbnail" so the type is signed.
    #[serde(default)]
    pub texture: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Humanoid {
    #[serde(default, rename = "humanBones")]
    pub human_bones: Vec<HumanBone>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HumanBone {
    #[serde(default)]
    pub bone: String,
    /// Nullable in practice: some exporters emit `-1` for unmapped bones.
    #[serde(default)]
    pub node: i32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BlendShapeMaster {
    #[serde(default, rename = "blendShapeGroups")]
    pub blend_shape_groups: Vec<BlendShapeGroup>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlendShapeGroup {
    #[serde(default)]
    pub name: String,
    #[serde(default, rename = "presetName")]
    pub preset_name: Option<String>,
    #[serde(default)]
    pub binds: Vec<BlendShapeBind>,
    #[serde(default, rename = "isBinary")]
    pub is_binary: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlendShapeBind {
    #[serde(default)]
    pub mesh: i32,
    #[serde(default)]
    pub index: i32,
    /// VRM 0.x spec defines the weight range as 0..=100
    /// (SkinnedMeshRenderer.SetBlendShapeWeight). The loader rescales
    /// to 0..=1 when building `ExpressionMorphBind`.
    #[serde(default)]
    pub weight: f32,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SecondaryAnimation {
    #[serde(default, rename = "boneGroups")]
    pub bone_groups: Vec<BoneGroup>,
    #[serde(default, rename = "colliderGroups")]
    pub collider_groups: Vec<ColliderGroup>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BoneGroup {
    #[serde(default)]
    pub comment: Option<String>,
    /// Field is spelled `stiffiness` (double-i) in the VRM 0.x spec.
    /// Preserved verbatim so files written by Unity-era exporters
    /// continue to parse.
    #[serde(default, rename = "stiffiness")]
    pub stiffiness: f32,
    #[serde(default, rename = "gravityPower")]
    pub gravity_power: f32,
    #[serde(default, rename = "gravityDir")]
    pub gravity_dir: Option<XyzVec>,
    #[serde(default, rename = "dragForce")]
    pub drag_force: f32,
    #[serde(default)]
    pub center: Option<i32>,
    #[serde(default, rename = "hitRadius")]
    pub hit_radius: f32,
    /// Each entry is a chain-root node index; children form the chain.
    #[serde(default)]
    pub bones: Vec<i32>,
    /// Indices into the sibling `colliderGroups` array.
    #[serde(default, rename = "colliderGroups")]
    pub collider_groups: Vec<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct XyzVec {
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
    #[serde(default)]
    pub z: f32,
}

impl XyzVec {
    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ColliderGroup {
    #[serde(default)]
    pub node: i32,
    #[serde(default)]
    pub colliders: Vec<Collider>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Collider {
    #[serde(default)]
    pub offset: Option<XyzVec>,
    #[serde(default)]
    pub radius: f32,
}
