//! VRM extension parsing — `VRMC_vrm` / `VRMC_springBone` (1.x) and
//! the legacy `VRM` root (0.x). Pure orchestration: each function takes
//! a deserialised `v0::*` or `v1::*` schema (see sibling modules) and
//! produces engine-side asset types (`HumanoidMap`, `ExpressionAssetSet`,
//! `SpringBoneAsset`, `ColliderAsset`, ...).
//!
//! The 0.x parser also handles the comma-split author list and the
//! `0..=100` → `0..=1` weight rescale; the 1.x parser handles the
//! preset-key / map-key naming inversion. Both translate to the same
//! `VrmMeta::spec_version` enum so downstream code is version-agnostic.
//!
//! Visibility: only [`parse_vrm_extensions`], [`resolve_thumbnail`],
//! [`ParsedVrm`], [`ThumbnailHint`] are reached from the loader in
//! `mod.rs`. The unit-test synthetic-extension cases also reach in
//! through [`parse_v0_extensions`] / [`parse_v1_extensions`].

use std::collections::HashMap;

use log::warn;

use super::{v0, v1};
use super::VrmLoadError;
use crate::asset::*;

// ---------------------------------------------------------------------------
// VRM extension dispatch — picks the right per-version parser.
// ---------------------------------------------------------------------------

pub(super) struct ParsedVrm {
    pub(super) meta: VrmMeta,
    pub(super) humanoid: Option<HumanoidMap>,
    pub(super) expressions: ExpressionAssetSet,
    pub(super) springs: Vec<SpringBoneAsset>,
    pub(super) colliders: Vec<ColliderAsset>,
    /// Where in the glTF document the VRM file says its thumbnail lives.
    /// Resolved into raw bytes after `parse_vrm_extensions` returns,
    /// where the caller has access to `gltf::Document` + the binary blob.
    pub(super) thumbnail_hint: ThumbnailHint,
}

impl ParsedVrm {
    fn empty() -> Self {
        Self {
            meta: VrmMeta::default(),
            humanoid: None,
            expressions: ExpressionAssetSet {
                expressions: vec![],
            },
            springs: vec![],
            colliders: vec![],
            thumbnail_hint: ThumbnailHint::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct ThumbnailHint {
    /// VRM 1.0: index into `doc.images()` (glTF *image*, not texture).
    pub(super) image_index: Option<usize>,
    /// VRM 0.x: index into `doc.textures()`. The texture's `.source()` is
    /// the image index we ultimately read.
    pub(super) texture_index: Option<usize>,
}

pub(super) fn parse_vrm_extensions(
    ext_map: &HashMap<String, serde_json::Value>,
    nodes: &[SkeletonNode],
    mesh_to_node: &HashMap<usize, usize>,
) -> Result<ParsedVrm, VrmLoadError> {
    // When both VRMC_vrm and VRM are present (some exporters bundle legacy
    // data for backwards compatibility), VRM 1.x is authoritative. Fall back
    // to the other path only if parsing the primary one fails outright.
    if let Some(raw) = ext_map.get("VRMC_vrm") {
        match parse_v1_extensions(raw, ext_map.get("VRMC_springBone")) {
            Ok(parsed) => return Ok(parsed),
            Err(e) => {
                warn!(
                    "VRM 1.x extension parse failed ({e}); falling back to VRM 0.x if available"
                );
            }
        }
    }
    if let Some(raw) = ext_map.get("VRM") {
        return parse_v0_extensions(raw, nodes, mesh_to_node);
    }
    // No recognised VRM extension: return glTF-as-avatar with no retargeting.
    warn!("no VRM extension found in glTF root; loading as glTF-only avatar");
    Ok(ParsedVrm::empty())
}

// ---------------------------------------------------------------------------
// VRM 1.x parser
// ---------------------------------------------------------------------------

pub(super) fn parse_v1_extensions(
    vrmc_vrm_raw: &serde_json::Value,
    vrmc_spring_bone_raw: Option<&serde_json::Value>,
) -> Result<ParsedVrm, VrmLoadError> {
    let vrmc_vrm: v1::VrmExtension = serde_json::from_value(vrmc_vrm_raw.clone())
        .map_err(|e| {
            warn!("VRMC_vrm deserialisation failed: {e}");
            VrmLoadError::JsonParse(e)
        })?;

    let (meta, thumbnail_hint) = build_v1_meta(&vrmc_vrm);
    let humanoid = Some(build_v1_humanoid(&vrmc_vrm.humanoid)?);
    let expressions = vrmc_vrm
        .expressions
        .as_ref()
        .map(build_v1_expressions)
        .unwrap_or_else(|| ExpressionAssetSet {
            expressions: vec![],
        });

    let (springs, colliders) = match vrmc_spring_bone_raw {
        Some(raw) => match serde_json::from_value::<v1::SpringBoneExtension>(raw.clone()) {
            Ok(sb) => build_v1_springs(&sb),
            Err(e) => {
                warn!("VRMC_springBone parse failed: {e}");
                (vec![], vec![])
            }
        },
        None => (vec![], vec![]),
    };

    Ok(ParsedVrm {
        meta,
        humanoid,
        expressions,
        springs,
        colliders,
        thumbnail_hint,
    })
}

/// Resolve a [`ThumbnailHint`] to the encoded image bytes (PNG / JPEG)
/// embedded in the VRM. Returns `None` when the hint is empty, the index
/// is out of range, the image source is an external URI we can't read,
/// or the buffer slice is unreachable. Logs but does not error — a
/// missing thumbnail must not fail the avatar load.
pub(super) fn resolve_thumbnail(
    doc: &gltf::Document,
    blob: Option<&[u8]>,
    hint: &ThumbnailHint,
) -> Option<EmbeddedThumbnail> {
    let image = if let Some(image_idx) = hint.image_index {
        doc.images().nth(image_idx)
    } else if let Some(tex_idx) = hint.texture_index {
        doc.textures().nth(tex_idx).map(|tex| tex.source())
    } else {
        return None;
    }?;

    match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            let start = view.offset();
            let end = start + view.length();
            let bytes = blob.and_then(|b| b.get(start..end))?;
            Some(EmbeddedThumbnail {
                mime_type: mime_type.to_string(),
                bytes: bytes.to_vec(),
            })
        }
        gltf::image::Source::Uri { uri, .. } => {
            // VRM thumbnails are virtually always embedded in the .glb
            // binary chunk; URI references are unusual but legal here.
            log::warn!(
                "VRM thumbnail at uri '{uri}' is external; skipping (no source-relative resolver)",
            );
            None
        }
    }
}

fn build_v1_meta(ext: &v1::VrmExtension) -> (VrmMeta, ThumbnailHint) {
    let meta = ext.meta.clone().unwrap_or_default();
    let hint = ThumbnailHint {
        image_index: meta.thumbnail_image.map(|i| i as usize),
        texture_index: None,
    };
    let vrm_meta = VrmMeta {
        spec_version: VrmSpecVersion::V1,
        spec_version_raw: ext.spec_version.clone(),
        title: meta.name,
        authors: meta.authors.unwrap_or_default(),
        model_version: meta.version,
        contact_information: meta.contact_information,
        references: meta.references.unwrap_or_default(),
        license: meta.license_url,
        copyright_information: meta.copyright_information,
        thumbnail: None,
    };
    (vrm_meta, hint)
}

fn build_v1_humanoid(humanoid: &v1::Humanoid) -> Result<HumanoidMap, VrmLoadError> {
    let mut bone_map = HashMap::new();
    for (bone_name, bone_def) in &humanoid.human_bones {
        // Unknown bones (fingers, toes, jaw, eyes, ...) are accepted by the
        // VRM 1.0 spec but ignored by this retargeting pipeline. Skipping them
        // is friendlier than hard-erroring the entire load.
        if let Some(humanoid_bone) = parse_humanoid_bone(bone_name) {
            bone_map.insert(humanoid_bone, NodeId(bone_def.node as u64));
        }
    }
    Ok(HumanoidMap { bone_map })
}

fn build_v1_expressions(section: &v1::Expressions) -> ExpressionAssetSet {
    let mut expressions = Vec::new();

    if let Some(ref preset) = section.preset {
        for (name, expr) in preset.iter_named() {
            expressions.push(v1_expression_to_def(name, expr));
        }
    }
    if let Some(ref custom) = section.custom {
        for (name, expr) in custom {
            expressions.push(v1_expression_to_def(name, expr));
        }
    }

    ExpressionAssetSet { expressions }
}

fn v1_expression_to_def(name: &str, expr: &v1::Expression) -> ExpressionDef {
    let morph_binds = expr
        .morph_target_binds
        .iter()
        .map(|b| ExpressionMorphBind {
            node_index: b.node as usize,
            morph_target_index: b.index as usize,
            weight: b.weight,
        })
        .collect();

    ExpressionDef {
        name: name.to_string(),
        weight: 0.0,
        morph_binds,
    }
}

fn build_v1_springs(ext: &v1::SpringBoneExtension) -> (Vec<SpringBoneAsset>, Vec<ColliderAsset>) {
    // Colliders are numbered sequentially starting at 1 so that references from
    // spring collider_groups can resolve back via the index map below.
    let mut next_collider_id: u64 = 1;
    let colliders: Vec<ColliderAsset> = ext
        .colliders
        .as_ref()
        .map(|defs| {
            defs.iter()
                .map(|c| {
                    let (shape, offset) = if let Some(ref sphere) = c.shape.sphere {
                        (
                            ColliderShape::Sphere {
                                radius: sphere.radius,
                            },
                            sphere.offset.unwrap_or([0.0; 3]),
                        )
                    } else if let Some(ref capsule) = c.shape.capsule {
                        let offset = capsule.offset.unwrap_or([0.0; 3]);
                        let height = if let Some(tail) = capsule.tail {
                            let dx = tail[0] - offset[0];
                            let dy = tail[1] - offset[1];
                            let dz = tail[2] - offset[2];
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        } else {
                            0.0
                        };
                        (
                            ColliderShape::Capsule {
                                radius: capsule.radius,
                                height,
                            },
                            offset,
                        )
                    } else {
                        (ColliderShape::Sphere { radius: 0.0 }, [0.0; 3])
                    };
                    let id = ColliderId(next_collider_id);
                    next_collider_id += 1;
                    ColliderAsset {
                        id,
                        node: NodeId(c.node as u64),
                        shape,
                        offset,
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let collider_id_map: HashMap<u32, ColliderId> = ext
        .colliders
        .as_ref()
        .map(|defs| {
            defs.iter()
                .enumerate()
                .map(|(i, _)| (i as u32, ColliderId(i as u64 + 1)))
                .collect()
        })
        .unwrap_or_default();

    let collider_group_map: HashMap<u32, Vec<ColliderId>> = ext
        .collider_groups
        .as_ref()
        .map(|groups| {
            groups
                .iter()
                .enumerate()
                .map(|(gi, group)| {
                    let ids: Vec<ColliderId> = group
                        .colliders
                        .iter()
                        .filter_map(|ci| collider_id_map.get(ci).copied())
                        .collect();
                    (gi as u32, ids)
                })
                .collect()
        })
        .unwrap_or_default();

    let springs: Vec<SpringBoneAsset> = ext
        .springs
        .iter()
        .map(|spring| {
            let chain_root = spring
                .joints
                .first()
                .map(|j| NodeId(j.node as u64))
                .unwrap_or(NodeId(0));
            let joints: Vec<NodeId> = spring
                .joints
                .iter()
                .map(|j| NodeId(j.node as u64))
                .collect();
            let collider_refs: Vec<ColliderRef> = spring
                .collider_groups
                .as_ref()
                .map(|groups| {
                    groups
                        .iter()
                        .filter_map(|gi| {
                            collider_group_map
                                .get(gi)
                                .map(|ids| ids.iter().map(|&id| ColliderRef { id }))
                        })
                        .flatten()
                        .collect()
                })
                .unwrap_or_default();

            let first = spring.joints.first();
            let stiffness = first.map(|j| j.stiffness).unwrap_or(1.0);
            let drag_force = first.and_then(|j| j.drag_force).unwrap_or(0.5);
            let gravity_dir = first.and_then(|j| j.gravity_dir).unwrap_or([0.0, -1.0, 0.0]);
            let gravity_power = first.map(|j| j.gravity_power).unwrap_or(0.0);
            let radius = first.and_then(|j| j.hit_radius).unwrap_or(0.05);

            let joint_stiffness: Vec<f32> = spring.joints.iter().map(|j| j.stiffness).collect();
            let joint_drag: Vec<f32> = spring
                .joints
                .iter()
                .map(|j| j.drag_force.unwrap_or(drag_force))
                .collect();
            let joint_gravity_power: Vec<f32> =
                spring.joints.iter().map(|j| j.gravity_power).collect();

            SpringBoneAsset {
                chain_root,
                joints,
                stiffness,
                drag_force,
                gravity_dir,
                gravity_power,
                radius,
                collider_refs,
                joint_stiffness,
                joint_drag,
                joint_gravity_power,
            }
        })
        .collect();

    (springs, colliders)
}

// ---------------------------------------------------------------------------
// VRM 0.x parser
// ---------------------------------------------------------------------------

pub(super) fn parse_v0_extensions(
    vrm_raw: &serde_json::Value,
    nodes: &[SkeletonNode],
    mesh_to_node: &HashMap<usize, usize>,
) -> Result<ParsedVrm, VrmLoadError> {
    let root: v0::VrmRoot = serde_json::from_value(vrm_raw.clone()).map_err(|e| {
        warn!("VRM 0.x root extension parse failed: {e}");
        VrmLoadError::JsonParse(e)
    })?;

    let (meta, thumbnail_hint) = build_v0_meta(&root);
    let humanoid = root.humanoid.as_ref().map(build_v0_humanoid);
    let expressions = root
        .blend_shape_master
        .as_ref()
        .map(|bsm| build_v0_expressions(bsm, mesh_to_node))
        .unwrap_or_else(|| ExpressionAssetSet {
            expressions: vec![],
        });
    let (springs, colliders) = root
        .secondary_animation
        .as_ref()
        .map(|sa| build_v0_springs(sa, nodes))
        .unwrap_or_else(|| (vec![], vec![]));

    Ok(ParsedVrm {
        meta,
        humanoid,
        expressions,
        springs,
        colliders,
        thumbnail_hint,
    })
}

fn build_v0_meta(root: &v0::VrmRoot) -> (VrmMeta, ThumbnailHint) {
    let meta = root.meta.clone().unwrap_or_default();
    // VRM 0.x `author` is a single string; multi-author files put names as a
    // comma-separated list. Split so Inspector and library entries treat each
    // as its own name.
    let authors: Vec<String> = meta
        .author
        .map(|s| {
            s.split(',')
                .map(|a| a.trim().to_string())
                .filter(|a| !a.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let references = meta.reference.into_iter().collect();
    // Some VRM 0.x exporters emit `-1` for "no thumbnail"; guard against that.
    let texture_index = meta
        .texture
        .filter(|i| *i >= 0)
        .map(|i| i as usize);
    let hint = ThumbnailHint {
        image_index: None,
        texture_index,
    };
    let vrm_meta = VrmMeta {
        spec_version: VrmSpecVersion::V0,
        spec_version_raw: root.spec_version.clone(),
        title: meta.title,
        authors,
        model_version: meta.version,
        contact_information: meta.contact_information,
        references,
        license: meta.license_name.or(meta.other_license_url),
        copyright_information: None,
        thumbnail: None,
    };
    (vrm_meta, hint)
}

fn build_v0_humanoid(humanoid: &v0::Humanoid) -> HumanoidMap {
    let mut bone_map = HashMap::new();
    for entry in &humanoid.human_bones {
        if entry.node < 0 {
            continue;
        }
        if let Some(bone) = parse_humanoid_bone(&entry.bone) {
            bone_map.insert(bone, NodeId(entry.node as u64));
        }
    }
    HumanoidMap { bone_map }
}

fn build_v0_expressions(
    master: &v0::BlendShapeMaster,
    mesh_to_node: &HashMap<usize, usize>,
) -> ExpressionAssetSet {
    let mut expressions = Vec::with_capacity(master.blend_shape_groups.len());

    for group in &master.blend_shape_groups {
        // Prefer the preset name (mapped to the VRM 1.0 canonical key) so the
        // tracking layer finds the right slot. Fall back to the group name for
        // unknown/custom blendshapes.
        let name = group
            .preset_name
            .as_deref()
            .and_then(v0_preset_name_to_v1)
            .map(|s| s.to_string())
            .unwrap_or_else(|| group.name.clone());

        if name.is_empty() {
            continue;
        }

        let morph_binds: Vec<ExpressionMorphBind> = group
            .binds
            .iter()
            .filter_map(|bind| {
                if bind.mesh < 0 || bind.index < 0 {
                    return None;
                }
                let node_index = mesh_to_node.get(&(bind.mesh as usize))?;
                Some(ExpressionMorphBind {
                    node_index: *node_index,
                    morph_target_index: bind.index as usize,
                    // VRM 0.x weights are 0..=100; rescale to 0..=1.
                    weight: bind.weight / 100.0,
                })
            })
            .collect();

        expressions.push(ExpressionDef {
            name,
            weight: 0.0,
            morph_binds,
        });
    }

    ExpressionAssetSet { expressions }
}

/// Translate a VRM 0.x `presetName` value to the VRM 1.0 canonical expression
/// identifier used throughout the tracking/retargeting pipeline. Returns
/// `None` for unknown presets and for `"unknown"` (so callers fall back to
/// the group's `name`).
fn v0_preset_name_to_v1(preset: &str) -> Option<&'static str> {
    match preset {
        "neutral" => Some("neutral"),
        // Mouth / vowels
        "a" => Some("aa"),
        "i" => Some("ih"),
        "u" => Some("ou"),
        "e" => Some("ee"),
        "o" => Some("oh"),
        // Eye blinks
        "blink" => Some("blink"),
        "blink_l" => Some("blinkLeft"),
        "blink_r" => Some("blinkRight"),
        // Emotions
        "joy" => Some("happy"),
        "angry" => Some("angry"),
        "sorrow" => Some("sad"),
        "fun" => Some("relaxed"),
        // Gaze
        "lookup" => Some("lookUp"),
        "lookdown" => Some("lookDown"),
        "lookleft" => Some("lookLeft"),
        "lookright" => Some("lookRight"),
        _ => None,
    }
}

fn build_v0_springs(
    sa: &v0::SecondaryAnimation,
    nodes: &[SkeletonNode],
) -> (Vec<SpringBoneAsset>, Vec<ColliderAsset>) {
    // Flatten 0.x colliderGroups (which bundle spheres under a single node)
    // into per-group ColliderAsset lists that can be referenced by index.
    let mut next_collider_id: u64 = 1;
    let mut all_colliders: Vec<ColliderAsset> = Vec::new();
    let mut group_to_ids: Vec<Vec<ColliderId>> = Vec::with_capacity(sa.collider_groups.len());

    for group in &sa.collider_groups {
        if group.node < 0 {
            group_to_ids.push(vec![]);
            continue;
        }
        let node = NodeId(group.node as u64);
        let mut ids = Vec::with_capacity(group.colliders.len());
        for c in &group.colliders {
            let offset = c.offset.as_ref().map(|v| v.to_array()).unwrap_or([0.0; 3]);
            let id = ColliderId(next_collider_id);
            next_collider_id += 1;
            all_colliders.push(ColliderAsset {
                id,
                node,
                shape: ColliderShape::Sphere { radius: c.radius },
                offset,
            });
            ids.push(id);
        }
        group_to_ids.push(ids);
    }

    let mut springs: Vec<SpringBoneAsset> = Vec::new();
    for group in &sa.bone_groups {
        let collider_refs: Vec<ColliderRef> = group
            .collider_groups
            .iter()
            .filter_map(|gi| group_to_ids.get(*gi as usize))
            .flat_map(|ids| ids.iter().map(|&id| ColliderRef { id }))
            .collect();

        let gravity_dir = group
            .gravity_dir
            .as_ref()
            .map(|v| v.to_array())
            .unwrap_or([0.0, -1.0, 0.0]);
        let radius = if group.hit_radius > 0.0 {
            group.hit_radius
        } else {
            0.02
        };

        for &root_idx in &group.bones {
            if root_idx < 0 {
                continue;
            }
            let chain = walk_v0_chain(nodes, root_idx as usize);
            if chain.is_empty() {
                continue;
            }
            let chain_root = chain[0];
            let joint_count = chain.len();
            springs.push(SpringBoneAsset {
                chain_root,
                joints: chain,
                stiffness: group.stiffiness,
                drag_force: group.drag_force,
                gravity_dir,
                gravity_power: group.gravity_power,
                radius,
                collider_refs: collider_refs.clone(),
                // 0.x shares parameters across the whole chain, so broadcast
                // the group values into per-joint slots.
                joint_stiffness: vec![group.stiffiness; joint_count],
                joint_drag: vec![group.drag_force; joint_count],
                joint_gravity_power: vec![group.gravity_power; joint_count],
            });
        }
    }

    (springs, all_colliders)
}

/// Walk a VRM 0.x spring chain starting at `root_idx`, following the first
/// child at each step. Stops when a node has no children. Branching chains
/// are flattened to the primary strand — the common case (hair, skirt
/// strands) has a single child per node.
fn walk_v0_chain(nodes: &[SkeletonNode], root_idx: usize) -> Vec<NodeId> {
    let mut chain = Vec::new();
    let mut cursor = root_idx;
    loop {
        if cursor >= nodes.len() {
            break;
        }
        chain.push(NodeId(cursor as u64));
        let next = nodes[cursor].children.first().copied();
        match next {
            Some(NodeId(n)) => {
                let n = n as usize;
                // Guard against accidental cycles (malformed input).
                if n == cursor || chain.iter().any(|NodeId(c)| *c as usize == n) {
                    break;
                }
                cursor = n;
            }
            None => break,
        }
    }
    chain
}
fn parse_humanoid_bone(name: &str) -> Option<HumanoidBone> {
    match name {
        "hips" => Some(HumanoidBone::Hips),
        "spine" => Some(HumanoidBone::Spine),
        "chest" => Some(HumanoidBone::Chest),
        "upperChest" => Some(HumanoidBone::UpperChest),
        "neck" => Some(HumanoidBone::Neck),
        "head" => Some(HumanoidBone::Head),
        "leftShoulder" => Some(HumanoidBone::LeftShoulder),
        "leftUpperArm" => Some(HumanoidBone::LeftUpperArm),
        "leftLowerArm" => Some(HumanoidBone::LeftLowerArm),
        "leftHand" => Some(HumanoidBone::LeftHand),
        "rightShoulder" => Some(HumanoidBone::RightShoulder),
        "rightUpperArm" => Some(HumanoidBone::RightUpperArm),
        "rightLowerArm" => Some(HumanoidBone::RightLowerArm),
        "rightHand" => Some(HumanoidBone::RightHand),
        "leftUpperLeg" => Some(HumanoidBone::LeftUpperLeg),
        "leftLowerLeg" => Some(HumanoidBone::LeftLowerLeg),
        "leftFoot" => Some(HumanoidBone::LeftFoot),
        "rightUpperLeg" => Some(HumanoidBone::RightUpperLeg),
        "rightLowerLeg" => Some(HumanoidBone::RightLowerLeg),
        "rightFoot" => Some(HumanoidBone::RightFoot),

        // ----- Fingers -----
        "leftThumbProximal" => Some(HumanoidBone::LeftThumbProximal),
        "leftThumbIntermediate" | "leftThumbMetacarpal" => {
            Some(HumanoidBone::LeftThumbIntermediate)
        }
        "leftThumbDistal" => Some(HumanoidBone::LeftThumbDistal),
        "leftIndexProximal" => Some(HumanoidBone::LeftIndexProximal),
        "leftIndexIntermediate" => Some(HumanoidBone::LeftIndexIntermediate),
        "leftIndexDistal" => Some(HumanoidBone::LeftIndexDistal),
        "leftMiddleProximal" => Some(HumanoidBone::LeftMiddleProximal),
        "leftMiddleIntermediate" => Some(HumanoidBone::LeftMiddleIntermediate),
        "leftMiddleDistal" => Some(HumanoidBone::LeftMiddleDistal),
        "leftRingProximal" => Some(HumanoidBone::LeftRingProximal),
        "leftRingIntermediate" => Some(HumanoidBone::LeftRingIntermediate),
        "leftRingDistal" => Some(HumanoidBone::LeftRingDistal),
        "leftLittleProximal" => Some(HumanoidBone::LeftLittleProximal),
        "leftLittleIntermediate" => Some(HumanoidBone::LeftLittleIntermediate),
        "leftLittleDistal" => Some(HumanoidBone::LeftLittleDistal),

        "rightThumbProximal" => Some(HumanoidBone::RightThumbProximal),
        "rightThumbIntermediate" | "rightThumbMetacarpal" => {
            Some(HumanoidBone::RightThumbIntermediate)
        }
        "rightThumbDistal" => Some(HumanoidBone::RightThumbDistal),
        "rightIndexProximal" => Some(HumanoidBone::RightIndexProximal),
        "rightIndexIntermediate" => Some(HumanoidBone::RightIndexIntermediate),
        "rightIndexDistal" => Some(HumanoidBone::RightIndexDistal),
        "rightMiddleProximal" => Some(HumanoidBone::RightMiddleProximal),
        "rightMiddleIntermediate" => Some(HumanoidBone::RightMiddleIntermediate),
        "rightMiddleDistal" => Some(HumanoidBone::RightMiddleDistal),
        "rightRingProximal" => Some(HumanoidBone::RightRingProximal),
        "rightRingIntermediate" => Some(HumanoidBone::RightRingIntermediate),
        "rightRingDistal" => Some(HumanoidBone::RightRingDistal),
        "rightLittleProximal" => Some(HumanoidBone::RightLittleProximal),
        "rightLittleIntermediate" => Some(HumanoidBone::RightLittleIntermediate),
        "rightLittleDistal" => Some(HumanoidBone::RightLittleDistal),

        _ => None,
    }
}
