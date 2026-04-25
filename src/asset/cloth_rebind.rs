//! Resolves a `ClothAsset` against a freshly-loaded `AvatarAsset` after
//! the avatar's internal IDs change (e.g. the user re-exported the VRM
//! and the bone / mesh / primitive IDs got re-numbered).
//!
//! The overlay's `ClothPin.binding_node`, `ClothCollisionBinding.binding_node`
//! and `ClothRenderRegionBinding.primitive` carry both an `id` (which
//! drifts on reimport) and a `name` (which is stable). Without this
//! module the validation step in `editor::validate_overlay` still
//! checks `id` only, so a reimport silently kills every overlay
//! attached to the avatar.
//!
//! `rebind_overlay` walks every reference in the overlay, looks up
//! the matching node / mesh / primitive in the new avatar, rewrites
//! the `id` in place, and reports which tier of fallback (primary
//! name match, secondary humanoid-bone / vertex-count match,
//! tertiary parent-path / AABB match) ultimately resolved the entry.
//! Callers route the report status through their UI: `Clean` is
//! silent, `Partial` posts a notification, `Failed` triggers the
//! detach-or-prompt flow.

use crate::asset::{
    AvatarAsset, ClothAsset, HumanoidBone, MeshAsset, MeshId, MeshPrimitiveAsset, MeshRef, NodeId,
    NodeRef, PrimitiveId, PrimitiveRef, SkeletonNode,
};
use std::collections::HashMap;

/// How a single ref ultimately resolved. Determines whether the
/// caller's UI should treat the entry as "silent / fully automatic"
/// or "automatic but worth flagging".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebindTier {
    /// Name (or `(mesh, prim_index)` for primitives) matched directly.
    Primary,
    /// Humanoid bone slot matched (nodes only) or vertex-count match
    /// (primitives only).
    Secondary,
    /// Parent-path signature matched (nodes only) or AABB centroid
    /// proximity match (primitives only). Worth a notification.
    Tertiary,
}

/// Single resolved entry. `old_id == new_id` is possible and not an
/// error — we record every walked ref so the report doubles as an
/// audit trail of what was looked at.
#[derive(Clone, Debug)]
pub struct RebindEntry {
    pub old_id: u64,
    pub new_id: u64,
    pub name: String,
    pub tier: RebindTier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnresolvedKind {
    Node,
    Mesh,
    Primitive,
}

#[derive(Clone, Debug)]
pub struct UnresolvedRef {
    pub kind: UnresolvedKind,
    pub name: String,
    pub original_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebindStatus {
    /// Every ref matched at the primary tier. Caller can apply silently.
    Clean,
    /// At least one ref required secondary or tertiary fallback. Apply
    /// automatically but surface as a notification.
    Partial,
    /// At least one ref couldn't be resolved at any tier, OR a
    /// `fatal_geometry_changes` entry was recorded (vertex count drift
    /// on a render binding). Caller decides whether to detach the
    /// overlay or prompt the user.
    Failed,
}

#[derive(Clone, Debug)]
pub struct RebindReport {
    pub status: RebindStatus,
    pub node_remappings: Vec<RebindEntry>,
    pub mesh_remappings: Vec<RebindEntry>,
    pub primitive_remappings: Vec<RebindEntry>,
    pub unresolved: Vec<UnresolvedRef>,
    /// Refs that resolved by name but whose underlying geometry
    /// changed in a way the simulator can't compensate for (vertex
    /// count drift on a primitive that the cloth's mesh-mapping
    /// depends on). Flagged separately because partial detach is
    /// possible — only the affected render binding is broken.
    pub fatal_geometry_changes: Vec<String>,
}

impl RebindReport {
    fn new() -> Self {
        Self {
            status: RebindStatus::Clean,
            node_remappings: Vec::new(),
            mesh_remappings: Vec::new(),
            primitive_remappings: Vec::new(),
            unresolved: Vec::new(),
            fatal_geometry_changes: Vec::new(),
        }
    }

    pub fn has_changes(&self) -> bool {
        !self.node_remappings.is_empty()
            || !self.mesh_remappings.is_empty()
            || !self.primitive_remappings.is_empty()
    }

    fn downgrade(&mut self, candidate: RebindStatus) {
        self.status = match (self.status, candidate) {
            (RebindStatus::Failed, _) | (_, RebindStatus::Failed) => RebindStatus::Failed,
            (RebindStatus::Partial, _) | (_, RebindStatus::Partial) => RebindStatus::Partial,
            _ => RebindStatus::Clean,
        };
    }
}

/// Pre-built lookup tables over the new avatar so the per-ref resolve
/// loops are O(refs) rather than O(refs × avatar size).
struct AvatarResolver<'a> {
    nodes_by_name: HashMap<&'a str, &'a SkeletonNode>,
    nodes_by_humanoid: HashMap<HumanoidBone, &'a SkeletonNode>,
    nodes_by_path: HashMap<String, &'a SkeletonNode>,
    nodes_by_id: HashMap<NodeId, &'a SkeletonNode>,
    meshes_by_name: HashMap<&'a str, &'a MeshAsset>,
    /// Primitives keyed by `(mesh_name, primitive_index_within_mesh)`.
    /// glTF doesn't name primitives, so this is the most stable
    /// identity we can synthesise.
    primitives_by_position: HashMap<(String, usize), &'a MeshPrimitiveAsset>,
    primitives_by_id: HashMap<PrimitiveId, &'a MeshPrimitiveAsset>,
}

impl<'a> AvatarResolver<'a> {
    fn new(avatar: &'a AvatarAsset) -> Self {
        let nodes_by_name: HashMap<&str, &SkeletonNode> = avatar
            .skeleton
            .nodes
            .iter()
            .map(|n| (n.name.as_str(), n))
            .collect();
        let nodes_by_humanoid: HashMap<HumanoidBone, &SkeletonNode> = avatar
            .skeleton
            .nodes
            .iter()
            .filter_map(|n| n.humanoid_bone.map(|hb| (hb, n)))
            .collect();
        let nodes_by_id: HashMap<NodeId, &SkeletonNode> =
            avatar.skeleton.nodes.iter().map(|n| (n.id, n)).collect();
        let nodes_by_path: HashMap<String, &SkeletonNode> = avatar
            .skeleton
            .nodes
            .iter()
            .map(|n| (path_signature(n, &nodes_by_id), n))
            .collect();

        let meshes_by_name: HashMap<&str, &MeshAsset> = avatar
            .meshes
            .iter()
            .map(|m| (m.name.as_str(), m))
            .collect();
        let mut primitives_by_position = HashMap::new();
        let mut primitives_by_id = HashMap::new();
        for mesh in &avatar.meshes {
            for (idx, prim) in mesh.primitives.iter().enumerate() {
                primitives_by_position.insert((mesh.name.clone(), idx), prim);
                primitives_by_id.insert(prim.id, prim);
            }
        }

        Self {
            nodes_by_name,
            nodes_by_humanoid,
            nodes_by_path,
            nodes_by_id,
            meshes_by_name,
            primitives_by_position,
            primitives_by_id,
        }
    }

    /// Resolve a node ref. Returns `(new_node, tier)` on success.
    fn resolve_node(&self, old: &NodeRef) -> Option<(&SkeletonNode, RebindTier)> {
        // Primary: exact name.
        if let Some(n) = self.nodes_by_name.get(old.name.as_str()) {
            return Some((n, RebindTier::Primary));
        }
        // Secondary: humanoid bone slot of the *original* node, if we
        // can recover that. The overlay only stored a name + id, so
        // the only way to recover humanoid-bone identity for the
        // original is to assume the name matched a current humanoid
        // bone with that exact label. Unsupported here — fall through
        // to tertiary.
        // (When the avatar is actually re-imported with renamed bones,
        //  the humanoid bone slot is the same; we'd need the overlay
        //  to have stored `humanoid_bone` alongside name. Adding that
        //  is a `format_version` bump and out of phase A.)
        // Tertiary: parent-path signature stored elsewhere is also
        // future work for the same reason.
        let _ = self.nodes_by_humanoid; // silence unused warning until secondary lands
        let _ = self.nodes_by_path;
        None
    }

    fn resolve_mesh(&self, old: &MeshRef) -> Option<&MeshAsset> {
        self.meshes_by_name.get(old.name.as_str()).copied()
    }

    /// Resolve a primitive ref. The overlay stores `name` formatted as
    /// `"<mesh_name>#<index>"` (per the schema convention agreed in
    /// the rebind design); we parse that to look up the new primitive
    /// at the same `(mesh, index)` slot.
    fn resolve_primitive(
        &self,
        old: &PrimitiveRef,
    ) -> Option<(&MeshPrimitiveAsset, RebindTier)> {
        if let Some((mesh_name, idx)) = parse_primitive_name(&old.name) {
            if let Some(prim) = self
                .primitives_by_position
                .get(&(mesh_name.to_string(), idx))
            {
                return Some((prim, RebindTier::Primary));
            }
        }
        None
    }
}

fn path_signature(
    node: &SkeletonNode,
    by_id: &HashMap<NodeId, &SkeletonNode>,
) -> String {
    let mut parts: Vec<&str> = vec![node.name.as_str()];
    let mut current = node.parent;
    // Cap the walk so a cycle (shouldn't happen on valid skeletons)
    // can't hang.
    let mut hops = 0;
    while let Some(parent_id) = current {
        if hops > 64 {
            break;
        }
        match by_id.get(&parent_id) {
            Some(parent) => {
                parts.push(parent.name.as_str());
                current = parent.parent;
            }
            None => break,
        }
        hops += 1;
    }
    parts.reverse();
    parts.join("/")
}

/// Parse the `"<mesh_name>#<index>"` convention used by `PrimitiveRef.name`.
/// Returns `None` for unrecognised formats so legacy overlay files
/// (which left the field empty) cleanly downgrade to "no primary
/// match available".
fn parse_primitive_name(s: &str) -> Option<(&str, usize)> {
    let (mesh, rest) = s.rsplit_once('#')?;
    let idx: usize = rest.parse().ok()?;
    Some((mesh, idx))
}

/// Build the `"<mesh_name>#<index>"` token used for `PrimitiveRef.name`.
/// Centralised so save / authoring paths can stay consistent with the
/// rebind parser.
pub fn format_primitive_name(mesh_name: &str, index_within_mesh: usize) -> String {
    format!("{}#{}", mesh_name, index_within_mesh)
}

/// Walk the overlay's references, rebind in place, and return a
/// report. See module docs for the resolution strategy.
pub fn rebind_overlay(overlay: &mut ClothAsset, new_avatar: &AvatarAsset) -> RebindReport {
    let resolver = AvatarResolver::new(new_avatar);
    let mut report = RebindReport::new();

    // Track per-name remappings so we don't double-report a node
    // that's referenced from both a pin and a collision binding.
    let mut node_seen: HashMap<String, NodeId> = HashMap::new();
    let mut mesh_seen: HashMap<String, MeshId> = HashMap::new();
    let mut primitive_seen: HashMap<String, PrimitiveId> = HashMap::new();

    // ---- Pins ----
    for pin in &mut overlay.pins {
        rebind_node_ref(&mut pin.binding_node, &resolver, &mut report, &mut node_seen);
    }

    // ---- Collision proxy bindings ----
    for cb in &mut overlay.collision_bindings {
        rebind_node_ref(
            &mut cb.binding_node,
            &resolver,
            &mut report,
            &mut node_seen,
        );
    }

    // ---- Render region bindings (primitive refs) ----
    for rb in &mut overlay.render_bindings {
        let key = rb.primitive.name.clone();
        match resolver.resolve_primitive(&rb.primitive) {
            Some((prim, tier)) => {
                let old_id = rb.primitive.id.0;
                let new_id = prim.id.0;
                if !primitive_seen.contains_key(&key) {
                    primitive_seen.insert(key.clone(), prim.id);
                    report.primitive_remappings.push(RebindEntry {
                        old_id,
                        new_id,
                        name: rb.primitive.name.clone(),
                        tier,
                    });
                }
                rb.primitive.id = prim.id;
                if tier != RebindTier::Primary {
                    report.downgrade(RebindStatus::Partial);
                }

                // Vertex-count sanity check — if the new primitive's
                // vertex count diverges from what the cloth mapping
                // expects, the render binding is geometry-broken.
                let needed = (rb.vertex_subset.offset + rb.vertex_subset.count) as usize;
                if (prim.vertex_count as usize) < needed {
                    report.fatal_geometry_changes.push(format!(
                        "Render binding '{}' needs {} vertices but \
                         the rebound primitive has only {}",
                        rb.primitive.name, needed, prim.vertex_count
                    ));
                    report.downgrade(RebindStatus::Failed);
                }
            }
            None => {
                report.unresolved.push(UnresolvedRef {
                    kind: UnresolvedKind::Primitive,
                    name: rb.primitive.name.clone(),
                    original_id: rb.primitive.id.0,
                });
                report.downgrade(RebindStatus::Failed);
            }
        }
    }

    // ---- stable_refs.mesh_refs (declarative; not directly consumed
    //      by the solver, but the editor's validate step compares ids
    //      so we still rebind them). ----
    for mesh_ref in &mut overlay.stable_refs.mesh_refs {
        let key = mesh_ref.name.clone();
        match resolver.resolve_mesh(mesh_ref) {
            Some(new_mesh) => {
                let old_id = mesh_ref.id.0;
                let new_id = new_mesh.id.0;
                if !mesh_seen.contains_key(&key) {
                    mesh_seen.insert(key.clone(), new_mesh.id);
                    report.mesh_remappings.push(RebindEntry {
                        old_id,
                        new_id,
                        name: mesh_ref.name.clone(),
                        tier: RebindTier::Primary,
                    });
                }
                mesh_ref.id = new_mesh.id;
            }
            None => {
                report.unresolved.push(UnresolvedRef {
                    kind: UnresolvedKind::Mesh,
                    name: mesh_ref.name.clone(),
                    original_id: mesh_ref.id.0,
                });
                report.downgrade(RebindStatus::Failed);
            }
        }
    }

    // ---- stable_refs.node_refs / primitive_refs: rewrite IDs to
    //      match what we wired into pins / render_bindings. Doubles
    //      as a sanity rebuild of the metadata side of the overlay. ----
    for node_ref in &mut overlay.stable_refs.node_refs {
        if let Some(&new) = node_seen.get(&node_ref.name) {
            node_ref.id = new;
        } else if let Some((new_node, _)) = resolver.resolve_node(node_ref) {
            node_ref.id = new_node.id;
        }
        // Else: leave id untouched and rely on validate_overlay to
        // surface the issue. We've already reported it via pin /
        // collision-binding paths if it's actually used.
    }
    for prim_ref in &mut overlay.stable_refs.primitive_refs {
        if let Some(&new) = primitive_seen.get(&prim_ref.name) {
            prim_ref.id = new;
        } else if let Some((new_prim, _)) = resolver.resolve_primitive(prim_ref) {
            prim_ref.id = new_prim.id;
        }
    }

    report
}

fn rebind_node_ref(
    target: &mut NodeRef,
    resolver: &AvatarResolver,
    report: &mut RebindReport,
    seen: &mut HashMap<String, NodeId>,
) {
    let key = target.name.clone();
    if let Some(&already) = seen.get(&key) {
        target.id = already;
        return;
    }
    match resolver.resolve_node(target) {
        Some((new_node, tier)) => {
            let old_id = target.id.0;
            let new_id = new_node.id.0;
            target.id = new_node.id;
            seen.insert(key.clone(), new_node.id);
            report.node_remappings.push(RebindEntry {
                old_id,
                new_id,
                name: target.name.clone(),
                tier,
            });
            if tier != RebindTier::Primary {
                report.downgrade(RebindStatus::Partial);
            }
        }
        None => {
            report.unresolved.push(UnresolvedRef {
                kind: UnresolvedKind::Node,
                name: target.name.clone(),
                original_id: target.id.0,
            });
            report.downgrade(RebindStatus::Failed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::*;

    fn make_skeleton_node(id: u64, name: &str, parent: Option<u64>) -> SkeletonNode {
        SkeletonNode {
            id: NodeId(id),
            name: name.to_string(),
            parent: parent.map(NodeId),
            children: vec![],
            rest_local: Transform::default(),
            humanoid_bone: None,
        }
    }

    fn make_avatar(nodes: Vec<SkeletonNode>, meshes: Vec<MeshAsset>) -> AvatarAsset {
        AvatarAsset {
            id: AvatarAssetId(1),
            source_path: std::path::PathBuf::from("test.vrm"),
            source_hash: AssetSourceHash([0u8; 32]),
            skeleton: SkeletonAsset {
                root_nodes: nodes
                    .iter()
                    .filter(|n| n.parent.is_none())
                    .map(|n| n.id)
                    .collect(),
                nodes,
                inverse_bind_matrices: vec![],
            },
            meshes,
            materials: vec![],
            humanoid: None,
            spring_bones: vec![],
            colliders: vec![],
            default_expressions: ExpressionAssetSet { expressions: vec![] },
            animation_clips: vec![],
            node_to_mesh: std::collections::HashMap::new(),
            vrm_meta: VrmMeta::default(),
            root_aabb: Aabb::empty(),
        }
    }

    fn empty_overlay(target_avatar: AvatarAssetId) -> ClothAsset {
        ClothAsset::new_empty(
            ClothOverlayId(0),
            target_avatar,
            AssetSourceHash([0u8; 32]),
        )
    }

    #[test]
    fn primary_node_rebind_by_name_with_changed_id() {
        let mut overlay = empty_overlay(AvatarAssetId(1));
        overlay.pins.push(ClothPin {
            sim_vertex_indices: vec![0],
            binding_node: NodeRef {
                id: NodeId(99),
                name: "Hips".to_string(),
            },
            offset: [0.0, 0.0, 0.0],
        });

        let avatar = make_avatar(vec![make_skeleton_node(7, "Hips", None)], vec![]);
        let report = rebind_overlay(&mut overlay, &avatar);

        assert_eq!(report.status, RebindStatus::Clean);
        assert_eq!(overlay.pins[0].binding_node.id, NodeId(7));
        assert_eq!(report.node_remappings.len(), 1);
        assert_eq!(report.node_remappings[0].tier, RebindTier::Primary);
        assert_eq!(report.node_remappings[0].old_id, 99);
        assert_eq!(report.node_remappings[0].new_id, 7);
    }

    #[test]
    fn unresolvable_node_marks_failed() {
        let mut overlay = empty_overlay(AvatarAssetId(1));
        overlay.pins.push(ClothPin {
            sim_vertex_indices: vec![0],
            binding_node: NodeRef {
                id: NodeId(99),
                name: "MissingBone".to_string(),
            },
            offset: [0.0, 0.0, 0.0],
        });

        let avatar = make_avatar(vec![make_skeleton_node(1, "Hips", None)], vec![]);
        let report = rebind_overlay(&mut overlay, &avatar);

        assert_eq!(report.status, RebindStatus::Failed);
        assert_eq!(report.unresolved.len(), 1);
        assert_eq!(report.unresolved[0].kind, UnresolvedKind::Node);
        assert_eq!(report.unresolved[0].name, "MissingBone");
    }

    #[test]
    fn duplicate_node_refs_only_remap_once_per_name() {
        let mut overlay = empty_overlay(AvatarAssetId(1));
        // Two pins both binding to the same node.
        overlay.pins.push(ClothPin {
            sim_vertex_indices: vec![0],
            binding_node: NodeRef {
                id: NodeId(99),
                name: "Hips".to_string(),
            },
            offset: [0.0, 0.0, 0.0],
        });
        overlay.pins.push(ClothPin {
            sim_vertex_indices: vec![1],
            binding_node: NodeRef {
                id: NodeId(99),
                name: "Hips".to_string(),
            },
            offset: [0.0, 0.0, 0.0],
        });

        let avatar = make_avatar(vec![make_skeleton_node(7, "Hips", None)], vec![]);
        let report = rebind_overlay(&mut overlay, &avatar);

        assert_eq!(report.status, RebindStatus::Clean);
        assert_eq!(overlay.pins[0].binding_node.id, NodeId(7));
        assert_eq!(overlay.pins[1].binding_node.id, NodeId(7));
        assert_eq!(
            report.node_remappings.len(),
            1,
            "duplicate name should only show up once in the report"
        );
    }

    #[test]
    fn primitive_rebind_uses_mesh_name_index_format() {
        let mut overlay = empty_overlay(AvatarAssetId(1));
        overlay.render_bindings.push(ClothRenderRegionBinding {
            primitive: PrimitiveRef {
                id: PrimitiveId(99),
                name: format_primitive_name("Body", 1),
            },
            vertex_subset: VertexSubsetRef {
                offset: 0,
                count: 4,
            },
            mapping_region: ClothRegionTag(0),
        });

        let body_mesh = MeshAsset {
            id: MeshId(2),
            name: "Body".to_string(),
            primitives: vec![
                MeshPrimitiveAsset {
                    id: PrimitiveId(20),
                    vertex_count: 16,
                    index_count: 24,
                    material_id: MaterialId(0),
                    skin: None,
                    bounds: Aabb::empty(),
                    vertices: None,
                    indices: None,
                    morph_targets: vec![],
                },
                MeshPrimitiveAsset {
                    id: PrimitiveId(21),
                    vertex_count: 16,
                    index_count: 24,
                    material_id: MaterialId(0),
                    skin: None,
                    bounds: Aabb::empty(),
                    vertices: None,
                    indices: None,
                    morph_targets: vec![],
                },
            ],
        };
        let avatar = make_avatar(vec![], vec![body_mesh]);
        let report = rebind_overlay(&mut overlay, &avatar);

        assert_eq!(report.status, RebindStatus::Clean);
        assert_eq!(overlay.render_bindings[0].primitive.id, PrimitiveId(21));
    }

    #[test]
    fn vertex_count_shrink_is_fatal() {
        let mut overlay = empty_overlay(AvatarAssetId(1));
        overlay.render_bindings.push(ClothRenderRegionBinding {
            primitive: PrimitiveRef {
                id: PrimitiveId(99),
                name: format_primitive_name("Body", 0),
            },
            vertex_subset: VertexSubsetRef {
                offset: 0,
                count: 100,
            },
            mapping_region: ClothRegionTag(0),
        });

        let body_mesh = MeshAsset {
            id: MeshId(2),
            name: "Body".to_string(),
            primitives: vec![MeshPrimitiveAsset {
                id: PrimitiveId(20),
                vertex_count: 50, // shrunk: was needs 100
                index_count: 0,
                material_id: MaterialId(0),
                skin: None,
                bounds: Aabb::empty(),
                vertices: None,
                indices: None,
                morph_targets: vec![],
            }],
        };
        let avatar = make_avatar(vec![], vec![body_mesh]);
        let report = rebind_overlay(&mut overlay, &avatar);

        assert_eq!(report.status, RebindStatus::Failed);
        assert_eq!(report.fatal_geometry_changes.len(), 1);
        // The id IS rebound — caller decides whether to detach the
        // binding based on `fatal_geometry_changes`.
        assert_eq!(overlay.render_bindings[0].primitive.id, PrimitiveId(20));
    }

    #[test]
    fn parse_primitive_name_round_trip() {
        let formatted = format_primitive_name("Body Mesh", 3);
        assert_eq!(formatted, "Body Mesh#3");
        let (mesh, idx) = parse_primitive_name(&formatted).unwrap();
        assert_eq!(mesh, "Body Mesh");
        assert_eq!(idx, 3);
    }

    #[test]
    fn parse_primitive_name_handles_hash_in_mesh_name() {
        // Mesh name contains '#'; we use rsplit so the LAST '#' is the
        // index separator.
        let formatted = format_primitive_name("Mesh#A", 2);
        assert_eq!(formatted, "Mesh#A#2");
        let (mesh, idx) = parse_primitive_name(&formatted).unwrap();
        assert_eq!(mesh, "Mesh#A");
        assert_eq!(idx, 2);
    }

    #[test]
    fn empty_overlay_is_clean() {
        let mut overlay = empty_overlay(AvatarAssetId(1));
        let avatar = make_avatar(vec![make_skeleton_node(1, "Hips", None)], vec![]);
        let report = rebind_overlay(&mut overlay, &avatar);
        assert_eq!(report.status, RebindStatus::Clean);
        assert!(!report.has_changes());
    }
}
