#![allow(dead_code)]
pub mod cloth_authoring;

use log::info;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::asset::{
    AssetSourceHash, AvatarAsset, AvatarAssetId, ClothAsset, ClothOverlayId, MeshId, NodeId,
    PrimitiveId,
};
use crate::persistence::ClothOverlayFile;

pub struct EditorSession {
    target_avatar: Option<Arc<AvatarAsset>>,
    active_overlay: Option<ClothOverlayId>,
    pub dirty: bool,
    /// The in-memory cloth asset being edited.
    pub overlay_asset: Option<ClothAsset>,
    /// File path the overlay was last saved to / loaded from.
    pub overlay_path: Option<PathBuf>,
    /// Counter for generating unique overlay IDs within the session.
    next_overlay_id: u64,
}

impl EditorSession {
    pub fn new() -> Self {
        Self {
            target_avatar: None,
            active_overlay: None,
            dirty: false,
            overlay_asset: None,
            overlay_path: None,
            next_overlay_id: 1,
        }
    }

    pub fn open(&mut self, avatar: Arc<AvatarAsset>) {
        self.target_avatar = Some(avatar);
        self.active_overlay = None;
        self.overlay_asset = None;
        self.overlay_path = None;
        self.dirty = false;
    }

    pub fn create_overlay(
        &mut self,
        avatar_id: AvatarAssetId,
        avatar_hash: AssetSourceHash,
    ) -> ClothOverlayId {
        let id = ClothOverlayId(self.next_overlay_id);
        self.next_overlay_id += 1;
        self.active_overlay = Some(id);
        self.overlay_asset = Some(ClothAsset::new_empty(id, avatar_id, avatar_hash));
        self.dirty = true;
        id
    }

    /// Save the current overlay to a file at `path` (or the stored path).
    /// Builds a `ClothOverlayFile` from the current overlay asset, serialises
    /// it as JSON, and writes it to disk.
    pub fn save_overlay(&mut self, path: Option<&Path>) -> Result<(), String> {
        let save_path = match path.or(self.overlay_path.as_deref()) {
            Some(p) => p.to_path_buf(),
            None => return Err("No save path specified".to_string()),
        };

        let overlay_name = self
            .overlay_asset
            .as_ref()
            .map(|a| a.metadata.name.clone())
            .unwrap_or_else(|| "Untitled".to_string());

        let target_avatar_path = self
            .target_avatar
            .as_ref()
            .map(|a| a.source_path.to_string_lossy().into_owned());

        let file = ClothOverlayFile {
            format_version: crate::persistence::OVERLAY_FORMAT_VERSION,
            created_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
            last_saved_with: format!("VulVATAR {}", env!("CARGO_PKG_VERSION")),
            overlay_name,
            target_avatar_path,
            cloth_asset: self.overlay_asset.clone(),
        };

        let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        crate::persistence::atomic_write(&save_path, &json)?;

        self.overlay_path = Some(save_path.clone());
        self.dirty = false;
        info!("editor: saved cloth overlay to {}", save_path.display());
        Ok(())
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn set_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn active_overlay(&self) -> Option<ClothOverlayId> {
        self.active_overlay
    }

    // ── Overlay browser actions (M12) ────────────────────────────────

    /// Clone the current overlay with a new ID and " (Copy)" name suffix.
    pub fn duplicate_overlay(&mut self) -> Option<ClothOverlayId> {
        let src = self.overlay_asset.as_ref()?.clone();
        let new_id = ClothOverlayId(self.next_overlay_id);
        self.next_overlay_id += 1;

        let mut dup = src;
        dup.id = new_id;
        dup.metadata.name = format!("{} (Copy)", dup.metadata.name);

        self.overlay_asset = Some(dup);
        self.active_overlay = Some(new_id);
        self.overlay_path = None; // duplicated overlay has no saved path yet
        self.dirty = true;
        Some(new_id)
    }

    /// Rename the current overlay.
    pub fn rename_overlay(&mut self, new_name: String) {
        if let Some(ref mut asset) = self.overlay_asset {
            asset.metadata.name = new_name;
            self.dirty = true;
        }
    }

    /// Delete / clear the current overlay and mark the session dirty.
    pub fn delete_overlay(&mut self) {
        self.overlay_asset = None;
        self.active_overlay = None;
        self.overlay_path = None;
        self.dirty = true;
    }

    /// Validate the overlay against the avatar and return a list of error messages.
    /// An empty list means validation passed.
    pub fn validate_overlay(&self, overlay: &ClothAsset, avatar: &AvatarAsset) -> Vec<String> {
        let mut errors = Vec::new();

        // 1. Check that the overlay targets this avatar.
        if overlay.target_avatar != avatar.id {
            errors.push(format!(
                "Overlay targets avatar {} but loaded avatar is {}",
                overlay.target_avatar.0, avatar.id.0
            ));
        }

        // 2. Check that the source hash matches (detects reimported/changed avatars).
        if overlay.target_avatar_hash.0 != avatar.source_hash.0 {
            errors.push("Source hash mismatch: avatar may have been reimported".to_string());
        }

        // Collect the set of valid node IDs from the avatar skeleton.
        let valid_nodes: std::collections::HashSet<NodeId> =
            avatar.skeleton.nodes.iter().map(|n| n.id).collect();

        // Collect the set of valid mesh IDs from the avatar meshes.
        let valid_meshes: std::collections::HashSet<MeshId> =
            avatar.meshes.iter().map(|m| m.id).collect();

        // 3. All stable node refs must resolve to an existing skeleton node.
        for node_ref in &overlay.stable_refs.node_refs {
            if !valid_nodes.contains(&node_ref.id) {
                errors.push(format!(
                    "Node ref '{}' (id {}) not found in skeleton",
                    node_ref.name, node_ref.id.0
                ));
            }
        }

        // 4. All stable mesh refs must resolve to an existing avatar mesh.
        for mesh_ref in &overlay.stable_refs.mesh_refs {
            if !valid_meshes.contains(&mesh_ref.id) {
                errors.push(format!(
                    "Mesh ref '{}' (id {}) not found in avatar",
                    mesh_ref.name, mesh_ref.id.0
                ));
            }
        }

        // 5. All stable primitive refs must resolve to an existing primitive.
        let valid_primitives: std::collections::HashSet<PrimitiveId> = avatar
            .meshes
            .iter()
            .flat_map(|m| m.primitives.iter().map(|p| p.id))
            .collect();

        for prim_ref in &overlay.stable_refs.primitive_refs {
            if !valid_primitives.contains(&prim_ref.id) {
                errors.push(format!(
                    "Primitive ref '{}' (id {}) not found in avatar",
                    prim_ref.name, prim_ref.id.0
                ));
            }
        }

        // 6. Pin binding nodes must reference valid skeleton nodes.
        for pin in &overlay.pins {
            if !valid_nodes.contains(&pin.binding_node.id) {
                errors.push(format!(
                    "Pin binding node '{}' (id {}) not found in skeleton",
                    pin.binding_node.name, pin.binding_node.id.0
                ));
            }
        }

        // 7. Collision proxy binding nodes must reference valid skeleton nodes.
        for cb in &overlay.collision_bindings {
            if !valid_nodes.contains(&cb.binding_node.id) {
                errors.push(format!(
                    "Collision binding node '{}' (id {}) not found in skeleton",
                    cb.binding_node.name, cb.binding_node.id.0
                ));
            }
        }

        // 8. Render region binding primitives must reference valid primitives.
        for rb in &overlay.render_bindings {
            if !valid_primitives.contains(&rb.primitive.id) {
                errors.push(format!(
                    "Render binding primitive '{}' (id {}) not found in avatar",
                    rb.primitive.name, rb.primitive.id.0
                ));
            }
        }

        errors
    }
}
