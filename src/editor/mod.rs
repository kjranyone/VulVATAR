pub mod cloth_authoring;

use std::sync::Arc;

use crate::asset::{AvatarAsset, ClothAsset, ClothOverlayId};

pub struct EditorSession {
    target_avatar: Option<Arc<AvatarAsset>>,
    active_overlay: Option<ClothOverlayId>,
    dirty: bool,
}

impl EditorSession {
    pub fn new() -> Self {
        Self {
            target_avatar: None,
            active_overlay: None,
            dirty: false,
        }
    }

    pub fn open(&mut self, avatar: Arc<AvatarAsset>) {
        self.target_avatar = Some(avatar);
        self.active_overlay = None;
        self.dirty = false;
    }

    pub fn create_overlay(&mut self) -> ClothOverlayId {
        let id = ClothOverlayId(1);
        self.active_overlay = Some(id.clone());
        self.dirty = true;
        id
    }

    pub fn save_overlay(&mut self) -> bool {
        if self.dirty {
            self.dirty = false;
            println!("editor: saved cloth overlay");
            true
        } else {
            false
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn validate_overlay(&self, _overlay: &ClothAsset, _avatar: &AvatarAsset) -> bool {
        true
    }
}
