use crate::asset::{AvatarAsset, ClothAsset, ClothOverlayId, ColliderShape, NodeId, PrimitiveId};

#[derive(Clone, Debug)]
pub enum AuthoringState {
    Idle,
    RegionSelected(RegionSelection),
    Editing(ClothOverlayId),
    Previewing(ClothOverlayId),
}

#[derive(Clone, Debug)]
pub struct RegionSelection {
    pub target_primitive: PrimitiveId,
    pub selected_vertex_range: (u32, u32),
}

#[derive(Clone, Debug)]
pub struct ClothAuthoringSession {
    pub state: AuthoringState,
    pub target_overlay: Option<ClothOverlayId>,
    pub dirty: bool,
    pub validation_errors: Vec<AuthoringValidationError>,
}

impl ClothAuthoringSession {
    pub fn new() -> Self {
        Self {
            state: AuthoringState::Idle,
            target_overlay: None,
            dirty: false,
            validation_errors: vec![],
        }
    }

    pub fn select_region(&mut self, primitive: PrimitiveId, start: u32, end: u32) {
        self.state = AuthoringState::RegionSelected(RegionSelection {
            target_primitive: primitive,
            selected_vertex_range: (start, end),
        });
        self.dirty = true;
    }

    pub fn begin_editing(&mut self, overlay_id: ClothOverlayId) {
        self.state = AuthoringState::Editing(overlay_id.clone());
        self.target_overlay = Some(overlay_id);
    }

    pub fn begin_preview(&mut self) {
        if let Some(ref id) = self.target_overlay {
            let id = id.clone();
            self.state = AuthoringState::Previewing(id);
        }
    }

    pub fn reset_to_idle(&mut self) {
        self.state = AuthoringState::Idle;
    }

    pub fn validate(
        &self,
        overlay: &ClothAsset,
        avatar: &AvatarAsset,
    ) -> Vec<AuthoringValidationError> {
        let mut errors = vec![];

        for pin in &overlay.pins {
            let node_found = avatar
                .skeleton
                .nodes
                .iter()
                .any(|n| n.id == pin.binding_node.id);
            if !node_found {
                errors.push(AuthoringValidationError::UnresolvedNodeRef {
                    node_name: pin.binding_node.name.clone(),
                });
            }
        }

        if overlay.simulation_mesh.vertices.is_empty() {
            errors.push(AuthoringValidationError::EmptySimulationMesh);
        }

        errors
    }
}

#[derive(Clone, Debug)]
pub enum AuthoringValidationError {
    UnresolvedNodeRef { node_name: String },
    UnresolvedPrimitiveRef { primitive_name: String },
    EmptySimulationMesh,
    EmptyPinSet,
    NoCollisionProxies,
    InvalidConstraintRange,
}
