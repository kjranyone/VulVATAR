#[derive(Clone, Debug)]
pub struct ClothState {
    pub regions: Vec<ClothRegionState>,
}

#[derive(Clone, Debug)]
pub struct ClothRegionState {
    pub name: String,
    pub particle_positions: Vec<[f32; 3]>,
    pub particle_velocities: Vec<[f32; 3]>,
    pub pinned_particle_indices: Vec<usize>,
}

impl ClothState {
    pub fn from_assets(regions: &[crate::asset::vrm::ClothAsset]) -> Self {
        Self {
            regions: regions
                .iter()
                .map(|region| ClothRegionState {
                    name: region.name.clone(),
                    particle_positions: vec![[0.0, 0.0, 0.0]; region.particle_count],
                    particle_velocities: vec![[0.0, 0.0, 0.0]; region.particle_count],
                    pinned_particle_indices: region.pinned_particle_indices.clone(),
                })
                .collect(),
        }
    }
}
