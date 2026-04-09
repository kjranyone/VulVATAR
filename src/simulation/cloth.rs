#[derive(Clone, Debug)]
pub struct ClothSimTempBuffers {
    pub temp_positions: Vec<[f32; 3]>,
    pub correction_accumulator: Vec<[f32; 3]>,
}

impl ClothSimTempBuffers {
    pub fn new(particle_count: usize) -> Self {
        Self {
            temp_positions: vec![[0.0, 0.0, 0.0]; particle_count],
            correction_accumulator: vec![[0.0, 0.0, 0.0]; particle_count],
        }
    }
}
