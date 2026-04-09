pub struct VrmAssetLoader;

#[derive(Clone, Debug)]
pub struct VrmAsset {
    pub meta: VrmMeta,
    pub skeleton: SkeletonAsset,
    pub meshes: Vec<MeshAsset>,
    pub materials: Vec<MaterialAsset>,
    pub spring_bones: Vec<SpringBoneAsset>,
    pub colliders: Vec<ColliderAsset>,
    pub cloth_regions: Vec<ClothAsset>,
}

#[derive(Clone, Debug)]
pub struct VrmMeta {
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct SkeletonAsset {
    pub joints: Vec<JointAsset>,
}

#[derive(Clone, Debug)]
pub struct JointAsset {
    pub name: String,
    pub parent_index: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct MeshAsset {
    pub name: String,
    pub primitive_count: usize,
}

#[derive(Clone, Debug)]
pub struct MaterialAsset {
    pub name: String,
    pub shading_model: ShadingModel,
}

#[derive(Clone, Debug)]
pub enum ShadingModel {
    Unlit,
    MToonLike,
}

#[derive(Clone, Debug)]
pub struct SpringBoneAsset {
    pub name: String,
    pub joint_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ColliderAsset {
    pub name: String,
    pub attached_joint: usize,
}

#[derive(Clone, Debug)]
pub struct ClothAsset {
    pub name: String,
    pub particle_count: usize,
    pub pinned_particle_indices: Vec<usize>,
}

impl VrmAssetLoader {
    pub fn new() -> Self {
        Self
    }

    pub fn load(&self, path: &str) -> VrmAsset {
        println!("vrm: load stub for '{}'", path);

        VrmAsset {
            meta: VrmMeta {
                name: "Sample Avatar".to_string(),
            },
            skeleton: SkeletonAsset {
                joints: vec![
                    JointAsset {
                        name: "hips".to_string(),
                        parent_index: None,
                    },
                    JointAsset {
                        name: "head".to_string(),
                        parent_index: Some(0),
                    },
                ],
            },
            meshes: vec![MeshAsset {
                name: "Body".to_string(),
                primitive_count: 1,
            }],
            materials: vec![MaterialAsset {
                name: "BodyMaterial".to_string(),
                shading_model: ShadingModel::MToonLike,
            }],
            spring_bones: vec![SpringBoneAsset {
                name: "HairFront".to_string(),
                joint_indices: vec![1],
            }],
            colliders: vec![ColliderAsset {
                name: "HeadCollider".to_string(),
                attached_joint: 1,
            }],
            cloth_regions: vec![ClothAsset {
                name: "CapeEdge".to_string(),
                particle_count: 8,
                pinned_particle_indices: vec![0, 1],
            }],
        }
    }
}
