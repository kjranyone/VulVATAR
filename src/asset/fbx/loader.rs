use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use log::{info, warn};

use crate::asset::vrm::LoadStage;
use crate::asset::*;
use super::expression::build_expressions;
use super::humanoid::map_bone_name;
use super::texture::TextureResolver;

static NEXT_AVATAR_ID: AtomicU64 = AtomicU64::new(100_000);

#[derive(Debug)]
pub enum FbxLoadError {
    Io(std::io::Error),
    Ufbx(String),
    Other(String),
}

impl std::fmt::Display for FbxLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Ufbx(e) => write!(f, "FBX parse error: {}", e),
            Self::Other(e) => write!(f, "FBX load error: {}", e),
        }
    }
}

impl std::error::Error for FbxLoadError {}

impl From<std::io::Error> for FbxLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub struct FbxAssetLoader;

impl FbxAssetLoader {
    pub fn new() -> Self {
        Self
    }

    pub fn load(&self, path: &str) -> Result<Arc<AvatarAsset>, FbxLoadError> {
        self.load_with_progress(path, |_| {})
    }

    pub fn load_with_progress(
        &self,
        path: &str,
        on_progress: impl Fn(LoadStage),
    ) -> Result<Arc<AvatarAsset>, FbxLoadError> {
        let source_path = PathBuf::from(path);
        on_progress(LoadStage::Reading);
        let file_data = std::fs::read(&source_path)?;
        let source_hash = compute_hash(&file_data);

        // Try load from cache
        if let Ok(Some(mut cached)) = crate::asset::cache::try_load(&source_path) {
            info!("fbx avatar cache hit for {}", source_path.display());
            rehydrate_textures(&mut cached, &source_path);
            cached.id = AvatarAssetId(NEXT_AVATAR_ID.fetch_add(1, Ordering::Relaxed));
            cached.set_loaded_from_cache(true);
            return Ok(Arc::new(cached));
        }

        on_progress(LoadStage::Parsing);
        let mut opts = ufbx::LoadOpts::default();
        opts.target_axes = ufbx::CoordinateAxes::right_handed_y_up();
        opts.target_unit_meters = 1.0;
        opts.space_conversion = ufbx::SpaceConversion::ModifyGeometry;
        opts.generate_missing_normals = true;

        let scene = ufbx::load_file(path, opts)
            .map_err(|e| FbxLoadError::Ufbx(format!("{}: {}", e.description, e.info())))?;

        on_progress(LoadStage::Skeleton);
        // Build mapping from FBX element_id to array index (NodeId)
        let mut elem_to_node_idx: HashMap<u32, usize> = HashMap::with_capacity(scene.nodes.len());
        for (idx, node) in scene.nodes.iter().enumerate() {
            elem_to_node_idx.insert(node.element.element_id, idx);
        }

        // Build skeleton nodes
        let mut nodes = Vec::with_capacity(scene.nodes.len());
        let mut humanoid_bone_map = HashMap::new();

        for (idx, node) in scene.nodes.iter().enumerate() {
            let id = NodeId(idx as u64);
            let name = node.element.name.to_string();
            let parent = node
                .parent
                .as_ref()
                .and_then(|p| elem_to_node_idx.get(&p.element.element_id))
                .map(|&p_idx| NodeId(p_idx as u64));
            let children: Vec<NodeId> = node
                .children
                .iter()
                .filter_map(|c| elem_to_node_idx.get(&c.element.element_id))
                .map(|&c_idx| NodeId(c_idx as u64))
                .collect();

            let t = [
                node.local_transform.translation.x as f32,
                node.local_transform.translation.y as f32,
                node.local_transform.translation.z as f32,
            ];
            let r = [
                node.local_transform.rotation.x as f32,
                node.local_transform.rotation.y as f32,
                node.local_transform.rotation.z as f32,
                node.local_transform.rotation.w as f32,
            ];
            let s = [
                node.local_transform.scale.x as f32,
                node.local_transform.scale.y as f32,
                node.local_transform.scale.z as f32,
            ];

            let humanoid_bone = map_bone_name(&name);
            if let Some(hb) = humanoid_bone {
                humanoid_bone_map.entry(hb).or_insert(id);
            }

            nodes.push(SkeletonNode {
                id,
                name,
                parent,
                children,
                rest_local: Transform {
                    translation: t,
                    rotation: r,
                    scale: s,
                },
                humanoid_bone,
            });
        }

        // Find root nodes (nodes without parent)
        let root_nodes: Vec<NodeId> = nodes
            .iter()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();

        // Texture and Material setup
        let mut tex_resolver = TextureResolver::new(&source_path);
        let total_mats = scene.materials.len();
        on_progress(LoadStage::Materials {
            current: 0,
            total: total_mats,
        });

        let mut materials = Vec::with_capacity(total_mats);
        for (i, mat) in scene.materials.iter().enumerate() {
            let mat_id = MaterialId(mat.element.element_id as u64 + 1);
            let mat_name = mat.element.name.to_string();

            // Check if there is any texture linked in FBX material
            let mut base_color_tex_path: Option<String> = None;
            let mut embedded_tex_binding: Option<TextureBinding> = None;

            for prop in &mat.textures {
                if !prop.texture.content.is_empty() {
                    // Embedded texture
                    if let Ok(img) = image::load_from_memory(&prop.texture.content) {
                        use image::GenericImageView;
                        let (w, h) = img.dimensions();
                        let uri = format!("{}#embedded_tex_{}", source_path.display(), prop.texture.element.element_id);
                        embedded_tex_binding = Some(TextureBinding {
                            uri,
                            pixel_data: Some(Arc::new(img.to_rgba8().into_raw())),
                            dimensions: (w, h),
                        });
                        break;
                    }
                }

                let path = if !prop.texture.filename.is_empty() {
                    prop.texture.filename.to_string()
                } else if !prop.texture.relative_filename.is_empty() {
                    prop.texture.relative_filename.to_string()
                } else {
                    String::new()
                };

                if !path.is_empty() {
                    base_color_tex_path = Some(path);
                    break;
                }
            }

            let base_color_texture = embedded_tex_binding.or_else(|| {
                tex_resolver.resolve_and_load(
                    base_color_tex_path.as_deref(),
                    &mat_name,
                )
            });

            let base_mode = MaterialMode::ToonLike;
            let base_color = [1.0, 1.0, 1.0, 1.0];
            let lower_name = mat_name.to_lowercase();
            let is_transparent = lower_name.contains("transparent")
                || lower_name.contains("trans")
                || lower_name.contains("alpha")
                || lower_name.contains("blend");
            let alpha_mode = if is_transparent {
                AlphaMode::Mask(0.05)
            } else {
                AlphaMode::Opaque
            };
            let double_sided = true;

            let toon_params = ToonMaterialParams {
                ramp_threshold: 0.5,
                shadow_softness: 0.1,
                outline_width: 0.005,
                outline_color: [0.0, 0.0, 0.0],
            };

            materials.push(MaterialAsset {
                id: mat_id,
                name: mat_name,
                base_mode,
                base_color,
                alpha_mode,
                double_sided,
                texture_bindings: MaterialTextureSet {
                    base_color_texture,
                    normal_map_texture: None,
                    shade_ramp_texture: None,
                    emissive_texture: None,
                    matcap_texture: None,
                },
                toon_params,
                mtoon_params: None,
            });

            on_progress(LoadStage::Materials {
                current: i + 1,
                total: total_mats,
            });
        }

        on_progress(LoadStage::Meshes);
        // Build meshes, skin bindings, morph targets
        let mut meshes = Vec::new();
        let mut node_to_mesh = HashMap::new();
        let mut all_morph_targets = Vec::new();

        let mut next_mesh_id: u64 = 1;
        let mut next_prim_id: u64 = 1;

        // Inverse bind matrices per node
        let mut inverse_bind_matrices = vec![identity_matrix(); nodes.len()];
        let mut ibm_set = vec![false; nodes.len()];

        for mesh in scene.meshes.iter() {
            let mesh_id = MeshId(next_mesh_id);
            next_mesh_id += 1;

            let skin = mesh.skin_deformers.first();

            // Collect joints and IBMs for this skin
            let mut skin_joint_nodes = Vec::new();
            let mut skin_ibm = Vec::new();

            if let Some(skin_def) = skin {
                for cluster in &skin_def.clusters {
                    if let Some(ref bone_node) = cluster.bone_node {
                        let node_idx = elem_to_node_idx
                            .get(&bone_node.element.element_id)
                            .copied()
                            .unwrap_or(0);
                        let joint_node_id = NodeId(node_idx as u64);
                        skin_joint_nodes.push(joint_node_id);

                        // Convert ufbx::Matrix to Mat4 column-major
                        let m = matrix_to_mat4(&cluster.geometry_to_bone);
                        skin_ibm.push(m);

                        if node_idx < inverse_bind_matrices.len() && !ibm_set[node_idx] {
                            inverse_bind_matrices[node_idx] = m;
                            ibm_set[node_idx] = true;
                        }
                    }
                }
            }

            let skin_binding = if !skin_joint_nodes.is_empty() {
                Some(SkinBinding {
                    joint_nodes: skin_joint_nodes,
                    inverse_bind_matrices: skin_ibm,
                })
            } else {
                None
            };

            // Pre-process blend shape channel offsets: Vec<(channel_name, HashMap<vert_idx, offset>)>
            let blend_deformer = mesh.blend_deformers.first();
            let mut channel_data = Vec::new();
            if let Some(blend) = blend_deformer {
                for channel in &blend.channels {
                    let mut offset_map: HashMap<usize, [f32; 3]> = HashMap::new();
                    if let Some(ref shape) = channel.target_shape {
                        for (i, &v) in shape.offset_vertices.iter().enumerate() {
                            if let Some(offset) = shape.position_offsets.get(i) {
                                offset_map.insert(
                                    v as usize,
                                    [offset.x as f32, offset.y as f32, offset.z as f32],
                                );
                            }
                        }
                    }
                    channel_data.push((channel.element.name.to_string(), offset_map));
                }
            }

            // Process primitives by material parts
            let mut primitives = Vec::new();
            let mut tri_buffer = [0u32; 64];

            // If no material parts, treat the whole mesh as 1 part
            let parts: Vec<(MaterialId, Vec<usize>)> = if mesh.material_parts.is_empty() {
                let all_faces: Vec<usize> = (0..mesh.faces.len()).collect();
                vec![(MaterialId(1), all_faces)]
            } else {
                mesh.material_parts
                    .iter()
                    .map(|part| {
                        let mat_elem_id = mesh
                            .materials
                            .get(part.index as usize)
                            .map(|m| m.element.element_id)
                            .unwrap_or(part.index);
                        let mat_id = MaterialId(mat_elem_id as u64 + 1);
                        let face_indices: Vec<usize> =
                            part.face_indices.iter().map(|&f| f as usize).collect();
                        (mat_id, face_indices)
                    })
                    .collect()
            };

            // Find first node that references this mesh to compute world-space bounds
            let mesh_node = scene.nodes.iter().find(|n| {
                n.mesh.as_ref().map(|m| m.element.element_id) == Some(mesh.element.element_id)
            });
            let node_to_world = mesh_node.map(|n| &n.node_to_world);

            for (part_mat_id, face_indices) in parts {
                let prim_id = PrimitiveId(next_prim_id);
                next_prim_id += 1;

                let mut positions = Vec::new();
                let mut normals = Vec::new();
                let mut uvs = Vec::new();
                let mut joint_indices = Vec::new();
                let mut joint_weights = Vec::new();
                let mut indices = Vec::new();

                let mut prim_bounds = Aabb::empty();

                // Per-channel delta vectors for this primitive
                let mut morph_deltas: Vec<(String, Vec<Vec3>)> = channel_data
                    .iter()
                    .map(|(name, _)| (name.clone(), Vec::new()))
                    .collect();

                for &face_idx in &face_indices {
                    let face = mesh.faces[face_idx];
                    let num_triangles = mesh.triangulate_face(&mut tri_buffer, face);
                    let tri_indices = &tri_buffer[..(num_triangles * 3) as usize];

                    for &corner in tri_indices {
                        let corner = corner as usize;
                        let vert_idx = mesh.vertex_indices[corner] as usize;

                        // Position
                        let p = mesh.vertex_position[corner];
                        let pos = [p.x as f32, p.y as f32, p.z as f32];
                        positions.push(pos);

                        // Transform position to model/world space for accurate AABB
                        let world_pos = if let Some(m) = node_to_world {
                            [
                                (m.m00 * p.x + m.m01 * p.y + m.m02 * p.z + m.m03) as f32,
                                (m.m10 * p.x + m.m11 * p.y + m.m12 * p.z + m.m13) as f32,
                                (m.m20 * p.x + m.m21 * p.y + m.m22 * p.z + m.m23) as f32,
                            ]
                        } else {
                            pos
                        };
                        prim_bounds.expand(&Aabb { min: world_pos, max: world_pos });

                        // Normal
                        let n = if mesh.vertex_normal.exists {
                            let norm = mesh.vertex_normal[corner];
                            [norm.x as f32, norm.y as f32, norm.z as f32]
                        } else {
                            [0.0, 1.0, 0.0]
                        };
                        normals.push(n);

                        // UV
                        let uv = if mesh.vertex_uv.exists {
                            let u = mesh.vertex_uv[corner];
                            [u.x as f32, 1.0 - u.y as f32] // Y-flip for standard rendering
                        } else {
                            [0.0, 0.0]
                        };
                        uvs.push(uv);

                        // Skinning weights
                        let mut j_indices = [0u16; 4];
                        let mut j_weights = [0.0f32; 4];

                        if let Some(skin_def) = skin {
                            if vert_idx < skin_def.vertices.len() {
                                let sv = &skin_def.vertices[vert_idx];
                                let mut total_w = 0.0;
                                let num_w = (sv.num_weights as usize).min(4);
                                for w_idx in 0..num_w {
                                    let w = &skin_def.weights[sv.weight_begin as usize + w_idx];
                                    let cluster = &skin_def.clusters[w.cluster_index as usize];
                                    let node_idx = cluster
                                        .bone_node
                                        .as_ref()
                                        .and_then(|b| elem_to_node_idx.get(&b.element.element_id).copied())
                                        .unwrap_or(0);
                                    j_indices[w_idx] = node_idx as u16;
                                    j_weights[w_idx] = w.weight as f32;
                                    total_w += w.weight as f32;
                                }
                                if total_w > 1e-6 {
                                    for w in &mut j_weights {
                                        *w /= total_w;
                                    }
                                }
                            }
                        }
                        joint_indices.push(j_indices);
                        joint_weights.push(j_weights);

                        // Morph targets
                        for (c_idx, (_, ref offset_map)) in channel_data.iter().enumerate() {
                            let offset = offset_map.get(&vert_idx).copied().unwrap_or([0.0, 0.0, 0.0]);
                            morph_deltas[c_idx].1.push(offset);
                        }

                        indices.push(indices.len() as u32);
                    }
                }

                let vertex_count = positions.len() as u32;
                let index_count = indices.len() as u32;

                let morph_targets: Vec<MorphTargetDelta> = morph_deltas
                    .into_iter()
                    .map(|(name, pos_deltas)| MorphTargetDelta {
                        name,
                        position_deltas: pos_deltas,
                        normal_deltas: Vec::new(),
                    })
                    .collect();

                let prim_morph_count = morph_targets.len();
                if prim_morph_count > 0 {
                    info!(
                        "[fbx] mesh '{}' prim {} has {} morph targets",
                        mesh.element.name,
                        prim_id.0,
                        prim_morph_count
                    );
                }

                primitives.push(Arc::new(MeshPrimitiveAsset {
                    id: prim_id,
                    vertex_count,
                    index_count,
                    material_id: part_mat_id,
                    skin: skin_binding.clone(),
                    bounds: prim_bounds,
                    vertices: if vertex_count > 0 {
                        Some(VertexData {
                            positions,
                            normals,
                            uvs,
                            joint_indices,
                            joint_weights,
                        })
                    } else {
                        None
                    },
                    indices: if index_count > 0 { Some(indices) } else { None },
                    morph_targets,
                }));
            }

            // Find nodes referencing this mesh
            for node in &scene.nodes {
                if let Some(ref m) = node.mesh {
                    if m.element.element_id == mesh.element.element_id {
                        let node_idx = elem_to_node_idx
                            .get(&node.element.element_id)
                            .copied()
                            .unwrap_or(0);
                        let meshes_idx = meshes.len();
                        node_to_mesh.insert(node_idx, meshes_idx);

                        // Collect morph target references
                        if let Some(first_prim) = primitives.first() {
                            for (ti, mt) in first_prim.morph_targets.iter().enumerate() {
                                all_morph_targets.push((node_idx, ti, mt.name.clone()));
                            }
                        }
                    }
                }
            }

            meshes.push(MeshAsset {
                id: mesh_id,
                name: mesh.element.name.to_string(),
                primitives,
            });
        }

        // Build expressions from blendshapes
        let default_expressions = build_expressions(&all_morph_targets);

        // Root AABB
        let mut root_aabb = Aabb::empty();
        for mesh in &meshes {
            for prim in &mesh.primitives {
                root_aabb.expand(&prim.bounds);
            }
        }

        on_progress(LoadStage::Finalizing);

        let humanoid = if !humanoid_bone_map.is_empty() {
            Some(HumanoidMap {
                bone_map: humanoid_bone_map,
            })
        } else {
            None
        };

        let file_stem = source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("FBX Model")
            .to_string();

        let vrm_meta = VrmMeta {
            spec_version: VrmSpecVersion::Unknown,
            spec_version_raw: Some("FBX".into()),
            title: Some(file_stem),
            authors: vec![],
            model_version: None,
            contact_information: None,
            references: vec![],
            license: None,
            copyright_information: None,
            thumbnail: None,
        };

        let asset = AvatarAsset {
            id: AvatarAssetId(NEXT_AVATAR_ID.fetch_add(1, Ordering::Relaxed)),
            source_path: source_path.clone(),
            source_hash,
            skeleton: SkeletonAsset {
                nodes,
                root_nodes,
                inverse_bind_matrices,
            },
            meshes,
            materials,
            humanoid,
            spring_bones: Vec::new(),
            colliders: Vec::new(),
            default_expressions,
            animation_clips: Vec::new(),
            node_to_mesh,
            vrm_meta,
            root_aabb,
            loaded_from_cache: false,
        };

        // Cache save
        if let Err(e) = crate::asset::cache::save(&source_path, &asset) {
            warn!("avatar cache: save for '{}' failed: {}", source_path.display(), e);
        }

        Ok(Arc::new(asset))
    }
}

fn matrix_to_mat4(m: &ufbx::Matrix) -> Mat4 {
    [
        [m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0],
        [m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0],
        [m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0],
        [m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0],
    ]
}

fn compute_hash(data: &[u8]) -> AssetSourceHash {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    AssetSourceHash(hash)
}

fn rehydrate_textures(asset: &mut AvatarAsset, source_path: &Path) {
    let mut tex_resolver = TextureResolver::new(source_path);
    for mat in &mut asset.materials {
        if let Some(ref mut tb) = mat.texture_bindings.base_color_texture {
            if tb.pixel_data.is_none() {
                let p = Path::new(&tb.uri);
                if p.is_file() {
                    if let Ok(img) = image::open(p) {
                        use image::GenericImageView;
                        let (w, h) = img.dimensions();
                        tb.dimensions = (w, h);
                        tb.pixel_data = Some(Arc::new(img.to_rgba8().into_raw()));
                        continue;
                    }
                }
                if let Some(new_binding) = tex_resolver.resolve_and_load(Some(&tb.uri), &mat.name) {
                    *tb = new_binding;
                }
            }
        }
    }
}
