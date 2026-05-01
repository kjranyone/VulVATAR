//! glTF / VRM-asset decoding helpers used by the loader orchestrator.
//!
//! Free functions that take borrowed `gltf::Document` views (plus the
//! optional .glb binary blob) and produce engine-side asset types
//! (`SkeletonNode`, `MeshAsset`, `MaterialAsset`, `SkinBinding`,
//! `AnimationClip`, ...). Pulled out of `mod.rs` so the loader's
//! `load_with_progress` orchestration is readable in isolation.
//!
//! The cache-hit `rehydrate_textures` path also lives here because it
//! reuses the same `texture_binding_from_info` decode that the regular
//! load uses.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::GenericImageView;
use log::warn;

use super::mtoon;
use super::LoadStage;
use crate::asset::*;

/// Walk every `TextureBinding` on `asset` and repopulate its
/// `pixel_data` from the freshly-parsed glTF document. Used by the
/// cache hit path: `TextureBinding.pixel_data` carries
/// `#[serde(skip)]` (see `src/asset/mod.rs`) so the cache file
/// stays compact, which means the renderer would otherwise see
/// `pixel_data == None` and fall back to its placeholder white
/// texture. The function reuses `texture_binding_from_info` so the
/// URI / decode path matches the regular load.
#[allow(clippy::type_complexity)]
pub(super) fn rehydrate_textures(
    asset: &mut AvatarAsset,
    gltf_doc: &gltf::Document,
    blob: Option<&[u8]>,
    source_path: &Path,
) {
    let mut decoded: HashMap<String, (Arc<Vec<u8>>, (u32, u32))> = HashMap::new();
    for texture in gltf_doc.textures() {
        let binding = texture_binding_from_info(&texture, blob, source_path);
        if let Some(pixels) = binding.pixel_data {
            decoded.insert(binding.uri, (pixels, binding.dimensions));
        }
    }

    let patch = |tb: &mut Option<TextureBinding>| {
        if let Some(tb) = tb.as_mut() {
            if let Some((pixels, dims)) = decoded.get(&tb.uri) {
                tb.pixel_data = Some(Arc::clone(pixels));
                tb.dimensions = *dims;
            }
        }
    };

    for mat in &mut asset.materials {
        patch(&mut mat.texture_bindings.base_color_texture);
        patch(&mut mat.texture_bindings.normal_map_texture);
        patch(&mut mat.texture_bindings.shade_ramp_texture);
        patch(&mut mat.texture_bindings.emissive_texture);
        patch(&mut mat.texture_bindings.matcap_texture);
    }
}
/// Pre-multiply each root node's rest_local by a 180° Y rotation, so the
/// whole skeleton subtree appears rotated 180° around Y in world space.
/// Equivalent to `three-vrm`'s `VRMUtils.rotateVRM0()`.
pub(super) fn bake_y_flip_on_roots(nodes: &mut [SkeletonNode], root_nodes: &[NodeId]) {
    // Y-axis 180° quaternion in xyzw layout: (0, 1, 0, 0).
    let y_flip: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
    for root_id in root_nodes {
        let idx = root_id.0 as usize;
        if idx >= nodes.len() {
            continue;
        }
        let node = &mut nodes[idx];
        // new_rot = Y180 * old_rot
        node.rest_local.rotation =
            crate::math_utils::quat_mul(&y_flip, &node.rest_local.rotation);
        // new_t = Y180 * old_t  →  flip X and Z, keep Y.
        node.rest_local.translation = [
            -node.rest_local.translation[0],
            node.rest_local.translation[1],
            -node.rest_local.translation[2],
        ];
    }
}

pub(super) fn build_skeleton_nodes(doc: &gltf::Document) -> (Vec<SkeletonNode>, HashMap<u32, String>) {
    let gltf_nodes: Vec<gltf::Node<'_>> = doc.nodes().collect();
    let node_count = gltf_nodes.len();

    let mut nodes = Vec::with_capacity(node_count);
    let mut name_map = HashMap::new();

    for gnode in &gltf_nodes {
        let idx = gnode.index();
        let name = gnode.name().unwrap_or("").to_string();
        name_map.insert(idx as u32, name.clone());

        let (t, r, s) = gnode.transform().decomposed();
        let parent = find_parent(&gltf_nodes, idx);
        let children: Vec<NodeId> =
            gnode.children().map(|c| NodeId(c.index() as u64)).collect();

        nodes.push(SkeletonNode {
            id: NodeId(idx as u64),
            name,
            parent,
            children,
            rest_local: Transform {
                translation: t,
                rotation: r,
                scale: s,
            },
            humanoid_bone: None,
        });
    }

    (nodes, name_map)
}

pub(super) fn find_parent(gltf_nodes: &[gltf::Node<'_>], target_idx: usize) -> Option<NodeId> {
    for gnode in gltf_nodes {
        for child in gnode.children() {
            if child.index() == target_idx {
                return Some(NodeId(gnode.index() as u64));
            }
        }
    }
    None
}

pub(super) fn build_meshes(doc: &gltf::Document, blob: Option<&[u8]>) -> Vec<MeshAsset> {
    let mut next_mesh_id: u64 = 1;
    let mut next_prim_id: u64 = 1;

    let buffers: Vec<gltf::buffer::Data> = doc
        .buffers()
        .map(|buf| {
            let data = match buf.source() {
                gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                gltf::buffer::Source::Uri(_uri) => {
                    warn!(
                        "[vrm] External buffer URIs not supported for mesh loading, \
                         buffer {} will be empty",
                        buf.index()
                    );
                    vec![]
                }
            };
            gltf::buffer::Data(data)
        })
        .collect();

    doc.meshes()
        .map(|gmesh| {
            let mesh_id = MeshId(next_mesh_id);
            next_mesh_id += 1;

            let primitives: Vec<MeshPrimitiveAsset> = gmesh
                .primitives()
                .map(|prim| {
                    let prim_id = PrimitiveId(next_prim_id);
                    next_prim_id += 1;

                    let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

                    let positions: Vec<[f32; 3]> = reader
                        .read_positions()
                        .map(|iter| iter.collect())
                        .unwrap_or_default();
                    let vertex_count = positions.len() as u32;

                    let normals: Vec<[f32; 3]> = reader
                        .read_normals()
                        .map(|iter| iter.collect())
                        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count as usize]);

                    let uvs: Vec<[f32; 2]> = reader
                        .read_tex_coords(0)
                        .map(|tc| tc.into_f32().collect())
                        .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count as usize]);

                    let joint_indices: Vec<[u16; 4]> = reader
                        .read_joints(0)
                        .map(|j| j.into_u16().collect())
                        .unwrap_or_default();

                    let joint_weights: Vec<[f32; 4]> = reader
                        .read_weights(0)
                        .map(|w| w.into_f32().collect())
                        .unwrap_or_default();

                    let indices: Vec<u32> = reader
                        .read_indices()
                        .map(|idx| idx.into_u32().collect())
                        .unwrap_or_default();
                    let index_count = indices.len() as u32;

                    let material_id = prim
                        .material()
                        .index()
                        .map(|i| MaterialId(i as u64 + 1))
                        .unwrap_or(MaterialId(0));

                    let bb = prim.bounding_box();
                    let bounds = Aabb {
                        min: bb.min,
                        max: bb.max,
                    };

                    let vertices = if vertex_count > 0 {
                        Some(VertexData {
                            positions,
                            normals,
                            uvs,
                            joint_indices,
                            joint_weights,
                        })
                    } else {
                        None
                    };

                    let idx_data = if index_count > 0 { Some(indices) } else { None };

                    // Parse morph targets (blend shapes).
                    let target_names: Vec<String> = gmesh
                        .extras()
                        .as_ref()
                        .and_then(|raw| {
                            let v: serde_json::Value =
                                serde_json::from_str(raw.get()).ok()?;
                            let arr = v.get("targetNames")?.as_array()?;
                            Some(
                                arr.iter()
                                    .filter_map(|s| s.as_str().map(String::from))
                                    .collect(),
                            )
                        })
                        .unwrap_or_default();

                    let get_buf_data = |buffer: gltf::Buffer<'_>| -> Option<&[u8]> {
                        buffers.get(buffer.index()).map(|d| d.0.as_slice())
                    };

                    let morph_targets: Vec<MorphTargetDelta> = prim
                        .morph_targets()
                        .enumerate()
                        .map(|(ti, mt)| {
                            let pos_d: Vec<[f32; 3]> = mt
                                .positions()
                                .and_then(|acc| {
                                    let iter = gltf::accessor::Iter::<[f32; 3]>::new(
                                        acc,
                                        get_buf_data,
                                    )?;
                                    Some(iter.collect())
                                })
                                .unwrap_or_default();
                            let norm_d: Vec<[f32; 3]> = mt
                                .normals()
                                .and_then(|acc| {
                                    let iter = gltf::accessor::Iter::<[f32; 3]>::new(
                                        acc,
                                        get_buf_data,
                                    )?;
                                    Some(iter.collect())
                                })
                                .unwrap_or_default();
                            let name = target_names
                                .get(ti)
                                .cloned()
                                .unwrap_or_else(|| format!("target_{}", ti));
                            MorphTargetDelta {
                                name,
                                position_deltas: pos_d,
                                normal_deltas: norm_d,
                            }
                        })
                        .collect();

                    if !morph_targets.is_empty() {
                        log::info!(
                            "[vrm] mesh '{}' prim {} has {} morph targets",
                            gmesh.name().unwrap_or("unnamed"),
                            prim_id.0,
                            morph_targets.len(),
                        );
                    }

                    MeshPrimitiveAsset {
                        id: prim_id,
                        vertex_count,
                        index_count,
                        material_id,
                        skin: None,
                        bounds,
                        vertices,
                        indices: idx_data,
                        morph_targets,
                    }
                })
                .collect();

            MeshAsset {
                id: mesh_id,
                name: gmesh.name().unwrap_or("unnamed").to_string(),
                primitives,
            }
        })
        .collect()
}

pub(super) fn resolve_texture_index(
    doc: &gltf::Document,
    index: usize,
    blob: Option<&[u8]>,
    source_path: &Path,
) -> Option<TextureBinding> {
    doc.textures()
        .nth(index)
        .map(|tex| texture_binding_from_info(&tex, blob, source_path))
}

pub(super) fn build_materials(
    doc: &gltf::Document,
    blob: Option<&[u8]>,
    source_path: &Path,
    on_progress: &dyn Fn(LoadStage),
) -> Vec<MaterialAsset> {
    let total = doc.materials().count();
    on_progress(LoadStage::Materials { current: 0, total });
    doc.materials()
        .enumerate()
        .map(|(i, gmat)| {
            let pbr = gmat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();

            let gltf_alpha = gmat.alpha_mode();
            let alpha_mode = match gltf_alpha {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => {
                    AlphaMode::Mask(gmat.alpha_cutoff().unwrap_or(0.5))
                }
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };
            let double_sided = gmat.double_sided();

            let mtoon_ext = gmat
                .extensions()
                .and_then(|ext| ext.get("VRMC_materials_mtoon"));
            let has_mtoon = mtoon_ext.is_some();

            let base_mode = if has_mtoon {
                MaterialMode::ToonLike
            } else {
                MaterialMode::SimpleLit
            };

            let base_color_texture = pbr.base_color_texture().map(|info| {
                texture_binding_from_info(&info.texture(), blob, source_path)
            });

            let normal_map_texture = gmat.normal_texture().map(|info| {
                texture_binding_from_info(&info.texture(), blob, source_path)
            });

            let mut shade_ramp_texture = None;
            let mut emissive_texture_binding = None;
            let mut matcap_texture_binding = None;

            let (toon_params, mtoon_params) = if let Some(mtoon) = mtoon_ext {
                let (toon, mut staged, tex_indices) = mtoon::parse_mtoon_params(mtoon);

                shade_ramp_texture = tex_indices
                    .shade_color_texture
                    .and_then(|idx| resolve_texture_index(doc, idx, blob, source_path));

                emissive_texture_binding = tex_indices
                    .emissive_texture
                    .and_then(|idx| resolve_texture_index(doc, idx, blob, source_path));

                matcap_texture_binding = tex_indices
                    .matcap_texture
                    .and_then(|idx| resolve_texture_index(doc, idx, blob, source_path));

                staged.emissive_texture =
                    emissive_texture_binding
                        .as_ref()
                        .map(|tb| MtoonTextureSlot {
                            uri: tb.uri.clone(),
                        });
                staged.rim_texture = tex_indices
                    .rim_texture
                    .and_then(|idx| resolve_texture_index(doc, idx, blob, source_path))
                    .map(|tb| MtoonTextureSlot { uri: tb.uri });
                staged.matcap_texture =
                    matcap_texture_binding.as_ref().map(|tb| MtoonTextureSlot {
                        uri: tb.uri.clone(),
                    });
                staged.uv_anim_mask_texture = tex_indices
                    .uv_anim_mask_texture
                    .and_then(|idx| resolve_texture_index(doc, idx, blob, source_path))
                    .map(|tb| MtoonTextureSlot { uri: tb.uri });

                (toon, Some(staged))
            } else {
                (
                    ToonMaterialParams {
                        ramp_threshold: 0.5,
                        shadow_softness: 0.1,
                        outline_width: 0.01,
                        outline_color: [0.0, 0.0, 0.0],
                    },
                    None,
                )
            };

            let asset = MaterialAsset {
                id: MaterialId(i as u64 + 1),
                name: gmat.name().unwrap_or("unnamed").to_string(),
                base_mode,
                base_color,
                alpha_mode,
                double_sided,
                texture_bindings: MaterialTextureSet {
                    base_color_texture,
                    normal_map_texture,
                    shade_ramp_texture,
                    emissive_texture: emissive_texture_binding,
                    matcap_texture: matcap_texture_binding,
                },
                toon_params,
                mtoon_params,
            };
            on_progress(LoadStage::Materials {
                current: i + 1,
                total,
            });
            asset
        })
        .inspect(|mat| {
            let tb = &mat.texture_bindings;
            let base = tb.base_color_texture.as_ref();
            log::info!(
                "[vrm] material #{} '{}' mode={:?} base_tex={} ({}x{})",
                mat.id.0,
                mat.name,
                mat.base_mode,
                base.map_or("none", |t| &t.uri),
                base.map_or(0, |t| t.dimensions.0),
                base.map_or(0, |t| t.dimensions.1),
            );
        })
        .collect()
}

pub(super) fn texture_binding_from_info(
    texture: &gltf::Texture<'_>,
    blob: Option<&[u8]>,
    source_path: &Path,
) -> TextureBinding {
    let base_dir = source_path.parent();
    let default_name = format!("texture_{}", texture.index());
    let texture_name = texture.source().name().unwrap_or(default_name.as_str());

    let source = texture.source().source();
    let (uri, pixel_data, dimensions) = match source {
        gltf::image::Source::View { view, .. } => {
            let start = view.offset();
            let end = start + view.length();
            let cache_key = format!("{}#{}", source_path.display(), texture_name);
            match blob.and_then(|b| b.get(start..end)) {
                Some(encoded) => match image::load_from_memory(encoded) {
                    Ok(img) => {
                        let (w, h) = img.dimensions();
                        log::info!(
                            "[vrm] texture '{}' loaded from buffer: {}x{} ({} bytes)",
                            cache_key,
                            w,
                            h,
                            encoded.len()
                        );
                        (cache_key, Some(Arc::new(img.to_rgba8().into_raw())), (w, h))
                    }
                    Err(e) => {
                        log::warn!("[vrm] texture '{}' decode failed: {}", cache_key, e);
                        (cache_key, None, (0, 0))
                    }
                },
                None => {
                    log::warn!(
                        "[vrm] texture '{}' buffer range {}..{} out of bounds",
                        cache_key,
                        start,
                        end
                    );
                    (cache_key, None, (0, 0))
                }
            }
        }
        gltf::image::Source::Uri { uri: file_uri, .. } => {
            let resolved_path = base_dir
                .map(|dir| dir.join(file_uri))
                .unwrap_or_else(|| PathBuf::from(file_uri));
            let cache_key = resolved_path.to_string_lossy().into_owned();
            match image::open(&resolved_path) {
                Ok(img) => {
                    let (w, h) = img.dimensions();
                    log::info!(
                        "[vrm] texture '{}' loaded from uri '{}': {}x{}",
                        texture_name,
                        resolved_path.display(),
                        w,
                        h
                    );
                    (cache_key, Some(Arc::new(img.to_rgba8().into_raw())), (w, h))
                }
                Err(e) => {
                    log::warn!(
                        "[vrm] texture '{}' file open failed for '{}': {}",
                        texture_name,
                        resolved_path.display(),
                        e
                    );
                    (cache_key, None, (0, 0))
                }
            }
        }
    };

    TextureBinding {
        uri,
        pixel_data,
        dimensions,
    }
}

pub(super) fn build_skin_bindings(doc: &gltf::Document, blob: Option<&[u8]>) -> Vec<(usize, SkinBinding)> {
    let buffers: Vec<gltf::buffer::Data> = doc
        .buffers()
        .map(|buf| {
            let data = match buf.source() {
                gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                gltf::buffer::Source::Uri(_uri) => {
                    warn!(
                        "[vrm] External buffer URIs not supported for skin loading, \
                         buffer {} will be empty",
                        buf.index()
                    );
                    vec![]
                }
            };
            gltf::buffer::Data(data)
        })
        .collect();

    doc.skins()
        .enumerate()
        .map(|(skin_index, skin)| {
            let joint_nodes: Vec<NodeId> =
                skin.joints().map(|j| NodeId(j.index() as u64)).collect();

            let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
            let inverse_bind_matrices: Vec<Mat4> = reader
                .read_inverse_bind_matrices()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| {
                    warn!(
                        "[vrm] Skin {} has no inverse bind matrices, using identity",
                        skin_index
                    );
                    vec![identity_matrix(); joint_nodes.len()]
                });

            (
                skin_index,
                SkinBinding {
                    joint_nodes,
                    inverse_bind_matrices,
                },
            )
        })
        .collect()
}

pub(super) fn assign_skins_to_meshes(
    doc: &gltf::Document,
    meshes: Vec<MeshAsset>,
    skins: &[(usize, SkinBinding)],
) -> Vec<MeshAsset> {
    // Build a map from glTF mesh index to skin index by examining nodes.
    // Each node can reference both a mesh and a skin.
    let mut mesh_to_skin: HashMap<usize, usize> = HashMap::new();
    for node in doc.nodes() {
        if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
            mesh_to_skin.insert(mesh.index(), skin.index());
        }
    }

    meshes
        .into_iter()
        .enumerate()
        .map(|(mesh_idx, mut mesh)| {
            let skin_binding = mesh_to_skin
                .get(&mesh_idx)
                .and_then(|skin_idx| skins.get(*skin_idx).map(|(_, s)| s));

            let effective_skin = skin_binding.or_else(|| skins.first().map(|(_, s)| s));

            if let Some(skin) = effective_skin {
                for prim in &mut mesh.primitives {
                    // Remap vertex joint indices from "skin-relative"
                    // (index into this skin's joint_nodes list) to
                    // "node-relative" (glTF node index). This lets the
                    // renderer use a single skinning matrix array shared
                    // across all skins in the avatar — critical for VRM
                    // models that ship separate skins for face/body/etc.
                    if let Some(ref mut vd) = prim.vertices {
                        for ji in vd.joint_indices.iter_mut() {
                            for slot in ji.iter_mut() {
                                let joint_idx = *slot as usize;
                                if let Some(NodeId(node_id)) =
                                    skin.joint_nodes.get(joint_idx)
                                {
                                    *slot = *node_id as u16;
                                }
                            }
                        }
                    }
                    prim.skin = Some(skin.clone());
                }
            }
            mesh
        })
        .collect()
}

pub(super) fn build_animation_clips(doc: &gltf::Document, blob: Option<&[u8]>) -> Vec<AnimationClip> {
    let buffers: Vec<gltf::buffer::Data> = doc
        .buffers()
        .map(|buf| {
            let data = match buf.source() {
                gltf::buffer::Source::Bin => blob.unwrap_or(&[]).to_vec(),
                gltf::buffer::Source::Uri(_) => vec![],
            };
            gltf::buffer::Data(data)
        })
        .collect();

    let mut clips = Vec::new();
    for (anim_idx, anim) in doc.animations().enumerate() {
        let name = anim
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("Animation_{}", anim_idx));

        let mut channels = Vec::new();
        let mut duration: f32 = 0.0;

        for channel in anim.channels() {
            let target = channel.target();
            let target_node = NodeId(target.node().index() as u64);

            let property = match target.property() {
                gltf::animation::Property::Translation => AnimationProperty::Translation,
                gltf::animation::Property::Rotation => AnimationProperty::Rotation,
                gltf::animation::Property::Scale => AnimationProperty::Scale,
                gltf::animation::Property::MorphTargetWeights => continue,
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

            let timestamps: Vec<f32> = reader
                .read_inputs()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            if let Some(&last_t) = timestamps.last() {
                if last_t > duration {
                    duration = last_t;
                }
            }

            let sampler = channel.sampler();
            let interp = match sampler.interpolation() {
                gltf::animation::Interpolation::Step => InterpolationMode::Step,
                gltf::animation::Interpolation::Linear => InterpolationMode::Linear,
                gltf::animation::Interpolation::CubicSpline => InterpolationMode::CubicSpline,
            };

            let outputs = reader.read_outputs();
            let keyframes: Vec<Keyframe> = match outputs {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    let values: Vec<[f32; 3]> = iter.collect();
                    build_keyframes_vec3(&timestamps, &values, interp)
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(rotations)) => {
                    let values: Vec<[f32; 4]> = rotations.into_f32().collect();
                    build_keyframes_vec4(&timestamps, &values, interp)
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    let values: Vec<[f32; 3]> = iter.collect();
                    build_keyframes_vec3(&timestamps, &values, interp)
                }
                _ => continue,
            };

            channels.push(AnimationChannel {
                target_node,
                property,
                keyframes,
            });
        }

        if channels.is_empty() {
            continue;
        }

        clips.push(AnimationClip {
            id: AnimationClipId(anim_idx as u64 + 1),
            name,
            duration,
            channels,
        });
    }

    clips
}

/// Build keyframes from vec3 outputs (translation/scale).
/// For CubicSpline, every 3 consecutive values are [in_tangent, value, out_tangent].
pub(super) fn build_keyframes_vec3(
    timestamps: &[f32],
    values: &[[f32; 3]],
    interp: InterpolationMode,
) -> Vec<Keyframe> {
    if interp == InterpolationMode::CubicSpline {
        // CubicSpline: 3 values per keyframe (in_tangent, value, out_tangent)
        timestamps
            .iter()
            .enumerate()
            .filter_map(|(i, &time)| {
                let base = i * 3;
                if base + 2 >= values.len() {
                    return None;
                }
                let in_t = values[base];
                let val = values[base + 1];
                let out_t = values[base + 2];
                Some(Keyframe {
                    time,
                    value: [val[0], val[1], val[2], 0.0],
                    tangents: Some([
                        [in_t[0], in_t[1], in_t[2], 0.0],
                        [out_t[0], out_t[1], out_t[2], 0.0],
                    ]),
                    interpolation: interp,
                })
            })
            .collect()
    } else {
        timestamps
            .iter()
            .zip(values.iter())
            .map(|(&time, val)| Keyframe {
                time,
                value: [val[0], val[1], val[2], 0.0],
                tangents: None,
                interpolation: interp,
            })
            .collect()
    }
}

/// Build keyframes from vec4 outputs (rotation quaternion).
pub(super) fn build_keyframes_vec4(
    timestamps: &[f32],
    values: &[[f32; 4]],
    interp: InterpolationMode,
) -> Vec<Keyframe> {
    if interp == InterpolationMode::CubicSpline {
        timestamps
            .iter()
            .enumerate()
            .filter_map(|(i, &time)| {
                let base = i * 3;
                if base + 2 >= values.len() {
                    return None;
                }
                Some(Keyframe {
                    time,
                    value: values[base + 1],
                    tangents: Some([values[base], values[base + 2]]),
                    interpolation: interp,
                })
            })
            .collect()
    } else {
        timestamps
            .iter()
            .zip(values.iter())
            .map(|(&time, &val)| Keyframe {
                time,
                value: val,
                tangents: None,
                interpolation: interp,
            })
            .collect()
    }
}
