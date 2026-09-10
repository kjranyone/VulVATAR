use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use image::GenericImageView;
use log::{info, warn};

use crate::asset::TextureBinding;

pub struct TextureResolver {
    search_dirs: Vec<PathBuf>,
    discovered_images: HashMap<String, PathBuf>,
    loaded_cache: HashMap<PathBuf, TextureBinding>,
}

impl TextureResolver {
    /// Initialize with candidate search directories relative to the FBX file path.
    pub fn new(fbx_path: &Path) -> Self {
        let mut search_dirs = Vec::new();
        let fbx_dir = fbx_path.parent().unwrap_or_else(|| Path::new("."));
        search_dirs.push(fbx_dir.to_path_buf());

        // Also add parent directory and its common subdirectories
        if let Some(parent) = fbx_dir.parent() {
            search_dirs.push(parent.to_path_buf());

            let candidates = [
                "Texture",
                "Textures",
                "texture",
                "textures",
                "Texture/PNG",
                "Texture/png",
                "Textures/PNG",
                "Textures/png",
                "Texture/PSD",
                "tex",
                "Images",
                "materials",
            ];
            for sub in &candidates {
                let candidate_path = parent.join(sub);
                if candidate_path.is_dir() {
                    search_dirs.push(candidate_path);
                }
            }
        }

        // Subdirectories of fbx_dir
        for sub in &["Texture", "Textures", "texture", "textures", "PNG", "png"] {
            let candidate_path = fbx_dir.join(sub);
            if candidate_path.is_dir() {
                search_dirs.push(candidate_path);
            }
        }

        // Index all image files found in the candidate search directories
        let mut discovered_images = HashMap::new();
        let image_extensions = ["png", "jpg", "jpeg", "tga", "bmp", "webp"];

        for dir in &search_dirs {
            index_directory_images(dir, &image_extensions, &mut discovered_images, 0, 3);
        }

        Self {
            search_dirs,
            discovered_images,
            loaded_cache: HashMap::new(),
        }
    }

    /// Try to resolve and load a texture given an FBX texture path and/or material name.
    pub fn resolve_and_load(
        &mut self,
        fbx_tex_path: Option<&str>,
        mat_name: &str,
    ) -> Option<TextureBinding> {
        let resolved_file = self.find_file(fbx_tex_path, mat_name)?;

        if let Some(cached) = self.loaded_cache.get(&resolved_file) {
            return Some(cached.clone());
        }

        match image::open(&resolved_file) {
            Ok(img) => {
                let (w, h) = img.dimensions();
                let uri = resolved_file.to_string_lossy().into_owned();
                info!(
                    "[fbx] texture loaded for '{}': {} ({}x{})",
                    mat_name,
                    resolved_file.display(),
                    w,
                    h
                );
                let binding = TextureBinding {
                    uri,
                    pixel_data: Some(Arc::new(img.to_rgba8().into_raw())),
                    dimensions: (w, h),
                };
                self.loaded_cache.insert(resolved_file, binding.clone());
                Some(binding)
            }
            Err(e) => {
                warn!(
                    "[fbx] failed to open image file '{}': {}",
                    resolved_file.display(),
                    e
                );
                None
            }
        }
    }

    fn find_file(&self, fbx_tex_path: Option<&str>, mat_name: &str) -> Option<PathBuf> {
        // 1. Direct path check from FBX texture path
        if let Some(path_str) = fbx_tex_path {
            let p = Path::new(path_str);
            if p.is_file() {
                return Some(p.to_path_buf());
            }

            // Check relative to search dirs
            for dir in &self.search_dirs {
                let joined = dir.join(p);
                if joined.is_file() {
                    return Some(joined);
                }
            }

            // Check by filename only
            if let Some(file_name) = p.file_name().and_then(|n| n.to_str()) {
                let norm = file_name.to_lowercase();
                if let Some(found) = self.discovered_images.get(&norm) {
                    return Some(found.clone());
                }
                // Check without extension
                let stem = Path::new(file_name)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if let Some(found) = self.discovered_images.get(&stem) {
                    return Some(found.clone());
                }
            }
        }

        // 2. Check by material name
        let clean_mat = mat_name
            .rsplit('|')
            .next()
            .unwrap_or(mat_name)
            .rsplit(':')
            .next()
            .unwrap_or(mat_name)
            .to_lowercase();

        // Exact match with stem or full name
        if let Some(found) = self.discovered_images.get(&clean_mat) {
            return Some(found.clone());
        }

        // Match with "_" normalized
        let norm_mat = clean_mat.replace('-', "_");
        for (k, path) in &self.discovered_images {
            if k == &norm_mat || k.starts_with(&norm_mat) || norm_mat.starts_with(k) {
                return Some(path.clone());
            }
        }

        // Token-based matching: strip common qualifiers like transparent, opaque, mask, mat, etc.
        let tokens: Vec<&str> = norm_mat
            .split('_')
            .filter(|&t| !t.is_empty() && t != "transparent" && t != "trans" && t != "alpha" && t != "mat" && t != "material" && t != "mask")
            .collect();
        for &tok in tokens.iter().rev() {
            if tok.len() >= 3 {
                for (k, path) in &self.discovered_images {
                    if k.contains(tok) {
                        return Some(path.clone());
                    }
                }
            }
        }

        None
    }
}

fn index_directory_images(
    dir: &Path,
    extensions: &[&str],
    out_map: &mut HashMap<String, PathBuf>,
    current_depth: usize,
    max_depth: usize,
) {
    if current_depth > max_depth {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            index_directory_images(&path, extensions, out_map, current_depth + 1, max_depth);
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.iter().any(|&e| e.eq_ignore_ascii_case(ext)) {
                    if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                        let lower_name = file_name.to_lowercase();
                        out_map.entry(lower_name).or_insert_with(|| path.clone());

                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            let lower_stem = stem.to_lowercase();
                            out_map.entry(lower_stem).or_insert_with(|| path.clone());
                        }
                    }
                }
            }
        }
    }
}
