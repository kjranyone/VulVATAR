#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AvatarLibraryEntry {
    pub path: PathBuf,
    pub name: String,
    pub source_hash: Option<Vec<u8>>,
    pub last_loaded: Option<String>,
    pub tags: Vec<String>,
    pub notes: String,
    pub favorite: bool,
    pub mesh_count: Option<usize>,
    pub material_count: Option<usize>,
    pub spring_chain_count: Option<usize>,
    pub collider_count: Option<usize>,
    pub has_humanoid: Option<bool>,
    pub category: Option<String>,
    pub vrm_title: Option<String>,
    pub vrm_author: Option<String>,
    pub vrm_version: Option<String>,
    pub thumbnail_path: Option<PathBuf>,
}

impl AvatarLibraryEntry {
    pub fn from_path(path: &Path) -> Self {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Unknown".to_string());
        Self {
            path: path.to_path_buf(),
            name,
            source_hash: None,
            last_loaded: None,
            tags: Vec::new(),
            notes: String::new(),
            favorite: false,
            mesh_count: None,
            material_count: None,
            spring_chain_count: None,
            collider_count: None,
            has_humanoid: None,
            category: None,
            vrm_title: None,
            vrm_author: None,
            vrm_version: None,
            thumbnail_path: None,
        }
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    pub fn update_from_asset(&mut self, asset: &crate::asset::AvatarAsset) {
        self.update_from_asset_inner(asset, None);
    }

    /// Same as [`Self::update_from_asset`] but additionally writes the
    /// VRM-embedded thumbnail (if any) to `thumbnail_dir/<safe_name>.<ext>`
    /// and stores its path on `self.thumbnail_path`. The directory is
    /// created if missing. Errors are logged but never fail the update —
    /// the avatar is still importable when the thumbnail can't be saved.
    pub fn update_from_asset_with_thumbnail_dir(
        &mut self,
        asset: &crate::asset::AvatarAsset,
        thumbnail_dir: &Path,
    ) {
        self.update_from_asset_inner(asset, Some(thumbnail_dir));
    }

    fn update_from_asset_inner(
        &mut self,
        asset: &crate::asset::AvatarAsset,
        thumbnail_dir: Option<&Path>,
    ) {
        self.name = asset
            .source_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.name.clone());
        self.source_hash = Some(asset.source_hash.0.to_vec());
        self.mesh_count = Some(asset.meshes.len());
        self.material_count = Some(asset.materials.len());
        self.spring_chain_count = Some(asset.spring_bones.len());
        self.collider_count = Some(asset.colliders.len());
        self.has_humanoid = Some(asset.humanoid.is_some());
        self.last_loaded = Some(chrono_free_timestamp());

        let meta = &asset.vrm_meta;
        self.vrm_title = meta.title.clone();
        self.vrm_author = if meta.authors.is_empty() {
            None
        } else {
            Some(meta.authors.join(", "))
        };
        self.vrm_version = Some(meta.spec_version.label().to_string());

        if let (Some(dir), Some(thumb)) = (thumbnail_dir, meta.thumbnail.as_ref()) {
            match write_vrm_thumbnail(dir, &self.name, thumb) {
                Ok(path) => self.thumbnail_path = Some(path),
                Err(e) => log::warn!(
                    "library: failed to write VRM thumbnail for '{}' to {}: {}",
                    self.name,
                    dir.display(),
                    e
                ),
            }
        }
    }
}

fn write_vrm_thumbnail(
    dir: &Path,
    avatar_name: &str,
    thumb: &crate::asset::EmbeddedThumbnail,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    // Sanitise the filename the same way ThumbnailGenerator does so the
    // two paths are interoperable. Only ASCII alphanumeric, '-', '_'.
    let safe: String = avatar_name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{}.{}", safe, thumb.extension()));
    std::fs::write(&path, &thumb.bytes)?;
    Ok(path)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AvatarLibrary {
    pub entries: Vec<AvatarLibraryEntry>,
}

fn chrono_free_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

impl AvatarLibrary {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, entry: AvatarLibraryEntry) {
        if let Some(existing) = self.find_by_path_mut(&entry.path) {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
    }

    pub fn remove(&mut self, path: &Path) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.path != path);
        self.entries.len() < before
    }

    pub fn find_by_path(&self, path: &Path) -> Option<&AvatarLibraryEntry> {
        self.entries.iter().find(|e| e.path == path)
    }

    fn find_by_path_mut(&mut self, path: &Path) -> Option<&mut AvatarLibraryEntry> {
        self.entries.iter_mut().find(|e| e.path == path)
    }

    pub fn search(&self, query: &str) -> Vec<&AvatarLibraryEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                let name_match = e.name.to_lowercase().contains(&q);
                let path_match = e.path.to_string_lossy().to_lowercase().contains(&q);
                let tag_match = e.tags.iter().any(|t| t.to_lowercase().contains(&q));
                name_match || path_match || tag_match
            })
            .collect()
    }

    pub fn favorites(&self) -> Vec<&AvatarLibraryEntry> {
        self.entries.iter().filter(|e| e.favorite).collect()
    }

    pub fn existing(&self) -> Vec<&AvatarLibraryEntry> {
        self.entries.iter().filter(|e| e.exists()).collect()
    }

    pub fn purge_missing(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.exists());
        before - self.entries.len()
    }

    pub fn sort_by_name(&mut self) {
        self.entries
            .sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    }

    pub fn sort_by_last_loaded(&mut self) {
        self.entries
            .sort_by(|a, b| b.last_loaded.cmp(&a.last_loaded));
    }

    pub fn sort_favorites_first(&mut self) {
        self.entries.sort_by(|a, b| b.favorite.cmp(&a.favorite));
    }

    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| e.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    pub fn by_category(&self, category: &str) -> Vec<&AvatarLibraryEntry> {
        self.entries
            .iter()
            .filter(|e| e.category.as_deref() == Some(category))
            .collect()
    }

    pub fn export_catalog(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize library catalog: {}", e))
    }

    pub fn import_catalog(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("failed to parse library catalog: {}", e))
    }
}

impl Default for AvatarLibrary {
    fn default() -> Self {
        Self::new()
    }
}
