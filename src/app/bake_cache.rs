use log::info;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BakeCacheEntry {
    pub avatar_name: String,
    pub avatar_hash: Vec<u8>,
    pub cache_type: BakeCacheType,
    pub frame_count: u32,
    pub resolution: [u32; 2],
    pub created_at: u64,
    pub file_size_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BakeCacheType {
    AnimationSequence,
    ExpressionMap,
    ClothRestPose,
    ThumbnailBatch,
}

#[derive(Clone, Debug)]
pub struct BakeCacheConfig {
    pub output_dir: PathBuf,
    pub format: BakeOutputFormat,
    pub overwrite: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BakeOutputFormat {
    PngSequence,
    RgbaRaw,
}

impl Default for BakeCacheConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("bake_cache"),
            format: BakeOutputFormat::PngSequence,
            overwrite: false,
        }
    }
}

pub struct BakeCacheManager {
    config: BakeCacheConfig,
    catalog: Vec<BakeCacheEntry>,
}

impl BakeCacheManager {
    pub fn new(config: BakeCacheConfig) -> Self {
        let catalog = Self::load_catalog(&config.output_dir);
        Self { config, catalog }
    }

    pub fn catalog(&self) -> &[BakeCacheEntry] {
        &self.catalog
    }

    pub fn config(&self) -> &BakeCacheConfig {
        &self.config
    }

    pub fn output_dir(&self) -> &Path {
        &self.config.output_dir
    }

    pub fn bake_animation(
        &mut self,
        avatar_name: &str,
        avatar_hash: &[u8],
        frame_count: u32,
        resolution: [u32; 2],
        frame_data_provider: &dyn Fn(u32) -> Vec<u8>,
    ) -> Result<BakeCacheEntry, String> {
        if resolution[0] == 0 || resolution[1] == 0 {
            return Err("resolution must be non-zero".to_string());
        }
        let safe_name: String = avatar_name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let output_subdir = self.config.output_dir.join(format!(
            "anim_{}_{}x{}",
            safe_name, resolution[0], resolution[1]
        ));

        std::fs::create_dir_all(&output_subdir)
            .map_err(|e| format!("failed to create bake dir: {}", e))?;

        let mut total_size = 0u64;
        for frame_idx in 0..frame_count {
            let pixel_data = frame_data_provider(frame_idx);
            let filename = format!("frame_{:06}.rgba", frame_idx);

            match self.config.format {
                BakeOutputFormat::RgbaRaw => {
                    let path = output_subdir.join(&filename);
                    std::fs::write(&path, &pixel_data)
                        .map_err(|e| format!("write frame {} failed: {}", frame_idx, e))?;
                    total_size += pixel_data.len() as u64;
                }
                BakeOutputFormat::PngSequence => {
                    let path = output_subdir.join(format!("frame_{:06}.png", frame_idx));
                    let img = image::RgbaImage::from_raw(resolution[0], resolution[1], pixel_data)
                        .ok_or("failed to create image for frame")?;
                    img.save(&path)
                        .map_err(|e| format!("save png frame {} failed: {}", frame_idx, e))?;
                    total_size += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                }
            }
        }

        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = BakeCacheEntry {
            avatar_name: avatar_name.to_string(),
            avatar_hash: avatar_hash.to_vec(),
            cache_type: BakeCacheType::AnimationSequence,
            frame_count,
            resolution,
            created_at: now_secs,
            file_size_bytes: total_size,
        };

        info!(
            "bake: cached {} frames ({} bytes) for '{}'",
            frame_count, total_size, avatar_name
        );

        self.catalog.push(entry.clone());
        self.save_catalog()?;
        Ok(entry)
    }

    pub fn find_by_avatar(&self, avatar_name: &str) -> Vec<&BakeCacheEntry> {
        self.catalog
            .iter()
            .filter(|e| e.avatar_name == avatar_name)
            .collect()
    }

    pub fn find_by_type(&self, cache_type: &BakeCacheType) -> Vec<&BakeCacheEntry> {
        self.catalog
            .iter()
            .filter(|e| &e.cache_type == cache_type)
            .collect()
    }

    pub fn purge_entry(&mut self, avatar_name: &str, cache_type: &BakeCacheType) -> usize {
        let before = self.catalog.len();
        self.catalog
            .retain(|e| !(e.avatar_name == avatar_name && &e.cache_type == cache_type));
        let removed = before - self.catalog.len();
        if removed > 0 {
            let _ = self.save_catalog();
        }
        removed
    }

    pub fn purge_all(&mut self) -> usize {
        let count = self.catalog.len();
        self.catalog.clear();
        let _ = std::fs::remove_dir_all(&self.config.output_dir);
        let _ = std::fs::create_dir_all(&self.config.output_dir);
        let _ = self.save_catalog();
        count
    }

    fn load_catalog(dir: &Path) -> Vec<BakeCacheEntry> {
        let path = dir.join("catalog.json");
        if !path.exists() {
            return Vec::new();
        }
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        serde_json::from_str(&data).unwrap_or_default()
    }

    fn save_catalog(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| format!("create catalog dir: {}", e))?;
        let path = self.config.output_dir.join("catalog.json");
        let json = serde_json::to_string(&self.catalog)
            .map_err(|e| format!("serialize catalog: {}", e))?;
        let tmp_path = path.with_extension("json.tmp");
        std::fs::write(&tmp_path, &json).map_err(|e| format!("write catalog tmp: {}", e))?;
        std::fs::rename(&tmp_path, &path).map_err(|e| format!("rename catalog: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_cache_default_config() {
        let config = BakeCacheConfig::default();
        assert_eq!(config.output_dir, PathBuf::from("bake_cache"));
        assert_eq!(config.format, BakeOutputFormat::PngSequence);
        assert!(!config.overwrite);
    }

    #[test]
    fn bake_cache_empty_catalog() {
        let dir = std::env::temp_dir().join("vulvatar_bake_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let mgr = BakeCacheManager::new(BakeCacheConfig {
            output_dir: dir.clone(),
            ..Default::default()
        });
        assert!(mgr.catalog().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bake_cache_bake_animation() {
        let dir = std::env::temp_dir().join("vulvatar_bake_test_bake");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let mut mgr = BakeCacheManager::new(BakeCacheConfig {
            output_dir: dir.clone(),
            format: BakeOutputFormat::RgbaRaw,
            ..Default::default()
        });

        let resolution = [4u32, 4u32];
        let pixel_count = (resolution[0] * resolution[1] * 4) as usize;
        let entry = mgr
            .bake_animation("test_avatar", &[1, 2, 3, 4], 3, resolution, &|_frame| {
                vec![128u8; pixel_count]
            })
            .unwrap();

        assert_eq!(entry.frame_count, 3);
        assert_eq!(entry.resolution, resolution);
        assert!(entry.file_size_bytes > 0);
        assert_eq!(mgr.catalog().len(), 1);

        let found = mgr.find_by_avatar("test_avatar");
        assert_eq!(found.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bake_cache_purge_entry() {
        let dir = std::env::temp_dir().join("vulvatar_bake_test_purge");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::create_dir_all(&dir);

        let mut mgr = BakeCacheManager::new(BakeCacheConfig {
            output_dir: dir.clone(),
            format: BakeOutputFormat::RgbaRaw,
            ..Default::default()
        });

        let resolution = [2u32, 2u32];
        let pixel_count = (resolution[0] * resolution[1] * 4) as usize;
        let _ = mgr.bake_animation("avatar_a", &[1], 1, resolution, &|_| vec![0u8; pixel_count]);
        let _ = mgr.bake_animation("avatar_b", &[2], 1, resolution, &|_| vec![0u8; pixel_count]);

        assert_eq!(mgr.catalog().len(), 2);
        let removed = mgr.purge_entry("avatar_a", &BakeCacheType::AnimationSequence);
        assert_eq!(removed, 1);
        assert_eq!(mgr.catalog().len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
