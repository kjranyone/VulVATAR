#![allow(dead_code)]
use log::info;
use std::path::{Path, PathBuf};

pub const THUMBNAIL_WIDTH: u32 = 128;
pub const THUMBNAIL_HEIGHT: u32 = 128;

#[derive(Clone, Debug)]
pub struct ThumbnailRequest {
    pub avatar_name: String,
    pub width: u32,
    pub height: u32,
    pub background_color: [f32; 4],
}

impl Default for ThumbnailRequest {
    fn default() -> Self {
        Self {
            avatar_name: String::new(),
            width: THUMBNAIL_WIDTH,
            height: THUMBNAIL_HEIGHT,
            background_color: [0.2, 0.2, 0.2, 1.0],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ThumbnailResult {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

pub struct ThumbnailGenerator {
    cache: Vec<(String, PathBuf)>,
    output_dir: PathBuf,
}

impl ThumbnailGenerator {
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            cache: Vec::new(),
            output_dir,
        }
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    pub fn set_output_dir(&mut self, dir: PathBuf) {
        self.output_dir = dir;
    }

    pub fn generate_placeholder(&self, request: &ThumbnailRequest) -> ThumbnailResult {
        let w = request.width.max(1) as usize;
        let h = request.height.max(1) as usize;
        let bg = request.background_color;
        let r = (bg[0] * 255.0) as u8;
        let g = (bg[1] * 255.0) as u8;
        let b = (bg[2] * 255.0) as u8;
        let a = (bg[3] * 255.0) as u8;

        let mut pixels = vec![0u8; w * h * 4];
        let cx = w / 2;
        let cy = h / 2;
        let radius = w.min(h) / 3;

        for y in 0..h {
            for x in 0..w {
                let dx = x as i32 - cx as i32;
                let dy = y as i32 - cy as i32;
                let inside = (dx * dx + dy * dy) <= (radius * radius) as i32;
                let idx = (y * w + x) * 4;
                if inside {
                    let dist = ((dx * dx + dy * dy) as f32).sqrt() / radius as f32;
                    let shade = ((1.0 - dist) * 180.0 + 40.0) as u8;
                    pixels[idx] = shade;
                    pixels[idx + 1] = shade;
                    pixels[idx + 2] = shade;
                    pixels[idx + 3] = 255;
                } else {
                    pixels[idx] = r;
                    pixels[idx + 1] = g;
                    pixels[idx + 2] = b;
                    pixels[idx + 3] = a;
                }
            }
        }

        let char_count = request.avatar_name.chars().count();
        let mut label_x = cx.saturating_sub(char_count / 2 * 5);
        for (i, ch) in request.avatar_name.chars().enumerate() {
            if i < 20 && label_x + 5 <= w {
                Self::draw_char(&mut pixels, w, ch, label_x, cy + radius + 4);
                label_x += 5;
            }
        }

        ThumbnailResult {
            width: w as u32,
            height: h as u32,
            pixels,
        }
    }

    fn draw_char(pixels: &mut [u8], width: usize, ch: char, x: usize, y: usize) {
        let glyph = Self::glyph_5x7(ch);
        let height = glyph.len() / 5;
        for row in 0..height {
            for col in 0..5usize {
                if glyph[row * 5 + col] {
                    let px = x + col;
                    let py = y + row;
                    if px < width && py * width + px < pixels.len() / 4 {
                        let idx = (py * width + px) * 4;
                        if idx + 3 < pixels.len() {
                            pixels[idx] = 220;
                            pixels[idx + 1] = 220;
                            pixels[idx + 2] = 220;
                            pixels[idx + 3] = 255;
                        }
                    }
                }
            }
        }
    }

    fn glyph_5x7(ch: char) -> &'static [bool; 35] {
        static EMPTY: [bool; 35] = [false; 35];
        match ch {
            'A' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, true, true, false, false, false, true, true, true, true, true, true,
                true, false, false, false, true, true, false, false, false, true,
            ],
            'B' => &[
                true, true, true, true, false, true, false, false, false, true, true, false, false,
                false, true, true, true, true, true, false, true, false, false, false, true, true,
                false, false, false, true, true, true, true, true, false,
            ],
            'C' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, false, true, false, false, false, false, true, false, false, false,
                false, true, false, false, false, true, false, true, true, true, false,
            ],
            'D' => &[
                true, true, true, false, false, true, false, false, true, false, true, false,
                false, false, true, true, false, false, false, true, true, false, false, false,
                true, true, false, false, true, false, true, true, true, false, false,
            ],
            'E' => &[
                true, true, true, true, true, true, false, false, false, false, true, false, false,
                false, false, true, true, true, true, false, true, false, false, false, false,
                true, false, false, false, false, true, true, true, true, true,
            ],
            'F' => &[
                true, true, true, true, true, true, false, false, false, false, true, false, false,
                false, false, true, true, true, true, false, true, false, false, false, false,
                true, false, false, false, false, true, false, false, false, false,
            ],
            'G' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, false, true, false, true, true, true, true, false, false, false,
                true, true, false, false, false, true, false, true, true, true, false,
            ],
            'H' => &[
                true, false, false, false, true, true, false, false, false, true, true, false,
                false, false, true, true, true, true, true, true, true, false, false, false, true,
                true, false, false, false, true, true, false, false, false, true,
            ],
            'I' => &[
                false, true, true, true, false, false, false, true, false, false, false, false,
                true, false, false, false, false, true, false, false, false, false, true, false,
                false, false, false, true, false, false, false, true, true, true, false,
            ],
            'J' => &[
                false, false, true, true, true, false, false, false, true, false, false, false,
                false, true, false, false, false, false, true, false, true, false, false, true,
                false, true, false, false, true, false, false, true, true, false, false,
            ],
            'K' => &[
                true, false, false, false, true, true, false, false, true, false, true, false,
                true, false, false, true, true, false, false, false, true, false, true, false,
                false, true, false, false, true, false, true, false, false, false, true,
            ],
            'L' => &[
                true, false, false, false, false, true, false, false, false, false, true, false,
                false, false, false, true, false, false, false, false, true, false, false, false,
                false, true, false, false, false, false, true, true, true, true, true,
            ],
            'M' => &[
                true, false, false, false, true, true, true, false, true, true, true, false, true,
                false, true, true, false, false, false, true, true, false, false, false, true,
                true, false, false, false, true, true, false, false, false, true,
            ],
            'N' => &[
                true, false, false, false, true, true, true, false, false, true, true, false, true,
                false, true, true, false, false, true, true, true, false, false, false, true, true,
                false, false, false, true, true, false, false, false, true,
            ],
            'O' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, true, true, false, false, false, true, true, false, false, false,
                true, true, false, false, false, true, false, true, true, true, false,
            ],
            'P' => &[
                true, true, true, true, false, true, false, false, false, true, true, false, false,
                false, true, true, true, true, true, false, true, false, false, false, false, true,
                false, false, false, false, true, false, false, false, false,
            ],
            'Q' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, true, true, false, false, false, true, true, false, true, false,
                true, true, false, false, true, false, false, true, true, true, true,
            ],
            'R' => &[
                true, true, true, true, false, true, false, false, false, true, true, false, false,
                false, true, true, true, true, true, false, true, false, true, false, false, true,
                false, false, true, false, true, false, false, false, true,
            ],
            'S' => &[
                false, true, true, true, true, true, false, false, false, false, true, false,
                false, false, false, false, true, true, true, false, false, false, false, false,
                true, false, false, false, false, true, true, true, true, true, false,
            ],
            'T' => &[
                true, true, true, true, true, false, false, true, false, false, false, false, true,
                false, false, false, false, true, false, false, false, false, true, false, false,
                false, false, true, false, false, false, false, true, false, false,
            ],
            'U' => &[
                true, false, false, false, true, true, false, false, false, true, true, false,
                false, false, true, true, false, false, false, true, true, false, false, false,
                true, true, false, false, false, true, false, true, true, true, false,
            ],
            'V' => &[
                true, false, false, false, true, true, false, false, false, true, true, false,
                false, false, true, true, false, false, false, true, false, true, false, true,
                false, false, true, false, true, false, false, false, true, false, false,
            ],
            'W' => &[
                true, false, false, false, true, true, false, false, false, true, true, false,
                false, false, true, true, false, true, false, true, true, true, false, true, true,
                true, false, false, false, true, true, false, false, false, true,
            ],
            'X' => &[
                true, false, false, false, true, true, false, false, false, true, false, true,
                false, true, false, false, false, true, false, false, false, true, false, true,
                false, true, false, false, false, true, true, false, false, false, true,
            ],
            'Y' => &[
                true, false, false, false, true, false, true, false, true, false, false, false,
                true, false, false, false, false, true, false, false, false, false, true, false,
                false, false, false, true, false, false, false, false, true, false, false,
            ],
            'Z' => &[
                true, true, true, true, true, false, false, false, true, false, false, false, true,
                false, false, false, true, false, false, false, true, false, false, false, false,
                true, false, false, false, false, true, true, true, true, true,
            ],
            '0' => &[
                false, true, true, true, false, true, false, false, true, true, true, false, true,
                false, true, true, true, false, false, true, true, false, false, false, true, true,
                false, false, false, true, false, true, true, true, false,
            ],
            '1' => &[
                false, false, true, false, false, false, true, true, false, false, false, false,
                true, false, false, false, false, true, false, false, false, false, true, false,
                false, false, false, true, false, false, false, true, true, true, false,
            ],
            '2' => &[
                false, true, true, true, false, true, false, false, false, true, false, false,
                false, false, true, false, false, true, true, false, false, true, false, false,
                false, true, false, false, false, false, true, true, true, true, true,
            ],
            '3' => &[
                false, true, true, true, false, true, false, false, false, true, false, false,
                false, false, true, false, false, true, true, false, false, false, false, false,
                true, true, false, false, false, true, false, true, true, true, false,
            ],
            '4' => &[
                true, false, false, true, false, true, false, false, true, false, true, false,
                false, true, false, true, true, true, true, true, false, false, false, true, false,
                false, false, false, true, false, false, false, false, true, false,
            ],
            '5' => &[
                true, true, true, true, true, true, false, false, false, false, true, true, true,
                true, false, false, false, false, false, true, false, false, false, false, true,
                true, false, false, false, true, false, true, true, true, false,
            ],
            '6' => &[
                false, true, true, true, false, true, false, false, false, false, true, false,
                false, false, false, true, true, true, true, false, true, false, false, false,
                true, true, false, false, false, true, false, true, true, true, false,
            ],
            '7' => &[
                true, true, true, true, true, false, false, false, false, true, false, false,
                false, true, false, false, false, true, false, false, false, true, false, false,
                false, false, true, false, false, false, false, true, false, false, false,
            ],
            '8' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, true, false, true, true, true, false, true, false, false, false,
                true, true, false, false, false, true, false, true, true, true, false,
            ],
            '9' => &[
                false, true, true, true, false, true, false, false, false, true, true, false,
                false, false, true, false, true, true, true, true, false, false, false, false,
                true, false, false, false, false, true, false, true, true, true, false,
            ],
            '-' => &[
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, true, true, true, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false,
            ],
            '_' => &[
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, true, true,
            ],
            '.' => &[
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, true, false, false, false, false, true, false, false,
            ],
            ' ' => &[
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false,
            ],
            _ => &EMPTY,
        }
    }

    /// Generate the default placeholder for `avatar_name` and write it
    /// to [`Self::thumbnail_path_for`]. Returns the path on success.
    /// Failures are logged but never fatal — every avatar should still
    /// load even if thumbnail I/O is broken.
    pub fn generate_and_save_placeholder(&self, avatar_name: &str) -> Option<PathBuf> {
        let request = ThumbnailRequest {
            avatar_name: avatar_name.to_string(),
            ..Default::default()
        };
        let result = self.generate_placeholder(&request);
        let path = self.thumbnail_path_for(avatar_name);
        match self.save_to_disk(&result, &path) {
            Ok(()) => Some(path),
            Err(e) => {
                log::warn!(
                    "thumbnail: failed to save placeholder for '{}': {}",
                    avatar_name,
                    e
                );
                None
            }
        }
    }

    pub fn save_to_disk(&self, result: &ThumbnailResult, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create thumbnail dir: {}", e))?;
        }

        let img = image::RgbaImage::from_raw(result.width, result.height, result.pixels.clone())
            .ok_or("failed to create image from thumbnail pixels")?;

        img.save(path)
            .map_err(|e| format!("failed to save thumbnail: {}", e))?;

        info!("thumbnail: saved to {}", path.display());
        Ok(())
    }

    pub fn thumbnail_path_for(&self, avatar_name: &str) -> PathBuf {
        let safe_name =
            avatar_name.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");
        self.output_dir.join(format!("{}.png", safe_name))
    }

    pub fn cache_thumbnail(&mut self, avatar_name: &str, path: &Path) {
        let entry = self.cache.iter().position(|(n, _)| n == avatar_name);
        match entry {
            Some(idx) => self.cache[idx].1 = path.to_path_buf(),
            None => self
                .cache
                .push((avatar_name.to_string(), path.to_path_buf())),
        }
    }

    pub fn get_cached(&self, avatar_name: &str) -> Option<&Path> {
        self.cache
            .iter()
            .find(|(n, _)| n == avatar_name)
            .map(|(_, p)| p.as_path())
    }

    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for ThumbnailGenerator {
    fn default() -> Self {
        Self::new(PathBuf::from("thumbnails"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumbnail_default_request() {
        let req = ThumbnailRequest::default();
        assert_eq!(req.width, THUMBNAIL_WIDTH);
        assert_eq!(req.height, THUMBNAIL_HEIGHT);
    }

    #[test]
    fn thumbnail_placeholder_dimensions() {
        let gen = ThumbnailGenerator::default();
        let result = gen.generate_placeholder(&ThumbnailRequest::default());
        assert_eq!(result.width, THUMBNAIL_WIDTH);
        assert_eq!(result.height, THUMBNAIL_HEIGHT);
        assert_eq!(
            result.pixels.len(),
            (THUMBNAIL_WIDTH * THUMBNAIL_HEIGHT * 4) as usize
        );
    }

    #[test]
    fn thumbnail_placeholder_custom_size() {
        let gen = ThumbnailGenerator::default();
        let result = gen.generate_placeholder(&ThumbnailRequest {
            width: 64,
            height: 64,
            ..Default::default()
        });
        assert_eq!(result.width, 64);
        assert_eq!(result.height, 64);
        assert_eq!(result.pixels.len(), 64 * 64 * 4);
    }

    #[test]
    fn thumbnail_cache_store_and_retrieve() {
        let mut gen = ThumbnailGenerator::default();
        gen.cache_thumbnail("test_avatar", Path::new("thumbnails/test_avatar.png"));
        assert_eq!(gen.cache_len(), 1);
        let cached = gen.get_cached("test_avatar");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), Path::new("thumbnails/test_avatar.png"));
    }

    #[test]
    fn thumbnail_cache_overwrite() {
        let mut gen = ThumbnailGenerator::default();
        gen.cache_thumbnail("avatar", Path::new("old.png"));
        gen.cache_thumbnail("avatar", Path::new("new.png"));
        assert_eq!(gen.cache_len(), 1);
        assert_eq!(gen.get_cached("avatar").unwrap(), Path::new("new.png"));
    }

    #[test]
    fn thumbnail_cache_clear() {
        let mut gen = ThumbnailGenerator::default();
        gen.cache_thumbnail("a", Path::new("a.png"));
        gen.cache_thumbnail("b", Path::new("b.png"));
        gen.clear_cache();
        assert_eq!(gen.cache_len(), 0);
        assert!(gen.get_cached("a").is_none());
    }

    #[test]
    fn thumbnail_path_sanitizes_name() {
        let gen = ThumbnailGenerator::new(PathBuf::from("out"));
        let path = gen.thumbnail_path_for("My Avatar (v2).vrm");
        assert!(path.to_string_lossy().contains("My_Avatar__v2__vrm.png"));
    }

    #[test]
    fn thumbnail_save_to_disk() {
        let dir = std::env::temp_dir().join("vulvatar_thumbnail_test");
        let gen = ThumbnailGenerator::new(dir.clone());
        let result = gen.generate_placeholder(&ThumbnailRequest {
            avatar_name: "test".to_string(),
            ..Default::default()
        });
        let path = dir.join("test.png");
        let save_result = gen.save_to_disk(&result, &path);
        assert!(save_result.is_ok());
        assert!(path.exists());

        let loaded = image::open(&path).unwrap();
        assert_eq!(loaded.width(), THUMBNAIL_WIDTH);
        assert_eq!(loaded.height(), THUMBNAIL_HEIGHT);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
