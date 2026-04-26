//! Persistent on-disk cache for parsed `AvatarAsset`.
//!
//! The first load of a VRM does the full glTF parse + VRM-extension parse +
//! material decode + skeleton/morph build. The result is then bincode-encoded
//! into `%APPDATA%\VulVATAR\cache\<source_hash>.vvtcache` so subsequent
//! loads of the same VRM bytes can skip those expensive steps.
//!
//! Wire layout:
//! ```text
//! [VvtCacheHeader (bincode)] [AvatarAsset body (bincode)]
//! ```
//!
//! Texture pixel data is **not** included in the body (Option B from
//! `plan/handover.md`): `TextureBinding.pixel_data` carries
//! `#[serde(skip)]`, so the cache file stays small. Cache loaders MUST
//! re-decode textures from the on-disk VRM before handing the asset to
//! the renderer — see `vrm.rs`'s cache-rehydration helper.
//!
//! Validation walks the header fields in order of cost:
//!  1. magic + cache_version (cheap, rejects obviously-stale files)
//!  2. parser_version (matches `CARGO_PKG_VERSION`; bumped on every release,
//!     so a fresh build invalidates cached files automatically)
//!  3. source size + mtime (avoids the SHA-256 cost when the file hasn't
//!     changed)
//!  4. source SHA-256 (if size+mtime didn't match, we recompute)

use crate::asset::{AssetSourceHash, AvatarAsset};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// File magic — `VVTCACHE` little-endian.
const VVT_CACHE_MAGIC: [u8; 8] = *b"VVTCACHE";
/// Bump on incompatible header / wire-format changes.
const VVT_CACHE_VERSION: u32 = 1;
/// Default cap for `evict_to_count`. Configurable from callers.
pub const DEFAULT_MAX_CACHE_ENTRIES: usize = 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VvtCacheHeader {
    pub magic: [u8; 8],
    pub cache_version: u32,
    pub parser_version: String,
    pub source_hash: [u8; 32],
    pub source_size_bytes: u64,
    pub source_mtime_unix: u64,
}

impl VvtCacheHeader {
    fn for_source(source_hash: [u8; 32], source_size_bytes: u64, source_mtime_unix: u64) -> Self {
        Self {
            magic: VVT_CACHE_MAGIC,
            cache_version: VVT_CACHE_VERSION,
            parser_version: env!("CARGO_PKG_VERSION").to_string(),
            source_hash,
            source_size_bytes,
            source_mtime_unix,
        }
    }
}

/// Filesystem path the cache file for `source_hash` would live at. The
/// file may or may not exist — call [`try_load`] to actually read it.
pub fn cache_path_for_hash(source_hash: &[u8; 32]) -> PathBuf {
    let mut p = crate::persistence::cache_dir();
    p.push(format!("{}.vvtcache", hex_string(source_hash)));
    p
}

fn hex_string(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Try to load a cached `AvatarAsset` for the given source VRM. Returns
/// `Ok(None)` for a clean cache miss (file absent / stale / corrupt) so
/// the loader can fall through to the parse path; only structural
/// programming errors surface as `Err`.
pub fn try_load(source_path: &Path) -> Result<Option<AvatarAsset>, String> {
    let metadata = match fs::metadata(source_path) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let source_size_bytes = metadata.len();
    let source_mtime_unix = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Cheap probe before reading the (potentially expensive) full file:
    // walk the cache_dir looking for any header whose size+mtime match.
    // For the common case — same path + bytes between runs — the SHA-256
    // recompute cost is avoided entirely. Worst case (size or mtime
    // mismatch on a same-bytes file) we recompute and still get a hit.
    let cache_dir = crate::persistence::cache_dir();
    if !cache_dir.exists() {
        return Ok(None);
    }

    // We don't yet know the source_hash — read it lazily after a header
    // mismatch on size/mtime. Iterate all entries because the on-disk
    // filename is keyed by hash, not by source path.
    let entries = match fs::read_dir(&cache_dir) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };

    let mut computed_hash: Option<[u8; 32]> = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("vvtcache") {
            continue;
        }

        let (header, body_offset) = match read_header(&path) {
            Ok(parts) => parts,
            Err(_) => continue,
        };
        if !is_header_self_consistent(&header) {
            continue;
        }
        if header.source_size_bytes != source_size_bytes {
            continue;
        }
        if header.source_mtime_unix == source_mtime_unix {
            // Same size + mtime: trust without recomputing the hash.
            return read_body_at(&path, body_offset).map(Some);
        }
        // Size matches but mtime drifted (touch / copy-with-attrs). Fall
        // back to the SHA-256 check — cheaper to compute once and cache
        // for any further candidates in this dir scan.
        let actual_hash = match computed_hash {
            Some(h) => h,
            None => match compute_source_hash(source_path) {
                Ok(h) => {
                    computed_hash = Some(h);
                    h
                }
                Err(_) => return Ok(None),
            },
        };
        if header.source_hash == actual_hash {
            return read_body_at(&path, body_offset).map(Some);
        }
    }

    Ok(None)
}

/// Persist `asset` to disk under its source hash. Best-effort — caller
/// should ignore failure (logged) and proceed with the freshly parsed
/// asset. Subsequent loads will simply miss until the next save attempt.
pub fn save(source_path: &Path, asset: &AvatarAsset) -> Result<(), String> {
    let metadata = fs::metadata(source_path)
        .map_err(|e| format!("cache save: stat '{}' failed: {}", source_path.display(), e))?;
    let source_size_bytes = metadata.len();
    let source_mtime_unix = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let header = VvtCacheHeader::for_source(asset.source_hash.0, source_size_bytes, source_mtime_unix);

    let cache_dir = crate::persistence::cache_dir();
    fs::create_dir_all(&cache_dir).map_err(|e| format!("cache save: mkdir: {}", e))?;
    let target_path = cache_path_for_hash(&asset.source_hash.0);
    let tmp_path = target_path.with_extension("vvtcache.tmp");

    let header_bytes = bincode::serialize(&header)
        .map_err(|e| format!("cache save: header serialize: {}", e))?;
    let body_bytes = bincode::serialize(asset)
        .map_err(|e| format!("cache save: body serialize: {}", e))?;

    {
        let mut f = fs::File::create(&tmp_path)
            .map_err(|e| format!("cache save: create tmp '{}': {}", tmp_path.display(), e))?;
        // 4-byte length prefix so the reader knows where the body starts.
        let header_len = (header_bytes.len() as u32).to_le_bytes();
        f.write_all(&header_len)
            .map_err(|e| format!("cache save: write header len: {}", e))?;
        f.write_all(&header_bytes)
            .map_err(|e| format!("cache save: write header: {}", e))?;
        f.write_all(&body_bytes)
            .map_err(|e| format!("cache save: write body: {}", e))?;
        f.flush()
            .map_err(|e| format!("cache save: flush: {}", e))?;
    }
    fs::rename(&tmp_path, &target_path)
        .map_err(|e| format!("cache save: rename: {}", e))?;

    info!(
        "cache: saved '{}' ({} bytes header + {} bytes body)",
        target_path.display(),
        header_bytes.len(),
        body_bytes.len()
    );
    Ok(())
}

fn read_header(path: &Path) -> Result<(VvtCacheHeader, u64), String> {
    let mut f = fs::File::open(path).map_err(|e| format!("open: {}", e))?;
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)
        .map_err(|e| format!("read header len: {}", e))?;
    let header_len = u32::from_le_bytes(len_buf) as usize;
    if header_len > 1_000_000 {
        return Err(format!("header len {} too big", header_len));
    }
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes)
        .map_err(|e| format!("read header: {}", e))?;
    let header: VvtCacheHeader = bincode::deserialize(&header_bytes)
        .map_err(|e| format!("deserialize header: {}", e))?;
    Ok((header, 4 + header_len as u64))
}

fn read_body_at(path: &Path, body_offset: u64) -> Result<AvatarAsset, String> {
    use std::io::{Seek, SeekFrom};
    let mut f = fs::File::open(path).map_err(|e| format!("open body: {}", e))?;
    f.seek(SeekFrom::Start(body_offset))
        .map_err(|e| format!("seek body: {}", e))?;
    let mut body_bytes = Vec::new();
    f.read_to_end(&mut body_bytes)
        .map_err(|e| format!("read body: {}", e))?;
    bincode::deserialize::<AvatarAsset>(&body_bytes).map_err(|e| format!("deserialize body: {}", e))
}

fn is_header_self_consistent(header: &VvtCacheHeader) -> bool {
    if header.magic != VVT_CACHE_MAGIC {
        return false;
    }
    if header.cache_version != VVT_CACHE_VERSION {
        return false;
    }
    // Parser version is `CARGO_PKG_VERSION`; bumped on every release. A
    // mismatch invalidates the cache because the underlying schema may
    // have changed without us bumping `VVT_CACHE_VERSION`.
    if header.parser_version != env!("CARGO_PKG_VERSION") {
        debug!(
            "cache: parser version mismatch: cached={} current={}",
            header.parser_version,
            env!("CARGO_PKG_VERSION")
        );
        return false;
    }
    true
}

/// SHA-256 the source file once. Mirrors the hashing the loader already
/// does in `VrmAssetLoader::load`, kept inline here so a cache lookup
/// doesn't need to round-trip through the parser.
fn compute_source_hash(source_path: &Path) -> Result<[u8; 32], String> {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(source_path).map_err(|e| format!("read source: {}", e))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    Ok(out)
}

/// Re-compute the hash for an asset that was loaded from cache. Used by
/// the loader after rehydrating textures, in case the assets get a
/// runtime ID assigned that differs from what was cached.
pub fn rehash_source(source_path: &Path) -> Option<AssetSourceHash> {
    compute_source_hash(source_path).ok().map(AssetSourceHash)
}

// ---------------------------------------------------------------------------
// Maintenance: total size + clear + eviction sweep for the UI / startup
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub entry_count: usize,
    pub total_bytes: u64,
}

/// Walk the cache dir and total `.vvtcache` byte usage. Used by the
/// inspector's Cache section so the user knows what to expect from a
/// "Clear" press without opening Explorer.
pub fn stats() -> CacheStats {
    let dir = crate::persistence::cache_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return CacheStats::default(),
    };
    let mut count = 0usize;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        if entry.path().extension().and_then(|s| s.to_str()) != Some("vvtcache") {
            continue;
        }
        count += 1;
        bytes += entry.metadata().map(|m| m.len()).unwrap_or(0);
    }
    CacheStats {
        entry_count: count,
        total_bytes: bytes,
    }
}

/// Remove every `.vvtcache` file. Called from the inspector "Clear"
/// button. Returns the number of entries removed.
pub fn clear_all() -> Result<usize, String> {
    let dir = crate::persistence::cache_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Ok(0),
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("vvtcache") {
            continue;
        }
        if fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    info!("cache: cleared {} entries", removed);
    Ok(removed)
}

/// Drop `.vvtcache` files past the most-recently-modified `max_count`.
/// Called from `Application::new` on startup so the cache doesn't grow
/// unbounded. Errors are logged + swallowed because eviction is
/// best-effort.
pub fn evict_to_count(max_count: usize) {
    let dir = crate::persistence::cache_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut files: Vec<(PathBuf, SystemTime)> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("vvtcache") {
                return None;
            }
            let mtime = e
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            Some((path, mtime))
        })
        .collect();
    if files.len() <= max_count {
        return;
    }
    // Newest first: keep the head, drop the tail.
    files.sort_by(|a, b| b.1.cmp(&a.1));
    let mut evicted = 0;
    for (path, _) in files.into_iter().skip(max_count) {
        if let Err(e) = fs::remove_file(&path) {
            warn!("cache: evict '{}' failed: {}", path.display(), e);
        } else {
            evicted += 1;
        }
    }
    if evicted > 0 {
        info!(
            "cache: evicted {} entries (cap {})",
            evicted, max_count
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_self_consistent() {
        let h = VvtCacheHeader::for_source([0u8; 32], 0, 0);
        assert!(is_header_self_consistent(&h));
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut h = VvtCacheHeader::for_source([0u8; 32], 0, 0);
        h.magic = *b"NOTACACH";
        assert!(!is_header_self_consistent(&h));
    }

    #[test]
    fn header_rejects_old_cache_version() {
        let mut h = VvtCacheHeader::for_source([0u8; 32], 0, 0);
        h.cache_version = VVT_CACHE_VERSION + 1;
        assert!(!is_header_self_consistent(&h));
    }

    #[test]
    fn cache_path_uses_hex_hash() {
        let hash = [0u8; 32];
        let p = cache_path_for_hash(&hash);
        assert!(p.to_string_lossy().contains("0000000000"));
        assert!(p.extension().and_then(|s| s.to_str()) == Some("vvtcache"));
    }
}
