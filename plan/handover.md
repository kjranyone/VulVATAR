# T10 Backlog Handover — Output Color Space & Offline Bake

This is the resume sheet for the two T10 items that didn't fit into a
single implementation session: **Output color space (sRGB / Linear
sRGB)** and **Offline bake / cache generation**. Both have a design
recorded here; both still need decisions before code lands.

The rest of the T10 backlog (avatar library, tracking, cloth, multi-
overlay, version migration, GPU inference, real-render thumbnails,
partial rebinding) shipped — see `plan/T10-future-extensions.md` for
status and the git history (`git log --oneline -- plan/`) for what
landed when.

## Context for a fresh session

If you're picking this up cold:

- The codebase is split into `src/app` (state + render-thread
  control), `src/renderer` (Vulkano-based renderer with offscreen
  targets, output export, thumbnail rendering), `src/gui` (eframe /
  egui UI), `src/asset` (VRM loading + cloth schema + the recently
  added `cloth_rebind`), `src/tracking` (CIGPose ONNX inference + an
  optional YOLOX person crop), `src/output` (file-backed shared
  memory virtual camera + image sequence sinks), and
  `src/persistence` (project files, library, cloth overlay,
  recovery snapshots, watched folders).
- Render thread architecture lives in `src/app/render_thread.rs`
  with a `RenderCommand` enum + matching response channels. The
  recent thumbnail work is the canonical pattern for adding a new
  render-thread command without breaking the pipelined main viewport
  readback flow.
- Persistence has a real two-stage migration framework
  (`load_project` / `load_cloth_overlay` go through
  `serde_json::Value` first, version-check, optionally migrate, then
  deserialise). Format constants live near the top of
  `src/persistence.rs`.

---

## Item 1 — Output Color Space (sRGB / Linear sRGB)

### What's there now

`src/renderer/output_export.rs` hardcodes `color_space: "srgb"` in
`ExportMetadata`, which is what eventually ends up on
`OutputFrame.export_metadata` and (for the MF virtual camera) on the
file-backed frame buffer. The GUI inspector exposes a `color_space_index`
combo that drives `OutputGuiState.output_color_space_index` (0 = sRGB,
1 = Linear sRGB), but the index is captured into `ProjectState`
already and goes nowhere downstream — the renderer ignores it, and so
does the MF camera DLL.

The sink chain that consumes `RenderColorSpace` from
`OutputTargetRequest` does exist in the renderer's output_export
codepath, but the actual render target format and shader gamma are
not wired to that field.

### What needs to land

Four layers, all of which need to agree on one of two states (sRGB or
Linear sRGB) per frame:

1. **Render target format**. The offscreen color attachment is created
   somewhere in `VulkanRenderer::resize` / target allocation
   (`src/renderer/targets.rs`). For sRGB output we want an sRGB
   format (e.g. `B8G8R8A8_SRGB`); for linear we want the linear
   variant (e.g. `B8G8R8A8_UNORM`). When the requested color space
   changes, the targets must be recreated.
2. **MToon shader output gamma**. The fragment shader path in
   `src/renderer/pipeline.rs` (and any others that write to the color
   target) ends with a final color value. With an sRGB-format
   attachment, the GPU performs linear → sRGB encoding on store, so
   the shader should write *linear* color. With a UNORM attachment
   in "Linear sRGB" output mode, the shader should *also* write
   linear and skip any final encoding. Most existing shader paths
   assume sRGB target and don't do an explicit encode/decode, which
   means switching to UNORM may show a darkened image until the
   pipeline path is audited.
3. **MF sample attributes**. When the virtual camera path is on, the
   producer side writes RGBA bytes to the file-backed frame buffer
   (`src/output/frame_sink.rs::Win32FileBackedSharedMemorySink`). The
   consumer side is `vulvatar-mf-camera/src/media_stream.rs` which
   constructs the IMFSample. Two attributes describe the colour:
   `MF_MT_VIDEO_PRIMARIES` and `MF_MT_TRANSFER_FUNCTION`. They're
   currently unset, which means clients (Meet, Teams, OBS) infer
   sRGB. For Linear sRGB output we'd need to set
   `MFVideoPrimaries_BT709` + `MFVideoTransFunc_Linear` on the
   sample's media type.
4. **GUI active-ness**. Once 1–3 are wired, the inspector combo
   actually does something. Until then it should keep working as a
   stored preference (no functional change).

### Decision points (need user input before coding)

- **a. UNORM (with shader-side encode) vs sRGB-format target**: which
  is the canonical path? UNORM + manual encode gives more control
  (tone mapping, exposure curves) but every shader needs to be
  audited. sRGB format gives the GPU an automatic linear→sRGB
  encode at store. Recommendation: sRGB-format target as default
  (matches current behaviour); Linear sRGB option recreates targets
  with UNORM and adds a "no encode" variant of the final blit.
- **b. Premultiplied alpha and color space interaction**: when both
  the alpha mode (Premultiplied / Straight) and color space (sRGB /
  Linear) can change independently, the resulting matrix is 4
  combinations. Need a regression image for each before declaring
  the work done. A small CLI (`src/bin/render_matrix.rs`?) that
  renders the same scene to all 4 outputs and writes them as PNGs
  would catch this.
- **c. Default for new projects**: sRGB. Don't change for existing
  projects (`color_space_index` already persists; serde-default
  yields 0 = sRGB, which is correct).

### Suggested staging

Order matters here because each phase needs the previous one's
plumbing to actually do something visible.

- **Stage 1 — Plumbing only.** Take the GUI's `color_space_index`,
  feed it through to `OutputTargetRequest.color_space` already on
  `RenderFrameInput`. Wire the renderer to read it. No actual
  format changes; the field is just observed and logged. This
  proves the routing without breaking anything.
- **Stage 2 — Render target format selection.** When
  `OutputTargetRequest.color_space` changes between renders, recreate
  the offscreen color targets with the corresponding format
  (`B8G8R8A8_SRGB` ↔ `B8G8R8A8_UNORM`). Surfaces in
  `src/renderer/targets.rs` (look for the `ImageCreateInfo` for the
  color attachment). Visual regression is expected on the UNORM
  branch; that's where stage 3 picks up.
- **Stage 3 — Shader gamma audit.** For each fragment shader that
  writes to the color target, decide whether it needs an explicit
  linear→sRGB encode at the end. With UNORM target, NONE of them
  should do an encode (the data should remain linear). With sRGB
  target, NONE either (the GPU does it on store). The "encode at the
  end" path probably doesn't exist today, so the audit may be a
  no-op other than confirming each shader writes linear.
- **Stage 4 — MF sample attributes.** Modify `media_stream.rs` in
  `vulvatar-mf-camera` to read the producer's intended color space
  from a new field in the file-backed frame buffer header (header is
  defined in `src/output/frame_sink.rs` and consumed in
  `vulvatar-mf-camera/src/shared_memory.rs`). 4-byte field at the
  end of the header would do — bump some reserved bytes. Set
  `MF_MT_VIDEO_PRIMARIES` and `MF_MT_TRANSFER_FUNCTION` on the
  sample accordingly.
- **Stage 5 — Validation.** Build the regression CLI from
  decision-point (b) and lock it into a manual smoke test. CI is
  unlikely to be useful here (needs a GPU), so a "before I close
  the PR I rendered this scene to all 4 outputs and they look
  right" checklist is the bar.

### Files that will change

| File | What |
|---|---|
| `src/renderer/frame_input.rs` | Already has `RenderColorSpace`; nothing to change. |
| `src/renderer/output_export.rs` | Replace hardcoded `"srgb"` with the requested space. |
| `src/renderer/targets.rs` | Pick format based on color space; recreate on change. |
| `src/renderer/mod.rs` | Resize / recreate trigger when color space changes mid-session. |
| `src/renderer/pipeline.rs` | Audit shader output (most likely no-op). |
| `src/output/frame_sink.rs` | Header field for color space metadata. |
| `vulvatar-mf-camera/src/shared_memory.rs` | Mirror header field. |
| `vulvatar-mf-camera/src/media_stream.rs` | Set MF media type attributes. |
| `src/gui/inspector.rs` | Tooltip on the existing combo explaining the trade-off. |
| `src/bin/render_matrix.rs` (new) | Validation CLI. |

### Risk

Medium-high. Color management is full of sign mistakes. The audit
in stage 3 is the most error-prone step; expect a "shipped sRGB,
client says it's washed out" feedback cycle. The validation CLI
mitigates this but only if it's used.

---

## Item 2 — Offline Bake / Cache Generation

### Scope is undecided

This is the open-scope item. "Offline bake" can mean three different
things, each with different value and effort:

- **B1. Avatar load cache.** Cache the parsed VRM (mesh data, skin
  bindings, materials, spring bones, colliders) in
  `%APPDATA%\VulVATAR\cache\<asset_hash>.vvtcache` so the second
  load of the same VRM is 5–10× faster. The bottleneck on cold load
  is glTF parsing + VRM extension parsing + material decoding;
  these are all deterministic given the source bytes.

  - Effort: medium. Mostly serialization plumbing.
  - User value: high. VRMs are getting larger and the dev loop is
    a lot of "load avatar, tweak code, restart, reload avatar".
  - Risk: low. The cache is keyed by asset hash + parser version; a
    parser change invalidates everything.

- **B2. Cloth rest-pose bake.** Run the cloth solver for N steps from
  rest pose until the simulation has settled, then store the final
  particle positions in the overlay so the first few frames of a
  new attach don't show a visible "jiggle into place".

  - Effort: small. The solver step already exists; this is
    "iterate it until convergence at save time, store the result".
  - User value: low–medium. Most overlays don't have a visible
    initial jiggle.
  - Risk: low. Settled positions are an optional optimization;
    when absent, current behavior (start from rest, settle at
    runtime) takes over.

- **B3. Library thumbnail batch.** Add a UI action that walks every
  library entry, loads the avatar, kicks the existing real-render
  thumbnail flow, and writes the result. Effectively a "regenerate
  all thumbnails" menu item.

  - Effort: small. The plumbing all exists post-thumbnail-work.
  - User value: low. Thumbnails are already auto-generated on
    avatar load.
  - Risk: low. Just calls existing functions on a loop.

### Recommendation

**B1 is the highest-value option.** B2 is close-to-irrelevant for
typical authoring. B3 is a small QoL win but not worth its own task.

Pick B1 for the next implementation session. The text below is the
sketch of how it would land.

### B1 — Avatar Load Cache: design

**Storage path**: `%APPDATA%\VulVATAR\cache\<source_hash>.vvtcache`.
The hash is `AvatarAsset.source_hash` which already exists and is
SHA-256 of the VRM bytes. Self-keyed; the cache file stands alone.

**Wire format**: bincode (probably), wrapping a struct that mirrors
the live `AvatarAsset` minus the `Arc`s. Mesh primitives, skin
bindings, materials, humanoid map, spring bones, colliders,
expression set, animation clips, root_aabb, vrm_meta. NOT the
runtime fields (`id`, `next_avatar_instance_id`).

The gotcha is that several `AvatarAsset` substructs aren't currently
`Serialize` / `Deserialize`. Adding the derives needs an audit of
each one — `MeshPrimitiveAsset` carries `Option<VertexData>` which
is `Vec<Vec3>` etc., already Serializable. Materials have texture
arcs that may or may not be in scope. The cache should probably
*not* include the decoded texture pixel data (those are large and
re-decodable from the original VRM); reference textures by index
and re-decode from the on-disk VRM on cache load.

**Cache header**:
```rust
struct VvtCacheHeader {
    magic: [u8; 8],           // b"VVTCACHE"
    cache_version: u32,        // bump when schema changes
    parser_version: String,    // env!("CARGO_PKG_VERSION") — invalidates on rebuild
    source_hash: [u8; 32],
    source_size_bytes: u64,
    source_mtime_unix: u64,
}
```

**Validation on load**:
1. File exists at the expected path.
2. Magic + cache_version match.
3. parser_version matches current build (otherwise cache is stale —
   schema may have changed). Strict equality, not semver.
4. source_hash matches the on-disk VRM's hash. (Recomputing the
   hash isn't free; cheaper to compare size + mtime first and only
   recompute hash on size+mtime mismatch.)

**Insertion point**: `VrmAssetLoader::load(path)` in
`src/asset/vrm.rs`. Pseudocode:

```rust
pub fn load(&self, path: &str) -> Result<Arc<AvatarAsset>, VrmLoadError> {
    let metadata = std::fs::metadata(path)?;
    let cache_key = derive_cache_key(path, &metadata);

    if let Some(cached) = try_load_cache(&cache_key) {
        if cached.is_valid_for(path, &metadata) {
            return Ok(cached.into_avatar_asset());
        }
    }

    let asset = self.parse_full(path)?;
    let _ = save_cache(&cache_key, &asset); // best-effort
    Ok(asset)
}
```

The cache miss path is identical to today's load path; the cache hit
path skips the (expensive) glTF parse / VRM extension parse / material
decode work.

**What about textures?** Two options:
- **Option A**: include decoded texture RGBA bytes in the cache.
  Fastest reload but cache files are huge (8K texture = 256 MB
  decoded). Probably not worth it.
- **Option B**: re-decode textures from the source VRM on every
  load, even with a cache hit. Texture decoding is fast compared to
  glTF parsing for moderately-sized VRMs. Recommendation: Option B.

**Cache eviction**: keep the last N entries by mtime, with N
configurable. Default 20. Run a sweep at startup.

**Files that change for B1**:

| File | What |
|---|---|
| `src/asset/vrm.rs` | Hook the cache check into `VrmAssetLoader::load`. |
| `src/asset/cache.rs` (new) | All the cache (de)serialization logic. |
| `src/asset/mod.rs` | Add `Serialize` / `Deserialize` to the substructs that lack them; new `pub mod cache`. |
| `src/persistence.rs` | New `cache_dir() -> PathBuf` matching `thumbnails_dir()`. |

**Open decisions for B1**:

- **a. Bincode vs `serde_json`**: bincode is much faster + smaller
  but binary so debugging is harder. `serde_json` is human-readable
  but several × larger and slower. Recommendation: `bincode`. The
  cache files are computed artifacts; debugging would be done by
  invalidating + reparsing.
- **b. Where to gate it**: a feature flag (`avatar-load-cache`) so
  builds without it work the same as today, OR always-on with the
  cache silently no-op if creation fails. Recommendation:
  always-on, fallible — no point in a feature flag for a pure
  optimization.
- **c. Multiple-avatar cache invalidation when CIGPose model
  changes etc.**: irrelevant. This is per-avatar, not per-tracker.

### Files that will change

For B1 specifically, see the table above. B2 and B3 don't justify
their own files; they'd be small additions to existing modules.

### Risk

Low. Worst case the cache is silently stale and load falls back to
the parser path.

---

## Pick-up checklist

When the next session opens this file:

1. Decide which item to work on. Output color space and B1 are both
   stand-alone — they don't depend on each other.
2. For Output color space, sign off on decisions a / b / c above.
   Then start at Stage 1 of the staging plan.
3. For Offline bake, sign off on B1 vs B2 vs B3, then on the B1 open
   decisions a / b / c.
4. Either implementation will land in 1–3 commits. Update
   `plan/T10-future-extensions.md` to reflect status as you go.

If picking up after a long gap, check `git log --oneline -20` to see
what landed last; this file is the design state but not the live
state of the codebase.
