# Profiling the Render Pipeline

## Purpose

This document describes how to measure per-frame performance in VulVATAR
and lists the known bottlenecks and their mitigations.

Related documents:

- [threading-model.md](threading-model.md)
- [vulkano-renderer-design.md](vulkano-renderer-design.md)
- [gpu-runtime-roadmap.md](gpu-runtime-roadmap.md)

## Quick Start

1. Add instrumentation (see below).
2. Build and run with logging enabled:
   ```
   RUST_LOG=vulvatar=info cargo run
   ```
3. Redirect stderr to `profile/` (gitignored) for post-hoc analysis:
   ```
   RUST_LOG=vulvatar=info cargo run 2> profile/run_$(date +%Y%m%d_%H%M%S).log
   ```
4. Grep for the `PROFILE` or `HARVEST` tags in the log.

For GPU-runtime regression checks, also grep for `GPU_RUNTIME`. The renderer
logs cumulative steady-state counters every 60 frames; the field names
follow the transform prepass layout (single `TransformGpuData` slot per
primitive, fused skinning + morph + cloth dispatch):

```text
GPU_RUNTIME frame=600 transform_resources=42 ubo_writes=25140 \
cloth_writes=180 cloth_creations=2 cache_slots=42 readback_bytes=8294400
```

After avatar warm-up:

- `transform_resources` and `cloth_creations` should plateau (the former
  at total visible primitive count, the latter at the count of
  cloth-bearing primitives) — any continued growth means a slot is being
  evicted and recreated every frame.
- `cache_slots` mirrors `transform_resources` minus any user-driven
  cache clears.
- `ubo_writes` and `cloth_writes` keep growing because they represent
  cheap in-place updates: one `control_ubo` write per visible primitive
  per frame, and one `cloth_pos_ssbo` rewrite per cloth-bearing
  primitive each time the solver's `version` advances.

## Instrumentation Pattern

The profiling code is **not checked in** — it is added on-demand and removed
before committing.  Below is a copy-paste recipe.

### GUI thread (`src/gui/mod.rs`)

Add fields to `GuiApp`:

```rust
pub prof_gui_draw_ms: f64,
pub prof_viewport_upload_ms: f64,
pub prof_run_frame_ms: f64,
```

Wrap `app.run_frame()`:

```rust
let t_run = Instant::now();
self.app.run_frame(&frame_config);
self.prof_run_frame_ms = t_run.elapsed().as_secs_f64() * 1000.0;
```

Wrap the GUI draw section:

```rust
let t_gui = Instant::now();
top_bar::draw(ctx, self);
// ... other draw calls ...
viewport::draw(ctx, self);
self.prof_gui_draw_ms = t_gui.elapsed().as_secs_f64() * 1000.0;
```

### Viewport texture upload (`src/gui/viewport.rs`)

Wrap the `rendered_pixels()` → `handle.set()` block:

```rust
let t_vp = std::time::Instant::now();
let has_rendered_image = if let Some((pixels, extent)) = state.app.rendered_pixels() {
    // ... existing upload code ...
};
state.prof_viewport_upload_ms = t_vp.elapsed().as_secs_f64() * 1000.0;
```

### Application frame (`src/app/mod.rs`)

Add fields to `Application`:

```rust
pub prof_sim_ms: f64,
pub prof_render_submit_ms: f64,
pub prof_render_recv_ms: f64,
pub prof_process_ms: f64,
```

Wrap each section of `run_frame()` with `Instant::now()` / `elapsed()`.

### Render thread (`src/renderer/mod.rs`)

Add fields to `RenderStats`:

```rust
pub prof_harvest_ms: f64,
pub prof_cb_build_ms: f64,
pub prof_submit_ms: f64,
```

In `render()`, measure three phases:

- **harvest**: `harvest_pending_readback()` — time waiting on previous frame's
  GPU fence.
- **cb_build**: from after harvest to just before `then_execute()` — command
  buffer construction including skinning writes, descriptor binds, draw calls.
- **submit**: `then_execute()` + `then_signal_fence_and_flush()` — Vulkan queue
  submission overhead.

Inside `harvest_pending_readback()`, break down further:

```rust
let t0 = std::time::Instant::now();
(pending.wait_fn)()?;
let fence_ms = t0.elapsed().as_secs_f64() * 1000.0;

let t1 = std::time::Instant::now();
let pixel_data = pending.readback_buffer.read()...to_vec();
let copy_ms = t1.elapsed().as_secs_f64() * 1000.0;

log::info!("HARVEST fence_wait={:.1}ms copy={:.1}ms", fence_ms, copy_ms);
```

### Log output

Add a periodic summary in `update()`:

```rust
if self.frame_count % 60 == 0 && self.frame_count > 0 {
    let rs = &self.app.prof_render_stats;
    info!(
        "PROFILE frame={} total={:.1}ms | run_frame={:.1}ms (sim={:.1} ...) \
         | gui_draw={:.1}ms (vp_upload={:.1}) \
         | render: harvest={:.1} cb={:.1} gpu_submit={:.1}",
        ...
    );
}
```

### Debug environment variables

For A/B testing, you can gate code behind `std::env::var()`:

| Variable | Effect |
|---|---|
| `VULVATAR_SKIP_READBACK=1` | Skip `copy_image_to_buffer` + `copy_buffer` in the render CB |
| `VULVATAR_SKIP_DRAW=1` | Skip all draw calls (clear-only render pass) |
| `VULVATAR_LOW_RES=1` | Force 640x480 output resolution |

These are **not** checked in — add them temporarily in `render()` / `run_frame()`
when needed, then remove before committing.

## Render Benchmark (`bench_render`)

`src/bin/bench_render.rs` renders an avatar N times through the full
`VulkanRenderer` path with animated expression weights and head motion,
reporting per-frame wall time (the loop times `render()` only, and the
async fence means the wall approximates max(CPU submit, GPU exec)):

```powershell
cargo run --bin bench_render -- sample_data\YUMEKA_v1.0.3\FBX\Yumeka_v1.0.3.fbx 300 1920 1080
```

`VULVATAR_BENCH_EXPRS=N` sets how many expressions animate (morph cost
scales with the ACTIVE count only — see the compacted gather below).
`VULVATAR_BENCH_BG=1` enables the generative background (exercises the
uniform ring + its interaction with the CB cache). With
`VULVATAR_BENCH_DUMP=<dir>` every harvested frame's pixels are written as
raw RGBA; because the animation is deterministic in the frame index, two
runs differing only in a renderer knob must produce byte-identical dumps —
that is the equivalence gate used for the CB cache / pixel pool.

### Measured attribution (Yumeka v1.0.3, 23 prims / 252,789 verts, Arc B580, 1080p)

```text
full frame (2026-09, before CB cache / pixel pool)
                        ~5.2-5.9 ms median
  harvest (fence+8MB)    ~1.5-2.5 ms   <- to_vec alloc+copy dominated
  cb_build (CPU)         ~2.5-3.5 ms   <- dominated by vulkano builder.build() (~2 ms)
    prepass recording    ~0.3-0.5 ms
    scene+post record    ~0.2-0.35 ms
  submit                 ~0.35 ms
compute prepass (GPU)    ~1.4-1.7 ms   <- was ~1.7 ms before the compacted morph gather
draw pass                ~0.2 ms

full frame (2026-09, CB cache + pixel pool ON)
                        ~3.7 ms median / ~1.0 ms min
```

After the two CPU-side optimizations below, the 1080p median sits at the
GPU-execution floor (~3.7 ms ≈ max(CPU ~1 ms, GPU exec)); further CPU work
would not move the median. The two A/B knobs (default on) exist so a live
regression can be bisected without a rebuild:

| Knob | Off-switch | What it isolates |
|---|---|---|
| Command-buffer cache | `VULVATAR_CB_CACHE=0` | prepare/record split + shape-key reuse of built CBs |
| Pixel-Vec pool | `VULVATAR_PIXEL_POOL=0` | zero-allocation harvest of the readback ring |

Both were verified pixel-identical byte-for-byte across a 149-frame
animated `bench_render` run (`VULVATAR_BENCH_DUMP` A/B compare, including
generative-background frames and cache-hit steady state).

### Command-buffer cache (prepare/record split)

`render()` used to interleave CPU writes (camera, control UBOs, morph
weights, cloth controls, material uniforms) with Vulkan recording and pay
`AutoCommandBufferBuilder::build()`'s dependency resolution (~2 ms) every
frame. The loop is now split: `prepare_frame` (`src/renderer/frame_plan.rs`)
performs every CPU write and captures the dispatch structure + a shape key
hashing every command-buffer-visible identity (descriptor-set/pipeline Arc
pointers, dispatch counts, push-constant bits, clear colour, ring slots);
`get_or_build_frame_cb` returns the previously built command buffer when
the key matches. Cached buffers are recorded with
`CommandBufferUsage::SimultaneousUse` and pin their resources, so pointer
keys cannot go stale through allocator reuse; every site that swaps a
CB-visible resource clears the cache (resize, format/MSAA change, avatar
swap, transform-slot rebuild, skinning realloc, cloth-slot (re)build,
readback/depth buffer realloc).

Prerequisite: the generative background's push constants (which animate
via `time` + tracking anchors every frame) moved to a 2-slot uniform ring
(`background.rs`), so an animated background no longer changes the command
buffer — steady state hits ~113/120 frames (7 misses = the ring-shape
warm-up). Ring shapes multiply: camera 3 × readback 2 × bg 2; the LRU cap
is 16.

### Zero-allocation pixel harvest

`harvest_pending_readback` used to `read().to_vec()` a fresh full-frame
Vec every frame (8.3 MB at 1080p — large-block virtual alloc + first-touch
faults). The readback ring now keeps one owned Arc per slot; at harvest,
`Arc::try_unwrap` reclaims the previous frame's allocation when consumers
have dropped it (the common case — counters show ~117 reuses vs 2 allocs
per 120 frames) and the GPU bytes are copied into the warm pages. When a
consumer still holds the old Arc the unwrap fails and a fresh Vec is
allocated — correct, just slower. The depth-aspect readback buffer
(`validate_gt` benches) is likewise cached per extent instead of
reallocated every frame.

The remaining harvest cost is the single 8 MB memcpy out of the mapped
ring buffer (~0.4 ms) — the price of the `Arc<Vec<u8>>` handoff API
shared by the preview / output worker / sinks. A genuinely zero-copy
lease (consumers borrowing the mapped memory) would require widening
`ExportedPixelData::CpuReadback` past `Arc<Vec<u8>>` through every
consumer plus a tear-free lifetime contract (consumers currently may hold
a frame past one ring rotation); revisit only if the 0.4 ms matters.

### GPU cloth dispatch note

The cloth Verlet/XPBD/normal dispatch recording was mechanically mirrored
into the plan path and is exercised by `diagnose_cloth` on the CPU-solver
backend; the GPU-solver dispatch structure (`VULVATAR_CLOTH_GPU=1`) is
trace-verified but not yet runtime-benched — eyeball the first live GPU
cloth session after this change.

### Compacted morph gather

`transform_cs`'s per-vertex morph loop is bounded by the control
block's `target_count`, which used to be the FULL authored target count
(446 on Yumeka's face — 26,880 verts × 446 = ~12 M loop iterations per
frame that only load a zero weight and continue). The render loop now
compacts the gather to the active targets each frame: the active
targets' info rows are written into the head of `morph_infos` and the
matching weights into the head of `morph_weights`, and `target_count`
is set to the active count. The entries buffer is untouched (info rows
are absolute indices into it), so the shader performs the same binary
searches over a shorter loop. Verified pixel-identical (max diff 0 on
an expression-weighted render).

## Known Bottlenecks and Solutions

### 1. Intel Arc: slow `copy_image_to_buffer` to host memory (30 ms)

**Symptom**: `fence_wait` is ~30 ms regardless of scene complexity or
resolution.  Skipping all draw calls doesn't help; skipping the readback
copy drops it to 0.1 ms.

**Root cause**: On Intel Arc (B-series tested), `vkCmdCopyImageToBuffer`
targeting a `PREFER_HOST | HOST_RANDOM_ACCESS` buffer triggers a slow DMA
transfer path in the driver (~30 ms for 1920x1080 RGBA).

**Solution**: Two-stage readback.

1. `copy_image_to_buffer` → device-local + host-visible (resizable BAR) staging
   buffer.  This stays on-chip and completes in < 1 ms.
2. `copy_buffer` → `PREFER_HOST | HOST_RANDOM_ACCESS` readback buffer.  This is
   a GPU-side memcpy that also completes in < 1 ms.
3. CPU reads from the host-cached readback buffer (2 ms for `to_vec()`).

The BAR staging buffer falls back to plain host-visible if resizable BAR is
unavailable.

### 2. egui texture upload in debug builds (42 ms)

**Symptom**: `vp_upload` is ~42 ms.

**Root cause**: `egui::ColorImage::from_rgba_unmultiplied` and `handle.set()`
perform 8 MB copies in unoptimised (opt-level 0) code.  The egui/epaint
crates have tight inner loops that benefit enormously from compiler
optimisation.

**Solution**: Per-crate `opt-level = 2` in `Cargo.toml` for
egui, epaint, eframe, emath, egui-winit, egui_glow, and glow.  This brings
`vp_upload` down to ~2 ms without requiring a full release build.

### 3. Synchronous GPU readback blocking the render thread

**Symptom**: `fence.wait(None)` called in the same frame as the GPU
submission, blocking the render thread for the full GPU render time.

**Solution**: Async readback with `PendingReadbackState`.
The GPU fence from frame N is waited on at the start of frame N+1.  By that
point the GPU work has already completed, so the wait returns instantly.

## Typical Profile (Healthy)

```
PROFILE total=16.6ms
  run_frame=1.4ms (sim=0.0 submit=0.0 recv=0.0 process=0.3)
  gui_draw=2.6ms (vp_upload=2.4)
  render: harvest=1.5 cb=1.1 gpu_submit=0.2
  meshes=17 mats=17
```

60 FPS, vsync-limited.  All budget items well under 5 ms.

## GPU Runtime Soak

Use this after changes to morphs, cloth, output handoff, or GPU scheduling.

### Scenario

1. Start with:
   - face tracking on
   - cloth on
   - 1920x1080 output
   - virtual camera or another live output sink active
2. Run for at least 30 minutes with logs enabled:
   ```powershell
   $stamp = Get-Date -Format yyyyMMdd_HHmmss
   $env:RUST_LOG = "vulvatar=info"
   cargo run 2> "profile\\gpu_soak_$stamp.log"
   ```
3. Keep normal user motion in frame so morph weights and cloth keep updating.

### Inspect

- `GPU_RUNTIME`
  - `transform_resources`
  - `cloth_creations`
  - `cache_slots`
- render cadence / frame-time summaries
- output queue depth and dropped frames
- active handoff path / fallback warnings

### Healthy signature

- `transform_resources` stops growing after warm-up (the compute prepass
  reuses each per-primitive slot)
- `cloth_creations` stops after the first cloth-on frame (and only ticks
  again if the user toggles a cloth overlay on a primitive that wasn't
  previously cloth-bearing — that case evicts the slot and rebuilds)
- only `ubo_writes` and `cloth_writes` continue growing
- no device loss
- no tracking-worker restart refusal unless intentionally provoked
- no unexpected output fallback activation

### Failure signatures worth saving

- `VIDEO_TDR_FAILURE`, device loss, or sudden process termination
- `tracking: refusing restart` without a deliberate stop/start test
- `GPU_RUNTIME` creation counters that keep rising during steady-state motion
- output queue depth or dropped-frame count that trends upward without recovery
- handoff falling back from GPU to CPU/shared memory unexpectedly
