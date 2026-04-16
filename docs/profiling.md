# Profiling the Render Pipeline

## Purpose

This document describes how to measure per-frame performance in VulVATAR
and identifies the known bottlenecks discovered during initial tuning.

Related documents:

- [threading-model.md](threading-model.md)
- [renderer-implementation-brief.md](renderer-implementation-brief.md)

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

## Known Bottlenecks and Solutions

### 1. Intel Arc: slow `copy_image_to_buffer` to host memory (30 ms)

**Symptom**: `fence_wait` is ~30 ms regardless of scene complexity or
resolution.  Skipping all draw calls doesn't help; skipping the readback
copy drops it to 0.1 ms.

**Root cause**: On Intel Arc (B-series tested), `vkCmdCopyImageToBuffer`
targeting a `PREFER_HOST | HOST_RANDOM_ACCESS` buffer triggers a slow DMA
transfer path in the driver (~30 ms for 1920x1080 RGBA).

**Solution** (implemented): Two-stage readback.

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

**Solution** (implemented): Per-crate `opt-level = 2` in `Cargo.toml` for
egui, epaint, eframe, emath, egui-winit, egui_glow, and glow.  This brings
`vp_upload` down to ~2 ms without requiring a full release build.

### 3. Synchronous GPU readback blocking the render thread (historical)

**Symptom**: `fence.wait(None)` called in the same frame as the GPU
submission, blocking the render thread for the full GPU render time.

**Solution** (implemented): Async readback with `PendingReadbackState`.
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
