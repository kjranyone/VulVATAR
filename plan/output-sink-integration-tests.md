# Output sink end-to-end integration tests

Date: 2026-05-20

## Goal

Add end-to-end tests that cover the full `RenderExportMode → ExportedFrame
→ OutputFrame → OutputSinkWriter` path so a future refactor cannot silently
break the capability-routing decision landed in
[`afab5c5`](https://github.com/kjranyone/VulVATAR/commit/afab5c5).

Today's coverage is unit-level on each layer (export pool, frame_sink
capability match, router queue policy). What's missing is the *composed*
contract: when `FrameSink::SharedTextureFileStub` is active, an
`ExportedFrame { pixel_data: GpuFrameToken(...) }` produced by the renderer
must round-trip through `OutputRouter::publish` and land as a `VGTK` record
on disk — and when `FrameSink::SharedMemory` is active, the same renderer
output (under `CpuReadback`) must land as the Win32 file-backed bytes.

## Why

The P2-05 first slice introduced an enum/writer agreement (`FrameSink::
supports_gpu_tokens()` vs `OutputSinkWriter::supports_gpu_tokens()`) and a
runtime branch (`Application::run_frame` picks `GpuExport` vs `CpuReadback`
based on the active sink). A drift in any one of three places now silently
breaks live output:

1. someone adds a new `FrameSink` variant and forgets the enum-level method
2. someone changes the renderer's `export_gpu` to also emit a CPU shadow,
   and the router's diagnostics misclassify the path
3. someone changes the writer factory and produces a writer whose
   `supports_gpu_tokens()` answer no longer matches the enum

(1) is covered by `frame_sink_capability_matches_writer`. (2) and (3) are
currently not covered end-to-end and would fail in production without a
test catching them.

## Status

- [x] `frame_sink_capability_matches_writer` (enum/writer agreement)
- [x] `diagnostics_handoff_path_follows_published_frames` (router-level)
- [x] `shared_texture_bridge_writes_gpu_token_without_cpu_pixels` (sink-level)
- [x] composed test: `RenderExportMode::GpuExport` →
      `ExportedFrame::GpuFrameToken` → `OutputRouter::publish` → file is
      `VGTK` shape (no `pixel_data` written)
- [x] composed test: `RenderExportMode::CpuReadback` →
      `ExportedFrame::CpuReadback` → `OutputRouter::publish` → file is
      `frame_000000.png` (ImageSequence path)
- [x] regression test: changing the active sink at runtime via
      `OutputRouter::set_writer` routes subsequent frames to the new
      sink's artefact (and *not* the old one)

## Architecture target

The new tests live in `tests/` (integration crate) rather than `#[cfg(test)]`
modules, because the composed contract spans multiple crate modules and
benefits from being callable without `pub(crate)` visibility loosening:

```text
tests/output_sink_e2e.rs
  - build a fake ExportedFrame (no Vulkan involved)
  - feed it through OutputRouter::publish
  - inspect the on-disk artefact written by the worker thread
  - assert the magic bytes match what FrameSink:::supports_gpu_tokens implies
```

Vulkan / renderer is *not* exercised — these tests cover the
post-`ExportedFrame` contract. Renderer-side production of
`ExportedPixelData::GpuFrameToken` is already covered by
`OutputExporter`'s unit tests.

## Scope

In:
- `tests/output_sink_e2e.rs` (new)
- Possibly a small `pub` test helper on `OutputRouter` that exposes
  "wait for worker to finish frame N" so the test doesn't poll the file
  system with arbitrary timeouts

Out:
- Vulkan / GPU export verification — that's already covered at the unit
  level and the real verification is hardware (deferred)
- the MF DLL consumer side — covered by [`win32-sink-gpu-token-consumption.md`]

## Tasks

- [x] **T1** path injection seam: `OutputRouter::with_writer(sink, writer)`
      lets tests construct a writer (`ImageSequenceSink::new(temp_dir)`,
      `SharedMemoryFileSink::with_path(temp_path)`) with a unique temp
      path per test; `SharedMemoryFileSink::with_path` made pub
- [x] **T2** `e2e_gpu_token_round_trip` — publish a publishable
      `GpuFrameToken` via `OutputRouter::with_writer(SharedTextureFileStub,
      SharedMemoryFileSink::with_path(...))`; verify VGTK magic and
      72-byte header land on disk
- [x] **T3** `e2e_cpu_readback_round_trip` — publish an `OutputFrame`
      with `pixel_data` via `ImageSequence`; verify `frame_000000.png`
      lands at the configured path with the PNG signature
- [x] **T4** `e2e_sink_swap_routes_to_new_writer_within_one_frame` —
      start with `ImageSequence`, publish, hot-swap via
      `OutputRouter::set_writer`, publish, assert the second sink's
      artefact appears AND a subsequent file is NOT written under the
      first sink's directory
- [x] **T5** `OutputRouter::wait_until_worker_drains(timeout)` (cfg test)
      — sends a `WorkerMessage::Sync(ack)` sentinel through the
      sync_channel(1); when the ack arrives, every prior `Frame` has
      already been written, so the test never sleeps on arbitrary
      timeouts

## Done when

- the three composed tests above pass on CI
- a deliberate "break" (e.g. flipping `FrameSink::supports_gpu_tokens`
  for `SharedTextureFileStub` to `false`) causes at least one of them
  to fail loudly
- the tests do not depend on global filesystem state outside their own
  temp directory
