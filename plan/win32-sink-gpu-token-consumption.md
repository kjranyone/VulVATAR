# Win32 file-backed sink — true GPU token consumption (P2-05 phase 2)

Date: 2026-05-20
Roadmap reference: [`gpu-runtime-roadmap-tasks.md` P2-05](./gpu-runtime-roadmap-tasks.md)

## Goal

Extend the Win32 file-backed live sinks (`VirtualCamera`, `SharedMemory`,
both currently `Win32FileBackedSharedMemorySink`) to accept
`OutputFrame::gpu_token` directly — eliminating the per-frame CPU readback
that the P2-05 first slice still routes them through.

When this lands, the renderer can request `RenderExportMode::GpuExport`
for live virtual-camera output, the producer writes a shared D3D11/D3D12
texture handle to a sidecar file, the MF DLL consumer opens the shared
texture by handle, and no per-frame RGBA byte copy crosses the process
boundary on the steady-state path.

## Why

The P2-05 first slice ([`afab5c5`](../commits)) made the export path pick
`GpuExport` vs `CpuReadback` based on
`FrameSink::supports_gpu_tokens()`. Today only `SharedTextureFileStub`
returns `true`; the live Win32 sinks still need CPU bytes because the MF
DLL consumer reads raw RGBA from the file mapping. That is the largest
remaining per-frame copy in the live output path.

The compute-prepass migration already eliminated the per-frame allocator
churn on the rendering side. The next-biggest steady-state cost the
profiler attributes to *output* is the `Win32FileBackedSharedMemorySink`
write — `width * height * 4` bytes per published frame, twice per frame
for the dual-sink case (preview + virtual camera).

## Status

- [x] Capability-based routing (P2-05 phase 1)
- [x] `SharedTextureFileStub` consumes GPU tokens via VGTK records
- [x] Win32 sink GPU-token sidecar format (`<main>.vgtk` next to
      `<main>.bin`, atomic tmp+rename publish)
- [x] CPU-fallback parity test (token unpublishable → bytes path,
      sidecar untouched)
- [x] Lease lifetime upgraded to `SingleConsumerRetained`; Option A
      next-publish-release wired into `OutputRouter::publish`
- [ ] MF DLL update to open shared D3D textures by exported handle
      (out-of-tree, blocked on W8)
- [ ] Negotiation: producer-side fence vs DLL-side fence wait
      (deferred; landing back-channelled sync is part of a later slice)

Hardware verification (NVIDIA / AMD / Intel Arc) is **out of scope for
this plan** — that lands in a separate vendor-diff verification pass.

## Architecture target

```text
Renderer
   │
   ├── export_gpu produces GpuFrameToken { external_handle, sync, lease }
   │
   ▼
OutputRouter
   │
   ▼
Win32FileBackedSharedMemorySink::write_frame
   │
   ├── if frame.gpu_token.is_publishable():
   │     write sidecar file (VGTK-shaped, see below) containing
   │       - external_handle (HANDLE-as-u64)
   │       - keyed-mutex key (sync)
   │       - extent / color space / alpha mode
   │     do NOT write pixel bytes; main file is empty/unchanged
   │
   └── else (CPU fallback):
         existing path — copy pixel bytes into the file mapping
         (matches current behaviour exactly)

MF DLL (consumer)
   │
   ├── if sidecar carries a publishable token:
   │     OpenSharedResource1(handle) → ID3D11Texture2D
   │     acquire keyed mutex
   │     bind as IMFSample backing
   │     release mutex
   │
   └── else:
         existing path — read raw RGBA from main file mapping
```

The main file mapping stays exactly the same on the CPU-fallback path so a
**partial deploy** (new producer, old DLL) degrades cleanly: the DLL just
reads zeros / stale bytes for one or two frames, then either the user
restarts the DLL or the producer falls back when handle export fails.

## Sidecar file format (proposal)

Separate file: `%ProgramData%\VulVATAR\camera_frame_handle.bin` next to
the existing `camera_frame_buffer.bin`. Atomic write via tmp-rename.

```
offset  size  field
  0       4   magic "VGTK"
  4       4   version (u32, currently 1)
  8       4   width
 12       4   height
 16       4   handle_type (1 = Win32Kmt, 2 = D3D12Fence, …)
 20       4   sync_type   (0 None, 1 ProducerWaitComplete, 2 FenceValue, 3 SemaphoreHandle)
 24       8   external_handle (u64)
 32       8   sync_value      (u64)
 40       8   frame_id
 48       8   timestamp_nanos
 56       4   color_space (0 srgb, 1 linear)
 60       4   alpha_mode  (0 opaque, 1 premultiplied, 2 straight)
 64       4   flags       (bit 0: preserve alpha; bit 1: linear color)
 68       4   reserved
 72             — end of header
```

Identical shape to `SharedMemoryFileSink`'s VGTK encoder where possible,
so a future consumer factoring can read one format from either producer.

## Scope

In:
- `src/output/frame_sink.rs` — `Win32FileBackedSharedMemorySink` gets a
  GPU-token write path; `supports_gpu_tokens()` returns `true`
- `src/output/frame_sink.rs` — `FrameSink::supports_gpu_tokens()` for
  `VirtualCamera` and `SharedMemory` becomes `true`
- MF DLL repo (out-of-tree) — open shared handle, route to IMFSample
- `docs/output-interop.md` — sidecar format + DLL protocol

Out:
- D3D12 / Vulkan-external-fence cross-process signaling (start with
  keyed-mutex via Win32Kmt; the other variants land later)
- vendor-difference investigation (separate plan; hardware-only)
- replacing the existing file mapping (kept as fallback)

## Tasks

- [x] **W1** define the sidecar format (locked in this doc; documented
      in `docs/output-interop.md` "Win32 GPU-handle sidecar" section)
- [x] **W2** implement `encode_gpu_token_sidecar` in
      `Win32FileBackedSharedMemorySink`; write atomically to the sidecar
      path on a publishable token
- [x] **W3** when the token is not publishable, leave the sidecar
      untouched and write the existing RGBA bytes (parity)
- [x] **W4** flip `FrameSink::supports_gpu_tokens()` for `VirtualCamera`
      and `SharedMemory` to `true` (Windows-gated;
      `frame_sink_capability_matches_writer` covers the agreement)
- [x] **W5** unit test: token frame → sidecar exists, main buffer is
      stale-but-untouched
- [x] **W6** unit test: CPU-fallback frame → sidecar is *not* updated,
      main buffer receives bytes
- [x] **W7** document the sidecar protocol in `docs/output-interop.md`
- [ ] **W8** MF DLL changes (out-of-tree) — open sidecar, parse VGTK,
      `OpenSharedResource1`, keyed-mutex acquire/release, fall back to
      main buffer if sidecar is absent

## Resolved IPC decision (lease ack)

Option A (short-lived, next-publish-release) selected. `OutputRouter`
holds the most recent `SingleConsumerRetained` lease as
`pending_retained_lease` and releases it the moment the next frame
crosses the `try_send` boundary. No back-channel from the DLL is
required for the first slice; the protocol assumes the consumer reads
the latest published sidecar within one producer frame interval.

Implications:
- export pool capacity stays at 2 (current + previous = 2 leased slots
  worst case, matching `DEFAULT_EXPORT_IMAGE_POOL_CAPACITY`)
- a stalled consumer sees the second-latest GPU frame for one extra
  tick (same window as the byte-buffer's seq tearing-protection)
- shutdown flushes the bound lease so the slot is reclaimed before the
  router drops

## Risks

- **Handle lifetime**: a Win32 KMT handle is a process-scoped handle. The
  DLL must `DuplicateHandle` into its own process before the producer
  releases the lease, OR the producer must hold the lease until the DLL
  acknowledges. The current `FrameLease` model assumes producer-side
  release on `OutputRouter::drain_completed_gpu_leases` — that needs the
  DLL to send an acknowledgement, which today's IPC does not have.
  Resolve this *before* W2.
- **Format compatibility**: NV12 / RGBA conversion currently happens inside
  the DLL on the RGBA bytes. On the GPU path the DLL gets an RGBA texture
  and needs to convert via shader / VideoProcessor — extra DLL work.
- **First-frame race**: the DLL may open the sidecar before the texture
  has any content. Producer must signal "valid" via the sync field (e.g.
  `OutputSyncToken::ProducerWaitComplete` already guarantees the producer
  fence is signalled).
- **Lease acknowledgement IPC**: needs a back-channel from the DLL. Today
  the producer assumes single-consumer-immediate semantics; for GPU
  texture sharing that assumption breaks. Decide: short-lived (DLL must
  consume within one frame) vs retained (DLL acks).

## Done when

- a frame published with `RenderExportMode::GpuExport` lands in the
  sidecar file in `VGTK` shape, with a real exportable handle
- the MF DLL can open that handle, render the texture, and acknowledge
  the lease within one frame (no leak in `ExportImagePool::stats()`)
- toggling the sink at runtime between `VirtualCamera` (GPU) and
  `ImageSequence` (CPU) produces the right artefact each time
- on a token-publish failure (handle export fails), the sink visibly
  falls back to the RGBA byte path and the diagnostics panel says so

## Sequence

1. Resolve the lease-ack IPC question first (architecture decision)
2. Land sidecar producer (W1–W7) with the DLL still on CPU path
3. Verify producer doesn't regress CPU-fallback parity
4. Land DLL changes (W8)
5. Hand off to hardware verification pass (separate plan)
