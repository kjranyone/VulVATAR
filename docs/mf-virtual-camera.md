# MF Virtual Camera

## Purpose

This document defines the shipped Windows 11 MediaFoundation virtual
camera — the contract the `vulvatar-mf-camera` DLL satisfies for
Frame Server, and the host-side wiring that drives it.

The camera registers as `VulVATAR Virtual Camera` and shows up in any
Windows MF capture client: Google Meet, Microsoft Teams, Zoom, OBS,
Chrome, Edge, the Windows Camera app. Frames come from the running
VulVATAR renderer via a named shared-memory ring.

Related documents:

- [architecture.md](./architecture.md) — process-level overview
- [output-interop.md](./output-interop.md) — sink abstraction
- [threading-model.md](./threading-model.md) — GUI / render / worker threads
- [mf-virtual-camera-handover.md](./mf-virtual-camera-handover.md) —
  historical debugging log (superseded by this document)

## Architecture

Two processes, one OS service in the middle.

```
┌──────────────────────────────┐       ┌──────────────────────────────────┐
│  vulvatar.exe (interactive)  │       │  svchost.exe (FrameServer svc,   │
│                              │       │  runs as NT AUTHORITY\LocalService)│
│  renderer → OutputRouter     │       │                                  │
│     ↓                        │       │  loads vulvatar_mf_camera.dll    │
│  Win32NamedSharedMemorySink  │       │  via HKLM InprocServer32         │
│     writes Global\VulVATAR_  │       │     ↓                            │
│     VirtualCamera            │──────►│  IMFMediaSource → IMFMediaStream │
│                              │  shm  │     ↑                            │
│  MFCreateVirtualCamera       │       │  reads Global\VulVATAR_          │
│  registers source CLSID and  │       │  VirtualCamera on each           │
│  starts the virtual camera   │       │  RequestSample                   │
└──────────────────────────────┘       └──────────────────────────────────┘
                                                      ↑
                                                  FsProxy
                                                      ↑
                                          Google Meet / Chrome / Teams
```

- Renderer produces RGBA frames; `Win32NamedSharedMemorySink` writes a
  32-byte header + pixel data into a `Global\VulVATAR_VirtualCamera`
  file mapping (SDDL grants `LocalService` read).
- `MFCreateVirtualCamera` tells Windows to register our CLSID as a
  camera under `SWD\VCAMDEVAPI\{hash}`.
- When a client enumerates cameras and calls `ActivateObject`, Frame
  Server's svchost loads `vulvatar_mf_camera.dll` from HKLM, QIs the
  handful of interfaces listed below, negotiates a media type, and
  starts pulling samples.
- Each `Stream::RequestSample` reads the latest frame from the shared
  memory, converts to the negotiated subtype (NV12 or RGB32), stamps
  it with `MFGetSystemTime`, and queues a `MEMediaSample` event.

## COM contract (DLL side)

`DllGetClassObject(CLSID_VULVATAR_MEDIA_SOURCE)` returns a class
factory that builds a single `VulvatarMediaSource` object. The source
is the one COM identity exposing **all** of the interfaces below —
this is critical: Frame Server's FsProxy QIs multiple interfaces on
the activated object and expects a single attribute store that
survives across calls.

| Interface | Purpose | Why it matters |
|---|---|---|
| `IMFActivate` | Returned from `IClassFactory::CreateInstance`. `ActivateObject` is a self-QI. | `MFCreateVirtualCamera` creates the virtual camera by asking our factory for an `IMFActivate` and then calling `ActivateObject(IMFMediaSource)`. |
| `IMFMediaSource` / `IMFMediaSourceEx` | Standard MF source API (Start/Stop/Pause, presentation descriptor, stream attributes, D3D manager). | Core source contract. `Pause` must return `MF_E_INVALID_STATE_TRANSITION` for live sources. `SetD3DManager` is a no-op for our software path. |
| `IMFAttributes` | Same store as `GetSourceAttributes`. Frame Server QIs the source for `IMFAttributes` during catalog matching. | Missing this interface makes FsProxy tear the bridge down in microseconds (`WatchdogOperation: アクティブ化` completing in 30–40 hns) and `ReadSample` returns `MF_SOURCE_READERF_ENDOFSTREAM` immediately. |
| `IMFGetService` | Service dispatcher. We serve `IMFExtendedCameraController`. | Missing the vtable is fatal; returning `MF_E_UNSUPPORTED_SERVICE` for unknown GUIDs is fine (matches MS SimpleMediaSource). |
| `IMFExtendedCameraController` + `IMFExtendedCameraControl` | Modern camera-control path. We serve privacy (`id=8`) and ROI (`id=21`, `id=22`) as well-formed "no capabilities" / "privacy off" responses. | Windows 11 Frame Server probes extended controls before falling back to `IKsControl`. Missing well-formed replies fails `IMFVirtualCamera::Start`. |
| `IKsControl` | Legacy KS property / event plumbing. Only `PROPSETID_VIDCAP_CAMERACONTROL / KSPROPERTY_CAMERACONTROL_PRIVACY` is handled. | Frame Server issues a GET of the privacy shutter during Start; an incorrect reply (empty header, wrong size) fails registration. Reply as "privacy permanently off". |
| `IMFSampleAllocatorControl` | `SetDefaultAllocator` + `GetAllocatorUsage`. | Frame Server provides an `IMFVideoSampleAllocatorEx` for cross-process sample marshaling. We must report `MFSampleAllocatorUsage_UsesProvidedAllocator (0)` and allocate samples from the provided pool — CPU `MFCreateMemoryBuffer` samples can't cross the FrameServerPool boundary. |
| `IMFRealTimeClient` / `IMFRealTimeClientEx` | Signals that we cooperate with MF's work-queue worker threads. | Presence admits us into FrameServer's realtime scheduling. Implementation is a no-op. |

The stream object (`VulvatarMediaStream`, one per Start cycle)
implements:

| Interface | Purpose |
|---|---|
| `IMFMediaStream2` | Stream lifecycle events + state. |
| `IMFAttributes` | Forwards to the stream descriptor's attribute store (`MF_DEVICESTREAM_*`). |

The `IMFAttributes` forwarder for both objects lives in
[`vulvatar-mf-camera/src/attributes.rs`](../vulvatar-mf-camera/src/attributes.rs)
as the `forward_imfattributes!` macro so the 28-method boilerplate
can't drift between source and stream.

## Media type contract

### Source-scoped attributes

Set on `VulvatarMediaSource.attributes` — this is what
`GetSourceAttributes` returns **and** what `QI<IMFAttributes>(source)`
returns (same store).

| Attribute | Value | Reason |
|---|---|---|
| `MFT_TRANSFORM_CLSID_Attribute` | our CLSID | Frame Server catalog match. |
| `MF_DEVICEMFT_SENSORPROFILE_COLLECTION` | 1 profile (`KSCAMERAPROFILE_Legacy`) with `((RES==;FRT<=60,1;SUT==))` | Profile-unaware clients need a legacy profile or Start refuses. |

### Stream-scoped attributes

Set on `stream_descriptor.attributes` (obtained by casting the stream
descriptor to `IMFAttributes`). The stream's `IMFAttributes` forwarder
exposes the same store to clients that QI the stream.

| Attribute | Value | Reason |
|---|---|---|
| `MF_DEVICESTREAM_STREAM_CATEGORY` | `PINNAME_VIDEO_CAPTURE` | Marks the stream as the colour camera feed. |
| `MF_DEVICESTREAM_STREAM_ID` | 0 | Single stream, canonical id. |
| `MF_DEVICESTREAM_FRAMESERVER_SHARED` | 1 | Declares the stream as Frame Server-shareable (required for the `MFVirtualCameraType_SoftwareCameraSource` path). |
| `MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES` | `MFFrameSourceTypes_Color` | Colour (as opposed to IR, depth, etc.). |

### Per-`IMFMediaType` attributes

Each entry in `SUPPORTED_FORMATS` produces an `IMFMediaType` with:

- `MF_MT_MAJOR_TYPE = MFMediaType_Video`
- `MF_MT_SUBTYPE` ∈ {`MFVideoFormat_NV12`, `MFVideoFormat_RGB32`}
- `MF_MT_INTERLACE_MODE = MFVideoInterlace_Progressive`
- `MF_MT_ALL_SAMPLES_INDEPENDENT = 1`
- `MF_MT_FIXED_SIZE_SAMPLES = 1`
- `MF_MT_FRAME_SIZE = pack(width, height)`
- `MF_MT_FRAME_RATE = pack(fps, 1)`
- `MF_MT_PIXEL_ASPECT_RATIO = pack(1, 1)`
- `MF_MT_DEFAULT_STRIDE = width` (NV12) or `width * 4` (RGB32)
- `MF_MT_SAMPLE_SIZE = width * height * 3 / 2` (NV12) or `width * height * 4` (RGB32)
- `MF_MT_AVG_BITRATE = sample_size * 8 * fps`

Supported formats, in preference order:

```
NV12   1920×1080 @ 30   ← default, renderer-native (no resample)
NV12   1920×1080 @ 60
NV12   1280× 720 @ 60 / 30
NV12    960× 540 @ 30
NV12    640× 480 @ 30
NV12    320× 240 @ 30
RGB32  1920×1080 @ 30
RGB32  1280× 720 @ 30
RGB32   640× 480 @ 30
```

NV12 is first because Chrome / WebRTC reject sources that only offer
RGB32. RGB32 is kept for clients that want the alpha channel
(`AlphaMode::Preserve` in the host controls whether the sample's 4th
byte carries real alpha or is forced to `0xFF`).

## Sample delivery contract

`Stream::RequestSample` has two paths:

1. **Allocator path** (Frame Server-mediated clients — the common case):
   `SetDefaultAllocator` gave us an `IMFVideoSampleAllocatorEx`. We
   call `InitializeSampleAllocator(10, current_media_type)` once, then
   `AllocateSample()` per frame and fill the sample's buffer in place.
   Only allocator-pool samples are marshalable to the FrameServerPool.

2. **Direct path** (`cargo run --bin mfenum` or any
   `CoCreateInstance(CLSID)` consumer in-process):
   Build a fresh `IMFSample` via `MFCreateSample` + `MFCreateMemoryBuffer`.

### Stride-aware writes

Allocator-backed buffers usually come with row-stride padding
(e.g. `width=1920`, `pitch=2048`) for hardware alignment. Writing
contiguously under that assumption is **fatal** for NV12:

- Y plane rows land at `dst + row * 1920` but the consumer reads at
  `dst + row * 2048` → all Y-plane rows shifted.
- UV plane lands inside the Y plane's padding region at
  `dst + 1920 * 1080` instead of `dst + 2048 * 1080` → the real UV
  plane stays zero-initialised.
- Zero chroma in YCbCr→RGB conversion clamps R and B to 0 and leaves G
  alone, so the output is a uniform green frame.

The fix (`vulvatar-mf-camera/src/media_stream.rs`):

1. Try `buffer.cast::<IMF2DBuffer>()`; success gives us the real
   `scanline0` + `pitch` via `Lock2D`.
2. Y plane: `scanline0 + row * pitch`, write `width` Y bytes per row.
3. UV plane base: `scanline0 + height * pitch`. Each chroma row:
   `uv_base + chroma_row * pitch`, write `width` bytes (= `width/2`
   interleaved `(Cb, Cr)` pairs).
4. Fall back to contiguous 1D writes when the buffer doesn't implement
   `IMF2DBuffer` (only the `MFCreateMemoryBuffer` direct-probe path).

### Sample timing

- **Time** comes from `MFGetSystemTime()` (live clock) — MF's standard
  reference clock, already used by SourceReader / capture engine
  downstream.
- **Duration** is `10_000_000 / fps` hundred-ns units.
- **Token** (`MFSampleExtension_Token`) is copied back from the
  `RequestSample` argument when provided.

### Colour space

BT.601 limited range (Y 16–235, UV 16–240). The standard for
"consumer webcam" pipelines; matches what Meet / Teams / Chrome
expect as the default for a software camera.

## Shared memory ABI

**Backing.** Producer and consumer share frames through a **file-backed
memory mapping**:

```
%ProgramData%\VulVATAR\camera_frame_buffer.bin
```

Both processes open the file by path with FILE_SHARE_READ |
FILE_SHARE_WRITE | FILE_SHARE_DELETE, then call
`CreateFileMappingW(file_handle, ...)` with no kernel-namespace name.
The kernel detects that two mappings are backed by the same file region
and shares the underlying physical pages between them. Cross-session
data flow happens at the file system level — not the kernel object
namespace — even though FrameServer's svchost (Session 0,
`NT AUTHORITY\LocalService`) and the producer (Session 1+, interactive
user) live in different sessions.

**Why not a `Global\` named section.** That was the obvious approach
and it does not work for non-elevated developer flows. A normal
interactive user under UAC does *not* hold `SeCreateGlobalPrivilege` in
their filtered token, so any `CreateFileMappingW` with a `Global\<name>`
prefix returns `ERROR_ACCESS_DENIED (5)` regardless of
SECURITY_ATTRIBUTES, integrity label, or name uniqueness — verified
empirically:

| Backing | Name | Result |
|---|---|---|
| pagefile | `Global\fresh_uuid` | FAIL ERROR_ACCESS_DENIED |
| pagefile | `Local\fresh_uuid` | OK |
| pagefile | (anonymous) | OK |
| **file** | (anonymous) | **OK** ← used here |
| file | `Global\fresh_uuid` | FAIL ERROR_ACCESS_DENIED |

`Local\` would work but is per-session, so svchost in Session 0 can't
see it. The remaining viable choice is a file-backed unnamed mapping.
Microsoft documents this explicitly: a `Global\` create from a
non-Session-0 process without `SeCreateGlobalPrivilege` returns
`ERROR_ACCESS_DENIED`.

**File ACL.** `dev.ps1` option 7 creates the directory and grants
`NT AUTHORITY\LocalService:(OI)(CI)(RX)`. Inheritance ensures the
producer's `frame_buffer.bin` picks up the LocalService grant
automatically — no per-file icacls step needed.

**Cache behaviour.** The producer opens the file with
`FILE_ATTRIBUTE_TEMPORARY` so the cache manager keeps dirty pages in
RAM. With frame data overwritten 60 times per second, almost nothing
ever reaches disk; the file is functionally a RAM-resident IPC
buffer with a path.

**Wire format** (little-endian, 32-byte header followed by RGBA
pixels):

```
offset 0   u32  magic      = 0x5643_4D46  ("FMCV")
offset 4   u32  width
offset 8   u32  height
offset 12  u64  sequence   (incremented on every frame)
offset 20  u64  timestamp  (producer's frame timestamp; not used on read side)
offset 28  u32  flags      (bit 0 = PRESERVE_ALPHA; NV12 ignores)
offset 32  ...  width * height * 4 RGBA bytes
```

The producer extends the file via `set_len` whenever the configured
output resolution requires more bytes than the current file size. The
consumer detects extension by polling `metadata().len()` on each
`ensure_mapping` call and remaps if the file has grown.

## Registration + install

### HKLM CLSID registration

`HKLM\Software\Classes\CLSID\{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}`

- `(default)` = `"VulVATAR Virtual Camera"`
- `InprocServer32\(default)` = timestamped DLL path in
  `C:\Program Files\VulVATAR\VirtualCamera\`
- `InprocServer32\ThreadingModel` = `"Both"`

`MFCreateVirtualCamera` uses this CLSID for `source_id`; FrameServer
resolves it via HKLM when a client activates the device.

### DLL install location

`C:\Program Files\VulVATAR\VirtualCamera\vulvatar_mf_camera_system_<epoch_ms>.dll`

- Timestamped filename lets a new install land alongside any DLL
  FrameServer still has `LoadLibrary`-held.
- Filesystem DACL explicitly grants
  `NT AUTHORITY\LocalService:(RX)` — default Program Files ACL
  grants `BUILTIN\Users` only, which **excludes** LocalService.
  Without the grant, svchost's `LoadLibrary` fails with access
  denied and `ReadSample` returns ENDOFSTREAM.

### Install workflow

```powershell
# Elevated PowerShell:
.\dev.ps1                           # → select "install mf virtual camera (HKLM)"
# or non-interactively:
.\dev.ps1 -Select N                 # where N is the menu index
```

`Install-MfCameraSystem` does, in order:

1. `Build-MfCameraDll` — `cargo build -p vulvatar-mf-camera` in the
   non-elevated caller shell (keeps `target/` out of Admin ownership)
2. If the caller isn't elevated: `Start-Process -Verb RunAs` launches
   a new PowerShell with `-RegisterMfCamera`, prompting for UAC once.
3. In the elevated shell, `Register-MfCameraSystem` runs:
   - Copy the new DLL to `Program Files` under a fresh timestamp
   - `icacls` grant `LocalService:(RX)` + verify the grant landed
   - Create `%ProgramData%\VulVATAR\` and grant inherited
     `LocalService:(OI)(CI)(RX)` for the frame-buffer discovery file
   - Remove any legacy HKCU registration
   - Write HKLM `CLSID` + `InprocServer32` + `ThreadingModel=Both`
   - **Stop and restart the `FrameServer` service** — svchost pins a
     DLL's `IClassFactory` for its lifetime, so a fresh HKLM
     registration alone leaves Meet/Chrome consuming the previously-
     loaded DLL until the next reboot.

### Host-side registration

`src/output/mf_virtual_camera.rs` calls `MFCreateVirtualCamera` with
`MFVirtualCameraType_SoftwareCameraSource +
MFVirtualCameraLifetime_Session + MFVirtualCameraAccess_CurrentUser`.
Lifetime = session means the virtual camera disappears when
`vulvatar.exe` exits — no stale registrations between runs.

## Build + dev workflow

```bash
# DLL only (fast):
cargo build -p vulvatar-mf-camera

# Diagnostic tool:
cargo run --bin mfenum
#   prints enumerated cameras + tries both direct-probe and
#   FS-mediated ReadSample loops.

# Full app:
cargo run
```

After rebuilding, the new DLL is at
`target/debug/vulvatar_mf_camera.dll` and is loaded into
`vulvatar.exe` directly (e.g. when `mfenum` does its direct probe).
FrameServer only picks up the new DLL after a reinstall via
`dev.ps1 install`.

## Diagnostics

### Trace log

The DLL appends one line per COM entry point / KS property / sample
to:

```
C:\Users\Public\vulvatar_mf_camera.log
```

Public is writable by both the interactive user (`vulvatar.exe`) and
`LocalService` (FrameServer svchost), so entries from all processes
hitting the DLL land in one file.

Each line is:

```
<epoch_ms> pid=<pid> tid=<tid> <message>
```

The first line per process is:

```
DllGetClassObject enter marker=<BUILD_MARKER> vulvatar-mf-camera\src\lib.rs:<line>
```

`BUILD_MARKER` is a short hand-written string in
`vulvatar-mf-camera/src/lib.rs` that gets bumped whenever the
contract changes. Check it in the log to confirm the installed DLL
matches the committed tree.

### FrameServer event log

```powershell
Get-WinEvent -LogName 'Microsoft-Windows-MF-FrameServer/Camera_FrameServer' `
  -MaxEvents 30 |
  Format-List TimeCreated, Id, LevelDisplayName, Message
```

Healthy for our camera:

- `Event 6` (FsProxy 初期化開始) with the `swd#vcamdevapi#...` symlink
- `Event 2` (SetOutputType 開始) with the negotiated media type
- `Event 3` (SetOutputType Stop) `hr=0x0`
- `Event 7` (FsProxy 初期化停止) `hr=0x0`
- `WatchdogOperation: 初期化` with `DurationToCompletionInHns` in the
  low thousands (milliseconds).

Broken:

- `WatchdogOperation: アクティブ化` with `DurationToCompletionInHns` in
  the tens of hns (microseconds) → FsProxy short-circuited before
  actually talking to our DLL. Usually means an interface is missing
  from the source's `#[implement(...)]` list, or the `IMFAttributes`
  store isn't shared between the source identity FrameServer wrote to
  and the one it's QI-ing now.

### Autostart env vars

For headless repro or quick install-and-test cycles:

- `VULVATAR_AUTOSTART_VIRTUAL_CAMERA=1` — GUI starts with the
  VirtualCamera sink already active.
- `VULVATAR_AUTOSTART_AVATAR=<path.vrm>` — GUI loads the avatar at
  the given path immediately after bootstrap.

## Known limitations

- **Single stream only.** The source exposes one video stream; no
  audio, no secondary streams (IR, depth, metadata).
- **Format change mid-life requires Stop + Start.** The stream is
  dropped on `Source::Stop` so the next `Start` builds a fresh one
  with a fresh format cache and a fresh allocator-initialised flag.
  Live renegotiation without a Stop is not supported.
- **BT.601 limited range only.** No BT.709, no full range.
- **Nearest-neighbour resampling** when producer resolution ≠
  negotiated size. Cheap, GPU-free, fine for a virtual camera; if
  quality becomes a concern, swap for bilinear.
- **CPU-backed samples only.** `SetD3DManager` is accepted but
  ignored; every frame is built from a CPU buffer. If we ever add a
  D3D-resident path, it plugs in at `fill_allocated_sample`.
- **No aggregation.** `IClassFactory::CreateInstance(outer != null)`
  returns `CLASS_E_NOAGGREGATION`.

## References

### Microsoft docs

- [MFCreateVirtualCamera](https://learn.microsoft.com/en-us/windows/win32/api/mfvirtualcamera/nf-mfvirtualcamera-mfcreatevirtualcamera)
- [Frame Server custom media source requirements](https://github.com/MicrosoftDocs/windows-driver-docs/blob/staging/windows-driver-docs-pr/stream/frame-server-custom-media-source.md)
- [MFSampleAllocatorUsage](https://learn.microsoft.com/en-us/windows/win32/api/mfidl/ne-mfidl-mfsampleallocatorusage)
- [IMF2DBuffer::Lock2D](https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/nf-mfobjects-imf2dbuffer-lock2d)

### Reference implementations

- [smourier/VCamSample](https://github.com/smourier/VCamSample) —
  the C++ reference for the Windows 11 software camera source; the
  one-identity pattern (Activator = MediaSource = IMFAttributes) and
  the `IMF2DBuffer` stride handling both follow its lead.
- [qing-wang/MFVCamSource](https://github.com/qing-wang/MFVCamSource) —
  alternative reference, useful for cross-checking specific attribute
  values.

### Relevant local files

| File | Role |
|---|---|
| `vulvatar-mf-camera/src/lib.rs` | DLL entry points, `BUILD_MARKER`. |
| `vulvatar-mf-camera/src/class_factory.rs` | `DllGetClassObject` → source as IMFActivate. |
| `vulvatar-mf-camera/src/media_source.rs` | `VulvatarMediaSource` — every COM interface above. |
| `vulvatar-mf-camera/src/media_stream.rs` | `VulvatarMediaStream` — RequestSample, stride-aware writers. |
| `vulvatar-mf-camera/src/attributes.rs` | `forward_imfattributes!` macro. |
| `vulvatar-mf-camera/src/shared_memory.rs` | Shared-memory reader (DLL side). |
| `vulvatar-mf-camera/src/registry.rs` | `DllRegisterServer` per-user path (unused in the shipped Program-Files install). |
| `vulvatar-mf-camera/src/trace.rs` | File-based tracing to `C:\Users\Public\`. |
| `src/output/frame_sink.rs` | `Win32NamedSharedMemorySink` (host writer side). |
| `src/output/mf_virtual_camera.rs` | `MFCreateVirtualCamera` wrapper, HKLM self-registration. |
| `src/bin/mfenum.rs` | Diagnostic tool (enumerate + direct probe + FS-mediated ReadSample). |
| `dev.ps1` | `Install-MfCameraSystem` — build, copy, ACL, HKLM, FrameServer restart. |
