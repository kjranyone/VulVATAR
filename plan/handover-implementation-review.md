# VulVATAR implementation review handover

Date: 2026-04-26

## Current status

The default build is not completely broken at the type level, but the
implementation is inconsistent around feature flags and virtual-camera output.
The highest-risk area is the output stack: MediaFoundation virtual camera,
legacy DirectShow code, and shared-memory frame transport are overlapping in
ways that can break runtime behavior even when `cargo check` passes.

Commands already run:

```powershell
cargo check --workspace
cargo test --workspace
cargo check --workspace --features inference-gpu
cargo check --workspace --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
```

Results:

- `cargo check --workspace`: passed.
- `cargo test --workspace`: passed on the current tree (`149 passed`, `1 ignored`).
- `cargo check --workspace --features inference-gpu`: passed.
- `cargo check --workspace --no-default-features`: failed.
- `cargo clippy --workspace --all-targets -- -D warnings`: failed with many warnings-as-errors; most are style, but there is at least one COM/threading concern in `vulvatar-mf-camera`.

## Worktree state observed

Uncommitted files observed during review:

```text
M src/asset/vrm.rs
M src/gui/mod.rs
M src/output/dshow_source.rs
M src/output/frame_sink.rs
M src/output/mod.rs
M src/output/virtual_camera_native.rs
M src/simulation/cloth_solver.rs
?? src/output/shmem_security.rs
```

Treat these as user/current-work changes. Do not revert them blindly.

## Findings

### 1. Windows no-default-features build is broken

Severity: High

`windows` is an optional dependency enabled by the `virtual-camera` feature:

- `Cargo.toml`: `virtual-camera = ["windows"]`

But `src/single_instance.rs` imports and uses `windows::...` whenever
`target_os = "windows"`, regardless of whether the feature is enabled.

Failing command:

```powershell
cargo check --workspace --no-default-features
```

Representative errors:

```text
use of unresolved module or unlinked crate `windows`
src\single_instance.rs:6:5
src\single_instance.rs:17:20
```

Likely fixes:

- Make `windows` a non-optional dependency if single-instance support is always required on Windows.
- Or split single-instance Win32 support into its own feature, e.g. `single-instance-windows = ["windows"]`.
- Or gate the Win32 implementation with `all(target_os = "windows", feature = "virtual-camera")` and provide a non-Win32 fallback when the feature is off.

Recommendation: make `windows` non-optional if this app is Windows-first. It is already used outside the virtual-camera domain.

### 2. MediaFoundation and DirectShow share the same CLSID and registry slot

Severity: High

The MF virtual camera and legacy DirectShow code both use:

```text
{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}
```

Files:

- `src/output/mf_virtual_camera.rs`
- `src/output/directshow_filter.rs`
- `src/dshow_dll.rs`
- `vulvatar-mf-camera/src/lib.rs`

MF registers `InprocServer32` to a timestamped `vulvatar_mf_camera_*.dll` under
`Program Files\VulVATAR\VirtualCamera`.

DirectShow registration writes the same CLSID's `InprocServer32` to:

```text
vulvatar_dshow_filter.dll
```

Problem:

Running DirectShow registration can overwrite the MF COM server registration.
After that, `MFCreateVirtualCamera` can load the wrong DLL or fail to activate
the source.

Recommendation:

- Remove or quarantine the legacy DirectShow registration path.
- If DirectShow must remain, give it a distinct CLSID and a separate DLL path.
- Prefer a single supported path: MediaFoundation virtual camera plus
  `vulvatar-mf-camera`.

### 3. `src/dshow_dll.rs` is not a valid COM server

Severity: High

`src/dshow_dll.rs` returns raw Rust struct pointers as COM interfaces:

- `DllGetClassObject` returns `Box<VulVatarClassFactory>` as `IClassFactory`.
- `CreateInstance` returns `DirectShowSourceFilter` as the requested object.

There is no valid COM vtable, refcount implementation, QueryInterface behavior,
or IID-specific object layout for the returned pointers.

Problem:

This can appear to compile and still be unusable or undefined at runtime when
a real COM client tries to consume it.

Recommendation:

- Delete this legacy DLL path if MediaFoundation is the intended output.
- If it must exist, reimplement it with `windows::core::implement` and real COM
  interface implementations.
- Do not let it share CLSID or registry entries with the MF source.

### 4. Shared-memory frame transport has no publish synchronization

Severity: Medium

Producer:

- `src/output/frame_sink.rs`
- `Win32FileBackedSharedMemorySink::write_frame`

Consumer:

- `vulvatar-mf-camera/src/shared_memory.rs`
- `SharedMemoryReader::read_latest`

The producer writes header first and pixel payload second. The consumer reads
the header and immediately copies the payload. There is no seqlock, commit
marker, generation check, or double buffering.

Problem:

The consumer can read a partially-written frame:

- new width/height/sequence with old pixels
- half-copied payload
- transient corrupt frame under load

Recommendation:

Implement a simple seqlock-style protocol:

```text
offset 12 sequence:
  producer writes odd sequence before payload
  producer writes payload
  producer writes even sequence after payload

consumer:
  read seq1
  if odd: skip
  copy header + payload
  read seq2
  accept only if seq1 == seq2 and even
```

Alternative: double-buffer the payload and publish an atomic active-buffer
index last.

### 5. VirtualCamera sink switch can leave runtime in a false-success state

Severity: Medium

`Application::ensure_output_sink_runtime` swaps `OutputRouter` before starting
the MF virtual camera:

```rust
if self.output.active_sink() != &sink {
    self.output.shutdown();
    self.output = OutputRouter::new(sink.clone());
}

// later: MfVirtualCamera::start()
```

If `MfVirtualCamera::start()` fails, active sink remains `VirtualCamera`.
The GUI requested/active mismatch will not show, because both now say
VirtualCamera, but no MF camera is actually running.

Problem:

The app may keep writing camera frames to the file-backed buffer while the
system camera does not exist.

Recommendation:

- For `FrameSink::VirtualCamera`, start MF camera first.
- Only replace `OutputRouter` after MF startup succeeds.
- On failure, leave the previous active sink in place, or explicitly roll back
  to `SharedMemory`.
- Add a test for "MF start fails -> active sink remains old sink".

### 6. Output queue policy is not actually bounded

Severity: Medium

`OutputRouter` has a local `VecDeque` with queue policy, but after each
publish it immediately pops one frame and sends it into an unbounded
`mpsc::channel`.

Problem:

If a sink is slower than the renderer, frames can accumulate in the channel
regardless of `ReplaceLatest`, `DropOldest`, etc. The local queue appears
bounded, but the real queue is unbounded.

Recommendation:

- Replace `mpsc::channel` with `sync_channel(1)` plus `try_send`.
- Or use a shared `Arc<Mutex<Option<OutputFrame>>>` latest-frame slot and have
  the worker poll/park on a condvar.
- Make the selected queue policy apply to the actual worker boundary, not only
  the pre-send local queue.

## Clippy notes

`cargo clippy --workspace --all-targets -- -D warnings` currently fails with
many lints. Most are not behavioral, but these should be reviewed rather than
blindly suppressed:

- `vulvatar-mf-camera/src/media_source.rs`: `Arc<Mutex<Option<IMFVideoSampleAllocatorEx>>>`
  is flagged as `Arc` over non-`Send`/`Sync` data. Confirm COM allocator thread
  affinity and whether access can happen from multiple MF callback threads.
- `src/dshow_dll.rs` unsafe exported COM functions lack safety docs, but the
  bigger issue is that the COM implementation itself is not valid.

For handover, do not spend time fixing all style lints first. Fix the runtime
architecture issues above.

## Recommended repair order

1. Decide output support boundary:
   - Preferred: MediaFoundation virtual camera only.
   - Remove or fully isolate DirectShow code.

2. Fix CLSID ownership:
   - One CLSID per COM server.
   - MF source CLSID must only point at `vulvatar_mf_camera.dll`.

3. Fix `ensure_output_sink_runtime` transactionality:
   - Start MF camera before swapping active output router.
   - Roll back cleanly on failure.

4. Add shared-memory publish synchronization:
   - Use seqlock or double buffering.
   - Add producer/consumer unit tests for torn-frame prevention if possible.

5. Fix feature gating:
   - `cargo check --workspace --no-default-features` should pass or the
     unsupported feature combination should be explicitly removed from support.

6. Fix actual output backpressure:
   - Replace the unbounded channel or make the worker consume latest-only.

7. Only then clean clippy/style fallout.

## Verification target after repair

Run:

```powershell
cargo check --workspace
cargo test --workspace
cargo check --workspace --no-default-features
cargo check --workspace --features inference-gpu
```

For virtual camera runtime verification:

```powershell
cargo build -p vulvatar-mf-camera
cargo run
```

Then enable VirtualCamera sink and confirm:

- `VulVATAR Virtual Camera` appears in a camera client.
- `%ProgramData%\VulVATAR\camera_frame_buffer.bin` updates while output is active.
- `C:\Users\Public\vulvatar_mf_camera.log` shows frames being read.
- Switching away from VirtualCamera removes/stops the MF virtual camera.

## Notes from license review

README now states:

- VulVATAR: GPL-3.0
- MediaPipe Pose / Hand ONNX exports (via `opencv/*_mediapipe` on
  Hugging Face): Apache-2.0
- ONNX Runtime: MIT

This is unrelated to the implementation breakage, but should remain in the
release checklist for binary distribution.
