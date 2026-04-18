# MF Virtual Camera — handover for MF virtual-camera registration

Status: updated 2026-04-17. The `E_NOINTERFACE` block was traced to missing
modern camera-control service support and has been addressed in code by
implementing `IMFExtendedCameraController` / `IMFExtendedCameraControl` plus
well-formed ROI_CONFIGCAPS payloads. The next observed failure is
`HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND)` (0x80070003), which lines up with
the main app's old HKCU COM self-registration: Windows' FrameServerMonitor /
FrameServer services require the media-source COM server to be visible under
HKLM. The main-side registrar now targets HKLM, reuses an existing HKLM
registration for normal non-elevated runs, and fails explicitly if no usable
HKLM registration exists.

## Goal

When the user selects VirtualCamera output in the app, Windows Camera / OBS /
Teams should list **VulVATAR Virtual Camera** as a selectable device. Frames
(task #10 in the internal tracker) come later — right now we're stuck at
registration.

## Architecture (short)

```
┌───────────────────────────┐           ┌─────────────────────────────────┐
│  vulvatar.exe             │           │  frameserver.exe (Windows)       │
│                           │           │                                  │
│  output/mf_virtual_camera │           │  IMFVirtualCamera::Start         │
│    MFStartup()            │           │    CoCreateInstance(CLSID)       │
│    ensure_dll_registered  │──HKLM──►  │     → DllGetClassObject          │
│    MFCreateVirtualCamera ─┼───────────┼─►    → CreateInstance(IMFActivate)│
│    cam.Start()  ────── probes / start◄─┼─    → ActivateObject(IMFMSrcEx)  │
│                           │           │     probing calls ↓              │
│                           │           │     FAILS silently               │
└───────────────────────────┘           └─────────────────────────────────┘
                                                       ↕
                                            vulvatar_mf_camera.dll
                                            (IMFActivate + IMFMediaSourceEx
                                             + IKsControl + IMFGetService +
                                             IMFSampleAllocatorControl +
                                             IMFExtendedCameraController;
                                             stream = IMFMediaStream2)
```

- DLL: `vulvatar-mf-camera/` (cdylib)
- Main-side registrar / `MFCreateVirtualCamera` caller: `src/output/mf_virtual_camera.rs`
- HKLM class registration written by elevated run/install, path = **system copy**
  `%ProgramFiles%/VulVATAR/VirtualCamera/vulvatar_mf_camera_<ms>.dll` so cargo
  can overwrite the canonical build output without Frame Server holding it and
  Frame Server does not need to load code from the developer tree
- CLSID: `{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}`
- Source ID passed to `MFCreateVirtualCamera`: the CLSID string above
- Lifetime: `MFVirtualCameraLifetime_Session`
- Type: `MFVirtualCameraType_SoftwareCameraSource`
- Access: `MFVirtualCameraAccess_CurrentUser`

## Dev workflow

After changing the MF camera DLL, install the DLL once from an elevated
PowerShell:

```
dev.ps1  → install mf virtual camera (HKLM)
```

The install command builds the camera DLL with static CRT (`+crt-static`) and
copies it under `%ProgramFiles%\VulVATAR\VirtualCamera\` before writing HKLM.
This avoids `IMFVirtualCamera::Start → 0x8007007E` caused by Frame Server
failing to load the DLL or a dependent module from `target\debug`.
The installer and runtime also remove the legacy HKCU CLSID registration;
otherwise HKCR resolves HKCU before HKLM and COM may still try to load the old
`target\debug\mf-camera-runtime\...dll`.

Normal runs can then be non-elevated:

```
dev.ps1  → run (debug)
```

No manual regsvr32 needed. To fully reset registration, run the uninstall
command from an elevated PowerShell so both HKCU legacy and HKLM keys can be
removed:

```
dev.ps1  → uninstall mf virtual camera
```

## Trace log

The DLL writes to `%TEMP%\vulvatar_mf_camera.log` (see
`vulvatar-mf-camera/src/trace.rs`). Each method entry on the source /
stream / activate emits a line, including KSIDENTIFIER set+id+flags for
`IKsControl` probes. The DLL also stamps a `BUILD_MARKER` string into
`DllGetClassObject` so a stale-copy scenario is easy to spot.

Clear before each run:

```powershell
Remove-Item "$env:TEMP\vulvatar_mf_camera.log" -ErrorAction SilentlyContinue
```

## Recent trace, with IID lookup

After adding `IMFExtendedCameraController`, the DLL reaches the privacy KS
probes without the previous `E_NOINTERFACE` service failure:

```
DllGetClassObject enter marker=extended-camera-controller ...
DllGetClassObject: query riid=00000001-0000-0000-C000-000000000046 → OK    (IID_IClassFactory)
ClassFactory::CreateInstance iid=7FEE9E9A-4A89-47A6-899C-B6A53A70FB67      (IID_IMFActivate)
ClassFactory::CreateInstance: activate.query → OK
Activate::ActivateObject riid=3C9B2EB9-86D5-4514-A394-F56664F9F0D8         (IID_IMFMediaSourceEx)
Activate::ActivateObject: constructing new source
Activate::ActivateObject: source.query → OK
SourceEx::GetSourceAttributes
Source::GetService guid=GUID_NULL riid=2032C7EF-76F6-492A-94F3-4A81F69380CC    ← UNIDENTIFIED
Source::CreatePresentationDescriptor
SourceEx::GetStreamAttributes id=0
SourceEx::GetSourceAttributes × 2
Source::GetService guid=GUID_NULL riid=B91EBFEE-CA03-4AF4-8A82-A31752F4A0FC    (IMFExtendedCameraController)
Source::GetService IMFExtendedCameraController → S_OK
Source::GetExtendedCameraControl stream=4294967295 property=21
ExtendedControl::LockPayload property=21 len=16
ExtendedControl::UnlockPayload property=21
Source::KsProperty set=PROPSETID_VIDCAP_CAMERACONTROL id=8 (PRIVACY) flags=0x1 (GET)
  handling PRIVACY ... data_len=40
  PRIVACY GET → S_OK (EXTENDEDPROP, 40-byte header + ULONGLONG value)
Source::KsEvent set=PROPSETID_VIDCAP_CAMERACONTROL id=8 flags=0x200 (BASICSUPPORT)
  data_len=32
  PRIVACY BASICSUPPORT → S_OK (32-byte EXTENDEDPROP_HEADER with Size=32, Flags=Capability=PRIVACY_OFF=2)
```

The latest remaining blocker is now registration visibility. If the app logs
`RegCreateKeyExW(...): WIN32_ERROR(5)`, HKLM has not been installed yet and
the process is not elevated. Run `dev.ps1 → install mf virtual camera (HKLM)`
from an elevated PowerShell.

If the app logs `IMFVirtualCamera::Start failed: ... (0x8007007E)`, check that
HKCR no longer points at `target\debug\mf-camera-runtime`. HKCU overrides HKLM
in HKCR, so remove:
`HKCU\Software\Classes\CLSID\{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}`.
Re-run the elevated install command so `InprocServer32` points under
`%ProgramFiles%\VulVATAR\VirtualCamera`.

## What has already been tried

| # | Change                                                          | Outcome         |
|---|-----------------------------------------------------------------|-----------------|
| 1 | Register DLL under HKCU with InprocServer32                     | Early probes load; final service path still fails |
| 2 | Add `IMFActivate` wrapper class (CreateInstance returns it)     | Unlocks probing |
| 3 | Source implements `IMFMediaSourceEx`, `IKsControl`, `IMFGetService`, `IMFSampleAllocatorControl` | Probing proceeds |
| 4 | Stream implements `IMFMediaStream2`                             | Still advances  |
| 5 | Set `MF_DEVICESTREAM_FRAMESERVER_SHARED=1` + category etc. on stream descriptor and stream attrs | Still advances  |
| 6 | Set `MF_VIRTUALCAMERA_CONFIGURATION_APP_PACKAGE_FAMILY_NAME=""` on source attrs | Still advances |
| 7 | `IKsControl` unknown sets → `HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND)` not `E_NOTIMPL` | Still advances |
| 8 | `IMFGetService` → `MF_E_UNSUPPORTED_SERVICE`                    | Still advances  |
| 9 | Cache source inside activate                                    | Still advances  |
| 10| `KSPROPERTY_CAMERACONTROL_PRIVACY` GET responds with 40-byte `KSCAMERA_EXTENDEDPROP_HEADER` + `ULONGLONG Value=0`, Flags=Capability=`KSCAMERA_EXTENDEDPROP_PRIVACY_OFF (2)` | KsProperty passes |
| 11| Event for same (set, id) BASICSUPPORT responds with 32-byte `KSCAMERA_EXTENDEDPROP_HEADER`, Size=32, Flags=Capability=PRIVACY_OFF | Event passes |
| 12| Implement `IMFExtendedCameraController` / `IMFExtendedCameraControl` and return well-formed empty ROI payloads | Clears previous `E_NOINTERFACE` path |
| 13| Move runtime COM registration from HKCU to HKLM and add one-time installer | Required for FrameServerMonitor / FrameServer visibility |
| 14| Move installed DLL from `target\debug\mf-camera-runtime` to `%ProgramFiles%\VulVATAR\VirtualCamera` and build with static CRT for install | Addresses `Start → 0x8007007E` before DLL trace starts |
| 15| Remove stale HKCU CLSID registration before using HKLM | Fixes HKCR resolving the old `target\debug` DLL despite a good HKLM install |

All changes are in the repo. The current focus is validating the HKLM install
path and then checking whether `IMFVirtualCamera::Start` proceeds to
`Source::Start`.

## Diagnostics I already did

- **Event Viewer sweep** (30 minutes, all channels, Error+Warning) for the
  CLSID / "vulvatar" / "virtual camera" / `MFCreateVirtualCamera` strings
  → 0 hits. MF does not log this failure.
- **Process-level DLL check**: `BUILD_MARKER` in the trace shows the fresh
  DLL *is* being loaded; Frame Server is not using a cached old copy.
- MS `Windows-Camera/Samples/VirtualCamera/VirtualCameraMediaSource/` used
  as reference throughout. Interfaces and attributes match.

## Likely next suspects

1. **Run elevated and verify HKLM registration.**
   `IMFExtendedCameraController` is now implemented. A smoke test reaches
   the privacy KS probes without `E_NOINTERFACE`; the remaining
   `0x80070003` path failure is consistent with missing HKLM visibility for
   FrameServerMonitor / FrameServer. Verify the elevated install writes:
   `HKLM\Software\Classes\CLSID\{B5F1C320-2B8F-4A9C-9BDC-43B0E8E6B2E1}\InprocServer32`
   to `%ProgramFiles%\VulVATAR\VirtualCamera\...`.

2. **Identify the unknown IID `2032C7EF-76F6-492A-94F3-4A81F69380CC`.**
   Not in windows-rs 0.58 or 0.62. Probably a Windows 11 SDK-only interface
   for virtual-camera orchestration. Searching the Windows SDK (`mfidl.h`,
   `mfapi.h`, `mfobjects.h`, `mfidl_p.h`, `mferror.h`, even
   `wincodecsdk.h`) for the GUID would identify it. Once identified, we can
   decide whether to implement it.

3. **Compare against a working virtual camera.** OBS Virtual Camera,
   XSplit VCam and Snap Camera all register under MF on Windows 11 and
   work. Dumping their registry entries (InprocServer32 path + sibling
   keys), running Process Monitor while their virtual camera is opened by
   Windows Camera, and diffing against our setup, would highlight any
   registration or category we are still missing.

4. **Windows Studio Effects camera sample.** MS shipped a newer sample for
   Windows 11's Studio Effects which targets the current Frame Server
   requirements more directly than `SimpleMediaSource`. Not yet located,
   but worth finding.

5. **Enable MFTrace ETW logging** to capture Frame Server's internal error
   when it rejects our source. `mftrace.exe` from the Windows SDK attaches
   to the Frame Server process and emits structured logs that often
   include the exact `HRESULT` and the callsite. This would end the
   guessing.

6. **Try `KSCAMERA_EXTENDEDPROP_PRIVACY_ON|OFF = 3` as `Capability`**
   instead of just `PRIVACY_OFF = 2`. Capability is a mask; reporting "both
   states supported" may satisfy Frame Server even though the current
   `Flags` says off.

7. **Attach to Frame Server with a debugger.** svchost-hosted; need
   `tasklist /svc /fi "services eq CameraEngineService"` to find the PID,
   then `WinDbg /p <pid>` and break on `mfcore.dll!CMFVirtualCamera::Start`.
   Heavy, but foolproof.

## Concrete files to read

| File                                          | What's there                                     |
|-----------------------------------------------|--------------------------------------------------|
| `vulvatar-mf-camera/src/lib.rs`               | DLL exports, `DllGetClassObject`, `BUILD_MARKER` |
| `vulvatar-mf-camera/src/class_factory.rs`     | `CreateInstance` returns the activate            |
| `vulvatar-mf-camera/src/activate.rs`          | `IMFActivate` + 28-method `IMFAttributes` delegation to internal `MFCreateAttributes` instance, source cached for repeated ActivateObject |
| `vulvatar-mf-camera/src/media_source.rs`      | The big one. All the Windows-11-specific probing handlers live here |
| `vulvatar-mf-camera/src/media_stream.rs`      | `IMFMediaStream2`                                |
| `vulvatar-mf-camera/src/trace.rs`             | File-based tracing                               |
| `vulvatar-mf-camera/src/registry.rs`          | DllRegisterServer / DllUnregisterServer          |
| `src/output/mf_virtual_camera.rs`             | Main-app side: `MFCreateVirtualCamera`, HKLM self-registration, scratch-copy logic |

## Memory notes

Two relevant entries in the auto-memory system:

- `project_mf_virtual_camera.md` — contract that the DLL must satisfy for
  Windows 11 Frame Server (interfaces + stream attrs + privacy KS responses).
  Update this when the current issue is solved so the knowledge persists.
- `feedback_no_stopgap.md` — the user wants the proper architecture
  implemented directly, not staged fallbacks. No "just make it work
  temporarily" patches.

## Git log for this thread

```
189594b Fill EXTENDEDPROP_HEADER in KsEvent BASICSUPPORT reply
683224b Advertise PRIVACY event BASICSUPPORT
cae7809 Reply to PRIVACY GET with KSCAMERA_EXTENDEDPROP layout
fa01d8f Trace PRIVACY KsProperty path + build marker
0b3a925 Answer PROPSETID_VIDCAP_CAMERACONTROL / PRIVACY GET
b7b77f3 Populate source attributes + log KS property-set GUIDs
92e6458 Cache media source instance inside the activate
6e398ce KsProperty/KsMethod/KsEvent: return ERROR_SET_NOT_FOUND
9a6ab96 Add IMFActivate wrapper + tracing
2ff2dea Set MF_DEVICESTREAM_FRAMESERVER_SHARED and friends
90b7dd2 Match MS SimpleMediaSource interface set
5f9e3bd Implement IMFGetService stub on the virtual camera source
0e29e7d Implement IKsControl on the virtual camera media source
f0e40d6 Add IMFMediaSourceEx + scratch-copy DLL registration
7ac21af Auto-register MF virtual camera DLL for dev builds
b8ffcf2 Wire MFCreateVirtualCamera + presentation descriptor
8f2d699 Add vulvatar-mf-camera cdylib skeleton
```

Each commit message explains which exact probe / interface / attribute was
added and why — read them top-to-bottom for the debugging timeline.

## Ask the next agent should answer first

Before writing code, read the most recent
`%TEMP%\vulvatar_mf_camera.log` entry from the user and confirm the
`BUILD_MARKER` matches the committed tree (so we know the DLL is fresh).
If the marker is stale, the user needs to restart Windows Camera Frame
Server or reboot — Frame Server is keeping an old scratch-copy alive.
