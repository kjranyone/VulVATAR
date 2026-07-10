# Building the `realsense` feature (RealSense D435 depth capture)

The `realsense` cargo feature links `realsense-rust` → `realsense-sys` →
the native **librealsense2** SDK. It is **off by default**; the normal
build is unaffected. This wiring is fiddly on Windows because
`realsense-sys` discovers the SDK purely through `pkg-config` (a Unix
convention), so a few pieces must be provided by hand.

## System prerequisites

| Dependency | This machine | How |
| --- | --- | --- |
| Intel RealSense SDK 2.0 | `C:\Users\kojiro\Documents\RealSense SDK 2.0` (**non-default path**, has a space) | Intel/librealsense Windows installer |
| LLVM / libclang | `C:\Program Files\LLVM` | `winget install LLVM.LLVM` (needed by `buildtime-bindgen`) |
| pkg-config | `%LOCALAPPDATA%\Microsoft\WinGet\Links\pkg-config.exe` | `winget install bloodrock.pkg-config-lite` |

`buildtime-bindgen` regenerates the FFI bindings against the **installed
SDK's** headers (2.58) rather than the crate's vendored 2.56.5 set, so a
version bump on the SDK is fine.

## The hand-written `realsense2.pc`

The SDK ships no `.pc` file, so `build-support/pkgconfig/realsense2.pc`
supplies one. Two non-obvious details:

1. **Use the 8.3 short path** for `prefix`
   (`C:/Users/kojiro/DOCUME~1/REALSE~1.0`) — the real path contains a
   space (`RealSense SDK 2.0`) which breaks pkg-config / linker arg
   splitting.
2. **`Cflags` must define `DLL_FOLDER`**
   (`-DDLL_FOLDER=${prefix}/bin/x64`). `realsense-sys`'s `build.rs`
   indexes `library.defines["DLL_FOLDER"]` unconditionally and panics
   `no entry found for key` without it. It uses that folder to copy
   `realsense2.dll` into `target/<profile>/deps/`.

Regenerate it (short path is machine-specific) with:

```powershell
$fso = New-Object -ComObject Scripting.FileSystemObject
$prefix = ($fso.GetFolder("C:\Users\kojiro\Documents\RealSense SDK 2.0").ShortPath) -replace '\\','/'
# write build-support/pkgconfig/realsense2.pc with that $prefix (see git history)
```

## Build & run

Set three env vars, then build with the feature:

```powershell
$env:PKG_CONFIG_PATH = "$PWD\build-support\pkgconfig"
$env:LIBCLANG_PATH   = "C:\Program Files\LLVM\bin"
$env:PATH            = "$env:LOCALAPPDATA\Microsoft\WinGet\Links;C:\Users\kojiro\Documents\RealSense SDK 2.0\bin\x64;$env:PATH"

cargo run --bin diagnose_realsense --features realsense
# -> Context::new() OK — realsense2 linked.
#    connected devices: N
```

- `PKG_CONFIG_PATH` → our `.pc`; `LIBCLANG_PATH` → bindgen; SDK `bin\x64`
  on PATH → `realsense2.dll` at runtime (the `build.rs` deps-copy covers
  `cargo test`, but a `cargo run` bin loads from its own dir + PATH).
- `diagnose_realsense` is a feature-gated smoke test that only proves the
  native toolchain (link + device enumeration), not streaming.

`dev.ps1` wires these three env vars up automatically: use the
**`build (realsense)`** / **`run (realsense)`** menu entries (helper
`Invoke-CargoRealsense`). `PKG_CONFIG_PATH` resolves to the repo's
`build-support\pkgconfig`; the SDK root defaults to
`%USERPROFILE%\Documents\RealSense SDK 2.0` and LLVM to
`%ProgramFiles%\LLVM\bin`, both overridable via
`$env:VULVATAR_REALSENSE_SDK` / `$env:LIBCLANG_PATH`.
