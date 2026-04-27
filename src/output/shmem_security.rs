//! Permissive DACL helper for `Global\` named shared sections.
//!
//! When the interactive-user producer (this process) and a non-interactive
//! consumer (FrameServer-hosted MF DLL or a DirectShow filter loaded into
//! another user's process) race to be the first to call
//! `CreateFileMappingW` on the same name, whichever side wins applies its
//! own DACL to the kernel section object. With `lpFileMappingAttributes =
//! NULL` the kernel uses the creator's default DACL, which denies the
//! other side write access — manifesting as `ERROR_ACCESS_DENIED` (5) on
//! the loser's `CreateFileMappingW` even though the names match.
//!
//! Implemented with raw FFI rather than the `windows` crate so this
//! helper can be linked from `frame_sink.rs::win32_shmem` (gated by
//! `cfg(target_os = "windows")` only — the `windows` crate is brought
//! in by the optional `virtual-camera` feature). Callers that go through
//! the `windows` crate's `CreateFileMappingW` cast `as_ptr()` to
//! `*const windows::Win32::Security::SECURITY_ATTRIBUTES`; the layouts
//! are identical (the static_assertions block below verifies this when
//! the `windows` crate is available).

use std::ffi::{c_void, OsStr};
use std::os::windows::ffi::OsStrExt;
use std::ptr;

const SDDL_REVISION_1: u32 = 1;
// GA -> SYSTEM, BUILTIN\Administrators, Interactive Users (the producer);
// GR -> LocalService (the FrameServer-hosted consumer reads the frames).
const SDDL: &str = "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GA;;;IU)(A;;GR;;;LS)";

/// Layout-compatible mirror of Win32 `SECURITY_ATTRIBUTES`. Matches
/// `windows::Win32::Security::SECURITY_ATTRIBUTES` byte-for-byte; see
/// the static_assertions block at the bottom of this file.
#[repr(C)]
pub struct RawSecurityAttributes {
    pub n_length: u32,
    pub security_descriptor: *mut c_void,
    pub inherit_handle: i32,
}

extern "system" {
    fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
        string_security_descriptor: *const u16,
        string_sd_revision: u32,
        security_descriptor: *mut *mut c_void,
        security_descriptor_size: *mut u32,
    ) -> i32;

    fn LocalFree(mem: *mut c_void) -> *mut c_void;
}

pub struct PermissiveSharedMemorySecurity {
    descriptor: *mut c_void,
    attributes: RawSecurityAttributes,
}

impl PermissiveSharedMemorySecurity {
    pub fn new() -> Result<Self, String> {
        let wide: Vec<u16> = OsStr::new(SDDL)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut descriptor: *mut c_void = ptr::null_mut();
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide.as_ptr(),
                SDDL_REVISION_1,
                &mut descriptor,
                ptr::null_mut(),
            )
        };
        if ok == 0 || descriptor.is_null() {
            return Err(
                "ConvertStringSecurityDescriptorToSecurityDescriptorW failed".into(),
            );
        }

        let attributes = RawSecurityAttributes {
            n_length: std::mem::size_of::<RawSecurityAttributes>() as u32,
            security_descriptor: descriptor,
            inherit_handle: 0,
        };
        Ok(Self {
            descriptor,
            attributes,
        })
    }

    /// Pointer for callers using the raw extern `CreateFileMappingW`
    /// declaration in `frame_sink.rs::win32_shmem`, which takes
    /// `*mut SecurityAttributes`.
    pub fn as_mut_ptr(&mut self) -> *mut RawSecurityAttributes {
        &mut self.attributes
    }

    /// Pointer for callers using the `windows` crate's
    /// `CreateFileMappingW`, which takes
    /// `Option<*const SECURITY_ATTRIBUTES>`. Layout-compat with our
    /// `RawSecurityAttributes` is asserted at compile time below.
    #[cfg(feature = "virtual-camera")]
    pub fn as_win_ptr(&self) -> *const windows::Win32::Security::SECURITY_ATTRIBUTES {
        (&self.attributes as *const RawSecurityAttributes).cast()
    }
}

impl Drop for PermissiveSharedMemorySecurity {
    fn drop(&mut self) {
        if !self.descriptor.is_null() {
            unsafe {
                let _ = LocalFree(self.descriptor);
            }
            self.descriptor = ptr::null_mut();
        }
    }
}

// Compile-time guarantee that the cast in `as_win_ptr` is sound.
#[cfg(feature = "virtual-camera")]
const _: () = {
    use windows::Win32::Security::SECURITY_ATTRIBUTES;
    assert!(
        std::mem::size_of::<RawSecurityAttributes>()
            == std::mem::size_of::<SECURITY_ATTRIBUTES>()
    );
    assert!(
        std::mem::align_of::<RawSecurityAttributes>()
            == std::mem::align_of::<SECURITY_ATTRIBUTES>()
    );
};
