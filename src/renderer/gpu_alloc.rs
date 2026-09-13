//! Shared GPU buffer allocation helpers.
//!
//! Every SSBO/UBO allocation in the renderer repeats the same
//! `Buffer::from_iter` / `from_data` / `new_slice` +
//! `BufferCreateInfo` + `AllocationCreateInfo` + `map_err` block. These
//! helpers keep that block in one place so a call site states only the
//! buffer's residency (host-writable vs device-local), usage flags,
//! payload, and a label for the error string.
//!
//! Residency convention used across the compute prepass:
//! - `host_*`: `PREFER_DEVICE | HOST_SEQUENTIAL_WRITE` — anything the
//!   CPU writes at least once after creation (base vertex data, morph
//!   deltas, per-frame weight / control blocks).
//! - `device_*`: `PREFER_DEVICE` alone — GPU-only payloads such as
//!   compute output VBOs.

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

fn host_visible_alloc() -> AllocationCreateInfo {
    AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
    }
}

fn device_local_alloc() -> AllocationCreateInfo {
    AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
        ..Default::default()
    }
}

fn buffer_info(usage: BufferUsage) -> BufferCreateInfo {
    BufferCreateInfo {
        usage,
        ..Default::default()
    }
}

/// Host-writable buffer filled from an iterator.
pub(super) fn host_buffer<T, I>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    usage: BufferUsage,
    iter: I,
    label: &str,
) -> Result<Subbuffer<[T]>, String>
where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    Buffer::from_iter(
        memory_allocator.clone(),
        buffer_info(usage),
        host_visible_alloc(),
        iter,
    )
    .map_err(|e| format!("renderer: {label} alloc failed: {e}"))
}

/// Host-writable uniform buffer from a single value (control blocks).
pub(super) fn host_ubo<T: BufferContents>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    value: T,
    label: &str,
) -> Result<Subbuffer<T>, String> {
    Buffer::from_data(
        memory_allocator.clone(),
        buffer_info(BufferUsage::UNIFORM_BUFFER),
        host_visible_alloc(),
        value,
    )
    .map_err(|e| format!("renderer: {label} alloc failed: {e}"))
}

/// Host-READABLE uninitialised buffer of `len` elements (diagnostic
/// staging: the recording half copies a device-local compute output
/// into it, and the CPU maps it after the frame fence). Requires the
/// `HOST_RANDOM_ACCESS` filter — `HOST_SEQUENTIAL_WRITE` memory is
/// write-only by contract.
pub(super) fn host_read_slice<T: BufferContents>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    len: u64,
    label: &str,
) -> Result<Subbuffer<[T]>, String> {
    Buffer::new_slice::<T>(
        memory_allocator.clone(),
        buffer_info(BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER),
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        len,
    )
    .map_err(|e| format!("renderer: {label} alloc failed: {e}"))
}

/// Device-local uninitialised buffer of `len` elements (compute output).
pub(super) fn device_slice<T: BufferContents>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    usage: BufferUsage,
    len: u64,
    label: &str,
) -> Result<Subbuffer<[T]>, String> {
    Buffer::new_slice::<T>(
        memory_allocator.clone(),
        buffer_info(usage),
        device_local_alloc(),
        len,
    )
    .map_err(|e| format!("renderer: {label} alloc failed: {e}"))
}

/// Return the shared 1-element stub stored in `slot`, allocating it on
/// first use. Vulkano refuses zero-element storage buffers, so stubs
/// stand in for "this binding has no real data" descriptor slots; the
/// corresponding control flag keeps the shader from reading them.
pub(super) fn get_or_init_stub<T>(
    slot: &mut Option<Subbuffer<[T]>>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    usage: BufferUsage,
    init: T,
    label: &str,
) -> Result<Subbuffer<[T]>, String>
where
    T: BufferContents + Clone,
{
    if let Some(ref stub) = *slot {
        return Ok(stub.clone());
    }
    let stub = host_buffer(memory_allocator, usage, [init].into_iter(), label)?;
    *slot = Some(stub.clone());
    Ok(stub)
}
