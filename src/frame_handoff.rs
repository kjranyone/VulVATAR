//! Neutral contract shared by renderer export and output routing.
//!
//! Renderer code owns the concrete Vulkan resources. Output code should only
//! see compact metadata plus an explicit lease/synchronization contract.

/// Unique identifier for an exported resource.
pub type ExportedResourceId = u64;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OutputFrameId(pub u64);

#[derive(Clone, Debug)]
pub enum OutputColorSpace {
    Srgb,
    LinearSrgb,
}

#[derive(Clone, Debug)]
pub enum AlphaMode {
    Opaque,
    Premultiplied,
    Straight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum HandoffPath {
    GpuSharedFrame,
    CpuReadback,
    SharedMemory,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    RequestedCpuReadback,
    ExternalHandleUnavailable,
    ExportPoolSaturated,
    MissingExternalHandle,
}

#[derive(Clone, Debug)]
pub enum ExternalHandleType {
    /// The frame has no external handle yet; useful while the GPU path is
    /// being wired up, but not a valid cross-process handoff on its own.
    Unavailable,
    Win32Kmt,
    D3D12Fence,
    VkSemaphore,
    SharedMemoryHandle,
}

#[derive(Clone, Debug)]
pub enum OutputSyncToken {
    None,
    /// The producer waited for render completion before publishing the token.
    /// Safe for the current handoff shape, but not a cross-process GPU wait
    /// primitive; a future zero-copy sink should prefer an exported fence or
    /// semaphore handle instead.
    ProducerWaitComplete,
    FenceValue(u64),
    SemaphoreHandle(u64),
}

#[derive(Clone, Debug)]
pub enum FrameLifetimeContract {
    SingleConsumerImmediate,
    SingleConsumerRetained,
}

#[derive(Clone, Debug)]
pub struct FrameLease {
    pub lease_id: u64,
    pub lifetime: FrameLifetimeContract,
}

#[derive(Clone, Debug)]
pub struct GpuFrameToken {
    pub resource_id: ExportedResourceId,
    pub handle_type: ExternalHandleType,
    pub external_handle: Option<u64>,
    pub sync: OutputSyncToken,
    pub lease: FrameLease,
}

impl GpuFrameToken {
    pub fn has_external_handle(&self) -> bool {
        self.external_handle.is_some()
    }
}
