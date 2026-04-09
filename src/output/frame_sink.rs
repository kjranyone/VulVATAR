#[derive(Clone, Debug)]
pub enum FrameSink {
    VirtualCamera,
    SharedTexture,
    SharedMemory,
    ImageSequence,
}

#[derive(Clone, Debug)]
pub enum FrameSinkQueuePolicy {
    DropOldest,
    DropNewest,
    ReplaceLatest,
    BlockNotAllowed,
}
