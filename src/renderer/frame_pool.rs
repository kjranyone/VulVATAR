#![allow(dead_code)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct PoolFrame {
    pub data: Vec<u8>,
    pub extent: [u32; 2],
    pub frame_index: u64,
    in_use: Arc<AtomicBool>,
}

impl PoolFrame {
    pub fn in_use_flag(&self) -> Arc<AtomicBool> {
        self.in_use.clone()
    }

    pub fn is_in_use(&self) -> bool {
        self.in_use.load(Ordering::Acquire)
    }

    pub fn reset(&mut self, width: u32, height: u32, frame_index: u64) {
        self.data.clear();
        self.extent = [width, height];
        self.frame_index = frame_index;
        self.in_use.store(true, Ordering::Release);
    }
}

pub struct FramePool {
    slots: Vec<PoolFrame>,
    capacity: usize,
    next_index: u64,
}

const DEFAULT_CAPACITY: usize = 2;

impl FramePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            capacity: capacity.max(DEFAULT_CAPACITY),
            next_index: 0,
        }
    }

    pub fn acquire_or_create(&mut self, width: u32, height: u32) -> usize {
        let needed = (width as usize) * (height as usize) * 4;

        for (i, frame) in self.slots.iter().enumerate() {
            if !frame.in_use.load(Ordering::Acquire) && frame.data.capacity() >= needed {
                self.slots[i].reset(width, height, self.next_index);
                self.next_index += 1;
                return i;
            }
        }

        if self.slots.len() < self.capacity {
            let idx = self.slots.len();
            let frame = PoolFrame {
                data: Vec::with_capacity(needed),
                extent: [width, height],
                frame_index: self.next_index,
                in_use: Arc::new(AtomicBool::new(true)),
            };
            self.slots.push(frame);
            self.next_index += 1;
            return idx;
        }

        for (i, frame) in self.slots.iter().enumerate() {
            if !frame.in_use.load(Ordering::Acquire) {
                self.slots[i].data = Vec::with_capacity(needed);
                self.slots[i].reset(width, height, self.next_index);
                self.next_index += 1;
                return i;
            }
        }

        let lru_idx = self
            .slots
            .iter()
            .enumerate()
            .min_by_key(|(_, f)| f.frame_index)
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.slots[lru_idx].data = Vec::with_capacity(needed);
        self.slots[lru_idx].reset(width, height, self.next_index);
        self.next_index += 1;
        lru_idx
    }

    pub fn release(&self, frame_index: u64) {
        for frame in &self.slots {
            if frame.frame_index == frame_index {
                frame.in_use.store(false, Ordering::Release);
                return;
            }
        }
    }

    pub fn release_by_flag(&self, flag: &AtomicBool) {
        flag.store(false, Ordering::Release);
    }

    pub fn get(&self, index: usize) -> Option<&PoolFrame> {
        self.slots.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut PoolFrame> {
        self.slots.get_mut(index)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|f| !f.in_use.load(Ordering::Acquire))
            .count()
    }
}

impl Default for FramePool {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}
