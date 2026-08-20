//! Bounded single-slot "latest-only" inbox for async worker threads.
//!
//! The YOLOX person-detection worker only ever needs the *freshest*
//! input frame: intermediate frames that pile up while the worker is
//! busy are stale and should be discarded. This must hold under GPU
//! stalls too, so the inbox has to be bounded.
//!
//! A `sync_channel(1)` + `try_send` gets this backwards — when the single
//! slot is full it drops the *newest* frame and the worker later consumes
//! the *stale* buffered one, inverting the freshness guarantee. An
//! unbounded channel with a worker-side drain loop keeps freshness but can
//! accumulate full-frame RGB buffers without limit during a stall.
//!
//! `LatestCell` gets both: `put` overwrites the slot with the newest item
//! (dropping any pending one), so the worker always takes the freshest
//! submission and memory is bounded to a single item. It replaces the
//! duplicated bounded-submit/drain logic both workers used to carry.

use std::sync::{Arc, Condvar, Mutex};

struct State<T> {
    item: Option<T>,
    closed: bool,
}

pub struct LatestCell<T> {
    state: Mutex<State<T>>,
    cv: Condvar,
}

impl<T> LatestCell<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                item: None,
                closed: false,
            }),
            cv: Condvar::new(),
        })
    }

    /// Store `item` as the latest input, dropping any previously-pending
    /// item that the worker had not yet taken. No-op once [`Self::close`]
    /// has been called (the worker is shutting down).
    pub fn put(&self, item: T) {
        let mut s = self.state.lock().unwrap();
        if s.closed {
            return;
        }
        s.item = Some(item);
        drop(s);
        self.cv.notify_one();
    }

    /// Block until an item is available, then take it. Returns `None`
    /// once the cell has been closed and drained — the worker loop's exit
    /// signal (equivalent to a channel `recv` returning `Err`).
    pub fn take_blocking(&self) -> Option<T> {
        let mut s = self.state.lock().unwrap();
        loop {
            if let Some(item) = s.item.take() {
                return Some(item);
            }
            if s.closed {
                return None;
            }
            s = self.cv.wait(s).unwrap();
        }
    }

    /// Signal the worker to exit: any current `take_blocking` wakes and
    /// returns `None` (once its slot is drained). Idempotent. The owner
    /// must call this before joining the worker thread — unlike a channel,
    /// dropping the sender handle does not wake a blocked `take_blocking`.
    pub fn close(&self) {
        let mut s = self.state.lock().unwrap();
        s.closed = true;
        drop(s);
        self.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_keeps_only_the_latest() {
        let cell = LatestCell::new();
        cell.put(1);
        cell.put(2);
        cell.put(3);
        assert_eq!(cell.take_blocking(), Some(3));
    }

    #[test]
    fn close_drains_then_signals_exit() {
        let cell = LatestCell::new();
        cell.put(7);
        cell.close();
        // Pending item is still delivered before the exit signal.
        assert_eq!(cell.take_blocking(), Some(7));
        // Then the closed cell reports exit rather than blocking forever.
        assert_eq!(cell.take_blocking(), None);
    }

    #[test]
    fn put_after_close_is_ignored() {
        let cell = LatestCell::new();
        cell.close();
        cell.put(1);
        assert_eq!(cell.take_blocking(), None);
    }

    #[test]
    fn worker_thread_takes_latest_and_exits_on_close() {
        use std::sync::mpsc;
        use std::thread;
        let cell = LatestCell::<u32>::new();
        let worker_cell = Arc::clone(&cell);
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            while let Some(v) = worker_cell.take_blocking() {
                tx.send(v).unwrap();
            }
        });
        cell.put(10);
        assert_eq!(rx.recv().unwrap(), 10);
        cell.close();
        handle.join().unwrap();
    }
}
