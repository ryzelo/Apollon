//! Diagnostics ($APOLLON_SEARCH_STATS): thread time per kind of work, summed over every call. Parallel calls
//! add up, so the parts can exceed the wall-clock time. Off: one cached bool check per call.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

pub const MATMUL: usize = 0;
pub const ATTN: usize = 1;
pub const CUT: usize = 2;
pub const RUN: usize = 3;
pub const BATCH_SETUP: usize = 4;
pub const READ: usize = 5;
const NAMES: [&str; 6] = ["matrix products (CPU)", "attention (whole, incl. its products)", "cut applied to rows/columns", "Model::run (whole)", "run_batch setup (cut rows, K/V copies)", "readout (KL)"];
static T: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static N: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("APOLLON_SEARCH_STATS").is_some())
}

pub struct Guard(usize, Option<Instant>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(s) = self.1 {
            T[self.0].fetch_add(s.elapsed().as_nanos() as u64, Ordering::Relaxed);
            N[self.0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// time from here to the end of the scope
#[inline]
pub fn time(slot: usize) -> Guard {
    Guard(slot, on().then(Instant::now))
}

/// add a measured duration
pub fn add(slot: usize, secs: f64) {
    if on() {
        T[slot].fetch_add((secs * 1e9) as u64, Ordering::Relaxed);
        N[slot].fetch_add(1, Ordering::Relaxed);
    }
}

/// row-layers of the batches: run in total, and run although the candidate of the row changes nothing
/// before a later layer (it starts later than the batch)
static ROW_LAYERS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub fn row_layers(total: u64, idle: u64) {
    if on() {
        ROW_LAYERS[0].fetch_add(total, Ordering::Relaxed);
        ROW_LAYERS[1].fetch_add(idle, Ordering::Relaxed);
    }
}

pub fn report(label: &str) {
    if !on() {
        return;
    }
    let (rt, ri) = (ROW_LAYERS[0].load(Ordering::Relaxed), ROW_LAYERS[1].load(Ordering::Relaxed));
    if rt > 0 {
        eprintln!("diag: batches ({label}): {rt} row-layers, {ri} of them before the row's own first change ({:.1}%)", 100.0 * ri as f64 / rt as f64);
    }
    eprintln!("diag: thread time so far ({label}):");
    for i in 0..NAMES.len() {
        let n = N[i].load(Ordering::Relaxed);
        if n > 0 {
            eprintln!("diag:   {:<40} {:>9.1} s {:>10} calls", NAMES[i], T[i].load(Ordering::Relaxed) as f64 / 1e9, n);
        }
    }
}
