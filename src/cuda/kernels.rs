// Rust CUDA device kernels (Phase G).
//
// Compiled to PTX with the `nvptx64-nvidia-cuda` target from this same crate
// via `scripts/ptx_shim.rs` (see docs/CUDA.md).  `no_std`, no heap, no
// panic/unwind, explicit bounded indexing.  Only byte-moving kernels so far;
// arithmetic kernels will import the exact-semantics constants.
//
// Kernels use the `extern "ptx-kernel"` ABI so rustc emits real launchable
// `.visible .entry` functions (verified empirically on the pinned nightly).

use core::arch::nvptx::*;

fn linear_tid() -> u64 {
    // Wrap-free arithmetic: geometry values are bounded by launch
    // validation on the host, and indices never overflow u64.
    unsafe { _thread_idx_x() as u64 + (_block_idx_x() as u64) * (_block_dim_x() as u64) }
}

/// Grid-stride fill kernel: writes `pattern` into `dst[y*w + x]` for a
/// canonical row-major surface.  Deterministic; exactness is trivially byte
/// equality with the host `fill_row` kernels (the kernel only moves bytes).
///
/// # Safety
/// `dst` must point to `w*h` writable bytes for the launch geometry; the host
/// wrapper validates bounds and launch sizes before launching.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vgf_fill_u8(
    dst: *mut u8,
    w: u32,
    h: u32,
    pattern: u8,
    total_threads: u32,
) {
    let mut i = linear_tid();
    let n = (w as u64) * (h as u64);
    let stride = total_threads as u64;
    while i < n {
        // SAFETY: caller validated dst bounds for this launch.
        unsafe { *dst.add(i as usize) = pattern };
        i += stride;
    }
}

/// Exact RGBA8 row copy kernel (opaque sprite blit): copies `n` bytes from
/// `src` to `dst`.  Byte-exact by construction; the arithmetic-free analog of
/// the AVX2/AVX-512 copy kernels.
///
/// # Safety
/// `dst`/`src` must point to `n` valid bytes each; the host validates launch
/// geometry before launching.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn vgf_copy_u8(
    dst: *mut u8,
    src: *const u8,
    n: u32,
    total_threads: u32,
) {
    let mut i = linear_tid();
    while i < n as u64 {
        // SAFETY: caller validated bounds for this launch.
        unsafe { *dst.add(i as usize) = *src.add(i as usize) };
        i += total_threads as u64;
    }
}
