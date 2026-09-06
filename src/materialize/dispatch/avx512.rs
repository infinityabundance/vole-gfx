//! AVX-512 backend (Phase E).
//!
//! Independent measured backend, not an assumed winner: wider stores and
//! masked tails reduce instructions, but clock-rate/port effects are
//! measured by the courts.  The kernels only move bytes (identical to the
//! AVX2 kernels' contract), so byte-parity with the oracle holds by
//! construction; only the width differs.

use crate::limits::Reject;
use crate::materialize::blocks::BlockShape;
use crate::materialize::dispatch::avx2::{self, Kernels};
use crate::materialize::scalar::Materialized;
use crate::observation::ObservationRequest;
use crate::state::Scene;

/// Required subfeatures for the byte kernels: F (32-bit ops) + BW (byte ops).
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512() -> bool {
    false
}

/// Exact 64-byte SIMD copy; tail byte-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn copy_row_avx512_kernel(dst: &mut [u8], src: &[u8]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0usize;
    while i + 64 <= n {
        // SAFETY: caller guarantees disjoint valid slices of length n.
        let v = unsafe { _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i) };
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, v) };
        i += 64;
    }
    while i < n {
        dst[i] = src[i];
        i += 1;
    }
}

/// Exact 64-byte SIMD fill; tail byte-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn fill_row_avx512_kernel(dst: &mut [u8], pattern: &[u8]) {
    use core::arch::x86_64::*;
    let bps = pattern.len();
    let n = dst.len();
    if bps == 4 {
        let p = u32::from_le_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);
        let v = _mm512_set1_epi32(p as i32);
        let mut i = 0usize;
        while i + 64 <= n {
            // SAFETY: dst has >= 64 bytes remaining.
            unsafe { _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, v) };
            i += 64;
        }
        while i < n {
            dst[i] = pattern[i % 4];
            i += 1;
        }
    } else {
        for px in dst.chunks_exact_mut(bps) {
            px.copy_from_slice(pattern);
        }
    }
}

fn copy_row_avx512(dst: &mut [u8], src: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: disjoint output/object-row slices; caller guarantees
            // non-overlap.
            unsafe { copy_row_avx512_kernel(dst, src) };
            return;
        }
    }
    dst.copy_from_slice(src);
}

fn fill_row_avx512(dst: &mut [u8], pattern: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: dst is a mutable slice of the output buffer.
            unsafe { fill_row_avx512_kernel(dst, pattern) };
            return;
        }
    }
    let bps = pattern.len();
    for px in dst.chunks_exact_mut(bps) {
        px.copy_from_slice(pattern);
    }
}

/// AVX-512 simple path (same eligibility and engine as AVX2).
pub fn materialize_simple(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<Option<Materialized>, Reject> {
    if has_avx512() {
        let k = Kernels {
            copy_row: copy_row_avx512,
            fill_row: fill_row_avx512,
        };
        avx2::simple_impl(scene, req, shape, &k)
    } else {
        Ok(None)
    }
}
