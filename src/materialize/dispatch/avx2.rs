//! AVX2 backend (Phase D).
//!
//! Strategy: exact kernels over the primitives that profiling shows dominate
//! the court workloads — row blits of fully-opaque, integer-translation
//! instances and row fills of opaque colors, evaluated by the shared
//! simple-composition engine (`crate::materialize::simple`).  In the eligible
//! subset composition reduces to ordered memory writes, which is byte-for-byte
//! what the scalar oracle computes.  All other documents fall back (`None`) to
//! the block/scalar paths; the dispatcher and courts record which path ran.
//!
//! SIMD never changes arithmetic — there is no arithmetic here beyond exact
//! copies/fills, which is exactly why the subset is provably equivalent.  The
//! kernels are pure byte-lane movers; the scalar-simple ablation row
//! (`simple::materialize_simple_scalar`) provides the no-SIMD baseline of the
//! same engine so the courts can decompose specialization gain from SIMD gain.

use crate::materialize::blocks::BlockShape;
use crate::materialize::scalar::Materialized;
use crate::materialize::simple::{Kernels, simple_impl};
use crate::observation::ObservationRequest;
use crate::state::Scene;

/// x86 detection at the coarse dispatch boundary (never per pixel).
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Exact 32-byte SIMD copy; tail handled byte-wise.  Result identical to a
/// plain copy (no arithmetic involved), so no semantic divergence is
/// possible; SIMD only reduces instruction count and memory ops.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx2_kernel(dst: &mut [u8], src: &[u8]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: caller guarantees dst/src are valid disjoint slices of
        // length n; both sides are in-bounds for 32-byte loads/stores.
        let v = unsafe { _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i) };
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v) };
        i += 32;
    }
    while i < n {
        dst[i] = src[i];
        i += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn copy_row_avx2_kernel(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

fn copy_row_avx2(dst: &mut [u8], src: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: dst/src are disjoint slices of length n (caller
            // guarantees non-overlap: dst is the output buffer, src is an
            // object row).
            unsafe { copy_row_avx2_kernel(dst, src) };
            return;
        }
    }
    dst.copy_from_slice(src);
}

/// Exact 32-byte SIMD fill of a row slice already sized to `pattern` repeats;
/// tail handled byte-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_row_avx2_kernel(dst: &mut [u8], pattern: &[u8]) {
    use core::arch::x86_64::*;
    let bps = pattern.len();
    let n = dst.len();
    if bps == 4 {
        let p = u32::from_le_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);
        let v = _mm256_set1_epi32(p as i32);
        let mut i = 0usize;
        while i + 32 <= n {
            // SAFETY: dst is a mutable slice with >= 32 bytes remaining.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v) };
            i += 32;
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

fn fill_row_avx2(dst: &mut [u8], pattern: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: dst is a mutable slice of the output buffer.
            unsafe { fill_row_avx2_kernel(dst, pattern) };
            return;
        }
    }
    let bps = pattern.len();
    for px in dst.chunks_exact_mut(bps) {
        px.copy_from_slice(pattern);
    }
}

/// Attempt the exact AVX2 simple-composition path for an eligible document.
/// Returns `None` when the document is outside the eligible subset (caller
/// falls back to the block/scalar path).
pub fn materialize_simple(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<Option<Materialized>, crate::limits::Reject> {
    if has_avx2() {
        let k = Kernels {
            copy_row: copy_row_avx2,
            fill_row: fill_row_avx2,
        };
        simple_impl(scene, req, shape, &k)
    } else {
        // No AVX2 on this host: report ineligibility rather than silently
        // running the scalar engine under an "avx2" label.  The dispatcher
        // only routes here after has_avx2(); the scalar-simple ablation row
        // (`simple::materialize_simple_scalar`) is the honest no-SIMD
        // version of this engine.
        Ok(None)
    }
}
