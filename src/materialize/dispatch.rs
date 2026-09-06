//! Backend dispatch (CPU).
//!
//! Coarse-grained runtime dispatch: features are detected once per request,
//! never per pixel.  `materialize_auto` tries the exact SIMD simple path
//! (opaque integer-translation composition) and falls back to the blocked
//! materializer, which itself is byte-identical to the scalar oracle.

pub mod avx2;
pub mod avx512;
pub mod rayon;

use super::blocks::BlockShape;
use super::scalar::Materialized;
use crate::limits::Reject;
use crate::observation::ObservationRequest;
use crate::state::Scene;

/// Which execution path a materialization took (recorded in receipts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    Scalar,
    Blocked,
    Avx2,
    Avx512,
    Rayon,
}

impl Path {
    pub fn name(self) -> &'static str {
        match self {
            Path::Scalar => "scalar",
            Path::Blocked => "blocked",
            Path::Avx2 => "avx2",
            Path::Avx512 => "avx512",
            Path::Rayon => "rayon",
        }
    }
}

/// Auto dispatch: AVX2 simple path → AVX-512 simple path → blocked/scalar.
///
/// Order follows the *measured* execution economics, not ISA width: the
/// phase-e receipt (cpu-backend-matrix, this host/court) recorded AVX-512 at
/// ~144 µs vs AVX2 at ~130 µs for the same byte-exact workload, so AVX-512 is
/// only consulted when AVX2 is unavailable or ineligible.  The chosen path is
/// always recorded so courts can re-derive the policy from receipts.
pub fn materialize_auto(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<(Materialized, Path), Reject> {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2::has_avx2()
            && let Some(m) = avx2::materialize_simple(scene, req, shape)?
        {
            return Ok((m, Path::Avx2));
        }
        if avx512::has_avx512()
            && let Some(m) = avx512::materialize_simple(scene, req, shape)?
        {
            return Ok((m, Path::Avx512));
        }
    }
    let mut bm =
        super::blocked::BlockMaterializer::new(scene, crate::state::index::DEFAULT_CELL_PX);
    let m = bm.materialize(req, shape)?;
    Ok((m, Path::Blocked))
}

/// Rayon + auto-SIMD throughput path (Phase F).
pub fn materialize_rayon_auto(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<(Materialized, Path), Reject> {
    let m = rayon::materialize_parallel(scene, req, shape)?;
    Ok((m, Path::Rayon))
}
