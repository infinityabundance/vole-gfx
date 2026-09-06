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

/// Auto dispatch: AVX-512 simple path → AVX2 simple path → blocked/scalar.
pub fn materialize_auto(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<(Materialized, Path), Reject> {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512::has_avx512()
            && let Some(m) = avx512::materialize_simple(scene, req, shape)? {
                return Ok((m, Path::Avx512));
            }
        if avx2::has_avx2()
            && let Some(m) = avx2::materialize_simple(scene, req, shape)? {
                return Ok((m, Path::Avx2));
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
