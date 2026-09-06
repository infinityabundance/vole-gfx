//! Rayon parallel backend (Phase F).
//!
//! Rayon operates **above** SIMD: the request is partitioned into coarse
//! independent display bands; each band runs the block materializer
//! (scalar/AVX2/AVX-512 underneath, per the path requested).  Bands are
//! independent: composition reads only the scene and writes only the band's
//! own output rows, so results are identical regardless of scheduling.
//!
//! Residual closure: bindings are processed per band in binding order; each
//! record affects one pixel, so per-band application is equivalent to the
//! whole-request pass (XOR records are applied once per pixel).

use crate::limits::Reject;
use crate::materialize::blocked::BlockMaterializer;
use crate::materialize::blocks::BlockShape;
use crate::materialize::scalar::{Counters, Materialized};
use crate::observation::{ObservationRequest, Output};
use crate::state::Scene;
use rayon::prelude::*;

/// Build a custom pool sized from `VOLE_GFX_THREADS` (default: logical CPUs).
pub fn pool() -> rayon::ThreadPool {
    let n = std::env::var("VOLE_GFX_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool")
}

/// Which SIMD width the rayon workers use per band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandSimd {
    Scalar,
    Avx2,
    Avx512,
    Auto,
}

impl BandSimd {
    pub fn name(self) -> &'static str {
        match self {
            BandSimd::Scalar => "rayon-scalar",
            BandSimd::Avx2 => "rayon-avx2",
            BandSimd::Avx512 => "rayon-avx512",
            BandSimd::Auto => "rayon-auto",
        }
    }
}

/// Materialize a request by splitting the domain into row bands of
/// `band_height` rows (full width), each processed on a rayon worker with the
/// block materializer at `shape` and the requested per-band SIMD path.
pub fn materialize_bands(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
    band_height: u32,
) -> Result<Materialized, Reject> {
    materialize_bands_with(scene, req, shape, band_height, BandSimd::Auto)
}

/// `materialize_bands` with an explicit per-band SIMD policy (for the
/// backend matrix: rayon-scalar / rayon-avx2 / rayon-avx512).
pub fn materialize_bands_with(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
    band_height: u32,
    simd: BandSimd,
) -> Result<Materialized, Reject> {
    req.validate()?;
    if let crate::observation::Domain::IrregularSamples(_) = &req.domain {
        // sparse domains are tiny by nature; run on the calling thread
        let mut bm = BlockMaterializer::new(scene, crate::state::index::DEFAULT_CELL_PX);
        return bm.materialize(req, shape);
    }
    let b = req.domain.bounds();
    let (w, h) = (b.width() as u32, b.height() as u32);
    if w == 0 || h == 0 {
        return Err(Reject::Degenerate);
    }
    let full = Output::new(w, h, req.format)?;
    let n_bands = h.div_ceil(band_height.max(1));
    let bands: Vec<(u32, crate::fixed::IRect)> = (0..n_bands)
        .map(|bi| {
            let y0 = bi * band_height;
            let y1 = (y0 + band_height).min(h);
            let rect = crate::fixed::IRect::new(b.x0, b.y0 + y0 as i32, b.x1, b.y0 + y1 as i32);
            (y0, rect)
        })
        .collect();

    let results: Vec<Result<Vec<u8>, Reject>> = bands
        .par_iter()
        .map(|&(_, band_rect)| {
            // Each band is its own sub-request over the same scene.
            let sub = crate::observation::ObservationRequest {
                time_ns: req.time_ns,
                view: req.view,
                domain: crate::observation::Domain::Rectangle {
                    x0: band_rect.x0,
                    y0: band_rect.y0,
                    w: band_rect.width() as u32,
                    h: band_rect.height() as u32,
                },
                sampling: req.sampling,
                format: req.format,
            };
            // per-band path selection
            let mut bm = BlockMaterializer::new(scene, crate::state::index::DEFAULT_CELL_PX);
            let m = match simd {
                BandSimd::Scalar => bm.materialize(&sub, shape)?,
                BandSimd::Avx2 if super::avx2::has_avx2() => {
                    match super::avx2::materialize_simple(scene, &sub, shape)? {
                        Some(m) => m,
                        None => bm.materialize(&sub, shape)?,
                    }
                }
                BandSimd::Avx512 if super::avx512::has_avx512() => {
                    match super::avx512::materialize_simple(scene, &sub, shape)? {
                        Some(m) => m,
                        None => bm.materialize(&sub, shape)?,
                    }
                }
                BandSimd::Auto => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        if super::avx512::has_avx512() {
                            if let Some(m) = super::avx512::materialize_simple(scene, &sub, shape)?
                            {
                                m
                            } else {
                                bm.materialize(&sub, shape)?
                            }
                        } else if super::avx2::has_avx2() {
                            if let Some(m) = super::avx2::materialize_simple(scene, &sub, shape)? {
                                m
                            } else {
                                bm.materialize(&sub, shape)?
                            }
                        } else {
                            bm.materialize(&sub, shape)?
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        bm.materialize(&sub, shape)?
                    }
                }
                _ => bm.materialize(&sub, shape)?,
            };
            Ok(m.output.data)
        })
        .collect();

    // deterministic merge in band order
    let mut out = full;
    let mut counters = Counters::default();
    let stride = w as usize * req.format.bytes_per_sample();
    for (band, result) in bands.iter().zip(results) {
        let (y0, _rect) = *band;
        let data = result?;
        let h0 = (y0 + band_height).min(h) - y0;
        let start = (y0 as usize) * stride;
        out.data[start..start + h0 as usize * stride]
            .copy_from_slice(&data[..h0 as usize * stride]);
        counters.samples += h0 as u64 * w as u64;
    }
    counters.samples = w as u64 * h as u64;
    Ok(Materialized {
        output: out,
        counters,
    })
}

/// Throughput convenience: pool + `materialize_bands` with sensible default
/// band height (a court-swept parameter).
pub fn materialize_parallel(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<Materialized, Reject> {
    let pool = pool();
    pool.install(|| materialize_bands(scene, req, shape, 64))
}

/// Which SIMD width the rayon workers would use underneath (for receipts).
pub fn worker_simd() -> &'static str {
    if super::avx512::has_avx512() {
        "avx512"
    } else if super::avx2::has_avx2() {
        "avx2"
    } else {
        "scalar"
    }
}
