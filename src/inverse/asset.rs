//! Raster asset input to the inverse compiler (Phase I).
//!
//! An `Asset` is a flat raster that the compiler tries to explain as bounded
//! procedural state + residual.  Assets are treated as code-value tables in
//! their own format: every comparison with a generated explanation happens in
//! the asset's code space, so a Gray8 asset compares gray codes and an Rgba8
//! asset compares RGBA codes (byte-exact reconstruction gate).

use crate::color::{ColorFormat, Rgba};
use crate::limits::Reject;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub w: u32,
    pub h: u32,
    pub format: ColorFormat,
    /// Row-major `w * h * bytes_per_sample` code values.
    pub data: Vec<u8>,
}

impl Asset {
    /// Build from a raster object's payload (must be exactly w*h samples).
    pub fn new(w: u32, h: u32, format: ColorFormat, data: Vec<u8>) -> Result<Asset, Reject> {
        let expect = w as u64 * h as u64 * format.bytes_per_sample() as u64;
        if w == 0 || h == 0 {
            return Err(Reject::Degenerate);
        }
        if w as u64 > crate::limits::MAX_DIMENSION as u64
            || h as u64 > crate::limits::MAX_DIMENSION as u64
        {
            return Err(Reject::DimensionTooLarge);
        }
        if data.len() as u64 != expect {
            return Err(Reject::PayloadMismatch);
        }
        Ok(Asset { w, h, format, data })
    }

    pub fn sample_count(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Index of pixel `(i, j)` in `data`.
    #[inline]
    pub fn offset(&self, i: u32, j: u32) -> usize {
        (j as usize * self.w as usize + i as usize) * self.format.bytes_per_sample()
    }

    /// Code bytes of pixel `(i, j)` (always 4 bytes; gray uses the low byte).
    #[inline]
    pub fn code_at(&self, i: u32, j: u32) -> [u8; 4] {
        let o = self.offset(i, j);
        let bps = self.format.bytes_per_sample();
        let mut c = [0u8; 4];
        c[..bps].copy_from_slice(&self.data[o..o + bps]);
        c
    }

    /// Pixel as RGBA (gray samples become `Rgba::gray`).
    #[inline]
    pub fn rgba_at(&self, i: u32, j: u32) -> Rgba {
        let c = self.code_at(i, j);
        match self.format {
            ColorFormat::Gray8 => Rgba::gray(c[0]),
            ColorFormat::Rgba8 => Rgba::from_bytes(c),
        }
    }

    /// Row slice of raw codes.
    pub fn row_codes(&self, j: u32) -> &[u8] {
        let bps = self.format.bytes_per_sample();
        let row = self.w as usize * bps;
        &self.data[j as usize * row..][..row]
    }

    /// Are two pixels' code values equal?
    #[inline]
    pub fn codes_equal(&self, i0: u32, j0: u32, i1: u32, j1: u32) -> bool {
        self.code_at(i0, j0) == self.code_at(i1, j1)
    }
}
