//! Explicit residual algebra (paper §residual).
//!
//! A residual binding corrects the *canonical surface* after procedural
//! composition.  U1 implements two exact algebras over literal (uncompressed)
//! sorted records:
//!
//! * `SPARSE_OVERWRITE`: each record replaces the sample value at `(x, y)`.
//! * `XOR`: each record xors `bps` bytes into the sample at `(x, y)`.
//!
//! Payload layout (normative, see `docs/IR.md`): `u64` record count, then per
//! record `u32 x`, `u32 y`, then `bps` value bytes.  Records must be sorted
//! strictly ascending by `(y, x)` — the canonical order.  Coordinates are
//! absolute pixels of the canonical surface; partial observation domains
//! apply only the intersecting records.
//!
//! Residual semantics are *never* implied: the algebra tag selects the rule.

pub mod algebra {
    /// Sparse overwrite.
    pub const SPARSE_OVERWRITE: u8 = 1;
    /// XOR.
    pub const XOR: u8 = 2;
}

use crate::color::ColorFormat;
use crate::fixed::IRect;
use crate::limits::Reject;

/// One decoded residual record (absolute pixel coordinates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub x: u32,
    pub y: u32,
}

/// Bytes per sample of the corrected surface.
pub fn bps(format: ColorFormat) -> usize {
    format.bytes_per_sample()
}

/// Parse and strictly validate a residual payload against its binding.
/// Deterministic; fails closed on the first malformed record.
pub fn validate_payload(
    algebra: u8,
    format: ColorFormat,
    region: &IRect,
    payload: &[u8],
) -> Result<(), Reject> {
    if algebra != algebra::SPARSE_OVERWRITE && algebra != algebra::XOR {
        return Err(Reject::UnknownTag);
    }
    let b = bps(format);
    let mut r = crate::ir::wire::Reader::new(payload);
    let count = r.u64().map_err(|_| Reject::Truncated)?;
    if count > crate::limits::MAX_RESIDUAL_RECORDS {
        return Err(Reject::CountExceedsLimit);
    }
    let mut prev_y: Option<u32> = None;
    let mut prev_x: u32 = 0;
    for _ in 0..count {
        let x = r.u32().map_err(|_| Reject::Truncated)?;
        let y = r.u32().map_err(|_| Reject::Truncated)?;
        let _val = r.bytes(b).map_err(|_| Reject::Truncated)?;
        // canonical sorted order (y asc, then x asc, strictly)
        if let Some(py) = prev_y
            && (y < py || (y == py && x <= prev_x))
        {
            return Err(Reject::NonCanonicalOrder);
        }
        prev_y = Some(y);
        prev_x = x;
        // in-region check
        if !(x >= region.x0 as u32
            && x < region.x1 as u32
            && y >= region.y0 as u32
            && y < region.y1 as u32)
        {
            return Err(Reject::ResidualRegionInvalid);
        }
    }
    if !r.done() {
        return Err(Reject::PayloadMismatch);
    }
    Ok(())
}

/// Append one record to a payload being built (must be pushed in canonical
/// (y,x) order).
pub fn push_record(out: &mut Vec<u8>, x: u32, y: u32, value: &[u8]) {
    out.extend_from_slice(&x.to_le_bytes());
    out.extend_from_slice(&y.to_le_bytes());
    out.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_validation_ok() {
        let region = IRect::new(0, 0, 4, 4);
        let mut p = Vec::new();
        p.extend_from_slice(&2u64.to_le_bytes());
        push_record(&mut p, 0, 0, &[1, 2, 3, 4]);
        push_record(&mut p, 1, 0, &[5, 6, 7, 8]);
        assert!(
            validate_payload(algebra::SPARSE_OVERWRITE, ColorFormat::Rgba8, &region, &p).is_ok()
        );
    }

    #[test]
    fn payload_validation_rejects_unsorted() {
        let region = IRect::new(0, 0, 4, 4);
        let mut p = Vec::new();
        p.extend_from_slice(&2u64.to_le_bytes());
        push_record(&mut p, 1, 0, &[1, 2, 3, 4]);
        push_record(&mut p, 0, 0, &[5, 6, 7, 8]);
        assert_eq!(
            validate_payload(algebra::SPARSE_OVERWRITE, ColorFormat::Rgba8, &region, &p),
            Err(Reject::NonCanonicalOrder)
        );
    }

    #[test]
    fn payload_validation_rejects_out_of_region() {
        let region = IRect::new(0, 0, 2, 2);
        let mut p = Vec::new();
        p.extend_from_slice(&1u64.to_le_bytes());
        push_record(&mut p, 5, 5, &[1, 2, 3, 4]);
        assert_eq!(
            validate_payload(algebra::SPARSE_OVERWRITE, ColorFormat::Rgba8, &region, &p),
            Err(Reject::ResidualRegionInvalid)
        );
    }

    #[test]
    fn payload_validation_rejects_bad_algebra() {
        let region = IRect::new(0, 0, 1, 1);
        assert_eq!(
            validate_payload(99, ColorFormat::Rgba8, &region, &[]),
            Err(Reject::UnknownTag)
        );
    }
}
