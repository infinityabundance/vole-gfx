//! Observation model.
//!
//! A materializer answers an `ObservationRequest`: a spatial domain of a
//! canonical surface sampled at one explicit time coordinate.  Frames, frame
//! rates, and resolutions are **not** normative state properties; a request
//! carries the time, the domain, and the output profile explicitly
//! (paper §Materialization, §Profiles).

pub mod domain;
pub mod request;
pub mod sampling;

pub use domain::Domain;
pub use request::{ObservationRequest, View};
pub use sampling::{SamplePosition, SamplingProfile};

/// Output buffer for a materialized observation.
///
/// `w`, `h` are in samples; `data` is row-major in the requested format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub w: u32,
    pub h: u32,
    pub format: crate::color::ColorFormat,
    pub data: Vec<u8>,
}

impl Output {
    pub fn new(
        w: u32,
        h: u32,
        format: crate::color::ColorFormat,
    ) -> Result<Output, crate::limits::Reject> {
        if w == 0 || h == 0 {
            return Err(crate::limits::Reject::Degenerate);
        }
        let samples = w as u64 * h as u64;
        if samples > crate::limits::MAX_OUTPUT_SAMPLES {
            return Err(crate::limits::Reject::BytesExceedsLimit);
        }
        let bytes = samples * format.bytes_per_sample() as u64;
        if bytes > crate::limits::MAX_OUTPUT_BYTES {
            return Err(crate::limits::Reject::BytesExceedsLimit);
        }
        Ok(Output {
            w,
            h,
            format,
            data: vec![0; bytes as usize],
        })
    }

    /// Samples actually requested by the domain (area of the domain).
    pub fn sample_count(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Canonical SHA-256 of the output buffer (row-major codes).
    pub fn canonical_hash(&self) -> crate::hash::ContentId {
        crate::hash::sha256(&self.data)
    }
}
