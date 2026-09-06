//! Sampling cadence and sample-position semantics.

/// Sample positions available in the U1 exact profile.
///
/// Exact U1 defines one sample per output pixel at the pixel center; the
/// variants below exist so that the *future* quality-bounded profile and
/// supersampling research have a declared place without changing U1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplePosition {
    /// Sample at pixel centers (U1 exact; normative).
    PixelCenters,
}

/// Sampling profile of an observation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingProfile {
    pub position: SamplePosition,
    /// Samples per pixel per axis (1 in exact U1; reserved).
    pub supersample: u32,
}

impl Default for SamplingProfile {
    fn default() -> Self {
        SamplingProfile {
            position: SamplePosition::PixelCenters,
            supersample: 1,
        }
    }
}

impl SamplingProfile {
    /// The exact U1 sampling profile.
    pub const fn exact_u1() -> Self {
        SamplingProfile {
            position: SamplePosition::PixelCenters,
            supersample: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_exact() {
        assert_eq!(SamplingProfile::default(), SamplingProfile::exact_u1());
    }
}
