//! `ObservationRequest`: explicit time + view + spatial domain + sampling +
//! output profile.  There is no hidden "current frame"; every materialization
//! is a function of this request.

use super::Domain;
use super::sampling::SamplingProfile;
use crate::color::ColorFormat;
use crate::limits::Reject;

/// View state of a request.
///
/// U1 exact 2D profile: the view is the identity (surface pixels == scene
/// pixels).  A declared view transform belongs to a later profile; the type
/// exists so the API does not silently re-learn `render_frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct View {
    /// Scene-space offset of the canonical surface origin (Q16.16).
    pub pan_x: i32,
    pub pan_y: i32,
}

impl View {
    pub const fn identity() -> Self {
        View { pan_x: 0, pan_y: 0 }
    }
    pub fn is_identity(&self) -> bool {
        self.pan_x == 0 && self.pan_y == 0
    }
}

/// A complete observation request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationRequest {
    /// Observation time in integer nanoseconds since the document origin.
    /// Arbitrary valid values are allowed; cadence is not encoded anywhere.
    pub time_ns: u64,
    /// View (identity in U1 exact 2D).
    pub view: View,
    /// Spatial domain of the canonical surface.
    pub domain: Domain,
    /// Sampling profile (exact U1 = pixel centers, no supersampling).
    pub sampling: SamplingProfile,
    /// Output sample format.
    pub format: ColorFormat,
}

impl ObservationRequest {
    /// A full-surface request at time `t`.
    pub fn full_surface(t: u64, w: u32, h: u32, format: ColorFormat) -> ObservationRequest {
        ObservationRequest {
            time_ns: t,
            view: View::identity(),
            domain: Domain::FullSurface { w, h },
            sampling: SamplingProfile::exact_u1(),
            format,
        }
    }

    /// A rectangular-region request at time `t`.
    pub fn region(
        t: u64,
        x0: i32,
        y0: i32,
        w: u32,
        h: u32,
        format: ColorFormat,
    ) -> ObservationRequest {
        ObservationRequest {
            time_ns: t,
            view: View::identity(),
            domain: Domain::Rectangle { x0, y0, w, h },
            sampling: SamplingProfile::exact_u1(),
            format,
        }
    }

    pub fn validate(&self) -> Result<(), Reject> {
        self.domain.validate()?;
        let supported = matches!(self.format, ColorFormat::Gray8 | ColorFormat::Rgba8);
        if !supported {
            return Err(Reject::UnsupportedColor);
        }
        if self.sampling != SamplingProfile::exact_u1() {
            return Err(Reject::UnsupportedProfile);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_validate() {
        let r = ObservationRequest::full_surface(0, 64, 64, ColorFormat::Rgba8);
        assert!(r.validate().is_ok());
        let r2 = ObservationRequest::region(1_000, 20_000, 0, 8, 8, ColorFormat::Gray8);
        assert!(r2.validate().is_err());
    }

    #[test]
    fn time_is_explicit() {
        let a = ObservationRequest::full_surface(30_000_000, 8, 8, ColorFormat::Rgba8);
        let b = ObservationRequest::full_surface(16_666_667, 8, 8, ColorFormat::Rgba8);
        assert_ne!(a.time_ns, b.time_ns);
    }
}
