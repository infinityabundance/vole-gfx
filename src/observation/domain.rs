//! Spatial observation domains of the canonical surface.

use crate::fixed::IRect;
use crate::limits::Reject;

/// Spatial domain of an observation request, in canonical-surface pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Domain {
    /// One sample at pixel `(x, y)`.
    Sample { x: i32, y: i32 },
    /// One horizontal span of `n` samples starting at `(x, y)`.
    SampleSpan { x: i32, y: i32, n: u32 },
    /// Arbitrary rectangle `[x0, x0+w) × [y0, y0+h)`.
    Rectangle { x0: i32, y0: i32, w: u32, h: u32 },
    /// A tile at integer tile coordinates of the given tile size.
    Tile {
        tile_x: i32,
        tile_y: i32,
        tile_w: u32,
        tile_h: u32,
    },
    /// A scanline (one row of samples).
    Scanline { y: i32, x0: i32, x1: i32 },
    /// A contiguous band of scanlines.
    DisplayBand { y0: i32, h: u32, x0: i32, x1: i32 },
    /// Irregular set of discrete samples.
    IrregularSamples(Vec<(i32, i32)>),
    /// A viewport rectangle (alias for `Rectangle`, explicit name).
    Viewport { x0: i32, y0: i32, w: u32, h: u32 },
    /// The complete canonical surface.
    FullSurface { w: u32, h: u32 },
}

impl Domain {
    /// Bounding rectangle of the domain (the minimal region touched).
    pub fn bounds(&self) -> IRect {
        match *self {
            Domain::Sample { x, y } => IRect::new(x, y, x + 1, y + 1),
            Domain::SampleSpan { x, y, n } => IRect::new(x, y, x + n as i32, y + 1),
            Domain::Rectangle { x0, y0, w, h } => IRect::new(x0, y0, x0 + w as i32, y0 + h as i32),
            Domain::Tile {
                tile_x,
                tile_y,
                tile_w,
                tile_h,
            } => IRect::new(
                tile_x * tile_w as i32,
                tile_y * tile_h as i32,
                (tile_x + 1) * tile_w as i32,
                (tile_y + 1) * tile_h as i32,
            ),
            Domain::Scanline { y, x0, x1 } => IRect::new(x0, y, x1, y + 1),
            Domain::DisplayBand { y0, h, x0, x1 } => IRect::new(x0, y0, x1, y0 + h as i32),
            Domain::IrregularSamples(ref pts) => {
                let mut r = IRect::empty();
                for &(x, y) in pts {
                    r = r.union(IRect::new(x, y, x + 1, y + 1));
                }
                r
            }
            Domain::Viewport { x0, y0, w, h } => IRect::new(x0, y0, x0 + w as i32, y0 + h as i32),
            Domain::FullSurface { w, h } => IRect::new(0, 0, w as i32, h as i32),
        }
    }

    /// Number of samples the domain requests.
    pub fn requested_samples(&self) -> u64 {
        match *self {
            Domain::Sample { .. } => 1,
            Domain::SampleSpan { n, .. } => n as u64,
            Domain::Rectangle { w, h, .. } => w as u64 * h as u64,
            Domain::Tile { .. } => self.bounds().area(),
            Domain::Scanline { x0, x1, .. } => (x1 - x0).max(0) as u64,
            Domain::DisplayBand { h, x0, x1, .. } => (x1 - x0).max(0) as u64 * h as u64,
            Domain::IrregularSamples(ref pts) => pts.len() as u64,
            Domain::Viewport { w, h, .. } => w as u64 * h as u64,
            Domain::FullSurface { w, h } => w as u64 * h as u64,
        }
    }

    /// Validate against normative caps.
    pub fn validate(&self) -> Result<(), Reject> {
        let b = self.bounds();
        let (w, h) = (b.width().max(0) as u64, b.height().max(0) as u64);
        if w > crate::limits::MAX_DIMENSION as u64 || h > crate::limits::MAX_DIMENSION as u64 {
            return Err(Reject::DimensionTooLarge);
        }
        if self.requested_samples() > crate::limits::MAX_OUTPUT_SAMPLES {
            return Err(Reject::BytesExceedsLimit);
        }
        // All sample positions must live inside the U1 scene coordinate range.
        let max = crate::limits::MAX_SCENE_PX;
        if b.x0 < -max || b.y0 < -max || b.x1 > max || b.y1 > max {
            return Err(Reject::CoordinateOutOfRange);
        }
        if let Domain::IrregularSamples(ref pts) = *self {
            for &(x, y) in pts {
                if x < -max || y < -max || x > max || y > max {
                    return Err(Reject::CoordinateOutOfRange);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds() {
        let d = Domain::Rectangle {
            x0: 10,
            y0: 20,
            w: 5,
            h: 6,
        };
        assert_eq!(d.bounds(), IRect::new(10, 20, 15, 26));
        assert_eq!(d.requested_samples(), 30);
        let t = Domain::Tile {
            tile_x: 2,
            tile_y: 3,
            tile_w: 16,
            tile_h: 16,
        };
        assert_eq!(t.bounds(), IRect::new(32, 48, 48, 64));
        assert_eq!(t.requested_samples(), 256);
    }

    #[test]
    fn validation() {
        assert!(Domain::Sample { x: 0, y: 0 }.validate().is_ok());
        assert!(Domain::Sample { x: -100, y: 0 }.validate().is_ok());
        assert!(Domain::Sample { x: 20000, y: 0 }.validate().is_err());
        assert!(Domain::FullSurface { w: 1920, h: 1080 }.validate().is_ok());
        assert!(Domain::FullSurface { w: 1 << 20, h: 1 }.validate().is_err());
    }
}
