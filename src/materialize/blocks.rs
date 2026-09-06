//! Materialization blocks: the optimization unit is neither "pixel" nor
//! "whole frame".  Block shape is an explicit backend decision swept by the
//! benchmark courts (paper §Block materialization).

use crate::fixed::IRect;

/// A materialization block shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockShape {
    W8H8,
    W16H8,
    W16H16,
    W32H8,
    W32H16,
    /// Whole scanline segments of the given width (tail-aware).
    ScanlineSegment(u32),
    /// A display band of `h` rows over a given span (tail-aware).
    Band {
        h: u32,
        span: u32,
    },
}

/// The canonical block-shape sweep used by courts and autotuning.
pub const SWEEP: &[BlockShape] = &[
    BlockShape::W8H8,
    BlockShape::W16H8,
    BlockShape::W16H16,
    BlockShape::W32H8,
    BlockShape::W32H16,
    BlockShape::ScanlineSegment(64),
    BlockShape::Band { h: 32, span: 512 },
];

impl BlockShape {
    pub fn name(&self) -> &'static str {
        match self {
            BlockShape::W8H8 => "8x8",
            BlockShape::W16H8 => "16x8",
            BlockShape::W16H16 => "16x16",
            BlockShape::W32H8 => "32x8",
            BlockShape::W32H16 => "32x16",
            BlockShape::ScanlineSegment(_) => "scanline64",
            BlockShape::Band { .. } => "band32x512",
        }
    }

    /// Nominal tile width/height for uniform tiling.
    pub fn tile(&self) -> (u32, u32) {
        match self {
            BlockShape::W8H8 => (8, 8),
            BlockShape::W16H8 => (16, 8),
            BlockShape::W16H16 => (16, 16),
            BlockShape::W32H8 => (32, 8),
            BlockShape::W32H16 => (32, 16),
            BlockShape::ScanlineSegment(w) => (*w, 1),
            BlockShape::Band { h, span } => (*span, *h),
        }
    }
}

/// Decompose a domain rectangle into a deterministic, ordered list of blocks
/// (row-major, left-to-right, top-to-bottom), clipped to the domain.  The
/// decomposition is canonical: it does not depend on worker scheduling.
pub fn tile_domain(domain: IRect, shape: BlockShape) -> Vec<IRect> {
    let (tw, th) = shape.tile();
    let mut blocks = Vec::new();
    if domain.is_empty() {
        return blocks;
    }
    let (tw, th) = (tw.max(1) as i32, th.max(1) as i32);
    let mut y = domain.y0;
    while y < domain.y1 {
        let mut x = domain.x0;
        while x < domain.x1 {
            let b = IRect::new(x, y, (x + tw).min(domain.x1), (y + th).min(domain.y1));
            blocks.push(b);
            x += tw;
        }
        y += th;
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_8x8_covers_domain_exactly() {
        let domain = IRect::new(2, 3, 20, 11); // 18 x 8
        let blocks = tile_domain(domain, BlockShape::W8H8);
        let covered: u64 = blocks.iter().map(|b| b.area()).sum();
        assert_eq!(covered, domain.area());
        assert_eq!(blocks.len(), 3); // 3 columns x 1 row
        assert_eq!(blocks[0], IRect::new(2, 3, 10, 11));
        assert_eq!(blocks[2], IRect::new(18, 3, 20, 11));
    }

    #[test]
    fn tiling_handles_tails() {
        let domain = IRect::new(0, 0, 10, 10);
        let blocks = tile_domain(domain, BlockShape::W16H8);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], IRect::new(0, 0, 10, 8));
        assert_eq!(blocks[1], IRect::new(0, 8, 10, 10));
    }

    #[test]
    fn empty_domain_no_blocks() {
        assert!(tile_domain(IRect::empty(), BlockShape::W16H16).is_empty());
    }
}
