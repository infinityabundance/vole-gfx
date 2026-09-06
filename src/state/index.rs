//! Deterministic spatial dependency indexing (paper §Dependency closure).
//!
//! A `PlacementIndex` bins the request-time placed instances (by their
//! conservative pixel bounds) into a uniform grid.  An observation block
//! queries the grid and obtains a *candidate closure* — a superset of the
//! instances that can possibly cover any sample in the block.  The exact
//! per-sample test (clip + inverse-affine sampling) decides; the index only
//! removes provably irrelevant instances.
//!
//! The index is deterministic: grid queries return instance indices in
//! ascending instance order, independent of scheduling.

use super::{Scene, StructOp};
use crate::fixed::IRect;
use std::collections::HashMap;

/// Default grid cell size in pixels (backend decision; courts sweep it).
pub const DEFAULT_CELL_PX: i32 = 64;

/// Uniform-grid placement index over one resolved scene.
pub struct PlacementIndex {
    cell_px: i32,
    /// cell (cx, cy) -> ascending instance indices whose bounds touch it.
    cells: HashMap<(i32, i32), Vec<u32>>,
    /// instance bounds (mirrors scene.instances order) for false-positive
    /// measurement and candidate rect checks.
    bounds: Vec<Option<IRect>>,
    n_instances: usize,
}

impl PlacementIndex {
    pub fn build(scene: &Scene<'_>, cell_px: i32) -> PlacementIndex {
        let cell_px = cell_px.max(1);
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let mut bounds = Vec::with_capacity(scene.instances.len());
        for (idx, inst) in scene.instances.iter().enumerate() {
            let b = inst.bounds_px;
            bounds.push(b);
            let Some(b) = b else {
                // Unknown extent (non-raster object): keep in a sentinel cell
                // that every query includes.
                cells
                    .entry((i32::MIN, i32::MIN))
                    .or_default()
                    .push(idx as u32);
                continue;
            };
            let (cx0, cy0) = (b.x0 / cell_px, b.y0 / cell_px);
            let (cx1, cy1) = ((b.x1 - 1) / cell_px, (b.y1 - 1) / cell_px);
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    cells.entry((cx, cy)).or_default().push(idx as u32);
                }
            }
        }
        PlacementIndex {
            cell_px,
            cells,
            bounds,
            n_instances: scene.instances.len(),
        }
    }

    /// Candidate closure for a pixel rectangle: appends ascending instance
    /// indices that may cover samples inside `rect` to `out` (cleared first).
    pub fn candidates(&self, rect: IRect, out: &mut Vec<u32>) {
        out.clear();
        let (cx0, cy0) = (rect.x0 / self.cell_px, rect.y0 / self.cell_px);
        let (cx1, cy1) = ((rect.x1 - 1) / self.cell_px, (rect.y1 - 1) / self.cell_px);
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                if let Some(list) = self.cells.get(&(cx, cy)) {
                    out.extend_from_slice(list);
                }
            }
        }
        // unknown-extent instances (sentinel cell)
        if let Some(list) = self.cells.get(&(i32::MIN, i32::MIN)) {
            out.extend_from_slice(list);
        }
        // deduplicate ascending (instances may touch several cells)
        out.sort_unstable();
        out.dedup();
    }

    pub fn cell_px(&self) -> i32 {
        self.cell_px
    }

    pub fn instance_count(&self) -> usize {
        self.n_instances
    }

    /// Conservative pixel bounds of instance `i` (mirrors scene order).
    pub fn bounds(&self, i: usize) -> Option<IRect> {
        self.bounds[i]
    }

    /// How many of `cands` actually intersect `rect` (true positives w.r.t.
    /// the conservative bounds).
    pub fn intersecting(&self, cands: &[u32], rect: IRect) -> usize {
        cands
            .iter()
            .filter(|&&i| match self.bounds[i as usize] {
                Some(b) => !b.intersect(rect).is_empty(),
                None => true,
            })
            .count()
    }
}

/// Structural-op coverage index: per-op conservative pixel effect rect.
/// Fill/copy/move only affect samples inside their destination rect; a block
/// can skip ops whose effect rect does not intersect it.
pub fn op_effect_px(ops: &[StructOp<'_>]) -> Vec<Option<IRect>> {
    ops.iter()
        .map(|op| match op {
            StructOp::FillRect { rect, .. } => Some(rect.px_bounds()),
            StructOp::CopyRegion { src, dx, dy } => {
                let dst = shifted(src, *dx, *dy);
                Some(dst.px_bounds())
            }
            StructOp::MoveRegion { src, dx, dy } => {
                // A move clears its source and writes its destination: the
                // effect rect is the union of both.
                let dst = shifted(src, *dx, *dy);
                let src_b = src.px_bounds();
                Some(src_b.union(dst.px_bounds()))
            }
            StructOp::BindResidual { region, .. } => Some(*region),
        })
        .collect()
}

fn shifted(r: &crate::fixed::RectF, dx: i32, dy: i32) -> crate::fixed::RectF {
    crate::fixed::RectF {
        x0: r.x0.wrapping_add(dx),
        y0: r.y0.wrapping_add(dy),
        x1: r.x1.wrapping_add(dx),
        y1: r.y1.wrapping_add(dy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{ColorFormat, Rgba};
    use crate::fixed::Affine;
    use crate::ir::{Document, Instance, Object};

    fn scene_three() -> Scene<'static> {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 10,
            h: 10,
            data: vec![0u8; 400],
        });
        for (order, tx) in [(1u32, 0i32), (2, 100), (3, 400)] {
            d.instances.push(Instance {
                object: 0,
                order,
                layer: 0,
                transform: Affine::from_px_translation(tx, 0),
                palette: None,
                clip: None,
                trajectory: 0,
                visible: true,
            });
        }
        let boxed: &'static mut Document = Box::leak(Box::new(d));
        Scene::resolve(boxed, 0).unwrap()
    }

    #[test]
    fn candidates_cover_relevant_instances_only() {
        let s = scene_three();
        let idx = PlacementIndex::build(&s, 64);
        let mut out = Vec::new();
        idx.candidates(IRect::new(0, 0, 64, 64), &mut out);
        assert_eq!(out, vec![0], "only the instance at origin");
        assert_eq!(idx.intersecting(&out, IRect::new(0, 0, 64, 64)), 1);

        idx.candidates(IRect::new(95, 0, 160, 64), &mut out);
        assert_eq!(out, vec![1], "only the instance at x=100");
    }

    #[test]
    fn unknown_extent_instances_always_candidates() {
        let s = scene_three();
        let idx = PlacementIndex::build(&s, 64);
        let mut out = Vec::new();
        // a far-away block still includes nothing unknown (all raster)
        idx.candidates(IRect::new(5000, 5000, 5100, 5100), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn op_effect_rects() {
        let ops = vec![
            StructOp::FillRect {
                rect: crate::fixed::RectF::from_px(0, 0, 8, 8),
                color: Rgba::WHITE,
            },
            StructOp::CopyRegion {
                src: crate::fixed::RectF::from_px(0, 0, 8, 8),
                dx: crate::fixed::fp(20),
                dy: 0,
            },
        ];
        let fx = op_effect_px(&ops);
        assert_eq!(fx[0], Some(IRect::new(0, 0, 8, 8)));
        assert_eq!(fx[1], Some(IRect::new(20, 0, 28, 8)));
    }
}
