//! Block materialization with dependency indexing (Phase C).
//!
//! Same exact U1 semantics as the scalar oracle, structured as:
//!
//!   q → spatial lookup → minimal candidate closure D(block) →
//!     primitive materialization → composition → residual closure → output
//!
//! * Per block, instances are culled with the uniform-grid `PlacementIndex`
//!   and structural ops with their conservative effect rects.  The remaining
//!   candidate sets are proven supersets of what can affect any sample in the
//!   block, so output is byte-for-byte identical to the oracle.
//! * Copy/move prefix evaluation is memoized per `(pixel, op-bound)` — a pure
//!   function of the scene — so repeated prefix queries across samples are
//!   computed once.  Memoization never changes results, only work.
//! * Bounded execution (depth + frame budgets) matches the oracle's guards.

use super::blocks::{BlockShape, tile_domain};
use super::scalar::{Counters, Materialized, sample_placed};
use crate::color::{Rgba, over};
use crate::fixed::Vec2;
use crate::limits::Reject;
use crate::observation::{ObservationRequest, Output};
use crate::state::index::{PlacementIndex, op_effect_px};
use crate::state::{Scene, StructOp};
use std::collections::HashMap;

const COPY_CHAIN_LIMIT: u32 = 256;
const FOLD_FRAME_BUDGET: u32 = 1 << 16;

/// Per-materialization metrics of the block path (evidence inputs).
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockMetrics {
    pub blocks: u64,
    pub candidates_selected: u64,
    pub candidates_true_positive: u64,
    pub instance_draws: u64,
    pub ops_considered: u64,
}

/// Block materializer.  Holds scratch state; reuse across requests of the
/// same scene for zero-allocation steady state (clears memo maps).
pub struct BlockMaterializer<'a> {
    scene: &'a Scene<'a>,
    index: PlacementIndex,
    op_effects: Vec<Option<crate::fixed::IRect>>,
    scratch_candidates: Vec<u32>,
    memo: HashMap<(i32, i32, u32), Rgba>,
    base_memo: HashMap<(i32, i32), Rgba>,
    metrics: BlockMetrics,
}

impl<'a> BlockMaterializer<'a> {
    pub fn new(scene: &'a Scene<'a>, cell_px: i32) -> BlockMaterializer<'a> {
        let index = PlacementIndex::build(scene, cell_px);
        let op_effects = op_effect_px(&scene.ops);
        BlockMaterializer {
            scene,
            index,
            op_effects,
            scratch_candidates: Vec::new(),
            memo: HashMap::new(),
            base_memo: HashMap::new(),
            metrics: BlockMetrics::default(),
        }
    }

    pub fn metrics(&self) -> BlockMetrics {
        self.metrics
    }

    pub fn scene(&self) -> &'a Scene<'a> {
        self.scene
    }

    /// Materialize a request over deterministic blocks of `shape`.
    pub fn materialize(
        &mut self,
        req: &ObservationRequest,
        shape: BlockShape,
    ) -> Result<Materialized, Reject> {
        req.validate()?;
        self.memo.clear();
        self.base_memo.clear();
        self.metrics = BlockMetrics::default();

        if let crate::observation::Domain::IrregularSamples(pts) = &req.domain {
            return self.materialize_irregular(req, pts);
        }

        let b = req.domain.bounds();
        let (w, h) = (b.width() as u32, b.height() as u32);
        if w == 0 || h == 0 {
            return Err(Reject::Degenerate);
        }
        let mut out = Output::new(w, h, req.format)?;
        let mut counters = Counters::default();

        for block in tile_domain(b, shape) {
            self.metrics.blocks += 1;
            // candidate instances for the whole block (copy out of scratch so
            // the borrow does not block later self use)
            self.query(block);
            let block_cands = self.scratch_candidates.clone();
            self.metrics.candidates_selected += block_cands.len() as u64;
            self.metrics.candidates_true_positive +=
                self.index.intersecting(&block_cands, block) as u64;

            // ops whose effect rect touches this block (fills/copies/moves)
            let mut block_ops: Vec<(usize, &StructOp<'_>)> = Vec::new();
            for (k, eff) in self.op_effects.iter().enumerate() {
                let touches = match eff {
                    Some(r) => !r.intersect(block).is_empty(),
                    None => true,
                };
                if touches && !matches!(self.scene.ops[k], StructOp::BindResidual { .. }) {
                    block_ops.push((k, &self.scene.ops[k]));
                }
            }
            self.metrics.ops_considered += block_ops.len() as u64;

            let (bx0, by0) = (block.x0, block.y0);
            for j in 0..block.height() {
                let y = by0 + j;
                let row_base = ((by0 - b.y0 + j) as usize) * w as usize;
                for i in 0..block.width() {
                    let x = bx0 + i;
                    let rgba =
                        self.compose_sample(x, y, &block_cands, &block_ops, &mut counters)?;
                    let code = super::scalar::encode_code_pub(rgba, req.format);
                    let off = (row_base + (x - b.x0) as usize) * req.format.bytes_per_sample();
                    super::scalar::write_code_pub(&mut out.data, off, code);
                    counters.samples += 1;
                }
            }
        }

        // Residual closure: exactly once per request, in binding order, on
        // the canonical code values (identical rule and pass to the scalar
        // oracle — a per-block pass would double-apply XOR records).
        for op in &self.scene.ops {
            if let StructOp::BindResidual {
                algebra,
                region,
                format,
                payload,
            } = op
            {
                if *format != req.format {
                    counters.residual_format_skips += 1;
                    continue;
                }
                super::scalar::apply_residual_pub(
                    *algebra,
                    region,
                    req.format,
                    payload,
                    &mut out.data,
                    w,
                    h,
                    b.x0,
                    b.y0,
                    &mut counters,
                )?;
            }
        }
        Ok(Materialized {
            output: out,
            counters,
        })
    }

    fn materialize_irregular(
        &mut self,
        req: &ObservationRequest,
        pts: &[(i32, i32)],
    ) -> Result<Materialized, Reject> {
        let n = pts.len();
        if n == 0 {
            return Err(Reject::Degenerate);
        }
        let mut out = Output::new(n as u32, 1, req.format)?;
        let mut counters = Counters::default();
        for (k, &(x, y)) in pts.iter().enumerate() {
            let one = crate::fixed::IRect::new(x, y, x + 1, y + 1);
            self.query(one);
            let cands = self.scratch_candidates.clone();
            let block_ops: Vec<(usize, &StructOp<'_>)> = self
                .scene
                .ops
                .iter()
                .enumerate()
                .filter(|(k, op)| {
                    !matches!(op, StructOp::BindResidual { .. })
                        && matches!(&self.op_effects[*k], Some(r) if !r.intersect(one).is_empty())
                })
                .collect();
            let rgba = self.compose_sample(x, y, &cands, &block_ops, &mut counters)?;
            let code = super::scalar::encode_code_pub(rgba, req.format);
            let off = k * req.format.bytes_per_sample();
            super::scalar::write_code_pub(&mut out.data, off, code);
            counters.samples += 1;
            for op in &self.scene.ops {
                if let StructOp::BindResidual {
                    algebra,
                    region,
                    format,
                    payload,
                } = op
                {
                    if *format != req.format {
                        counters.residual_format_skips += 1;
                        continue;
                    }
                    super::scalar::apply_residual_at_pub(
                        *algebra,
                        region,
                        req.format,
                        payload,
                        x,
                        y,
                        &mut out.data[off..off + req.format.bytes_per_sample()],
                        &mut counters,
                    )?;
                }
            }
        }
        Ok(Materialized {
            output: out,
            counters,
        })
    }

    /// Query the placement index into the scratch buffer.
    fn query(&mut self, rect: crate::fixed::IRect) -> &[u32] {
        self.index.candidates(rect, &mut self.scratch_candidates);
        &self.scratch_candidates
    }

    /// Compose one sample from block-culled candidates and ops.
    fn compose_sample(
        &mut self,
        x: i32,
        y: i32,
        block_cands: &[u32],
        block_ops: &[(usize, &StructOp<'_>)],
        c: &mut Counters,
    ) -> Result<Rgba, Reject> {
        let cx = Vec2::sample_center(x, y);
        let base = self.draw_base_cands(cx, block_cands, c)?;
        let mut cur = base;
        let mut budget = FOLD_FRAME_BUDGET;
        for &(k, op) in block_ops {
            match op {
                StructOp::FillRect { rect, color } => {
                    if covers(cx, rect) {
                        cur = over(*color, cur);
                        c.ops_applied += 1;
                    }
                }
                StructOp::CopyRegion { src, dx, dy } => {
                    if covers_shifted(cx, src, *dx, *dy) {
                        let from = cx - Vec2::new(*dx, *dy);
                        cur = self.fold_prefix(from, k, 1, &mut budget, c)?;
                        c.ops_applied += 1;
                    }
                }
                StructOp::MoveRegion { src, dx, dy } => {
                    if covers_shifted(cx, src, *dx, *dy) {
                        let from = cx - Vec2::new(*dx, *dy);
                        cur = self.fold_prefix(from, k, 1, &mut budget, c)?;
                        c.ops_applied += 1;
                    } else if covers(cx, src) {
                        cur = Rgba::TRANSPARENT;
                        c.ops_applied += 1;
                    }
                }
                StructOp::BindResidual { .. } => {}
            }
        }
        Ok(cur)
    }

    /// Instance-composed base at a sample using the *block* candidate list
    /// (per-pixel exact tests happen inside).
    fn draw_base_cands(
        &self,
        cx: Vec2,
        block_cands: &[u32],
        c: &mut Counters,
    ) -> Result<Rgba, Reject> {
        let mut cur = Rgba::TRANSPARENT;
        for &inst_i in block_cands {
            c.instance_tests += 1;
            let inst = &self.scene.instances[inst_i as usize];
            if let Some(b) = inst.bounds_px {
                let px = (cx.x >> 16, cx.y >> 16);
                if !b.contains_px(px.0, px.1) {
                    continue;
                }
            }
            let Some(local) = sample_placed(self.scene, inst, cx) else {
                continue;
            };
            c.instance_draws += 1;
            cur = over(local, cur);
        }
        Ok(cur)
    }

    /// Memoized base at an arbitrary pixel (used by copy/move recursion).
    fn draw_base_memo(&mut self, cx: Vec2, c: &mut Counters) -> Result<Rgba, Reject> {
        let (px, py) = (cx.x >> 16, cx.y >> 16);
        if let Some(v) = self.base_memo.get(&(px, py)) {
            return Ok(*v);
        }
        let one = crate::fixed::IRect::new(px, py, px + 1, py + 1);
        self.index.candidates(one, &mut self.scratch_candidates);
        let scene = self.scene;
        let mut cur = Rgba::TRANSPARENT;
        let mut draws = 0u64;
        for &inst_i in &self.scratch_candidates {
            c.instance_tests += 1;
            let inst = &scene.instances[inst_i as usize];
            if let Some(b) = inst.bounds_px
                && !b.contains_px(px, py)
            {
                continue;
            }
            draws += 1;
            let Some(local) = sample_placed(scene, inst, cx) else {
                continue;
            };
            c.instance_draws += 1;
            cur = over(local, cur);
        }
        self.metrics.candidates_selected += self.scratch_candidates.len() as u64;
        self.metrics.instance_draws += draws;
        self.base_memo.insert((px, py), cur);
        Ok(cur)
    }

    /// Exact prefix-surface value at `cx` after ops `0..upto` (memoized).
    fn fold_prefix(
        &mut self,
        cx: Vec2,
        upto: usize,
        depth: u32,
        budget: &mut u32,
        c: &mut Counters,
    ) -> Result<Rgba, Reject> {
        if depth > COPY_CHAIN_LIMIT {
            return Err(Reject::DependencyTooDeep);
        }
        let (px, py) = (cx.x >> 16, cx.y >> 16);
        let key = (px, py, upto as u32);
        if self.memo.contains_key(&key) {
            return Ok(self.memo[&key]);
        }
        *budget = budget
            .checked_sub(1)
            .ok_or(Reject::ExecutionBudgetExceeded)?;
        let mut cur = self.draw_base_memo(cx, c)?;
        for k in 0..upto {
            let op = &self.scene.ops[k];
            match op {
                StructOp::FillRect { rect, color } => {
                    if covers(cx, rect) {
                        cur = over(*color, cur);
                        c.ops_applied += 1;
                    }
                }
                StructOp::CopyRegion { src, dx, dy } => {
                    if covers_shifted(cx, src, *dx, *dy) {
                        let from = cx - Vec2::new(*dx, *dy);
                        cur = self.fold_prefix(from, k, depth + 1, budget, c)?;
                        c.ops_applied += 1;
                    }
                }
                StructOp::MoveRegion { src, dx, dy } => {
                    if covers_shifted(cx, src, *dx, *dy) {
                        let from = cx - Vec2::new(*dx, *dy);
                        cur = self.fold_prefix(from, k, depth + 1, budget, c)?;
                        c.ops_applied += 1;
                    } else if covers(cx, src) {
                        cur = Rgba::TRANSPARENT;
                        c.ops_applied += 1;
                    }
                }
                StructOp::BindResidual { .. } => {}
            }
        }
        self.memo.insert(key, cur);
        Ok(cur)
    }
}

#[inline]
fn covers(cx: Vec2, r: &crate::fixed::RectF) -> bool {
    (cx.x as i64) >= (r.x0 as i64)
        && (cx.x as i64) < (r.x1 as i64)
        && (cx.y as i64) >= (r.y0 as i64)
        && (cx.y as i64) < (r.y1 as i64)
}

#[inline]
fn covers_shifted(cx: Vec2, r: &crate::fixed::RectF, dx: i32, dy: i32) -> bool {
    let x = cx.x as i64 - dx as i64;
    let y = cx.y as i64 - dy as i64;
    x >= r.x0 as i64 && x < r.x1 as i64 && y >= r.y0 as i64 && y < r.y1 as i64
}
