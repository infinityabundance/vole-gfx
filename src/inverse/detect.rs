//! Bounded scalar detectors (Phase I): each proposes a generator explanation
//! for an asset from cheap structural invariants, in the paper's search
//! order.  Proposals are *candidates*, not verdicts: `evaluate` measures the
//! exact residual of each proposal and the Pareto frontier keeps only
//! non-dominated explanations.  All work is deterministic and counted
//! (`search_work` units are pixel compares / hash evaluations).

use super::asset::Asset;
use crate::color::Rgba;
use crate::ir::{Document, Instance, Object};
use crate::procedural::build;
use crate::procedural::families::{palette, tiled};

/// Hard cap on any single period/tile dimension scanned by a detector
/// (bounded search; courts keep assets small, large assets fall back).
pub const PERIOD_SCAN_LIMIT: u32 = 4096;

/// A generator proposal: a self-contained document (one or more objects +
/// instances, e.g. a background field plus sprite overlay) that, materialized
/// at the asset's surface, is a candidate explanation.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub name: &'static str,
    pub doc: Document,
    /// Deterministic work units spent to produce this proposal.
    pub work: u64,
}

/// Build the canonical single-object document: `object` placed at the origin
/// (object extent == asset surface by construction in the detectors).
pub fn one_object_doc(object: Object) -> Document {
    let mut d = Document::new();
    d.objects.push(object);
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: crate::fixed::Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    d
}

/// Enumerate proposals for an asset in a deterministic order.
pub fn propose(asset: &Asset) -> Vec<Proposal> {
    let mut out = Vec::new();
    constant(asset, &mut out);
    let two = unique_color_count(asset) <= 2;
    if two {
        stripes_and_checker(asset, &mut out);
    }
    palette_bands(asset, &mut out);
    tiled(asset, &mut out);
    gradients(asset, &mut out);
    super::structure::propose(asset, &mut out); // Phase J structural reuse
    out
}

fn push(out: &mut Vec<Proposal>, name: &'static str, object: Object, work: u64) {
    out.push(Proposal {
        name,
        doc: one_object_doc(object),
        work,
    });
}

/// Number of distinct code values in the asset (bounded scan).
fn unique_color_count(asset: &Asset) -> usize {
    let mut set = Vec::new();
    for j in 0..asset.h {
        for i in 0..asset.w {
            let c = asset.code_at(i, j);
            if !set.contains(&c) {
                set.push(c);
                if set.len() > 256 {
                    return set.len();
                }
            }
        }
    }
    set.len()
}

/// Are all rows byte-identical to row 0?
fn rows_uniform(asset: &Asset) -> bool {
    let r0 = asset.row_codes(0);
    for j in 1..asset.h {
        if asset.row_codes(j) != r0 {
            return false;
        }
    }
    true
}

/// Are all columns byte-identical to column 0?
fn cols_uniform(asset: &Asset) -> bool {
    let bps = asset.format.bytes_per_sample();
    for j in 0..asset.h {
        let base = asset.offset(0, j);
        let expect = &asset.data[base..base + bps];
        for i in 1..asset.w {
            let o = asset.offset(i, j);
            if &asset.data[o..o + bps] != expect {
                return false;
            }
        }
    }
    true
}

/// Longest run of the *first* color of row 0 (u32; 0 if row 0 is empty).
fn first_run_x(asset: &Asset) -> u32 {
    let c0 = asset.code_at(0, 0);
    let mut n = 0u32;
    while n < asset.w && asset.code_at(n, 0) == c0 {
        n += 1;
    }
    n
}

fn constant(asset: &Asset, out: &mut Vec<Proposal>) {
    let c0 = asset.code_at(0, 0);
    // O(w*h) compare; counted
    let work = asset.sample_count();
    let mut uniform = true;
    for j in 0..asset.h {
        for i in 0..asset.w {
            if asset.code_at(i, j) != c0 {
                uniform = false;
                break;
            }
        }
        if !uniform {
            break;
        }
    }
    if uniform {
        let color = asset.rgba_at(0, 0);
        push(
            out,
            "constant",
            build::constant(asset.w, asset.h, color),
            work,
        );
    }
}

/// 2-color periodic families: vertical bands (stripe-x), horizontal bands
/// (stripe-y), and checker.
fn stripes_and_checker(asset: &Asset, out: &mut Vec<Proposal>) {
    let work = asset.sample_count();
    // palette order from row 0: c0 = pixel (0,0), c1 = the other color
    let c0 = asset.rgba_at(0, 0);
    let mut c1 = None;
    for j in 0..asset.h {
        for i in 0..asset.w {
            let c = asset.rgba_at(i, j);
            if c != c0 {
                c1 = Some(c);
                break;
            }
        }
        if c1.is_some() {
            break;
        }
    }
    let Some(c1) = c1 else { return };
    // vertical bands (color varies along x): rows uniform
    if rows_uniform(asset) {
        let px = first_run_x(asset);
        if px >= 1 && px < asset.w && px <= PERIOD_SCAN_LIMIT {
            push(
                out,
                "periodic-stripe-x",
                build::stripes(asset.w, asset.h, px, c0, c1),
                work,
            );
        }
    }
    // horizontal bands (color varies along y): columns uniform
    if cols_uniform(asset) {
        let c00 = asset.code_at(0, 0);
        let mut py = 0u32;
        while py < asset.h && asset.code_at(0, py) == c00 {
            py += 1;
        }
        if py >= 1 && py < asset.h && py <= PERIOD_SCAN_LIMIT {
            push(
                out,
                "periodic-stripe-y",
                build::stripe_y(asset.w, asset.h, py, c0, c1),
                work,
            );
        }
    }
    // checker: derive px from row 0 and py from column 0 when both vary
    let c00 = asset.code_at(0, 0);
    let px0 = first_run_x(asset);
    let mut py0 = 0u32;
    while py0 < asset.h && asset.code_at(0, py0) == c00 {
        py0 += 1;
    }
    if px0 >= 1 && px0 < asset.w && py0 >= 1 && py0 < asset.h {
        // px is the run of c0 along row 0 — but the checker's first c0 run
        // has length px only when py is a multiple of... verify the full
        // plane against the candidate and let evaluation arbitrate.
        push(
            out,
            "periodic-checker",
            build::checker(asset.w, asset.h, px0, py0, c0, c1),
            work,
        );
    }
}

/// Palette-band families: uniform rows whose row-0 pattern is an exact cycle
/// of n equal-width bands (band-x), and the transposed case (band-y).
fn palette_bands(asset: &Asset, out: &mut Vec<Proposal>) {
    if unique_color_count(asset) < 3 || unique_color_count(asset) > palette::MAX_ENTRIES as usize {
        return;
    }
    let work = asset.sample_count();
    if rows_uniform(asset) {
        // row-0 run lengths
        let mut runs: Vec<(Rgba, u32)> = Vec::new();
        let mut i = 0u32;
        let bps = asset.format.bytes_per_sample();
        while i < asset.w {
            let o = asset.offset(i, 0);
            let cur = Rgba::from_bytes({
                let mut c = [0u8; 4];
                c[..bps].copy_from_slice(&asset.data[o..o + bps]);
                c
            });
            let mut n = 0u32;
            while i + n < asset.w {
                let oo = asset.offset(i + n, 0);
                if asset.data[oo..oo + bps] != asset.data[o..o + bps] {
                    break;
                }
                n += 1;
            }
            runs.push((cur, n));
            i += n;
        }
        // perfect cycle: the sequence of run colors repeats with period n0,
        // every run has equal width
        let Some((first, first_w)) = runs.first().copied() else {
            return;
        };
        if first_w == 0 {
            return;
        }
        // find the first *later* run equal to the first (its position is the
        // cycle length); without one the whole row is one cycle
        let n0 = 1 + runs[1..]
            .iter()
            .position(|&(c, w)| c == first && w == first_w)
            .unwrap_or(runs.len() - 1);
        if !(2..=256).contains(&n0) {
            return;
        }
        let cycle = &runs[..n0];
        if cycle.iter().any(|&(_, w)| w != first_w) {
            return;
        }
        // the rest of the runs must repeat the cycle exactly
        let mut ok = true;
        for (k, &(c, w)) in runs[n0..].iter().enumerate() {
            if cycle[k % n0] != (c, w) {
                ok = false;
                break;
            }
        }
        if !ok {
            return;
        }
        if cycle.len() < 2 || cycle.len() > 256 {
            return;
        }
        let entries: Vec<Rgba> = cycle.iter().map(|&(c, _)| c).collect();
        push(
            out,
            "palette-band-x",
            build::palette_bands(asset.w, asset.h, first_w, entries),
            work,
        );
    }
}

/// Tiled: minimal horizontal + vertical period such that every pixel repeats
/// with the wrap rule `p(i, j) == p(i % tw, j % th)`.
fn tiled(asset: &Asset, out: &mut Vec<Proposal>) {
    let work = asset.sample_count();
    let max_tw = asset.w.min(PERIOD_SCAN_LIMIT + 1);
    let mut tw = 0u32;
    'w: for p in 1..max_tw {
        // every column i must equal column i - p for i >= p (wrap equivalent)
        for j in 0..asset.h {
            for i in p..asset.w {
                if !asset.codes_equal(i, j, i - p, j) {
                    continue 'w;
                }
            }
        }
        tw = p;
        break;
    }
    if tw == 0 || tw >= asset.w {
        return;
    }
    let max_th = asset.h.min(PERIOD_SCAN_LIMIT + 1);
    let mut th = 0u32;
    'h: for p in 1..max_th {
        for j in p..asset.h {
            for i in 0..tw {
                if !asset.codes_equal(i, j, i, j - p) {
                    continue 'h;
                }
            }
        }
        th = p;
        break;
    }
    if th == 0 || th >= asset.h || tw > tiled::MAX_TILE_DIM || th > tiled::MAX_TILE_DIM {
        return;
    }
    let tile = (0..th)
        .flat_map(|j| (0..tw).map(move |i| asset.rgba_at(i, j)))
        .collect();
    push(
        out,
        "tiled",
        build::tiled(asset.w, asset.h, tw, th, tile),
        work,
    );
}

/// Simple analytic gradients: uniform row/column ramps and the bilinear
/// corner fit (all measured against the asset by evaluation).
fn gradients(asset: &Asset, out: &mut Vec<Proposal>) {
    let work = asset.sample_count();
    if asset.w >= 2 && rows_uniform(asset) {
        let c0 = asset.rgba_at(0, 0);
        let c1 = asset.rgba_at(asset.w - 1, 0);
        if c0 != c1 {
            push(
                out,
                "gradient-h-ramp",
                build::linear_gradient(asset.w, asset.h, 0, 0, c0, (asset.w - 1) as i32, 0, c1),
                work,
            );
        }
    }
    if asset.h >= 2 && cols_uniform(asset) {
        let c0 = asset.rgba_at(0, 0);
        let c1 = asset.rgba_at(0, asset.h - 1);
        if c0 != c1 {
            push(
                out,
                "gradient-v-ramp",
                build::linear_gradient(asset.w, asset.h, 0, 0, c0, 0, (asset.h - 1) as i32, c1),
                work,
            );
        }
    }
    if asset.w >= 2 && asset.h >= 2 {
        push(
            out,
            "gradient-bilinear",
            build::bilinear_gradient(
                asset.w,
                asset.h,
                asset.rgba_at(0, 0),
                asset.rgba_at(asset.w - 1, 0),
                asset.rgba_at(0, asset.h - 1),
                asset.rgba_at(asset.w - 1, asset.h - 1),
            ),
            work,
        );
    }
}

/// The literal (original raster) fallback object.
pub fn literal_object(asset: &Asset) -> Object {
    Object::Raster {
        format: asset.format,
        w: asset.w,
        h: asset.h,
        data: asset.data.clone(),
    }
}
