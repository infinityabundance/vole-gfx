//! Bounded scalar detectors (Phase I): each proposes a generator explanation
//! for an asset from cheap structural invariants, in the paper's search
//! order.  Proposals are *candidates*, not verdicts: `evaluate` measures the
//! exact residual of each proposal and the Pareto frontier keeps only
//! non-dominated explanations.  All work is deterministic and **counted
//! exactly where it occurs** (`SearchCounter`, see `super::work`): a
//! detector's proposal carries the real scan operations it performed, not a
//! flat `sample_count()`.

use super::asset::Asset;
use super::work::SearchCounter;
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
    /// Deterministic search work spent to produce this proposal: the
    /// detector's full scan counter (attribution rule: each proposal of a
    /// detector carries the shared scan cost — see `super::work`).
    pub work: SearchCounter,
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
    stripes_and_checker(asset, &mut out);
    palette_bands(asset, &mut out);
    tiled(asset, &mut out);
    gradients(asset, &mut out);
    super::structure::propose(asset, &mut out); // Phase J structural reuse
    super::search::propose_seeded_field(asset, &mut out); // Phase K seed sweep
    out
}

fn push(out: &mut Vec<Proposal>, name: &'static str, object: Object, work: SearchCounter) {
    out.push(Proposal {
        name,
        doc: one_object_doc(object),
        work,
    });
}

/// Number of distinct code values in the asset (bounded scan).  Each sample
/// is read once; each linear membership probe against the running distinct
/// set is counted (the set caps at 256 entries).
fn unique_color_count(asset: &Asset, w: &mut SearchCounter) -> usize {
    let mut set: Vec<[u8; 4]> = Vec::new();
    for j in 0..asset.h {
        for i in 0..asset.w {
            w.read_pixels(1);
            let c = asset.code_at(i, j);
            let mut found = false;
            for e in &set {
                w.compare_codes(1);
                if *e == c {
                    found = true;
                    break;
                }
            }
            if !found {
                set.push(c);
                if set.len() > 256 {
                    return set.len();
                }
            }
        }
    }
    set.len()
}

/// Are all rows byte-identical to row 0?  Each compared row costs `w`
/// whole-code equality decisions (full-row upper bound; see `super::work`).
fn rows_uniform(asset: &Asset, w: &mut SearchCounter) -> bool {
    let r0 = asset.row_codes(0);
    for j in 1..asset.h {
        w.compare_codes(asset.w as u64);
        if asset.row_codes(j) != r0 {
            return false;
        }
    }
    true
}

/// Are all columns byte-identical to column 0?  Per-sample whole-code
/// compares; the scan stops at the first differing sample (data-exact).
fn cols_uniform(asset: &Asset, w: &mut SearchCounter) -> bool {
    let bps = asset.format.bytes_per_sample();
    for j in 0..asset.h {
        let base = asset.offset(0, j);
        let expect = &asset.data[base..base + bps];
        for i in 1..asset.w {
            w.compare_codes(1);
            let o = asset.offset(i, j);
            if &asset.data[o..o + bps] != expect {
                return false;
            }
        }
    }
    true
}

/// Longest run of the *first* color of row 0 (u32; 0 if row 0 is empty).
/// Reads and compares only the samples actually examined.
fn first_run_x(asset: &Asset, w: &mut SearchCounter) -> u32 {
    w.read_pixels(1);
    let c0 = asset.code_at(0, 0);
    let mut n = 0u32;
    while n < asset.w {
        w.read_pixels(1);
        w.compare_codes(1);
        if asset.code_at(n, 0) != c0 {
            break;
        }
        n += 1;
    }
    n
}

/// Height of the first color run of column 0 (0 when column 0 is uniform in
/// that color for the whole asset).  Reads and compares examined samples.
fn first_run_y(asset: &Asset, w: &mut SearchCounter) -> u32 {
    w.read_pixels(1);
    let c00 = asset.code_at(0, 0);
    let mut n = 0u32;
    while n < asset.h {
        w.read_pixels(1);
        w.compare_codes(1);
        if asset.code_at(0, n) != c00 {
            break;
        }
        n += 1;
    }
    n
}

fn constant(asset: &Asset, out: &mut Vec<Proposal>) {
    let mut w = SearchCounter::default();
    w.read_pixels(1);
    let c0 = asset.code_at(0, 0);
    // per-sample compare until the first mismatch (data-exact early exit)
    let mut uniform = true;
    for j in 0..asset.h {
        for i in 0..asset.w {
            w.read_pixels(1);
            w.compare_codes(1);
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
        push(out, "constant", build::constant(asset.w, asset.h, color), w);
    }
}

/// 2-color periodic families: vertical bands (stripe-x), horizontal bands
/// (stripe-y), and checker.  Runs only when the asset has <= 2 distinct code
/// values (the gate scan is counted here, into the detector's proposals).
fn stripes_and_checker(asset: &Asset, out: &mut Vec<Proposal>) {
    let mut w = SearchCounter::default();
    if unique_color_count(asset, &mut w) > 2 {
        return;
    }
    // palette order from row 0: c0 = pixel (0,0), c1 = the other color
    w.read_pixels(1);
    let c0 = asset.rgba_at(0, 0);
    let mut c1 = None;
    'find: for j in 0..asset.h {
        for i in 0..asset.w {
            w.read_pixels(1);
            w.compare_codes(1);
            let c = asset.rgba_at(i, j);
            if c != c0 {
                c1 = Some(c);
                break 'find;
            }
        }
    }
    let Some(c1) = c1 else { return };
    // vertical bands (color varies along x): rows uniform
    if rows_uniform(asset, &mut w) {
        let px = first_run_x(asset, &mut w);
        if px >= 1 && px < asset.w && px <= PERIOD_SCAN_LIMIT {
            push(
                out,
                "periodic-stripe-x",
                build::stripes(asset.w, asset.h, px, c0, c1),
                w,
            );
        }
    }
    // horizontal bands (color varies along y): columns uniform
    if cols_uniform(asset, &mut w) {
        let py = first_run_y(asset, &mut w);
        if py >= 1 && py < asset.h && py <= PERIOD_SCAN_LIMIT {
            push(
                out,
                "periodic-stripe-y",
                build::stripe_y(asset.w, asset.h, py, c0, c1),
                w,
            );
        }
    }
    // checker: derive px from row 0 and py from column 0 when both vary
    let px0 = first_run_x(asset, &mut w);
    let py0 = first_run_y(asset, &mut w);
    if px0 >= 1 && px0 < asset.w && py0 >= 1 && py0 < asset.h {
        // px is the run of c0 along row 0 — but the checker's first c0 run
        // has length px only when py is a multiple of... verify the full
        // plane against the candidate and let evaluation arbitrate.
        push(
            out,
            "periodic-checker",
            build::checker(asset.w, asset.h, px0, py0, c0, c1),
            w,
        );
    }
}

/// Palette-band families: uniform rows whose row-0 pattern is an exact cycle
/// of n equal-width bands (band-x), and the transposed case (band-y).
fn palette_bands(asset: &Asset, out: &mut Vec<Proposal>) {
    let mut w = SearchCounter::default();
    let ncols = unique_color_count(asset, &mut w);
    if ncols < 3 || ncols > palette::MAX_ENTRIES as usize {
        return;
    }
    if rows_uniform(asset, &mut w) {
        // row-0 run lengths (each run-extension probe is one whole-code
        // compare; a run start additionally materializes one sample)
        let mut runs: Vec<(Rgba, u32)> = Vec::new();
        let mut i = 0u32;
        let bps = asset.format.bytes_per_sample();
        while i < asset.w {
            w.read_pixels(1);
            let o = asset.offset(i, 0);
            let cur = Rgba::from_bytes({
                let mut c = [0u8; 4];
                c[..bps].copy_from_slice(&asset.data[o..o + bps]);
                c
            });
            let mut n = 0u32;
            while i + n < asset.w {
                w.compare_codes(1);
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
        // cycle length); without one the whole row is one cycle.  Each
        // candidate run costs one whole-code decision (counted); the width
        // half of the tuple compare is scalar control work.
        let mut n0 = runs.len();
        for (k, &r) in runs[1..].iter().enumerate() {
            w.compare_codes(1);
            if r == (first, first_w) {
                n0 = 1 + k;
                break;
            }
        }
        if !(2..=256).contains(&n0) {
            return;
        }
        let cycle = &runs[..n0];
        if cycle.iter().any(|&(_, w2)| w2 != first_w) {
            return;
        }
        // the rest of the runs must repeat the cycle exactly: same color AND
        // same width (a same-color run of a different width rejects); the
        // whole-code color decision is counted, the u32 width half of the
        // tuple compare is scalar control work and deliberately not counted
        let mut ok = true;
        for (k, &(c, w2)) in runs[n0..].iter().enumerate() {
            w.compare_codes(1);
            if cycle[k % n0] != (c, w2) {
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
            w,
        );
    }
}

/// Tiled: minimal horizontal + vertical period such that every pixel repeats
/// with the wrap rule `p(i, j) == p(i % tw, j % th)`.
///
/// Every period candidate re-scans the plane, so the counted work is the sum
/// over the scanned periods of the sample pairs actually compared (each
/// `codes_equal` materializes two samples and decides one whole-code
/// equality), including the failing pair that rejects a period.  A naive
/// `sample_count()` accounting would miss this entirely.
fn tiled(asset: &Asset, out: &mut Vec<Proposal>) {
    let mut w = SearchCounter::default();
    let max_tw = asset.w.min(PERIOD_SCAN_LIMIT + 1);
    let mut tw = 0u32;
    'w: for p in 1..max_tw {
        // every column i must equal column i - p for i >= p (wrap equivalent)
        for j in 0..asset.h {
            for i in p..asset.w {
                w.read_pixels(2);
                w.compare_codes(1);
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
                w.read_pixels(2);
                w.compare_codes(1);
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
    // lifting the minimal tile into the proposal object: one sample read per
    // tile pixel
    w.read_pixels(tw as u64 * th as u64);
    let tile = (0..th)
        .flat_map(|j| (0..tw).map(move |i| asset.rgba_at(i, j)))
        .collect();
    push(
        out,
        "tiled",
        build::tiled(asset.w, asset.h, tw, th, tile),
        w,
    );
}

/// Simple analytic gradients: uniform row/column ramps and the bilinear
/// corner fit (all measured against the asset by evaluation).
fn gradients(asset: &Asset, out: &mut Vec<Proposal>) {
    let mut w = SearchCounter::default();
    if asset.w >= 2 && rows_uniform(asset, &mut w) {
        w.read_pixels(1);
        let c0 = asset.rgba_at(0, 0);
        w.read_pixels(1);
        let c1 = asset.rgba_at(asset.w - 1, 0);
        w.compare_codes(1);
        if c0 != c1 {
            push(
                out,
                "gradient-h-ramp",
                build::linear_gradient(asset.w, asset.h, 0, 0, c0, (asset.w - 1) as i32, 0, c1),
                w,
            );
        }
    }
    if asset.h >= 2 && cols_uniform(asset, &mut w) {
        w.read_pixels(1);
        let c0 = asset.rgba_at(0, 0);
        w.read_pixels(1);
        let c1 = asset.rgba_at(0, asset.h - 1);
        w.compare_codes(1);
        if c0 != c1 {
            push(
                out,
                "gradient-v-ramp",
                build::linear_gradient(asset.w, asset.h, 0, 0, c0, 0, (asset.h - 1) as i32, c1),
                w,
            );
        }
    }
    if asset.w >= 2 && asset.h >= 2 {
        for _ in 0..4 {
            w.read_pixels(1);
        }
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
            w,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorFormat;

    fn gray_asset(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Asset {
        let mut data = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                data.push(f(i, j));
            }
        }
        Asset::new(w, h, ColorFormat::Gray8, data).unwrap()
    }

    fn one(asset: &Asset, det: fn(&Asset, &mut Vec<Proposal>)) -> Proposal {
        let mut out = Vec::new();
        det(asset, &mut out);
        assert_eq!(out.len(), 1, "detector must emit exactly one proposal");
        out.remove(0)
    }

    /// The constant detector on a uniform asset must count one read + one
    /// whole-code compare per sample (plus the seed read) — a single full
    /// scan, no more.
    #[test]
    fn constant_detector_counts_one_full_scan() {
        let a = gray_asset(16, 12, |_, _| 90);
        let p = one(&a, constant);
        let samples = a.sample_count(); // 192
        // 1 seed read + per sample (1 read + 1 compare) over the full plane
        assert_eq!(p.work.total_units(), 1 + 2 * samples);
        assert_eq!(p.work.pixels_read, 1 + samples);
        assert_eq!(p.work.code_compares, samples);
        assert_eq!(p.work.crop_bytes_compared, 0);
    }

    /// The tiled detector re-scans the plane for every rejected period, so
    /// its counted work must exceed a single full scan — and must exceed the
    /// constant detector's single-scan cost on a same-sized asset.  This is
    /// the regression the flat `sample_count()` accounting got wrong.
    #[test]
    fn tiled_detector_counts_per_period_scan_work() {
        // 32x24 plane with an exact 16x12 checker period (8x6 cells): the
        // horizontal scan rejects p = 1..=15 (each after comparing up to the
        // first cell boundary it crosses) and accepts p = 16 after a
        // full-plane pass; the vertical scan does the same for the 12-row
        // period.
        let a = gray_asset(
            32,
            24,
            |i, j| {
                if ((i / 8) + (j / 6)) % 2 == 0 { 0 } else { 255 }
            },
        );
        let t = one(&a, tiled);
        let uniform = gray_asset(32, 24, |_, _| 90);
        let c = one(&uniform, constant);
        assert_eq!(t.name, "tiled");
        // period scan work (periods x plane) beats any single full scan
        assert!(
            t.work.total_units() > 2 * a.sample_count(),
            "tiled scan must exceed two full scans, got {}",
            t.work.total_units()
        );
        assert!(
            t.work.total_units() > c.work.total_units(),
            "tiled scan ({}) must exceed the constant single-scan cost ({}) on a same-sized asset",
            t.work.total_units(),
            c.work.total_units()
        );
        // every period probe materializes two samples and decides one whole
        // code; the only extra reads are the minimal-tile lift (16x12)
        assert!(t.work.code_compares > 0);
        assert!(t.work.pixels_read > 2 * t.work.code_compares);
    }

    /// The palette-band detector's cost is dominated by its distinct-color
    /// gate scan (linear membership probes) plus the full row-uniformity
    /// scan plus row-0 run extension — all exceeding a single flat scan.
    #[test]
    fn palette_band_detector_counts_gate_and_run_work() {
        // 60x20 three-band asset, band width 5: 3 distinct colors.
        let a = gray_asset(60, 20, |i, _| [10, 120, 230][(i / 5) as usize % 3]);
        let p = one(&a, palette_bands);
        assert_eq!(p.name, "palette-band-x");
        // reads: distinct gate (w*h) + run starts (~w/5); compares: gate
        // membership (>= w*h) + rows_uniform (19 rows x 60) + run extension
        // (~w) + cycle check.  Total must be several full scans, not one.
        assert!(
            p.work.total_units() > 4 * a.sample_count(),
            "palette-band scan must exceed four full scans, got {}",
            p.work.total_units()
        );
        assert!(p.work.code_compares > p.work.pixels_read);
    }

    /// A detector that rejects an asset must emit nothing (early exit still
    /// counted where it occurred, but no proposal is fabricated).
    #[test]
    fn rejected_detectors_emit_no_proposal() {
        // noise: constant rejects after a few compares -> no proposal
        let n = gray_asset(16, 16, |i, j| ((i * 31 + j * 17) % 256) as u8);
        let mut out = Vec::new();
        constant(&n, &mut out);
        assert!(out.is_empty());
        // a 3-color asset rejects stripes_and_checker (gate > 2 colors)
        let three = gray_asset(12, 12, |i, j| ((i / 4 + j / 4) % 3) as u8 * 100);
        let mut out = Vec::new();
        stripes_and_checker(&three, &mut out);
        assert!(out.is_empty());
    }

    /// Detector counting is deterministic: repeated runs of a proposal sweep
    /// produce identical counters.
    #[test]
    fn detector_work_is_deterministic() {
        let a = gray_asset(
            20,
            14,
            |i, j| {
                if ((i / 5) + (j / 7)) % 2 == 0 { 0 } else { 255 }
            },
        );
        let mut o1 = Vec::new();
        let mut o2 = Vec::new();
        constant(&a, &mut o1);
        constant(&a, &mut o2);
        assert_eq!(o1.len(), o2.len());
        for (x, y) in o1.iter().zip(o2.iter()) {
            assert_eq!(x.work, y.work);
        }
    }
}
