//! Phase J: structural reuse detectors (fingerprints, sprite extraction).
//!
//! Detectors here produce *composite* proposals — documents with several
//! objects/instances — because reuse is inherently compositional: a
//! background field plus shared sprite objects placed by translation.
//!
//! Algorithm: pixels differing from the dominant (field) color are labeled
//! into 4-connected components; components are grouped by identical content
//! (same size + byte-equal crop).  The most frequent content class becomes a
//! shared raster object; one proposal places it at each of its component
//! origins (`sprite-repeat` when a class occurs >= 2 times, otherwise
//! `sprite-on-field`).  Interior background pixels inside a crop are stored
//! verbatim, so byte-exactness is preserved by construction and re-verified
//! by evaluation.  All scans are deterministic and **counted exactly where
//! they occur** (`SearchCounter`): every field-color sample read, every
//! flood-fill foreground probe, and every byte compared between crops.

use super::asset::Asset;
use super::detect::Proposal;
use super::work::SearchCounter;
use crate::color::{ColorFormat, Rgba};
use crate::ir::{Document, Instance, Object};
use crate::procedural::build;
use std::collections::BTreeMap;

/// Sprite side cap for the component crops (search bound).
const MAX_CROP_DIM: u32 = 512;

/// Normalized 4-byte code key of a sample (gray pads with zeros).
fn code_key(asset: &Asset, o: usize) -> [u8; 4] {
    let bps = asset.format.bytes_per_sample();
    let mut c = [0u8; 4];
    c[..bps].copy_from_slice(&asset.data[o..o + bps]);
    c
}

/// One foreground probe: materialize the sample's code and decide whether it
/// differs from the field color (1 read + 1 whole-code compare, counted).
#[inline]
fn is_fg(asset: &Asset, i: u32, j: u32, bg: [u8; 4], w: &mut SearchCounter) -> bool {
    w.read_pixels(1);
    let k = code_key(asset, asset.offset(i, j));
    w.compare_codes(1);
    k != bg
}

/// Most frequent code value (the field color), deterministically chosen.
///
/// Selection rule (normative for the inverse search): highest frequency;
/// ties broken by the **earliest row-major first occurrence** (a
/// representation-natural rule that never privileges a color's numerical
/// value); a further tie is impossible for distinct codes (a single pixel
/// has one code) and would fall back to the lower code value.
///
/// The accumulator is a `BTreeMap` (deterministic key order) so the winner
/// is process-independent; `HashMap` iteration order is deliberately
/// randomized and must never influence a search decision.  Each sample's
/// code materialization is counted (reads).
fn field_color(asset: &Asset, w: &mut SearchCounter) -> Option<[u8; 4]> {
    // code -> (count, first row-major index)
    let mut tab: BTreeMap<[u8; 4], (u64, u64)> = BTreeMap::new();
    for j in 0..asset.h {
        for i in 0..asset.w {
            let idx = (j as u64) * asset.w as u64 + i as u64;
            w.read_pixels(1);
            let k = code_key(asset, asset.offset(i, j));
            let e = tab.entry(k).or_insert((0, idx));
            e.0 += 1;
        }
    }
    tab.into_iter()
        .max_by_key(|&(_, (count, first))| (count, std::cmp::Reverse(first)))
        .map(|(c, _)| c)
}

/// A connected component of non-field pixels: its content crop rectangle and
/// origin.
#[derive(Clone)]
struct Component {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

/// Label 4-connected components of pixels != `bg`; returns their bounding
/// crops in deterministic (row-major seed) order.  Every foreground probe
/// (outer scan + flood-fill neighbor checks) is counted.
fn components(asset: &Asset, bg: [u8; 4], w: &mut SearchCounter) -> Vec<Component> {
    let (wd, h) = (asset.w as usize, asset.h as usize);
    let mut visited = vec![false; wd * h];
    let mut out = Vec::new();
    for j in 0..h {
        for i in 0..wd {
            if visited[j * wd + i] {
                continue;
            }
            if !is_fg(asset, i as u32, j as u32, bg, w) {
                continue;
            }
            // flood fill (bounded stack: at most n entries)
            let mut stack: Vec<(u32, u32)> = vec![(i as u32, j as u32)];
            visited[j * wd + i] = true;
            let (mut x0, mut y0, mut x1, mut y1) = (i as u32, j as u32, i as u32 + 1, j as u32 + 1);
            while let Some((x, y)) = stack.pop() {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx < asset.w && ny < asset.h && !visited[ny as usize * wd + nx as usize] {
                        visited[ny as usize * wd + nx as usize] = true;
                        if is_fg(asset, nx, ny, bg, w) {
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            out.push(Component { x0, y0, x1, y1 });
        }
    }
    out
}

/// Byte-equal content comparison of two same-size crops.  Counts every byte
/// actually compared (data-exact early exit at the first differing byte).
fn crops_equal(asset: &Asset, a: &Component, b: &Component, w: &mut SearchCounter) -> bool {
    debug_assert_eq!(a.x1 - a.x0, b.x1 - b.x0);
    debug_assert_eq!(a.y1 - a.y0, b.y1 - b.y0);
    let bps = asset.format.bytes_per_sample();
    let (cw, ch) = (a.x1 - a.x0, a.y1 - a.y0);
    for j in 0..ch {
        let oa_row = asset.offset(a.x0, a.y0 + j);
        let ob_row = asset.offset(b.x0, b.y0 + j);
        for i in 0..cw {
            let oa = oa_row + i as usize * bps;
            let ob = ob_row + i as usize * bps;
            for k in 0..bps {
                w.compare_crop_bytes(1);
                if asset.data[oa + k] != asset.data[ob + k] {
                    return false;
                }
            }
        }
    }
    true
}

/// A raster object of the crop contents (interior field pixels included).
/// Lifting the crop bytes into the proposal object is candidate
/// *construction* (bounded by content size; reflected in the candidate's
/// persistent bytes), so it is deliberately not counted as search work.
fn crop_object(asset: &Asset, c: &Component) -> Object {
    let (cw, ch) = (c.x1 - c.x0, c.y1 - c.y0);
    let bps = asset.format.bytes_per_sample();
    let mut data = Vec::with_capacity((cw * ch) as usize * bps);
    for j in c.y0..c.y1 {
        let row = asset.offset(c.x0, j);
        let row_len = cw as usize * bps;
        data.extend_from_slice(&asset.data[row..row + row_len]);
    }
    Object::Raster {
        format: asset.format,
        w: cw,
        h: ch,
        data,
    }
}

/// Composite doc: `field` covering the surface at layer 0, then a shared
/// `sprite` object placed at every `(dx, dy)` (layer 1, ascending orders).
fn composite_doc(field: Object, sprite: Object, at: &[(i32, i32)]) -> Document {
    let mut d = Document::new();
    d.objects.push(field);
    d.objects.push(sprite);
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
    for (k, &(dx, dy)) in at.iter().enumerate() {
        d.instances.push(Instance {
            object: 1,
            order: 2 + k as u32,
            layer: 1,
            transform: crate::fixed::Affine::from_px_translation(dx, dy),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
    }
    d
}

/// Structural detectors appended after the Phase I family detectors.
pub fn propose(asset: &Asset, out: &mut Vec<Proposal>) {
    if asset.w > 512 || asset.h > 512 {
        return; // search bounded; larger assets keep Phase I + fallback
    }
    let mut w = SearchCounter::default();
    let Some(bg) = field_color(asset, &mut w) else {
        return;
    };
    let comps = components(asset, bg, &mut w);
    if comps.is_empty() {
        return; // uniform: the constant detector covers it
    }
    let mut candidates: Vec<Component> = comps
        .into_iter()
        .filter(|c| {
            let (cw, ch) = (c.x1 - c.x0, c.y1 - c.y0);
            cw > 0 && ch > 0 && cw <= MAX_CROP_DIM && ch <= MAX_CROP_DIM
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    // group components by identical content (deterministic: first occurrence
    // order); each same-size crop pair is byte-compared (counted)
    let mut classes: Vec<(Component, Vec<Component>)> = Vec::new();
    loop {
        if candidates.is_empty() {
            break;
        }
        let first = candidates.remove(0);
        let mut members = vec![first.clone()];
        let mut rest = Vec::new();
        let (fcw, fch) = (first.x1 - first.x0, first.y1 - first.y0);
        for c in candidates {
            let (ccw, cch) = (c.x1 - c.x0, c.y1 - c.y0);
            if ccw == fcw && cch == fch && crops_equal(asset, &first, &c, &mut w) {
                members.push(c);
            } else {
                rest.push(c);
            }
        }
        candidates = rest;
        classes.push((first, members));
    }
    // the dominant class = most members (stable: ties keep first/top-left)
    classes.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));
    let Some((rep, members)) = classes.into_iter().next() else {
        return;
    };
    let (cw, ch) = (rep.x1 - rep.x0, rep.y1 - rep.y0);
    if cw == 0 || ch == 0 || cw > MAX_CROP_DIM || ch > MAX_CROP_DIM {
        return;
    }
    let bg_color = match asset.format {
        ColorFormat::Rgba8 => Rgba::from_bytes(bg),
        ColorFormat::Gray8 => Rgba::gray(bg[0]),
    };
    let field = build::constant(asset.w, asset.h, bg_color);
    let sprite = crop_object(asset, &rep);
    let at: Vec<(i32, i32)> = members.iter().map(|m| (m.x0 as i32, m.y0 as i32)).collect();
    if members.len() >= 2 {
        out.push(Proposal {
            name: "sprite-repeat",
            doc: composite_doc(field, sprite, &at),
            work: w,
        });
    } else {
        out.push(Proposal {
            name: "sprite-on-field",
            doc: composite_doc(field, sprite, &at),
            work: w,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset_of(data: &[[u8; 4]], w: u32, h: u32) -> Asset {
        let bytes: Vec<u8> = data.iter().flatten().copied().collect();
        Asset::new(w, h, ColorFormat::Rgba8, bytes).unwrap()
    }

    fn rgba(r: u8, g: u8, b: u8) -> [u8; 4] {
        [r, g, b, 255]
    }

    #[test]
    fn field_color_prefers_higher_frequency() {
        // 6 red, 2 green: red wins regardless of order of first appearance.
        let a = rgba(255, 0, 0);
        let b = rgba(0, 255, 0);
        // green appears first, but red is more frequent
        let data = vec![b, a, a, a, b, a, a, a];
        let asset = asset_of(&data, 8, 1);
        let mut w = SearchCounter::default();
        assert_eq!(field_color(&asset, &mut w), Some(a));
        // the whole surface was read once
        assert_eq!(w.pixels_read, asset.sample_count());
        assert_eq!(w.code_compares, 0);
    }

    /// Adversarial tie: two colors with EXACTLY equal frequency.  The winner
    /// must be the color whose first occurrence is earliest in row-major
    /// order, process-independently (BTreeMap selection, no HashMap
    /// iteration anywhere in the decision).
    #[test]
    fn field_color_tie_breaks_by_first_row_major_occurrence() {
        // 4 red then 4 green: equal counts; red first appears at index 0,
        // green at index 4 -> red must win.
        let a = rgba(255, 0, 0);
        let b = rgba(0, 0, 255);
        let mut row = Vec::new();
        for _ in 0..4 {
            row.push(a);
        }
        for _ in 0..4 {
            row.push(b);
        }
        let asset = asset_of(&row, 8, 1);
        let mut w = SearchCounter::default();
        assert_eq!(field_color(&asset, &mut w), Some(a));

        // interleaved tie: a,b,a,b,a,b,a,b: a first at 0, b first at 1 -> a
        let mut inter = Vec::new();
        for k in 0..8 {
            inter.push(if k % 2 == 0 { a } else { b });
        }
        let asset2 = asset_of(&inter, 8, 1);
        let mut w2 = SearchCounter::default();
        assert_eq!(field_color(&asset2, &mut w2), Some(a));

        // 2D tie across rows: top row all b, second row all a (equal counts,
        // same first column): b first at row 0 -> b wins by row-major order.
        let mut grid = Vec::new();
        for _ in 0..4 {
            grid.push(b); // row 0: b
        }
        for _ in 0..4 {
            grid.push(a); // row 1: a
        }
        let asset3 = asset_of(&grid, 4, 2);
        let mut w3 = SearchCounter::default();
        assert_eq!(field_color(&asset3, &mut w3), Some(b));
    }

    /// Repeated evaluation must be process-independent: same asset, many
    /// fresh accumulator instances, always the same winner.
    #[test]
    fn field_color_deterministic_across_repeated_runs() {
        let a = rgba(200, 0, 0);
        let b = rgba(0, 200, 0);
        let c = rgba(0, 0, 200);
        let mut data = Vec::new();
        // counts: a=8, b=8, c=4 with a's first occurrence earliest
        for _ in 0..4 {
            data.push(a);
        }
        for _ in 0..4 {
            data.push(b);
        }
        for _ in 0..4 {
            data.push(c);
        }
        for _ in 0..4 {
            data.push(a);
        }
        for _ in 0..4 {
            data.push(b);
        }
        let asset = asset_of(&data, 10, 2);
        for _ in 0..32 {
            let mut w = SearchCounter::default();
            assert_eq!(field_color(&asset, &mut w), Some(a));
        }
    }

    #[test]
    fn gray_tie_rule() {
        // gray asset: two gray levels at equal frequency; earlier row-major
        // occurrence wins (values 200 vs 40: the numeric value must NOT
        // decide the tie).
        let mut d = vec![200u8, 200, 40, 40];
        d.extend_from_slice(&[200, 200, 40, 40]);
        let asset = Asset::new(8, 1, ColorFormat::Gray8, d).unwrap();
        let mut w = SearchCounter::default();
        assert_eq!(field_color(&asset, &mut w), Some([200, 0, 0, 0]));
    }

    /// The component scan counts every foreground probe it performs: field
    /// probes (reads + compares) and flood-fill neighbor probes.  A
    /// structural proposal must therefore report more than a naive single
    /// pass over the surface.
    #[test]
    fn structural_scan_counts_probes_and_crop_bytes() {
        // 24x20 field of gray-ish blue with two identical 6x5 sprites
        let bg = [11u8, 22, 33, 255];
        let mut data: Vec<[u8; 4]> = Vec::new();
        for j in 0..20u32 {
            for i in 0..24u32 {
                let on_sprite = (4..10).contains(&i) && (3..8).contains(&j)
                    || (14..20).contains(&i) && (11..16).contains(&j);
                data.push(if on_sprite {
                    if (i + j) % 3 == 0 {
                        [200u8, 40, 40, 255]
                    } else {
                        [40, 200, 40, 255]
                    }
                } else {
                    bg
                });
            }
        }
        let bytes: Vec<u8> = data.iter().flatten().copied().collect();
        let asset = Asset::new(24, 20, ColorFormat::Rgba8, bytes).unwrap();

        let mut w = SearchCounter::default();
        assert_eq!(field_color(&asset, &mut w), Some(bg));
        let comps = components(&asset, bg, &mut w);
        assert_eq!(comps.len(), 2);
        // two identical crops must byte-compare exactly (6*5*4 bytes)
        let crop_before = w.crop_bytes_compared;
        let mut w2 = w;
        assert!(crops_equal(&asset, &comps[0], &comps[1], &mut w2));
        assert_eq!(
            w2.crop_bytes_compared - crop_before,
            6 * 5 * 4,
            "byte-exact crop equality must compare every byte of the crop"
        );

        // the whole pipeline reports real probe work, exceeding one sample
        // read per pixel (background probes + flood neighbors + crop bytes)
        let mut full = SearchCounter::default();
        field_color(&asset, &mut full);
        let comps2 = components(&asset, bg, &mut full);
        assert_eq!(comps2.len(), 2);
        assert!(
            full.total_units() > asset.sample_count(),
            "probe work {} should exceed one flat sample-count pass {}",
            full.total_units(),
            asset.sample_count()
        );
        assert!(full.code_compares > 0, "foreground probes must be counted");
    }
}
