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
//! by evaluation.  All scans are deterministic and work-counted.

use super::asset::Asset;
use super::detect::Proposal;
use crate::color::{ColorFormat, Rgba};
use crate::ir::{Document, Instance, Object};
use crate::procedural::build;
use std::collections::HashMap;

/// Sprite side cap for the component crops (search bound).
const MAX_CROP_DIM: u32 = 512;

/// Normalized 4-byte code key of a sample (gray pads with zeros).
fn code_key(asset: &Asset, o: usize) -> [u8; 4] {
    let bps = asset.format.bytes_per_sample();
    let mut c = [0u8; 4];
    c[..bps].copy_from_slice(&asset.data[o..o + bps]);
    c
}

/// Most frequent code value (the field color).  O(n) with a bounded table.
fn field_color(asset: &Asset) -> Option<[u8; 4]> {
    let mut counts: HashMap<[u8; 4], u64> = HashMap::new();
    for j in 0..asset.h {
        for i in 0..asset.w {
            *counts
                .entry(code_key(asset, asset.offset(i, j)))
                .or_default() += 1;
        }
    }
    counts.into_iter().max_by_key(|&(_, n)| n).map(|(c, _)| c)
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
/// crops in deterministic (row-major seed) order.
fn components(asset: &Asset, bg: [u8; 4]) -> Vec<Component> {
    let (w, h) = (asset.w as usize, asset.h as usize);
    let mut visited = vec![false; w * h];
    let mut out = Vec::new();
    let is_fg = |i: u32, j: u32| code_key(asset, asset.offset(i, j)) != bg;
    for j in 0..h {
        for i in 0..w {
            if visited[j * w + i] || !is_fg(i as u32, j as u32) {
                continue;
            }
            // flood fill (bounded stack: at most n entries)
            let mut stack: Vec<(u32, u32)> = vec![(i as u32, j as u32)];
            visited[j * w + i] = true;
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
                    if nx < asset.w && ny < asset.h && !visited[ny as usize * w + nx as usize] {
                        visited[ny as usize * w + nx as usize] = true;
                        if is_fg(nx, ny) {
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

/// Byte-equal content comparison of two same-size crops.
fn crops_equal(asset: &Asset, a: &Component, b: &Component) -> bool {
    debug_assert_eq!(a.x1 - a.x0, b.x1 - b.x0);
    debug_assert_eq!(a.y1 - a.y0, b.y1 - b.y0);
    let bps = asset.format.bytes_per_sample();
    let (cw, ch) = (a.x1 - a.x0, a.y1 - a.y0);
    for j in 0..ch {
        for i in 0..cw {
            let oa = asset.offset(a.x0 + i, a.y0 + j);
            let ob = asset.offset(b.x0 + i, b.y0 + j);
            if asset.data[oa..oa + bps] != asset.data[ob..ob + bps] {
                return false;
            }
        }
    }
    true
}

/// A raster object of the crop contents (interior field pixels included).
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
    let Some(bg) = field_color(asset) else {
        return;
    };
    let comps = components(asset, bg);
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
    // order)
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
            if ccw == fcw && cch == fch && crops_equal(asset, &first, &c) {
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
    let scan_work = asset.sample_count() * 2;
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
            work: scan_work,
        });
    } else {
        out.push(Proposal {
            name: "sprite-on-field",
            doc: composite_doc(field, sprite, &at),
            work: scan_work,
        });
    }
}
