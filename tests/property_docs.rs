//! Phase B: deterministic property tests over generated valid documents.
//!
//! Uses the in-crate split-mix RNG (no `rand` dependency: determinism is the
//! point).  Invariants checked per document:
//!   * validate → encode → decode → model equality → canonical-form pass;
//!   * materializing a partition of disjoint tiles equals the full surface
//!     (partial materialization correctness);
//!   * counters: samples evaluated == requested area exactly.

mod common;

use common::*;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::fixed::fp;
use vole_gfx::ir::{Document, Event, Op};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::{Domain, ObservationRequest};

fn random_doc(rng: &mut Rng, w: u32, h: u32) -> Document {
    let mut d = Document::new();
    let n_obj = 1 + rng.below(3);
    for _o in 0..n_obj {
        let ow = 1 + rng.below(w.min(8));
        let oh = 1 + rng.below(h.min(8));
        match rng.below(3) {
            0 => d.objects.push(checker(
                ow,
                oh,
                Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
                Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
            )),
            1 => {
                let mut data = Vec::new();
                for _j in 0..oh {
                    for i in 0..ow {
                        let v = (255 * i / ow) as u8;
                        data.extend_from_slice(&[v, v.wrapping_mul(2), 255 - v, 255]);
                    }
                }
                d.objects.push(vole_gfx::ir::Object::Raster {
                    format: ColorFormat::Rgba8,
                    w: ow,
                    h: oh,
                    data,
                });
            }
            _ => {
                let pal = d.objects.len() as u32;
                d.objects.push(vole_gfx::ir::Object::Palette {
                    entries: vec![
                        Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
                        Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
                        Rgba::WHITE,
                    ],
                });
                let mut idx = Vec::new();
                for _j in 0..oh {
                    for _i in 0..ow {
                        idx.push(rng.below(3));
                    }
                }
                d.objects.push(vole_gfx::ir::Object::IndexedRaster {
                    pal,
                    w: ow,
                    h: oh,
                    indices: idx,
                });
            }
        }
    }
    let mut order = 1u32;
    for o in 0..d.objects.len() as u32 {
        if matches!(d.objects[o as usize], vole_gfx::ir::Object::Palette { .. }) {
            continue; // palettes are not drawable
        }
        let tx = (rng.below(8) as i32) - 4;
        let ty = (rng.below(8) as i32) - 4;
        d.instances.push(placed(o, order, rng.below(3), tx, ty));
        order += 1;
    }
    let mut t = 1u64;
    for _ in 0..1 + rng.below(2) {
        let mut ops = vec![Op::FillRect {
            rect: vole_gfx::fixed::RectF::from_px(-4, -4, w as i32 + 4, h as i32 + 4),
            color: Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
        }];
        if rng.below(2) == 0 {
            let dx = (rng.below(6) as i32) - 3;
            ops.push(Op::CopyRegion {
                src: vole_gfx::fixed::RectF::from_px(0, 0, 4, 4),
                dx: fp(dx),
                dy: fp(0),
            });
        }
        d.events.push(Event { t, ops });
        t += 1 + rng.below(1000) as u64;
    }
    d
}

#[test]
fn random_docs_roundtrip_and_canonical() {
    let mut rng = Rng::new(0xB0BA_5EED_2026_0906);
    let mut checked = 0u32;
    for _ in 0..150u32 {
        let doc = random_doc(&mut rng, 16, 16);
        if vole_gfx::ir::validate::validate(&doc).is_err() {
            continue;
        }
        let bytes = vole_gfx::ir::encode::encode(&doc);
        let back = vole_gfx::ir::decode::decode(&bytes).unwrap();
        assert_eq!(back, doc, "decode(encode(x)) != x");
        vole_gfx::ir::canonical::check_canonical(&bytes).expect("canonical form");
        checked += 1;
    }
    assert!(checked >= 40, "expected >=40 checked docs, got {checked}");
}

#[test]
fn random_docs_tile_partition_equals_full() {
    let mut rng = Rng::new(0xDEAD_BEEF_1234_5678);
    let mut checked = 0u32;
    for _ in 0..200u32 {
        let doc = random_doc(&mut rng, 16, 16);
        if vole_gfx::ir::validate::validate(&doc).is_err() {
            continue;
        }
        let t = 5_000_000;
        let Ok(full) = materialize_document(&doc, &surface(t, 16, 16, ColorFormat::Rgba8)) else {
            continue;
        };
        for ty in 0..4i32 {
            for tx in 0..4i32 {
                let (x0, y0) = (tx * 4, ty * 4);
                let req = ObservationRequest::region(t, x0, y0, 4, 4, ColorFormat::Rgba8);
                let tile = materialize_document(&doc, &req).unwrap();
                assert_eq!(tile.counters.samples, 16, "samples == requested area");
                for j in 0..4usize {
                    for i in 0..4usize {
                        let fo = (((y0 + j as i32) * 16 + (x0 + i as i32)) * 4) as usize;
                        let po = (j * 4 + i) * 4;
                        assert_eq!(
                            &full.output.data[fo..fo + 4],
                            &tile.output.data[po..po + 4],
                            "tile ({tx},{ty}) px ({i},{j})"
                        );
                    }
                }
            }
        }
        checked += 1;
    }
    assert!(checked >= 30, "expected >=30 checked docs, got {checked}");
}

#[test]
fn random_docs_irregular_domain_matches() {
    let mut rng = Rng::new(0xFEED_FACE_00C0_FFEE);
    let mut checked = 0u32;
    for _ in 0..60u32 {
        let doc = random_doc(&mut rng, 16, 16);
        if vole_gfx::ir::validate::validate(&doc).is_err() {
            continue;
        }
        let t = 777;
        let Ok(full) = materialize_document(&doc, &surface(t, 16, 16, ColorFormat::Gray8)) else {
            continue;
        };
        // sparse 5-sample request
        let mut pts = Vec::new();
        for _ in 0..5 {
            pts.push(((rng.below(16) as i32), (rng.below(16) as i32)));
        }
        let req = ObservationRequest {
            time_ns: t,
            view: Default::default(),
            domain: Domain::IrregularSamples(pts.clone()),
            sampling: Default::default(),
            format: ColorFormat::Gray8,
        };
        let m = materialize_document(&doc, &req).unwrap();
        assert_eq!(m.output.data.len(), 5);
        for (k, &(x, y)) in pts.iter().enumerate() {
            let fo = (y * 16 + x) as usize;
            assert_eq!(full.output.data[fo], m.output.data[k], "sparse sample {k}");
        }
        checked += 1;
    }
    assert!(checked >= 15, "expected >=15 checked docs, got {checked}");
}
