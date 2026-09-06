//! Differential backend tests: every exact backend must reproduce the scalar
//! oracle byte-for-byte (canonical output hashes equal), across block shapes,
//! odd widths, unaligned request beginnings, tiny regions, and clipping.

mod common;

use common::*;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::fixed::{Affine, RectF, fp};
use vole_gfx::ir::{Document, Event, Instance, Op};
use vole_gfx::materialize::blocked::BlockMaterializer;
use vole_gfx::materialize::blocks::BlockShape;
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::{Domain, ObservationRequest};
use vole_gfx::state::Scene;

fn valid(doc: &Document) -> bool {
    vole_gfx::ir::validate::validate(doc).is_ok()
}

/// Sparse-ish scene: many small instances scattered over 320x180.
fn scattered_scene(seed: u64, count: usize) -> Document {
    let mut rng = Rng::new(seed);
    let mut d = Document::new();
    let mut objs = Vec::new();
    for _ in 0..6 {
        objs.push(checker(
            1 + rng.below(8),
            1 + rng.below(8),
            Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
            Rgba::new(rng.u8(), rng.u8(), rng.u8(), 255),
        ));
    }
    d.objects = objs;
    for o in 0..count as u32 {
        let obj = o % 6;
        d.instances.push(placed(
            obj,
            o,
            rng.below(4),
            (rng.below(320) as i32) - 8,
            (rng.below(180) as i32) - 8,
        ));
    }
    d.events.push(Event {
        t: 1,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(-8, -8, 328, 188),
            color: Rgba::new(24, 24, 24, 255),
        }],
    });
    d
}

fn assert_blocked_equals_scalar(
    doc: &Document,
    t: u64,
    req: &ObservationRequest,
    shape: BlockShape,
) {
    let scene = Scene::resolve(doc, t).expect("resolve");
    let scalar = materialize_document(doc, req).expect("scalar");
    let mut blocked = BlockMaterializer::new(&scene, 64);
    let b = blocked.materialize(req, shape).expect("blocked");
    if scalar.output.data != b.output.data {
        for i in 0..scalar.output.data.len() {
            if scalar.output.data[i] != b.output.data[i] {
                let w = req.domain.bounds().width() as usize;
                let px = i / 4;
                eprintln!(
                    "DBG shape {} first diff byte {i} pixel {},{}, scalar {:?} blocked {:?}",
                    shape.name(),
                    px % w,
                    px / w,
                    &scalar.output.data[i..i + 4],
                    &b.output.data[i..i + 4]
                );
                break;
            }
        }
    }
    assert_eq!(
        scalar.output.canonical_hash(),
        b.output.canonical_hash(),
        "hash mismatch for shape {}",
        shape.name()
    );
    assert_eq!(
        scalar.output.data,
        b.output.data,
        "bytes differ for shape {}",
        shape.name()
    );
}

#[test]
fn block_shapes_match_scalar_on_scenes() {
    let docs = [
        scene_v1(),
        scene_v2(),
        scene_v4(),
        scene_v5(),
        scattered_scene(99, 40),
        scattered_scene(1234, 120),
    ];
    let shapes = [
        BlockShape::W8H8,
        BlockShape::W16H8,
        BlockShape::W16H16,
        BlockShape::W32H8,
        BlockShape::ScanlineSegment(64),
    ];
    for doc in &docs {
        if valid(doc) {
            for shape in shapes {
                let req = ObservationRequest::full_surface(0, 320, 180, ColorFormat::Rgba8);
                assert_blocked_equals_scalar(doc, 0, &req, shape);
            }
        }
    }
}

#[test]
fn partial_domains_and_odd_shapes_match() {
    let doc = scattered_scene(7, 60);
    let t = 0;
    let shapes = [BlockShape::W16H8, BlockShape::W32H16];
    // odd width, odd height, unaligned origin, negative origin, tiny
    let domains = [
        Domain::Rectangle {
            x0: 1,
            y0: 1,
            w: 63,
            h: 47,
        },
        Domain::Rectangle {
            x0: 0,
            y0: 0,
            w: 3,
            h: 3,
        },
        Domain::Rectangle {
            x0: -5,
            y0: 3,
            w: 21,
            h: 9,
        },
        Domain::DisplayBand {
            y0: 10,
            h: 2,
            x0: 0,
            x1: 320,
        },
        Domain::Scanline {
            y: 0,
            x0: 4,
            x1: 100,
        },
        Domain::Sample { x: 40, y: 40 },
        Domain::SampleSpan { x: 10, y: 10, n: 7 },
        Domain::FullSurface { w: 320, h: 180 },
    ];
    for dom in domains {
        let req = ObservationRequest {
            time_ns: t,
            view: Default::default(),
            domain: dom,
            sampling: Default::default(),
            format: ColorFormat::Rgba8,
        };
        for shape in shapes {
            assert_blocked_equals_scalar(&doc, t, &req, shape);
        }
    }
}

#[test]
fn irregular_domains_match() {
    let doc = scattered_scene(31, 50);
    let t = 0;
    let mut pts = Vec::new();
    for k in 0..40u32 {
        pts.push(((k * 7 % 320) as i32, (k * 13 % 180) as i32));
    }
    let req = ObservationRequest {
        time_ns: t,
        view: Default::default(),
        domain: Domain::IrregularSamples(pts),
        sampling: Default::default(),
        format: ColorFormat::Gray8,
    };
    let scene = Scene::resolve(&doc, t).unwrap();
    let scalar = materialize_document(&doc, &req).unwrap();
    let mut blocked = BlockMaterializer::new(&scene, 64);
    let b = blocked.materialize(&req, BlockShape::W16H16).unwrap();
    assert_eq!(scalar.output.data, b.output.data);
}

#[test]
fn trajectory_times_match() {
    let doc = scene_v3();
    let t = 12_345_678;
    let req = ObservationRequest::full_surface(t, 64, 64, ColorFormat::Rgba8);
    assert_blocked_equals_scalar(&doc, t, &req, BlockShape::W16H16);
    let doc = scattered_scene(5, 30);
    for t in [0u64, 16_666_667, 123_456_789] {
        let req = ObservationRequest::full_surface(t, 128, 64, ColorFormat::Rgba8);
        assert_blocked_equals_scalar(&doc, t, &req, BlockShape::W16H8);
    }
}

#[test]
fn scaled_and_clipped_instances_match() {
    // scaling via fixed affine + palette override + clips
    let mut d = Document::new();
    d.objects.push(checker(
        8,
        8,
        Rgba::new(9, 250, 40, 255),
        Rgba::new(200, 30, 250, 255),
    ));
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine {
            a: fp(2),
            b: 0,
            tx: fp(8),
            c: 0,
            d: fp(3),
            ty: fp(2),
        },
        palette: None,
        clip: Some(RectF::from_px(8, 2, 20, 12)),
        trajectory: 0,
        visible: true,
    });
    let t = 0;
    for shape in [BlockShape::W16H16, BlockShape::W8H8] {
        let req = ObservationRequest::full_surface(0, 32, 24, ColorFormat::Rgba8);
        assert_blocked_equals_scalar(&d, t, &req, shape);
    }
}

#[test]
fn mirror_transform_matches() {
    // negative-scale (mirror) affine: det < 0 path
    let mut d = Document::new();
    d.objects.push(checker(
        4,
        4,
        Rgba::new(255, 200, 0, 255),
        Rgba::new(0, 40, 255, 255),
    ));
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine {
            a: -fp(1),
            b: 0,
            tx: fp(20),
            c: 0,
            d: fp(1),
            ty: fp(4),
        },
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    let t = 0;
    let req = ObservationRequest::full_surface(0, 32, 16, ColorFormat::Rgba8);
    assert_blocked_equals_scalar(&d, t, &req, BlockShape::W16H16);
}

#[test]
fn dense_scene_still_matches() {
    // everything overlapping: candidate culling must still be exact
    let mut rng = Rng::new(0xDA7A);
    let mut d = Document::new();
    d.objects
        .push(checker(3, 3, Rgba::WHITE, Rgba::OPAQUE_BLACK));
    for o in 0..50u32 {
        d.instances.push(placed(0, o, 0, 5, 5));
    }
    let _ = &mut rng;
    let req = ObservationRequest::full_surface(0, 16, 16, ColorFormat::Rgba8);
    assert_blocked_equals_scalar(&d, 0, &req, BlockShape::W8H8);
}
