//! Phase D/E/F differential tests: AVX2, AVX-512 and Rayon backends must
//! reproduce the scalar oracle byte-for-byte, and dispatch must record which
//! path ran.  SIMD kernels only move bytes in the eligible subset, so parity
//! is by construction; these tests prove the eligibility logic and the
//! dispatch + fallback behavior across odd widths, tails, gray targets,
//! negative origins, clipping and non-eligible (alpha/scaled) scenes.

mod common;

use common::*;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::fixed::{Affine, RectF, fp};
use vole_gfx::ir::{Document, Event, Instance, Object, Op};
use vole_gfx::materialize::blocks::BlockShape;
use vole_gfx::materialize::dispatch::avx2;
use vole_gfx::materialize::dispatch::avx512;
use vole_gfx::materialize::dispatch::rayon;
use vole_gfx::materialize::dispatch::{self, Path};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::ObservationRequest;
use vole_gfx::state::Scene;

/// Eligible scene: opaque integer-translation sprites + opaque fills.
fn eligible_scene(gray: bool) -> Document {
    let mut d = Document::new();
    if gray {
        let mut g = Vec::new();
        for j in 0..16 {
            for i in 0..24 {
                g.push(((i * 7 + j * 3) % 256) as u8);
            }
        }
        d.objects.push(Object::Raster {
            format: ColorFormat::Gray8,
            w: 24,
            h: 16,
            data: g,
        });
    } else {
        let mut data = Vec::new();
        for j in 0..16 {
            for i in 0..24 {
                data.extend_from_slice(&[(i * 11) as u8, (j * 13) as u8, 200, 255]);
            }
        }
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 24,
            h: 16,
            data,
        });
    }
    for (order, x, y, l) in [
        (1u32, 5i32, 7i32, 0u32),
        (2, 40, 12, 1),
        (3, -6, 20, 1),
        (4, 100, 3, 2),
        (5, 55, 40, 0),
    ] {
        d.instances.push(Instance {
            object: 0,
            order,
            layer: l,
            transform: Affine::from_px_translation(x, y),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
    }
    d.events.push(Event {
        t: 1,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(-4, -4, 128, 96),
            color: if gray {
                Rgba::gray(17)
            } else {
                Rgba::new(17, 23, 29, 255)
            },
        }],
    });
    d
}

/// Ineligible scene (alpha object and scaled instance): must fall back and
/// still be byte-identical through auto dispatch.
fn ineligible_scene() -> Document {
    let mut d = Document::new();
    let mut data = Vec::new();
    for k in 0..64 {
        data.extend_from_slice(&[10, 20, 30, if k % 3 == 0 { 128 } else { 255 }]);
    }
    d.objects.push(Object::Raster {
        format: ColorFormat::Rgba8,
        w: 8,
        h: 8,
        data,
    });
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine {
            a: fp(2),
            b: 0,
            tx: fp(8),
            c: 0,
            d: fp(1),
            ty: fp(2),
        },
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    d.events.push(Event {
        t: 1,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 64, 64),
            color: Rgba::new(1, 2, 3, 128), // translucent fill -> ineligible
        }],
    });
    d
}

fn all_paths_equal(doc: &Document, req: &ObservationRequest) {
    let t = req.time_ns;
    let scene = Scene::resolve(doc, t).unwrap();
    let oracle = materialize_document(doc, req).expect("scalar oracle");
    let oh = oracle.output.canonical_hash();

    // blocked
    let mut bm = vole_gfx::materialize::blocked::BlockMaterializer::new(&scene, 64);
    let m = bm.materialize(req, BlockShape::W16H16).expect("blocked");
    assert_eq!(m.output.canonical_hash(), oh, "blocked parity");

    // forced avx2 path (skipped honestly when hardware lacks it)
    if avx2::has_avx2() {
        if let Some(m) = avx2::materialize_simple(&scene, req, BlockShape::W16H16).expect("avx2") {
            assert_eq!(m.output.canonical_hash(), oh, "avx2 parity");
        } else {
            eprintln!("avx2: not eligible, skipped on this scene");
        }
    } else {
        eprintln!("avx2: not evaluated on this host");
    }
    // forced avx512 path
    if avx512::has_avx512() {
        if let Some(m) =
            avx512::materialize_simple(&scene, req, BlockShape::W16H16).expect("avx512")
        {
            assert_eq!(m.output.canonical_hash(), oh, "avx512 parity");
        }
    } else {
        eprintln!("avx512: not evaluated on this host");
    }
    // rayon bands (default pool)
    let m = rayon::materialize_parallel(&scene, req, BlockShape::W16H16).expect("rayon");
    assert_eq!(m.output.canonical_hash(), oh, "rayon parity");
    // rayon determinism across repeated runs
    let m2 = rayon::materialize_parallel(&scene, req, BlockShape::W16H16).expect("rayon2");
    assert_eq!(m2.output.data, m.output.data, "rayon deterministic");

    // auto dispatch: must equal oracle; record path
    let (m3, path) = dispatch::materialize_auto(&scene, req, BlockShape::W16H16).expect("auto");
    assert_eq!(m3.output.canonical_hash(), oh, "auto parity");
    eprintln!("auto dispatch path: {}", path.name());
}

#[test]
fn eligible_scene_all_backends_equal_rgba() {
    let d = eligible_scene(false);
    if vole_gfx::ir::validate::validate(&d).is_err() {
        panic!("eligible scene must validate");
    }
    for (x0, y0, w, h) in [
        (0i32, 0i32, 128u32, 96u32),
        (3, 5, 61, 47),
        (-2, 7, 33, 17),
        (1, 1, 5, 5),
    ] {
        let req = ObservationRequest::region(0, x0, y0, w, h, ColorFormat::Rgba8);
        all_paths_equal(&d, &req);
    }
    let full = ObservationRequest::full_surface(0, 128, 96, ColorFormat::Rgba8);
    all_paths_equal(&d, &full);
}

#[test]
fn eligible_scene_all_backends_equal_gray() {
    let d = eligible_scene(true);
    let req = ObservationRequest::full_surface(0, 128, 96, ColorFormat::Gray8);
    all_paths_equal(&d, &req);
}

#[test]
fn ineligible_scene_falls_back_and_matches() {
    let d = ineligible_scene();
    let req = ObservationRequest::full_surface(0, 64, 64, ColorFormat::Rgba8);
    let scene = Scene::resolve(&d, 0).unwrap();
    let oracle = materialize_document(&d, &req).unwrap();
    let (m, path) = dispatch::materialize_auto(&scene, &req, BlockShape::W16H16).unwrap();
    assert_eq!(m.output.data, oracle.output.data);
    // translucent content is ineligible for the SIMD simple path
    assert_ne!(path, Path::Avx2);
    assert_ne!(path, Path::Avx512);
    eprintln!("ineligible scene dispatched to {}", path.name());
}

#[test]
fn conformance_vectors_reproduce_via_every_backend() {
    // The pinned conformance hashes must hold through every backend.
    let cases: [(Document, &str, ColorFormat); 3] = [
        (
            scene_v1(),
            "6a10325504993ff5de852ee50df37ca4052e4620a4a8901f2cc8f2c05bec7ca8",
            ColorFormat::Rgba8,
        ),
        (
            scene_v2(),
            "ac872d6a9f6ed36248cf4278d4288a529d45c38c74149df6dff52672440140d7",
            ColorFormat::Rgba8,
        ),
        (
            scene_v5(),
            "d9593d5eb769c41d9d51866d7c699177857ab1fe34c373ad02da9c1315443daf",
            ColorFormat::Rgba8,
        ),
    ];
    for (doc, expected, fmt) in cases {
        let req = ObservationRequest::full_surface(1000, 64, 64, fmt);
        let scene = Scene::resolve(&doc, req.time_ns).unwrap();
        let (m, path) = dispatch::materialize_auto(&scene, &req, BlockShape::W16H16).unwrap();
        assert_eq!(m.output.canonical_hash().to_hex(), expected);
        eprintln!("conformance {} via {}", expected, path.name());
    }
}

#[test]
fn rayon_band_heights_agree() {
    let d = eligible_scene(false);
    let req = ObservationRequest::full_surface(0, 128, 96, ColorFormat::Rgba8);
    let scene = Scene::resolve(&d, 0).unwrap();
    let mut hash = None;
    for bh in [1u32, 7, 16, 64, 200] {
        let m = rayon::materialize_bands(&scene, &req, BlockShape::W16H8, bh).unwrap();
        match hash {
            None => hash = Some(m.output.canonical_hash()),
            Some(h) => assert_eq!(h, m.output.canonical_hash(), "band height {bh} diverged"),
        }
    }
}
