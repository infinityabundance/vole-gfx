//! Phase H tests: generator field objects (Γ(U, s, θ)).
//!
//! Covers: family parameter round-trips (canonical blob encode/decode),
//! semantic validation incl. referenced-object checks, exact materialization
//! parity across backends (scalar oracle / blocked / rayon / auto dispatch;
//! the constant-field fast path must be eligible and byte-identical), direct
//! observation semantics (region requests evaluate only requested samples —
//! the no-mandatory-re-baking boundary), scaled/clipped generator instances
//! on the scalar fallback, and adversarial parameter blobs (fail closed).

mod common;

use common::*;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::fixed::{Affine, IRect, RectF, fp};
use vole_gfx::ir::{Document, Event, Instance, Object, Op};
use vole_gfx::materialize::blocked::BlockMaterializer;
use vole_gfx::materialize::blocks::BlockShape;
use vole_gfx::materialize::dispatch::{self, rayon};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::{Domain, ObservationRequest};
use vole_gfx::procedural::build;
use vole_gfx::state::Scene;

fn all_backends_equal(doc: &Document, req: &ObservationRequest) {
    let t = req.time_ns;
    let scene = Scene::resolve(doc, t).unwrap();
    let oracle = materialize_document(doc, req).expect("scalar oracle");
    let oh = oracle.output.canonical_hash();

    let mut bm = BlockMaterializer::new(&scene, 64);
    let m = bm.materialize(req, BlockShape::W16H16).expect("blocked");
    assert_eq!(m.output.canonical_hash(), oh, "blocked parity");

    let m = rayon::materialize_parallel(&scene, req, BlockShape::W16H16).expect("rayon");
    assert_eq!(m.output.canonical_hash(), oh, "rayon parity");

    let (m, path) = dispatch::materialize_auto(&scene, req, BlockShape::W16H16).expect("auto");
    assert_eq!(m.output.canonical_hash(), oh, "auto parity");
    eprintln!("backend parity ok via {}", path.name());
}

/// Scene A: every family over a 192x192 surface (fractal + tiled + palette +
/// checker + gradient + disc + object-family + noise + constant, plus raster
/// sprites above, residual, and a trajectory-moved instance).
fn mixed_scene(time_ops: bool) -> Document {
    let mut d = Document::new();
    // 0: raster base sprite set (common helpers)
    d.objects.push(checker(
        8,
        8,
        Rgba::new(200, 30, 30, 255),
        Rgba::new(30, 30, 200, 255),
    ));
    // 1..11: generator fields
    d.objects
        .push(build::constant(192, 192, Rgba::new(9, 13, 17, 255)));
    d.objects.push(build::checker(
        192,
        192,
        8,
        8,
        Rgba::new(240, 240, 235, 255),
        Rgba::new(40, 44, 52, 255),
    ));
    d.objects.push(build::linear_gradient(
        192,
        192,
        0,
        0,
        Rgba::new(255, 0, 0, 255),
        191,
        191,
        Rgba::new(0, 0, 255, 255),
    ));
    d.objects.push(build::bilinear_gradient(
        192,
        192,
        Rgba::new(20, 20, 20, 255),
        Rgba::new(220, 20, 20, 255),
        Rgba::new(20, 220, 20, 255),
        Rgba::new(220, 220, 20, 255),
    ));
    d.objects.push(build::palette_bands(
        192,
        192,
        24,
        vec![
            Rgba::new(0, 0, 0, 255),
            Rgba::new(128, 0, 0, 255),
            Rgba::new(255, 255, 255, 255),
        ],
    ));
    d.objects.push(build::tiled(
        192,
        192,
        4,
        4,
        vec![
            Rgba::new(0, 0, 0, 255),
            Rgba::new(255, 0, 0, 255),
            Rgba::new(0, 255, 0, 255),
            Rgba::new(0, 0, 255, 255),
            Rgba::new(255, 255, 0, 255),
            Rgba::new(0, 255, 255, 255),
            Rgba::new(255, 0, 255, 255),
            Rgba::new(255, 255, 255, 255),
            Rgba::new(64, 64, 64, 255),
            Rgba::new(128, 0, 128, 255),
            Rgba::new(0, 128, 128, 255),
            Rgba::new(128, 128, 0, 255),
            Rgba::new(16, 16, 16, 255),
            Rgba::new(32, 32, 32, 255),
            Rgba::new(48, 48, 48, 255),
            Rgba::new(96, 96, 96, 255),
        ],
    ));
    d.objects.push(build::affine_reuse(
        48,
        48,
        0,
        Affine::identity(),
        Rgba::OPAQUE_BLACK,
    ));
    d.objects.push(build::noise_field(64, 64, 1234));
    d.objects.push(build::fractal(192, 192, 99, 12, 3, 49152));
    d.objects.push(build::disc(
        192,
        192,
        fp(96),
        fp(96),
        fp(60),
        Rgba::new(255, 255, 0, 255),
        Rgba::TRANSPARENT,
    ));
    d.objects.push(build::object_family(
        192,
        192,
        24,
        24,
        vec![0],
        Rgba::OPAQUE_BLACK,
    ));

    d.instances.push(placed(1, 1, 0, 0, 0)); // constant backdrop
    d.instances.push(placed(2, 2, 1, 0, 0)); // checker field
    d.instances.push(placed(6, 3, 2, 40, 90)); // affine reuse of sprite 0
    d.instances.push(placed(7, 4, 2, 130, 0)); // noise
    d.instances.push(placed(8, 5, 3, 0, 0)); // fractal over everything below
    // disc is transparent outside -> lets fractal show through (alpha path)
    d.instances.push(placed(9, 6, 4, 0, 0));
    // object-family grid over a corner
    d.instances.push(placed(10, 7, 5, 0, 0));
    // sprites on top
    d.instances.push(placed(0, 8, 6, 60, 20));
    d.events.push(Event {
        t: 0,
        ops: vec![Op::BindResidual(vole_gfx::ir::ResidualBind {
            algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
            region: IRect::new(10, 10, 12, 12),
            format: ColorFormat::Rgba8,
            payload: {
                let mut p = Vec::new();
                p.extend_from_slice(&1u64.to_le_bytes());
                vole_gfx::residual::push_record(&mut p, 10, 10, &[255, 0, 128, 255]);
                p
            },
        })],
    });
    if time_ops {
        // move the top sprite with a trajectory
        d.trajectories.push(vole_gfx::ir::Trajectory {
            kind: vole_gfx::ir::traj::LINEAR_TRANSLATION,
            keys: vec![
                vole_gfx::ir::TrajKey {
                    t: 0,
                    tx: fp(60),
                    ty: fp(20),
                },
                vole_gfx::ir::TrajKey {
                    t: 33_333_333,
                    tx: fp(100),
                    ty: fp(60),
                },
            ],
        });
        d.events.push(Event {
            t: 10,
            ops: vec![Op::InstTrajectory {
                instance: 8,
                trajectory: 1,
            }],
        });
    }
    d
}

#[test]
fn mixed_scene_validates_and_roundtrips() {
    let d = mixed_scene(false);
    vole_gfx::ir::validate::validate(&d).expect("mixed scene valid");
    let bytes = vole_gfx::ir::encode::encode(&d);
    vole_gfx::ir::canonical::check_canonical(&bytes).expect("canonical");
    let back = vole_gfx::ir::decode::decode(&bytes).expect("decode");
    assert_eq!(back, d);
}

#[test]
fn mixed_scene_all_backends_equal_rgba() {
    let d = mixed_scene(true);
    vole_gfx::ir::validate::validate(&d).unwrap();
    let req = ObservationRequest::full_surface(33_333_333, 192, 192, ColorFormat::Rgba8);
    all_backends_equal(&d, &req);
    // partial domains: tile + band + irregular sample set at later time
    let tile = ObservationRequest {
        time_ns: 16_000_000,
        view: vole_gfx::observation::View::identity(),
        domain: Domain::Tile {
            tile_x: 3,
            tile_y: 2,
            tile_w: 32,
            tile_h: 32,
        },
        sampling: vole_gfx::observation::SamplingProfile::exact_u1(),
        format: ColorFormat::Rgba8,
    };
    all_backends_equal(&d, &tile);
    let band = ObservationRequest {
        time_ns: 50_000_000,
        view: vole_gfx::observation::View::identity(),
        domain: Domain::DisplayBand {
            y0: 80,
            h: 40,
            x0: 0,
            x1: 192,
        },
        sampling: vole_gfx::observation::SamplingProfile::exact_u1(),
        format: ColorFormat::Rgba8,
    };
    all_backends_equal(&d, &band);
}

#[test]
fn mixed_scene_gray_and_irregular() {
    let d = mixed_scene(true);
    vole_gfx::ir::validate::validate(&d).unwrap();
    let gray = ObservationRequest::full_surface(25_000_000, 96, 96, ColorFormat::Gray8);
    all_backends_equal(&d, &gray);
    let irr = ObservationRequest {
        time_ns: 7_000_000,
        view: vole_gfx::observation::View::identity(),
        domain: Domain::IrregularSamples(vec![(0, 0), (191, 191), (95, 1), (3, 200), (200, 3)]),
        sampling: vole_gfx::observation::SamplingProfile::exact_u1(),
        format: ColorFormat::Rgba8,
    };
    all_backends_equal(&d, &irr);
}

/// Partial-domain requests must equal the crop of the full surface, and the
/// number of evaluated samples must track the requested domain (direct
/// observation; no whole-object re-baking).
#[test]
fn direct_observation_tracks_request_domain() {
    let mut d = Document::new();
    d.objects.push(build::fractal(512, 512, 7, 4, 4, 32768));
    d.instances.push(placed(0, 1, 0, 0, 0));
    vole_gfx::ir::validate::validate(&d).unwrap();

    let full = ObservationRequest::full_surface(0, 512, 512, ColorFormat::Rgba8);
    let fm = materialize_document(&d, &full).unwrap();

    // 1% region: 52x52 of 512x512 (approx 1%)
    let reg = ObservationRequest::region(0, 100, 100, 52, 52, ColorFormat::Rgba8);
    let rm = materialize_document(&d, &reg).unwrap();
    assert_eq!(rm.counters.samples, 52 * 52);
    // crop equality with the full-surface result
    for j in 0..52u32 {
        for i in 0..52u32 {
            let foff = (((100 + j as i32) as usize) * 512 + (100 + i as i32) as usize) * 4;
            let roff = ((j as usize) * 52 + i as usize) * 4;
            assert_eq!(
                &fm.output.data[foff..foff + 4],
                &rm.output.data[roff..roff + 4]
            );
        }
    }
    // a single sample must evaluate exactly one sample
    let one = ObservationRequest {
        time_ns: 0,
        view: vole_gfx::observation::View::identity(),
        domain: Domain::Sample { x: 300, y: 200 },
        sampling: vole_gfx::observation::SamplingProfile::exact_u1(),
        format: ColorFormat::Rgba8,
    };
    let om = materialize_document(&d, &one).unwrap();
    assert_eq!(om.counters.samples, 1);
    assert_eq!(om.output.data.len(), 4);
}

/// Constant generator fields are eligible for the fill-based fast path and
/// must stay byte-identical through scalar-simple / AVX2 / auto.
#[test]
fn constant_field_fast_path_parity() {
    let mut d = Document::new();
    d.objects
        .push(build::constant(256, 256, Rgba::new(30, 60, 90, 255)));
    d.objects
        .push(build::constant(128, 128, Rgba::new(255, 200, 0, 255)));
    d.instances.push(placed(0, 1, 0, 0, 0));
    d.instances.push(placed(1, 2, 1, 200, 200));
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(64, 64, 192, 192),
            color: Rgba::new(5, 5, 5, 255),
        }],
    });
    vole_gfx::ir::validate::validate(&d).unwrap();
    let req = ObservationRequest::full_surface(0, 400, 400, ColorFormat::Rgba8);
    let scene = Scene::resolve(&d, 0).unwrap();
    let oracle = materialize_document(&d, &req).unwrap();
    let oh = oracle.output.canonical_hash();

    // scalar-simple (no SIMD) engine must handle constant fills
    let m =
        vole_gfx::materialize::simple::materialize_simple_scalar(&scene, &req, BlockShape::W16H16)
            .expect("eligible")
            .expect("some output");
    assert_eq!(m.output.canonical_hash(), oh, "scalar-simple parity");

    // auto dispatch: AVX2 fills; still byte-identical
    let (m, path) = dispatch::materialize_auto(&scene, &req, BlockShape::W16H16).unwrap();
    assert_eq!(m.output.canonical_hash(), oh, "auto parity");
    assert_eq!(
        path,
        dispatch::Path::Avx2,
        "constant fields must dispatch to avx2"
    );
}

/// Scaled / clipped / translucent generator instances are outside the fast
/// path but must fall back to exact per-sample evaluation.
#[test]
fn transformed_generator_instances_fall_back_exact() {
    let mut d = Document::new();
    d.objects.push(build::checker(
        16,
        16,
        4,
        4,
        Rgba::new(0, 0, 0, 255),
        Rgba::WHITE,
    ));
    // scaled placement: identity-linear is broken -> ineligible
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::uniform_scale(fp(2)),
        palette: None,
        clip: Some(RectF::from_px(8, 8, 24, 24)),
        trajectory: 0,
        visible: true,
    });
    vole_gfx::ir::validate::validate(&d).unwrap();
    let req = ObservationRequest::full_surface(0, 48, 48, ColorFormat::Rgba8);
    let scene = Scene::resolve(&d, 0).unwrap();
    let oracle = materialize_document(&d, &req).unwrap();
    let (m, path) = dispatch::materialize_auto(&scene, &req, BlockShape::W16H16).unwrap();
    assert_eq!(m.output.data, oracle.output.data);
    assert_eq!(path, dispatch::Path::Blocked);
}

/// Raster equivalent of a checker field must materialize identically when
/// both are placed 1:1 (the inverse compiler's identity-check seed).
#[test]
fn field_matches_raster_equivalent() {
    let raster = checker(
        16,
        16,
        Rgba::new(10, 10, 10, 255),
        Rgba::new(250, 250, 250, 255),
    );
    let field = build::checker(
        16,
        16,
        1,
        1,
        Rgba::new(10, 10, 10, 255),
        Rgba::new(250, 250, 250, 255),
    );

    let mut dr = Document::new();
    dr.objects.push(raster);
    dr.instances.push(placed(0, 1, 0, 0, 0));

    let mut df = Document::new();
    df.objects.push(field);
    df.instances.push(placed(0, 1, 0, 0, 0));

    let req = ObservationRequest::full_surface(0, 16, 16, ColorFormat::Rgba8);
    let a = materialize_document(&dr, &req).unwrap();
    let b = materialize_document(&df, &req).unwrap();
    assert_eq!(a.output.data, b.output.data);
}

/// Validation of generator objects: unknown family, malformed params,
/// extent-relative violations, invalid references.
#[test]
fn adversarial_params_fail_closed() {
    use vole_gfx::limits::Reject;

    // unknown family tag
    let d = Document::new();
    let bad = Document {
        objects: vec![Object::GeneratorField {
            family: 200,
            w: 8,
            h: 8,
            params: vec![],
        }],
        ..d
    };
    assert_eq!(
        vole_gfx::ir::validate::validate(&bad),
        Err(Reject::UnknownTag)
    );

    // valid family, truncated params
    let mut bad2 = Document::new();
    bad2.objects.push(Object::GeneratorField {
        family: vole_gfx::procedural::family::CONSTANT,
        w: 8,
        h: 8,
        params: vec![0x12], // needs 4 bytes
    });
    assert!(vole_gfx::ir::validate::validate(&bad2).is_err());

    // oversize param blob (over the 64 KiB cap)
    let mut bad3 = Document::new();
    bad3.objects.push(Object::GeneratorField {
        family: vole_gfx::procedural::family::CONSTANT,
        w: 8,
        h: 8,
        params: vec![0; (vole_gfx::limits::MAX_GENERATOR_PARAMS + 1) as usize],
    });
    assert_eq!(
        vole_gfx::ir::validate::validate(&bad3),
        Err(Reject::BytesExceedsLimit)
    );

    // periodic: stripe-x with py != 1 is non-canonical
    let mut blob = Vec::new();
    blob.push(vole_gfx::procedural::families::periodic::MODE_STRIPE_X);
    blob.extend_from_slice(&2u32.to_le_bytes());
    blob.extend_from_slice(&3u32.to_le_bytes()); // py must be 1 for stripe-x
    blob.extend_from_slice(&Rgba::OPAQUE_BLACK.to_bytes());
    blob.extend_from_slice(&Rgba::WHITE.to_bytes());
    let mut bad4 = Document::new();
    bad4.objects.push(Object::GeneratorField {
        family: vole_gfx::procedural::family::PERIODIC,
        w: 8,
        h: 8,
        params: blob,
    });
    assert_eq!(
        vole_gfx::ir::validate::validate(&bad4),
        Err(Reject::NonCanonicalOrder)
    );

    // affine reuse referencing a palette (non-content) object
    let mut bad5 = Document::new();
    bad5.objects.push(Object::Palette {
        entries: vec![Rgba::WHITE],
    });
    let mut blob5 = Vec::new();
    blob5.extend_from_slice(&0u32.to_le_bytes()); // object 0 = palette
    let id = Affine::identity();
    for v in [id.a, id.b, id.tx, id.c, id.d, id.ty] {
        blob5.extend_from_slice(&v.to_le_bytes());
    }
    blob5.extend_from_slice(&Rgba::OPAQUE_BLACK.to_bytes());
    bad5.objects.push(Object::GeneratorField {
        family: vole_gfx::procedural::family::AFFINE_REUSE,
        w: 8,
        h: 8,
        params: blob5,
    });
    assert_eq!(
        vole_gfx::ir::validate::validate(&bad5),
        Err(Reject::InvalidIndex)
    );

    // affine reuse referencing a missing object
    let mut bad6 = Document::new();
    let mut blob6 = Vec::new();
    blob6.extend_from_slice(&7u32.to_le_bytes());
    let id = Affine::identity();
    for v in [id.a, id.b, id.tx, id.c, id.d, id.ty] {
        blob6.extend_from_slice(&v.to_le_bytes());
    }
    blob6.extend_from_slice(&Rgba::OPAQUE_BLACK.to_bytes());
    bad6.objects.push(Object::GeneratorField {
        family: vole_gfx::procedural::family::AFFINE_REUSE,
        w: 8,
        h: 8,
        params: blob6,
    });
    assert_eq!(
        vole_gfx::ir::validate::validate(&bad6),
        Err(Reject::InvalidIndex)
    );

    // decoder must reject an oversized declared generator payload before
    // allocation (hostile length field)
    let mut hdr = Vec::new();
    hdr.extend_from_slice(vole_gfx::limits::MAGIC);
    hdr.extend_from_slice(&vole_gfx::limits::FORMAT_VERSION.to_le_bytes());
    hdr.extend_from_slice(&(vole_gfx::limits::UNIVERSE_U1.len() as u32).to_le_bytes());
    hdr.extend_from_slice(vole_gfx::limits::UNIVERSE_U1.as_bytes());
    hdr.push(1); // exact profile
    let mut body = Vec::new();
    body.extend_from_slice(&1u64.to_le_bytes()); // one object
    body.push(vole_gfx::ir::kind::GENERATOR_FIELD);
    let mut payload = Vec::new();
    payload.push(vole_gfx::procedural::family::CONSTANT);
    payload.extend_from_slice(&8u32.to_le_bytes());
    payload.extend_from_slice(&8u32.to_le_bytes());
    payload.extend_from_slice(&(vole_gfx::limits::MAX_GENERATOR_PARAMS as u32 + 1).to_le_bytes());
    body.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    body.extend_from_slice(&payload);
    hdr.extend_from_slice(&(body.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&body);
    assert_eq!(
        vole_gfx::ir::decode::decode(&hdr),
        Err(Reject::BytesExceedsLimit)
    );
}

/// Work estimators are bounded and reported per family (scheduler input).
#[test]
fn work_estimates_bounded() {
    let d = mixed_scene(false);
    let scene = Scene::resolve(&d, 0).unwrap();
    for o in 0..d.objects.len() as u32 {
        if let Some(f) = scene.field_of(o) {
            let w = f.work();
            assert!(w >= 1, "family {} must cost >= 1", f.family_tag());
            assert!(
                w <= vole_gfx::limits::MAX_GENERATOR_WORK_PER_SAMPLE,
                "family {} work {w} over cap",
                f.family_tag()
            );
        }
    }
}

/// Pinned conformance hashes for generator-field scenes (Phase H): the mixed
/// ten-family scene at 33.3 ms (RGBA + Gray8) and at t=0.  Any semantic or
/// wire change to the generator families breaks these pins.
#[test]
fn pinned_procedural_conformance_hashes() {
    let d = mixed_scene(true);
    let rgba = ObservationRequest::full_surface(33_333_333, 192, 192, ColorFormat::Rgba8);
    let gray = ObservationRequest::full_surface(33_333_333, 192, 192, ColorFormat::Gray8);
    let t0 = ObservationRequest::full_surface(0, 192, 192, ColorFormat::Rgba8);
    let h1 = materialize_document(&d, &rgba)
        .unwrap()
        .output
        .canonical_hash()
        .to_hex();
    let h2 = materialize_document(&d, &gray)
        .unwrap()
        .output
        .canonical_hash()
        .to_hex();
    let h3 = materialize_document(&d, &t0)
        .unwrap()
        .output
        .canonical_hash()
        .to_hex();
    eprintln!("PH_H_RGBA_33MS={h1}");
    eprintln!("PH_H_GRAY_33MS={h2}");
    eprintln!("PH_H_RGBA_T0={h3}");
    assert_eq!(
        h1,
        "28963c5d69e7338e78bf221685225c47e7a9026039efeddb3cf57d31f419b67f"
    );
    assert_eq!(
        h2,
        "9fa141db13a74acb97a95416fe07a5ce90b18bff35d1df47a2a8404acd85398e"
    );
    assert_eq!(
        h3,
        "e40cac63ee2f4bfff860b90391afe4e23e9f7ed2c87a487951de19f0f4132b49"
    );
}
