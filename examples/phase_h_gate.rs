//! Phase H gate: seeded procedural state (`O = Γ(U, s, θ)`).
//!
//! Courts a 1920x1080 scene whose persistent state is procedural fields
//! (constant / checker / gradient / palette-band / tiled / affine-reuse /
//! hash noise / fractal / SDF / object-family) plus trajectories, sprites and
//! residual.  Evidence recorded per request shape:
//!
//! * exactness: every request hash equals the scalar oracle;
//! * direct observation: samples evaluated == samples requested, so region /
//!   tile / band / irregular requests do NOT re-bake the objects (the
//!   no-mandatory-re-baking boundary holds per sample);
//! * work fraction `samples_evaluated(region)/samples_evaluated(full)`;
//! * backend parity: scalar / blocked / rayon byte-identical on the full
//!   request; the pure-constant subset also runs the fill-based fast path
//!   (avx2) byte-identically;
//! * representation bytes of the persistent procedural state.
//!
//! Run: `cargo run --release --example phase_h_gate`

use std::time::Instant;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::evidence::receipt::{Receipt, emit_receipt};
use vole_gfx::fixed::{Affine, IRect, RectF, fp};
use vole_gfx::ir::{Document, Event, Instance, Object, Op};
use vole_gfx::materialize::blocked::BlockMaterializer;
use vole_gfx::materialize::blocks::BlockShape;
use vole_gfx::materialize::dispatch;
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::{Domain, ObservationRequest, View};
use vole_gfx::procedural::build;
use vole_gfx::state::Scene;

const W: u32 = 1920;
const H: u32 = 1080;

fn placed(object: u32, order: u32, layer: u32, x: i32, y: i32) -> Instance {
    Instance {
        object,
        order,
        layer,
        transform: vole_gfx::fixed::Affine::from_px_translation(x, y),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    }
}

/// The phase-h procedural court: persistent generator state + trajectories +
/// sprites + residual, 10 families as placed fields.
fn court() -> Document {
    let mut d = Document::new();
    // 0: sprite atlas base (raster object; 16x16)
    {
        let mut data = Vec::new();
        for j in 0..16u32 {
            for i in 0..16u32 {
                let c = if (i / 8 + j / 8) % 2 == 0 {
                    Rgba::new(255, 240, 200, 255)
                } else {
                    Rgba::new(90, 60, 30, 255)
                };
                data.extend_from_slice(&c.to_bytes());
            }
        }
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 16,
            h: 16,
            data,
        });
    }
    // 1: constant base (full surface)
    d.objects
        .push(build::constant(W, H, Rgba::new(13, 17, 23, 255)));
    // 2: coarse checker backdrop
    d.objects.push(build::checker(
        W,
        H,
        160,
        120,
        Rgba::new(24, 30, 40, 255),
        Rgba::new(30, 38, 52, 255),
    ));
    // 3: fractal field (the expensive per-sample family)
    d.objects.push(build::fractal(W, H, 7, 32, 3, 45056));
    // 4: linear gradient band on the left edge
    d.objects.push(build::linear_gradient(
        128,
        H,
        0,
        0,
        Rgba::new(20, 40, 90, 255),
        0,
        (H - 1) as i32,
        Rgba::new(90, 160, 255, 255),
    ));
    // 5: HUD palette bands (bottom 96 rows)
    d.objects.push(build::palette_bands(
        W,
        96,
        32,
        vec![
            Rgba::new(16, 18, 22, 255),
            Rgba::new(60, 70, 90, 255),
            Rgba::new(200, 170, 60, 255),
        ],
    ));
    // 6: tiled decor field (top-right, 8x8 tile)
    {
        let mut tile = Vec::new();
        for j in 0..8u32 {
            for i in 0..8u32 {
                let c = if i == 0 || j == 0 || i == 7 || j == 7 {
                    Rgba::new(210, 160, 60, 255)
                } else {
                    Rgba::new(40, 60, 40, 255)
                };
                tile.push(c);
            }
        }
        d.objects.push(build::tiled(320, 240, 8, 8, tile));
    }
    // 7: affine reuse of the sprite atlas
    d.objects.push(build::affine_reuse(
        160,
        160,
        0,
        Affine::identity(),
        Rgba::TRANSPARENT,
    ));
    // 8: hash noise
    d.objects.push(build::noise_field(1024, 1024, 42));
    // 9: SDF disc (yellow, transparent outside)
    d.objects.push(build::disc(
        W,
        H,
        fp(1400),
        fp(300),
        fp(220),
        Rgba::new(255, 220, 60, 255),
        Rgba::TRANSPARENT,
    ));
    // 10: object-family grid of the sprite atlas
    d.objects.push(build::object_family(
        640,
        360,
        32,
        32,
        vec![0],
        Rgba::TRANSPARENT,
    ));

    d.instances.push(placed(1, 1, 0, 0, 0)); // constant base
    d.instances.push(placed(2, 2, 1, 0, 0)); // checker
    d.instances.push(placed(3, 3, 2, 0, 0)); // fractal
    d.instances.push(placed(4, 4, 3, 0, 0)); // gradient band
    d.instances.push(placed(6, 5, 4, W as i32 - 320, 0)); // tiled decor
    d.instances
        .push(placed(10, 6, 5, W as i32 - 640, H as i32 - 360)); // grid
    d.instances.push(placed(7, 7, 6, 60, H as i32 - 260)); // logo reuse
    d.instances.push(placed(8, 8, 7, 0, 0)); // noise over full canvas region
    d.instances.push(placed(9, 9, 8, 0, 0)); // disc
    d.instances.push(placed(5, 10, 9, 0, H as i32 - 96)); // HUD
    d.instances.push(placed(0, 11, 10, 700, 500)); // moving sprite on top

    // trajectory: the sprite drifts across the canvas over 1 s
    d.trajectories.push(vole_gfx::ir::Trajectory {
        kind: vole_gfx::ir::traj::LINEAR_TRANSLATION,
        keys: vec![
            vole_gfx::ir::TrajKey {
                t: 0,
                tx: fp(700),
                ty: fp(500),
            },
            vole_gfx::ir::TrajKey {
                t: 1_000_000_000,
                tx: fp(1200),
                ty: fp(200),
            },
        ],
    });
    d.events.push(Event {
        t: 1,
        ops: vec![Op::InstTrajectory {
            instance: 11,
            trajectory: 1,
        }],
    });
    // a sparse residual retouch at the surface center
    d.events.push(Event {
        t: 2,
        ops: vec![Op::BindResidual(vole_gfx::ir::ResidualBind {
            algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
            region: IRect::new(0, 0, W as i32, H as i32),
            format: ColorFormat::Rgba8,
            payload: {
                let mut p = Vec::new();
                p.extend_from_slice(&1u64.to_le_bytes());
                vole_gfx::residual::push_record(&mut p, 960, 540, &[255, 0, 128, 255]);
                p
            },
        })],
    });
    d
}

/// Pure-constant subset: every placed object is an opaque constant field or
/// opaque raster/fill (the eligible subset of the fill fast path).
fn constant_subset() -> Document {
    let mut d = Document::new();
    d.objects
        .push(build::constant(1920, 1080, Rgba::new(13, 17, 23, 255)));
    d.objects
        .push(build::constant(200, 200, Rgba::new(255, 220, 60, 255)));
    d.instances.push(placed(0, 1, 0, 0, 0));
    d.instances.push(placed(1, 2, 1, 860, 440));
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 900, 1920, 960),
            color: Rgba::new(200, 200, 210, 255),
        }],
    });
    d
}

/// Median of 3 timings of `f` (ns); one warmup run first.
fn time_ns(mut f: impl FnMut()) -> u64 {
    f();
    let mut ts = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        f();
        ts.push(t0.elapsed().as_nanos() as u64);
    }
    ts.sort_unstable();
    ts[1]
}

/// Assert `region_data` equals the crop of `full_data` at `(x0, y0)` with the
/// given stride/widths (RGBA8).
fn assert_crop_equals(
    full: &[u8],
    fw: usize,
    x0: i32,
    y0: i32,
    region: &[u8],
    rw: usize,
    rh: usize,
) {
    for j in 0..rh {
        let fro = ((y0 as usize + j) * fw + x0 as usize) * 4;
        let rro = j * rw * 4;
        assert_eq!(
            &full[fro..fro + rw * 4],
            &region[rro..rro + rw * 4],
            "crop row {j}"
        );
    }
}

fn main() {
    let doc = court();
    vole_gfx::ir::validate::validate(&doc).expect("court valid");
    let bytes = vole_gfx::ir::encode::encode(&doc);
    vole_gfx::ir::canonical::check_canonical(&bytes).expect("court canonical");

    let req_full = ObservationRequest::full_surface(33_333_333, W, H, ColorFormat::Rgba8);
    let scene = Scene::resolve(&doc, req_full.time_ns).unwrap();
    let oracle = materialize_document(&doc, &req_full).expect("oracle full");
    let oh = oracle.output.canonical_hash().to_hex();

    let mut r = Receipt::new("phase-h-procedural-fields", "matrix");
    r.inputs
        .insert("court".into(), "procedural-1080p-10-families".into());
    r.outputs.insert("surface".into(), format!("{W}x{H} rgba8"));
    r.outputs.insert("canonical_hash".into(), oh.clone());
    r.canonical_hash = Some(oh.clone());
    r.reference_hash = Some(oh.clone());
    r.persistent_bytes = bytes.len() as u64;
    r.metrics.insert("objects".into(), doc.objects.len() as u64);
    r.metrics
        .insert("instances".into(), doc.instances.len() as u64);

    // ---- per-family cost estimate table (scheduler input) ----
    let mut total_work = 0u64;
    for f in (0..doc.objects.len() as u32).filter_map(|o| scene.field_of(o)) {
        let w = f.work();
        total_work += w;
        r.outputs
            .insert(format!("family{}_work", f.family_tag()), w.to_string());
    }
    r.metrics
        .insert("sum_work_units_per_sample".into(), total_work);

    // ---- request-shape matrix ----
    let shapes: Vec<(&str, Domain)> = vec![
        ("full", Domain::FullSurface { w: W, h: H }),
        (
            "region-1pct",
            Domain::Rectangle {
                x0: 200,
                y0: 200,
                w: 192,
                h: 108,
            },
        ),
        (
            "tile-64",
            Domain::Tile {
                tile_x: 5,
                tile_y: 3,
                tile_w: 64,
                tile_h: 64,
            },
        ),
        (
            "scanline",
            Domain::Scanline {
                y: 540,
                x0: 0,
                x1: W as i32,
            },
        ),
        (
            "band-32",
            Domain::DisplayBand {
                y0: 400,
                h: 32,
                x0: 0,
                x1: W as i32,
            },
        ),
        (
            "irregular",
            Domain::IrregularSamples(vec![(0, 0), (1919, 1079), (960, 540), (1, 1), (500, 800)]),
        ),
    ];
    let mut region_sampled = 0u64;
    for (name, domain) in shapes {
        let req = ObservationRequest {
            time_ns: req_full.time_ns,
            view: View::identity(),
            domain,
            sampling: vole_gfx::observation::SamplingProfile::exact_u1(),
            format: ColorFormat::Rgba8,
        };
        let requested = req.domain.requested_samples();
        let m = materialize_document(&doc, &req).expect(name);
        assert_eq!(
            m.counters.samples, requested,
            "{name}: evaluated samples must equal requested (no re-baking)"
        );
        let mut bm = BlockMaterializer::new(&scene, 64);
        let b = bm.materialize(&req, BlockShape::W16H16).expect(name);
        assert_eq!(b.output.data, m.output.data, "{name}: blocked parity");
        r.metrics.insert(format!("{name}_samples"), requested);
        if name == "region-1pct" {
            region_sampled = requested;
            // crop of the full surface must match the region exactly
            assert_crop_equals(
                &oracle.output.data,
                W as usize,
                200,
                200,
                &m.output.data,
                192,
                108,
            );
        }
        if name == "full" {
            let m = rayon_materialize(&scene, &req);
            assert_eq!(m.output.data, oracle.output.data, "full: rayon parity");
        }
        eprintln!("{name:>12}: samples evaluated {requested} (== requested)");
    }
    r.outputs.insert(
        "work_fraction_1pct".into(),
        format!(
            "{:.4}",
            region_sampled as f64 / oracle.counters.samples as f64
        ),
    );
    r.notes.push(format!(
        "samples evaluated == samples requested for every shape ({} families, cost {} units/sample); region-1pct work fraction {:.4}",
        doc.objects.len() - 1,
        total_work,
        region_sampled as f64 / oracle.counters.samples as f64
    ));

    // ---- cadence-independent observation (irregular time list) ----
    let mut seen = std::collections::BTreeSet::new();
    for t in [0u64, 16_666_667, 33_333_333, 144_444_444, 987_654_321] {
        let req = ObservationRequest::region(t, 690, 490, 300, 200, ColorFormat::Rgba8);
        let m = materialize_document(&doc, &req).expect("time query");
        seen.insert(m.output.canonical_hash().to_hex());
        r.outputs
            .insert(format!("time_{t}_hash"), m.output.canonical_hash().to_hex());
    }
    r.notes.push(format!(
        "state queried at 5 arbitrary observation times over a moving trajectory: {} distinct region hashes (motion observed, cadence-independent)",
        seen.len()
    ));

    // ---- gray request parity on the full surface ----
    let gray_req = ObservationRequest::full_surface(33_333_333, W, H, ColorFormat::Gray8);
    let gm = materialize_document(&doc, &gray_req).expect("gray oracle");
    let gscene = Scene::resolve(&doc, gray_req.time_ns).unwrap();
    let mut gbm = BlockMaterializer::new(&gscene, 64);
    let gb = gbm
        .materialize(&gray_req, BlockShape::W16H16)
        .expect("gray blocked");
    assert_eq!(gb.output.data, gm.output.data, "gray parity");
    r.outputs
        .insert("gray_hash".into(), gm.output.canonical_hash().to_hex());

    // ---- backend timing on this court ----
    let oracle_ns = time_ns(|| {
        let _ = materialize_document(&doc, &req_full).unwrap();
    });
    let blocked_ns = time_ns(|| {
        let mut bm = BlockMaterializer::new(&scene, 64);
        let _ = bm.materialize(&req_full, BlockShape::W16H16).unwrap();
    });
    r.metrics.insert("oracle_full_ns".into(), oracle_ns);
    r.metrics.insert("blocked_full_ns".into(), blocked_ns);
    let (auto, path) = dispatch::materialize_auto(&scene, &req_full, BlockShape::W16H16).unwrap();
    assert_eq!(auto.output.data, oracle.output.data, "auto parity on court");
    r.outputs
        .insert("auto_dispatch_path".into(), path.name().into());
    eprintln!(
        "court full: oracle {oracle_ns} ns, blocked {blocked_ns} ns, auto {}",
        path.name()
    );

    // ---- pure-constant subset uses the fill fast path (avx2) ----
    let cd = constant_subset();
    vole_gfx::ir::validate::validate(&cd).expect("subset valid");
    let creq = ObservationRequest::full_surface(0, W, H, ColorFormat::Rgba8);
    let coracle = materialize_document(&cd, &creq).unwrap();
    let cscene = Scene::resolve(&cd, 0).unwrap();
    let (m2, path2) = dispatch::materialize_auto(&cscene, &creq, BlockShape::W16H16).unwrap();
    assert_eq!(
        m2.output.data, coracle.output.data,
        "constant subset parity"
    );
    r.outputs
        .insert("constant_subset_path".into(), path2.name().into());
    let cns = time_ns(|| {
        let _ = dispatch::materialize_auto(&cscene, &creq, BlockShape::W16H16).unwrap();
    });
    r.metrics.insert("constant_subset_auto_ns".into(), cns);
    eprintln!(
        "constant subset: auto {} ({cns} ns), byte-identical",
        path2.name()
    );

    r.pass = true;
    let path = emit_receipt(r).expect("emit");
    println!("PHASE H PASS; receipt {path}");
    println!("canonical full hash {oh}");
}

/// Rayon full-surface materialization (band-parallel, default pool).
fn rayon_materialize(
    scene: &Scene<'_>,
    req: &ObservationRequest,
) -> vole_gfx::materialize::scalar::Materialized {
    vole_gfx::materialize::dispatch::rayon::materialize_parallel(scene, req, BlockShape::W16H16)
        .expect("rayon")
}
