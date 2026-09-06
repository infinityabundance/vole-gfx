//! Phase C gate: blocking + dependency indexing evidence.
//!
//! Proves byte-parity of the blocked materializer with the scalar oracle on a
//! 100+ object scene at 1920x1080, measures candidate-closure quality and
//! timing per block shape, and emits a receipt.
//!
//! Run: `cargo run --release --example phase_c_gate`

use std::time::Instant;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::evidence::receipt::{Receipt, emit_receipt};
use vole_gfx::fixed::{RectF, fp};
use vole_gfx::ir::{Document, Event, Instance, Object, Op};
use vole_gfx::materialize::blocked::BlockMaterializer;
use vole_gfx::materialize::blocks::BlockShape;
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::state::Scene;

fn rng_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn court_scene() -> Document {
    let mut seed = 0x51E5u64;
    let mut d = Document::new();
    // 8 tile objects
    for k in 0..8 {
        let w = 8 + (k % 4) * 8;
        let h = 8 + (k % 3) * 8;
        let mut data = Vec::new();
        for j in 0..h {
            for i in 0..w {
                let c = if (i + j + k) % 2 == 0 {
                    Rgba::new((k * 30) as u8, 120, (255 - k * 30) as u8, 255)
                } else {
                    Rgba::new(20, 20, 40, 255)
                };
                data.extend_from_slice(&c.to_bytes());
            }
        }
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w,
            h,
            data,
        });
    }
    // palette + indexed checker
    d.objects.push(Object::Palette {
        entries: vec![
            Rgba::OPAQUE_BLACK,
            Rgba::WHITE,
            Rgba::new(255, 60, 0, 255),
            Rgba::new(0, 200, 255, 255),
        ],
    });
    let mut idx = Vec::new();
    for j in 0..32 {
        for i in 0..32 {
            idx.push(((i / 8 + j / 8) % 4) as u32);
        }
    }
    d.objects.push(Object::IndexedRaster {
        pal: 8,
        w: 32,
        h: 32,
        indices: idx,
    });

    // 150 placed instances (background field + scattered sprites)
    for o in 0..150u32 {
        let obj = o % 9;
        let tx = ((rng_u64(&mut seed) % 1920) as i32) - 40;
        let ty = ((rng_u64(&mut seed) % 1080) as i32) - 40;
        d.instances.push(Instance {
            object: obj,
            order: o,
            layer: (o % 4),
            transform: vole_gfx::fixed::Affine::from_px_translation(tx, ty),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
    }
    // structural ops: large background fill, some region copies, sparse residual
    d.events.push(Event {
        t: 1,
        ops: vec![
            Op::FillRect {
                rect: RectF::from_px(0, 0, 1920, 1080),
                color: Rgba::new(24, 28, 34, 255),
            },
            Op::CopyRegion {
                src: RectF::from_px(0, 0, 256, 256),
                dx: fp(256),
                dy: fp(256),
            },
            Op::CopyRegion {
                src: RectF::from_px(512, 512, 768, 768),
                dx: fp(-256),
                dy: fp(256),
            },
        ],
    });
    let mut payload = Vec::new();
    payload.extend_from_slice(&3u64.to_le_bytes());
    vole_gfx::residual::push_record(&mut payload, 7, 9, &[255, 0, 255, 255]);
    vole_gfx::residual::push_record(&mut payload, 960, 540, &[255, 255, 0, 255]);
    vole_gfx::residual::push_record(&mut payload, 1900, 1000, &[9, 250, 9, 255]);
    d.events.push(Event {
        t: 2,
        ops: vec![Op::BindResidual(vole_gfx::ir::ResidualBind {
            algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
            region: vole_gfx::fixed::IRect::new(0, 0, 1920, 1080),
            format: ColorFormat::Rgba8,
            payload,
        })],
    });
    d
}

fn main() {
    let doc = court_scene();
    vole_gfx::ir::validate::validate(&doc).expect("valid scene");
    let bytes = vole_gfx::ir::encode::encode(&doc);
    let t = 16_666_667u64;

    // oracle hash at full surface
    let req =
        vole_gfx::observation::ObservationRequest::full_surface(t, 1920, 1080, ColorFormat::Rgba8);
    let scene = Scene::resolve(&doc, t).unwrap();
    let t0 = Instant::now();
    let scalar = materialize_document(&doc, &req).expect("scalar");
    let scalar_ns = t0.elapsed().as_nanos() as u64;

    let mut best = None;
    for shape in [
        BlockShape::W8H8,
        BlockShape::W16H8,
        BlockShape::W16H16,
        BlockShape::W32H8,
        BlockShape::ScanlineSegment(64),
    ] {
        let mut bm = BlockMaterializer::new(&scene, 64);
        let t1 = Instant::now();
        let m = bm.materialize(&req, shape).expect("blocked");
        let ns = t1.elapsed().as_nanos() as u64;
        assert_eq!(
            m.output.canonical_hash(),
            scalar.output.canonical_hash(),
            "parity for {}",
            shape.name()
        );
        let mets = bm.metrics();
        let cands_per_sample = mets.candidates_selected as f64 / m.output.sample_count() as f64;
        eprintln!(
            "shape {:>10}: {ns:>10} ns  candidates/sample {cands_per_sample:.3}  blocks {}  false-positive-rate {:.3}",
            shape.name(),
            mets.blocks,
            if mets.candidates_selected > 0 {
                1.0 - (mets.candidates_true_positive as f64 / mets.candidates_selected as f64)
            } else {
                0.0
            }
        );
        best = Some((shape, ns, mets));
    }
    let (best_shape, best_ns, mets) = best.unwrap();

    // partial-domain work fraction: 1% region of the surface
    let small =
        vole_gfx::observation::ObservationRequest::region(t, 0, 0, 192, 108, ColorFormat::Rgba8);
    let mut bm = BlockMaterializer::new(&scene, 64);
    let t2 = Instant::now();
    let m_small = bm.materialize(&small, BlockShape::W16H16).expect("small");
    let small_ns = t2.elapsed().as_nanos() as u64;
    let small_mets = bm.metrics();
    let work_fraction = small_ns as f64 / scalar_ns as f64;
    let hash_full = scalar.output.canonical_hash().to_hex();

    let mut r = Receipt::new("phase-c-blocking-indexing", "blocked-scalar");
    r.pass = true;
    r.inputs
        .insert("doc_objects".into(), doc.objects.len().to_string());
    r.inputs
        .insert("doc_instances".into(), doc.instances.len().to_string());
    r.inputs
        .insert("serialized_bytes".into(), bytes.len().to_string());
    r.outputs
        .insert("surface".into(), "1920x1080 rgba8 @ t=16666667".into());
    r.outputs.insert("canonical_hash".into(), hash_full.clone());
    r.canonical_hash = Some(hash_full.clone());
    r.reference_hash = Some(hash_full);
    r.backend = format!("scalar+blocked({})", best_shape.name());
    r.persistent_bytes = bytes.len() as u64;
    r.samples_requested = scalar.output.sample_count();
    r.samples_evaluated = scalar.counters.samples;
    r.metrics.insert("scalar_materialize_ns".into(), scalar_ns);
    r.metrics.insert("best_blocked_ns".into(), best_ns);
    r.metrics.insert("small_region_1pct_ns".into(), small_ns);
    r.metrics.insert("blocks".into(), mets.blocks);
    r.metrics
        .insert("candidates_selected".into(), mets.candidates_selected);
    r.metrics.insert(
        "candidates_true_positive".into(),
        mets.candidates_true_positive,
    );
    r.metrics
        .insert("instance_draws".into(), mets.instance_draws);
    r.metrics
        .insert("ops_considered".into(), mets.ops_considered);
    r.metrics
        .insert("small_region_samples".into(), m_small.output.sample_count());
    r.metrics.insert(
        "small_candidates_selected".into(),
        small_mets.candidates_selected,
    );
    r.notes.push(format!(
        "blocked == scalar byte-for-byte at 1920x1080 for all swept shapes; 1%% region work_fraction={work_fraction:.4}; candidate quality measured per receipt fields"
    ));
    let path = emit_receipt(r).expect("emit");
    println!(
        "PHASE C PASS; receipt {path}; scalar {scalar_ns} ns; best block {} {best_ns} ns; 1% region work fraction {work_fraction:.4}",
        best_shape.name()
    );
}
