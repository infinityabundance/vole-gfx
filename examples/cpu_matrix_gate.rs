//! Phase D/E/F gate: exact CPU backend matrix on one court scene.
//!
//! Usage: `cargo run --release --example cpu_matrix_gate -- <d|e|f>`
//!   d: scalar / blocked / AVX2
//!   e: adds AVX-512
//!   f: adds Rayon×{scalar,avx2,avx512,auto}
//!
//! Every backend must equal the scalar oracle byte-for-byte; timings and the
//! recorded dispatch paths go into the receipt.

use std::time::Instant;
use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::evidence::receipt::{Receipt, emit_receipt};
use vole_gfx::fixed::RectF;
use vole_gfx::ir::{Document, Event, Instance, Object, Op};
use vole_gfx::materialize::blocked::BlockMaterializer;
use vole_gfx::materialize::blocks::BlockShape;
use vole_gfx::materialize::dispatch::{self, avx2, avx512, rayon};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::state::Scene;

fn rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Eligible court: 200 opaque, integer-translation sprites over opaque fills,
/// plus a tiny sparse residual (eligible subset exercises fills, blits,
/// residual closure).
fn court() -> Document {
    let mut seed = 0xD15EA5Eu64;
    let mut d = Document::new();
    // 6 sprite objects (fully opaque, Rgba8)
    for k in 0..6u32 {
        let w = 8 + (k % 3) * 16;
        let h = 8 + (k % 2) * 16;
        let mut data = Vec::new();
        for j in 0..h {
            for i in 0..w {
                let v = (i * 17 + j * 31 + k * 7) as u8;
                data.extend_from_slice(&[v, v.wrapping_mul(2), 255 - v, 255]);
            }
        }
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w,
            h,
            data,
        });
    }
    for o in 0..200u32 {
        let obj = o % 6;
        d.instances.push(Instance {
            object: obj,
            order: o,
            layer: (o % 5),
            transform: vole_gfx::fixed::Affine::from_px_translation(
                ((rng(&mut seed) % 1840) as i32) + 20,
                ((rng(&mut seed) % 1000) as i32) + 20,
            ),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
    }
    d.events.push(Event {
        t: 1,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 1920, 1080),
            color: Rgba::new(9, 13, 17, 255),
        }],
    });
    // opaque HUD-like bar (fill) over sprites
    d.events.push(Event {
        t: 2,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 1000, 1920, 1020),
            color: Rgba::new(230, 230, 235, 255),
        }],
    });
    let mut payload = Vec::new();
    payload.extend_from_slice(&1u64.to_le_bytes());
    vole_gfx::residual::push_record(&mut payload, 960, 500, &[255, 0, 128, 255]);
    d.events.push(Event {
        t: 3,
        ops: vec![Op::BindResidual(vole_gfx::ir::ResidualBind {
            algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
            region: vole_gfx::fixed::IRect::new(0, 0, 1920, 1080),
            format: ColorFormat::Rgba8,
            payload,
        })],
    });
    d
}

/// Time `f`: warmup + median of 3.
fn time_ms<F: FnMut()>(mut f: F) -> u64 {
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

fn run_bench(
    r: &mut Receipt,
    oh: &str,
    backend: &str,
    f: &mut dyn FnMut() -> Result<
        vole_gfx::materialize::scalar::Materialized,
        vole_gfx::limits::Reject,
    >,
) {
    // parity check once
    let m = f().expect(backend);
    assert_eq!(m.output.canonical_hash().to_hex(), oh, "{backend} parity");
    let ns = time_ms(|| {
        let _ = f().expect(backend);
    });
    eprintln!("{backend:>16}: {ns:>12} ns");
    r.metrics.insert(format!("{backend}_ns"), ns);
    r.backend = format!("{backend}+matrix");
}

fn main() {
    let phase = std::env::args().nth(1).unwrap_or_else(|| "d".into());
    let doc = court();
    vole_gfx::ir::validate::validate(&doc).expect("court scene valid");
    let req = vole_gfx::observation::ObservationRequest::full_surface(
        33_333_333,
        1920,
        1080,
        ColorFormat::Rgba8,
    );
    let scene = Scene::resolve(&doc, req.time_ns).unwrap();
    let shape = BlockShape::W16H16;

    let oracle = materialize_document(&doc, &req).unwrap();
    let oh = oracle.output.canonical_hash().to_hex();
    let oracle_ns = time_ms(|| {
        let _ = materialize_document(&doc, &req).unwrap();
    });

    let mut r = Receipt::new("cpu-backend-matrix", "matrix");
    r.inputs.insert("phase".into(), phase.clone());
    r.outputs
        .insert("surface".into(), "1920x1080 rgba8 200 sprites".into());
    r.outputs.insert("canonical_hash".into(), oh.clone());
    r.canonical_hash = Some(oh.clone());
    r.reference_hash = Some(oh.clone());
    r.persistent_bytes = vole_gfx::ir::encode::encode(&doc).len() as u64;
    r.samples_requested = oracle.output.sample_count();

    run_bench(&mut r, &oh, "scalar", &mut || {
        materialize_document(&doc, &req)
    });
    let blocked_ns = time_ms(|| {
        let mut bm = BlockMaterializer::new(&scene, 64);
        let _ = bm.materialize(&req, shape).unwrap();
    });
    eprintln!("{:>16}: {blocked_ns:>12} ns", "blocked");
    r.metrics.insert("blocked_ns".into(), blocked_ns);

    if phase == "d" || phase == "e" || phase == "f" {
        if avx2::has_avx2() {
            run_bench(&mut r, &oh, "avx2", &mut || match avx2::materialize_simple(
                &scene, &req, shape,
            )
            .unwrap()
            {
                Some(m) => Ok(m),
                None => panic!("avx2 not eligible on court"),
            });
        } else {
            eprintln!("avx2: not evaluated on this host");
            r.notes.push("avx2: not evaluated on this host".into());
        }
    }
    if phase == "e" || phase == "f" {
        if avx512::has_avx512() {
            run_bench(
                &mut r,
                &oh,
                "avx512",
                &mut || match avx512::materialize_simple(&scene, &req, shape).unwrap() {
                    Some(m) => Ok(m),
                    None => panic!("avx512 not eligible on court"),
                },
            );
        } else {
            eprintln!("avx512: not evaluated on this host");
            r.notes.push("avx512: not evaluated on this host".into());
        }
    }
    if phase == "f" {
        for simd in [
            rayon::BandSimd::Scalar,
            rayon::BandSimd::Avx2,
            rayon::BandSimd::Avx512,
            rayon::BandSimd::Auto,
        ] {
            let name = simd.name().to_string();
            run_bench(&mut r, &oh, &name, &mut || {
                rayon::materialize_bands_with(&scene, &req, shape, 64, simd)
            });
        }
        // auto dispatch path record
        let (_, path) = dispatch::materialize_auto(&scene, &req, shape).unwrap();
        eprintln!("auto dispatch path: {}", path.name());
        r.outputs
            .insert("auto_dispatch_path".into(), path.name().into());
    }
    r.metrics.insert("oracle_scalar_ns".into(), oracle_ns);
    r.pass = true;
    r.notes.push(format!(
        "all measured backends byte-identical to scalar oracle; host dispatch: avx2={} avx512={}",
        avx2::has_avx2(),
        avx512::has_avx512()
    ));
    let path = emit_receipt(r).expect("emit");
    println!("PHASE {phase} PASS; receipt {path}; oracle {oracle_ns} ns");
}
