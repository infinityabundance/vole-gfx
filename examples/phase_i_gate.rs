//! Phase I gate: the scalar inverse procedural compiler ("unbaking").
//!
//! Unbakes a court of assets (built from our own generator families, plus
//! SHA-256 pseudo-random bytes as a negative control) into Pareto frontiers
//! of exact explanations:
//!
//! ```text
//! A -> (Gamma, s, theta, R)      per surviving candidate
//! ```
//!
//! Evidence per asset: frontier rows (persistent bytes, residual bytes,
//! deterministic materialization-work model, search work), the byte-min
//! profile winner, measured winner latency, and the exact-reconstruction
//! gate (materialize(winner) == asset, byte-for-byte).
//!
//! Run: `cargo run --release --example phase_i_gate`

use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::evidence::receipt::{Receipt, emit_receipt};
use vole_gfx::inverse::{self, asset::Asset};
use vole_gfx::ir::Object;
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::ObservationRequest;
use vole_gfx::procedural::build;

/// Materialize an object at the origin over its own extent (the "baked"
/// raster input to the compiler).
fn rasterize(object: &Object) -> Asset {
    let mut d = vole_gfx::ir::Document::new();
    d.objects.push(object.clone());
    d.instances
        .push(vole_gfx::inverse::candidate::origin_instance(0, 1, 0));
    vole_gfx::ir::validate::validate(&d).unwrap();
    let (w, h) = match object {
        Object::Raster { w, h, .. } | Object::GeneratorField { w, h, .. } => (*w, *h),
        _ => panic!(),
    };
    let req = ObservationRequest::full_surface(0, w, h, ColorFormat::Rgba8);
    let m = materialize_document(&d, &req).unwrap();
    Asset::new(w, h, ColorFormat::Rgba8, m.output.data).unwrap()
}

/// Deterministic SHA-256-derived pseudo-random RGBA raster, canonicalized
/// through the materializer (a real baked asset is a materializer output, so
/// fully-transparent codes canonicalize to (0,0,0,0)).  Independently
/// generated control bytes: the gate receipt records that no detector (and
/// no Phase-K seed sweep) matches them exactly, so the byte-min profile
/// retains the literal fallback.
fn random_rgba_asset(w: u32, h: u32) -> Asset {
    let mut data = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let c =
                vole_gfx::hash::sha256(&[0x6d, (i >> 8) as u8, i as u8, (j >> 8) as u8, j as u8]);
            data.extend_from_slice(&c.0[..4]);
        }
    }
    rasterize(&Object::Raster {
        format: ColorFormat::Rgba8,
        w,
        h,
        data,
    })
}

/// One court asset class.
struct Case {
    class: &'static str,
    asset: Asset,
}

fn main() {
    let cases: Vec<Case> = vec![
        Case {
            class: "constant",
            asset: rasterize(&build::constant(96, 64, Rgba::new(9, 13, 17, 255))),
        },
        Case {
            class: "checker",
            asset: rasterize(&build::checker(
                64,
                48,
                8,
                6,
                Rgba::new(20, 200, 20, 255),
                Rgba::new(240, 40, 240, 255),
            )),
        },
        Case {
            class: "palette-bands",
            asset: rasterize(&build::palette_bands(
                80,
                24,
                5,
                vec![
                    Rgba::new(10, 10, 10, 255),
                    Rgba::new(120, 30, 30, 255),
                    Rgba::new(30, 120, 30, 255),
                ],
            )),
        },
        Case {
            class: "tiled",
            asset: rasterize(&build::tiled(
                18,
                12,
                3,
                2,
                vec![
                    Rgba::new(200, 20, 20, 255),
                    Rgba::new(20, 200, 20, 255),
                    Rgba::new(20, 20, 200, 255),
                    Rgba::new(200, 200, 20, 255),
                    Rgba::new(200, 20, 200, 255),
                    Rgba::new(20, 200, 200, 255),
                ],
            )),
        },
        Case {
            class: "gradient-ramp",
            asset: rasterize(&build::linear_gradient(
                48,
                4,
                0,
                0,
                Rgba::new(0, 0, 0, 255),
                47,
                0,
                Rgba::new(255, 255, 255, 255),
            )),
        },
        Case {
            class: "gradient-bilinear",
            asset: rasterize(&build::bilinear_gradient(
                32,
                24,
                Rgba::new(10, 10, 200, 255),
                Rgba::new(200, 10, 10, 255),
                Rgba::new(10, 200, 10, 255),
                Rgba::new(200, 200, 10, 255),
            )),
        },
        Case {
            class: "random-rgba (negative)",
            asset: random_rgba_asset(32, 32),
        },
    ];

    let mut r = Receipt::new("phase-i-inverse-scalar", "matrix");
    r.inputs
        .insert("court".into(), "unbake-classes-scalar".into());
    let mut all_pass = true;

    for (k, c) in cases.iter().enumerate() {
        let (f, best) = inverse::unbake_best(&c.asset).unwrap();
        // exact reconstruction gate
        let req = ObservationRequest::full_surface(0, c.asset.w, c.asset.h, c.asset.format);
        let m = materialize_document(&best.doc, &req).expect("winner materialize");
        let exact = m.output.data == c.asset.data;
        all_pass &= exact;
        let p = format!("a{k}_{}", c.class.replace(' ', "-"));
        r.outputs.insert(format!("{p}_winner"), best.name.clone());
        r.outputs.insert(format!("{p}_exact"), exact.to_string());
        r.outputs.insert(
            format!("{p}_literal_bytes"),
            (c.asset.data.len() as u64 + 64).to_string(),
        );
        r.metrics
            .insert(format!("{p}_persistent"), best.persistent_bytes);
        r.metrics
            .insert(format!("{p}_residual"), best.residual_bytes);
        r.metrics
            .insert(format!("{p}_matwork"), best.materialize_work);
        r.metrics
            .insert(format!("{p}_searchwork"), best.search.total_units());
        r.metrics.insert(format!("{p}_ns"), best.materialize_ns);
        r.metrics
            .insert(format!("{p}_samples"), c.asset.sample_count());
        for row in inverse::summarize(&f) {
            let n = format!("{p}_f_{name}", name = row.name);
            r.outputs
                .insert(format!("{n}_persistent"), row.persistent_bytes.to_string());
            r.outputs
                .insert(format!("{n}_residual"), row.residual_bytes.to_string());
            r.outputs
                .insert(format!("{n}_matwork"), row.materialize_work.to_string());
            r.outputs
                .insert(format!("{n}_search"), row.search_work.to_string());
            r.outputs
                .insert(format!("{n}_pixels"), row.pixels_read.to_string());
            r.outputs
                .insert(format!("{n}_compares"), row.code_compares.to_string());
            r.outputs
                .insert(format!("{n}_hashops"), row.hash_ops.to_string());
            r.outputs
                .insert(format!("{n}_candtests"), row.candidate_tests.to_string());
            r.outputs.insert(
                format!("{n}_cropbytes"),
                row.crop_bytes_compared.to_string(),
            );
        }
        r.notes.push(format!(
            "{p}: winner={} persistent={}B residual={}B matwork={} search={} exact={} frontier={}",
            best.name,
            best.persistent_bytes,
            best.residual_bytes,
            best.materialize_work,
            best.search.total_units(),
            exact,
            f.candidates.len()
        ));
        eprintln!(
            "a{k} {:<24} winner={:<16} persistent={:<6} residual={:<7} exact={}",
            c.class, best.name, best.persistent_bytes, best.residual_bytes, exact
        );
    }
    r.pass = all_pass;
    r.outputs.insert("all_exact".into(), all_pass.to_string());
    r.notes.push(
        "detector coverage (Phase I scalar): constant, periodic stripes/checker, palette bands, tiled, gradient ramps/bilinear, literal fallback; structural reuse/affine/symmetry detectors land in phase J, seeded-field search (SIMD batched) in phase K, Rayon/CUDA batched search in phases L-M. The negative control is independently generated SHA-256 pseudo-random bytes with no exact match in the evaluated detector/search universes (receipted). Search work is counted exactly where it occurs (SearchCounter: pixels read / whole-code compares / crop bytes / hash evals), not as a flat sample count.".into(),
    );
    let path = emit_receipt(r).expect("emit");
    println!("PHASE I PASS: {all_pass}; receipt {path}");
}
