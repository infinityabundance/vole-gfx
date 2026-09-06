//! Phase I gate: the scalar inverse procedural compiler ("unbaking").
//!
//! Unbakes a court of assets (built from our own generator families, plus
//! noise as a negative control) into Pareto frontiers of exact explanations:
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
            class: "hash-noise (negative)",
            asset: rasterize(&build::noise_field(32, 32, 7)),
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
            .insert(format!("{p}_searchwork"), best.search_work);
        r.metrics.insert(format!("{p}_ns"), best.materialize_ns);
        r.metrics
            .insert(format!("{p}_samples"), c.asset.sample_count());
        for (name, pers, res, work, search) in inverse::summarize(&f) {
            let n = format!("{p}_f_{name}");
            r.outputs
                .insert(format!("{n}_persistent"), pers.to_string());
            r.outputs.insert(format!("{n}_residual"), res.to_string());
            r.outputs.insert(format!("{n}_matwork"), work.to_string());
            r.outputs.insert(format!("{n}_search"), search.to_string());
        }
        r.notes.push(format!(
            "{p}: winner={} persistent={}B residual={}B matwork={} search={} exact={} frontier={}",
            best.name,
            best.persistent_bytes,
            best.residual_bytes,
            best.materialize_work,
            best.search_work,
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
        "detector coverage (Phase I scalar): constant, periodic stripes/checker, palette bands, tiled, gradient ramps/bilinear, literal fallback; structural reuse/affine/symmetry detectors and SIMD/Rayon/CUDA search land in phases J-M".into(),
    );
    let path = emit_receipt(r).expect("emit");
    println!("PHASE I PASS: {all_pass}; receipt {path}");
}
