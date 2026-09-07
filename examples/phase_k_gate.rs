//! Phase K gate: seeded-field inverse search.
//!
//! Courts:
//! - deterministic gray-noise field assets (Gray8 and opaque-gray Rgba8,
//!   several seeds/extents) that the wired-in detector must unbake to the
//!   exact `seeded-field` explanation with zero residual;
//! - negative controls (SHA-256 random gray and random RGBA) and the fractal
//!   family (seeded but NOT part of the Phase K searched universe) that must
//!   fall back to literal;
//! - a colorful analytic asset (no gray surface) that must not trigger a
//!   seed sweep at all.
//!
//! For every gray-surface court the gate sweeps `[0, RANGE)` with the scalar
//! oracle and, when the host provides them, the AVX2 and AVX-512 batched
//! kernels, recording per-backend wall time and executed hash evaluations and
//! asserting the **accepted seed sets are identical across backends**.  The
//! auto-dispatch row records which path the policy constant selected.
//!
//! Run: `cargo run --release --example phase_k_gate`

use vole_gfx::color::ColorFormat;
use vole_gfx::evidence::receipt::{Receipt, emit_receipt};
use vole_gfx::inverse::search::{sweep_auto, sweep_avx2, sweep_avx512, sweep_scalar};
use vole_gfx::inverse::{self, asset::Asset};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::ObservationRequest;
use vole_gfx::procedural::build;

/// Deterministic-field raster helper in Gray8.
fn field_gray(w: u32, h: u32, seed: u64) -> Asset {
    let mut data = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            data.push(vole_gfx::inverse::search::gray_value(seed, i, j));
        }
    }
    Asset::new(w, h, ColorFormat::Gray8, data).unwrap()
}

/// Deterministic-field raster helper in opaque-gray Rgba8.
fn field_rgba(w: u32, h: u32, seed: u64) -> Asset {
    let g = field_gray(w, h, seed);
    let mut data = Vec::new();
    for &v in &g.data {
        data.extend_from_slice(&[v, v, v, 255]);
    }
    Asset::new(w, h, ColorFormat::Rgba8, data).unwrap()
}

/// SHA-256 pseudo-random bytes (outside every U1 generator family).
fn random_gray(w: u32, h: u32, tag: u8) -> Asset {
    let mut data = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let c =
                vole_gfx::hash::sha256(&[tag, (i >> 8) as u8, i as u8, (j >> 8) as u8, j as u8]);
            data.push(c.0[2] ^ c.0[13]);
        }
    }
    Asset::new(w, h, ColorFormat::Gray8, data).unwrap()
}

/// SHA-256 pseudo-random bytes canonicalized through the materializer (a
/// real baked asset is a materializer output) and outside every U1 generator
/// family.
fn random_rgba(w: u32, h: u32, tag: u8) -> Asset {
    let mut data = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let c =
                vole_gfx::hash::sha256(&[tag, (i >> 8) as u8, i as u8, (j >> 8) as u8, j as u8]);
            data.extend_from_slice(&c.0[..4]);
        }
    }
    let mut d = vole_gfx::ir::Document::new();
    d.objects.push(vole_gfx::ir::Object::Raster {
        format: ColorFormat::Rgba8,
        w,
        h,
        data,
    });
    d.instances
        .push(vole_gfx::inverse::candidate::origin_instance(0, 1, 0));
    rasterize_doc(&d, w, h, ColorFormat::Rgba8)
}

/// Materialize a doc over `w x h` at its format (court rasterizer).
fn rasterize_doc(d: &vole_gfx::ir::Document, w: u32, h: u32, fmt: ColorFormat) -> Asset {
    vole_gfx::ir::validate::validate(d).unwrap();
    let req = ObservationRequest::full_surface(0, w, h, fmt);
    let m = materialize_document(d, &req).unwrap();
    Asset::new(w, h, fmt, m.output.data).unwrap()
}

fn colorful_bilinear(w: u32, h: u32) -> Asset {
    let mut d = vole_gfx::ir::Document::new();
    d.objects.push(build::bilinear_gradient(
        w,
        h,
        vole_gfx::color::Rgba::new(10, 10, 200, 255),
        vole_gfx::color::Rgba::new(200, 10, 10, 255),
        vole_gfx::color::Rgba::new(10, 200, 10, 255),
        vole_gfx::color::Rgba::new(200, 200, 10, 255),
    ));
    d.instances
        .push(vole_gfx::inverse::candidate::origin_instance(0, 1, 0));
    rasterize_doc(&d, w, h, ColorFormat::Rgba8)
}

fn fractal_gray(w: u32, h: u32, seed: u64) -> Asset {
    let mut d = vole_gfx::ir::Document::new();
    d.objects.push(build::fractal(w, h, seed, 4, 3, 32768));
    d.instances
        .push(vole_gfx::inverse::candidate::origin_instance(0, 1, 0));
    rasterize_doc(&d, w, h, ColorFormat::Rgba8)
}

struct Case {
    class: &'static str,
    asset: Asset,
    expect: &'static str, // expected byte-min profile winner
}

fn median_ns(runs: usize, mut f: impl FnMut()) -> u64 {
    let mut v = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = std::time::Instant::now();
        f();
        v.push(t0.elapsed().as_nanos() as u64);
    }
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    let sweep_range: u64 = 1 << 20; // explicit court range for the backend rows
    let cases = [
        Case {
            class: "field-gray-32x24-s7",
            asset: field_gray(32, 24, 7),
            expect: "seeded-field",
        },
        Case {
            class: "field-gray-40x30-s4242",
            asset: field_gray(40, 30, 4242),
            expect: "seeded-field",
        },
        Case {
            class: "field-gray-17x9-s65535",
            asset: field_gray(17, 9, 65535),
            expect: "seeded-field",
        },
        Case {
            class: "field-rgba-gray-24x24-s99",
            asset: field_rgba(24, 24, 99),
            expect: "seeded-field",
        },
        Case {
            class: "negative-random-gray-24x24",
            asset: random_gray(24, 24, 0x1a),
            expect: "literal",
        },
        Case {
            class: "negative-random-rgba-24x24",
            asset: random_rgba(24, 24, 0x2b),
            expect: "literal",
        },
        Case {
            class: "fractal-gray-24x24 (not-searched family)",
            asset: fractal_gray(24, 24, 4242),
            expect: "literal",
        },
        Case {
            class: "colorful-bilinear (no gray surface)",
            asset: colorful_bilinear(24, 24),
            expect: "gradient-bilinear",
        },
    ];

    let mut r = Receipt::new("phase-k-seeded-field-search", "search");
    r.inputs
        .insert("sweep_range".into(), sweep_range.to_string());
    r.inputs.insert(
        "policy".into(),
        format!(
            "search auto prefer avx512: {}",
            vole_gfx::inverse::search::SEARCH_AUTO_PREFER_AVX512
        ),
    );
    let mut all_pass = true;

    for (k, c) in cases.iter().enumerate() {
        let p = format!("a{k}_{}", c.class);
        // ---- wiring: the wired-in detector via unbake (canonical scalar
        // sweep at DEFAULT_SEED_SWEEP_RANGE)
        let (f, best) = inverse::unbake_best(&c.asset).unwrap();
        let req = ObservationRequest::full_surface(0, c.asset.w, c.asset.h, c.asset.format);
        let m = materialize_document(&best.doc, &req).expect("winner materialize");
        let exact = m.output.data == c.asset.data;
        let winner_ok = best.name == c.expect;
        all_pass &= exact && winner_ok;
        r.outputs.insert(format!("{p}_winner"), best.name.clone());
        r.outputs
            .insert(format!("{p}_expect"), c.expect.to_string());
        r.outputs.insert(format!("{p}_exact"), exact.to_string());
        r.outputs
            .insert(format!("{p}_frontier"), f.candidates.len().to_string());
        r.metrics
            .insert(format!("{p}_persistent"), best.persistent_bytes);
        r.metrics
            .insert(format!("{p}_residual"), best.residual_bytes);
        r.metrics
            .insert(format!("{p}_matwork"), best.materialize_work);
        r.metrics
            .insert(format!("{p}_searchwork"), best.search.total_units());
        r.metrics
            .insert(format!("{p}_winner_ns"), best.materialize_ns);

        // ---- backend sweep rows (gray surfaces only)
        if let Some(scalar) = sweep_scalar(&c.asset, sweep_range) {
            let scalar_matches = scalar.matches.clone();
            r.outputs.insert(
                format!("{p}_scalar_matches"),
                format!("{:?}", scalar.matches),
            );
            r.metrics
                .insert(format!("{p}_scalar_hashes"), scalar.work.hash_ops);
            r.metrics.insert(
                format!("{p}_scalar_ns"),
                median_ns(3, || {
                    let _ = sweep_scalar(&c.asset, sweep_range);
                }),
            );
            let mut parity = true;
            if let Some(avx2) = sweep_avx2(&c.asset, sweep_range) {
                let m = avx2.matches.clone();
                parity &= m == scalar_matches;
                r.metrics
                    .insert(format!("{p}_avx2_hashes"), avx2.work.hash_ops);
                r.metrics.insert(
                    format!("{p}_avx2_ns"),
                    median_ns(5, || {
                        let _ = sweep_avx2(&c.asset, sweep_range);
                    }),
                );
                r.outputs
                    .insert(format!("{p}_avx2_matches"), format!("{m:?}"));
            }
            if let Some(avx512) = sweep_avx512(&c.asset, sweep_range) {
                let m = avx512.matches.clone();
                parity &= m == scalar_matches;
                r.metrics
                    .insert(format!("{p}_avx512_hashes"), avx512.work.hash_ops);
                r.metrics.insert(
                    format!("{p}_avx512_ns"),
                    median_ns(5, || {
                        let _ = sweep_avx512(&c.asset, sweep_range);
                    }),
                );
                r.outputs
                    .insert(format!("{p}_avx512_matches"), format!("{m:?}"));
            }
            // auto dispatch row
            if let Some(auto) = sweep_auto(&c.asset, sweep_range) {
                parity &= auto.matches == scalar_matches;
                r.outputs
                    .insert(format!("{p}_auto_path"), auto.backend.name().into());
                r.metrics.insert(
                    format!("{p}_auto_ns"),
                    auto_median_ns(&c.asset, sweep_range),
                );
                r.metrics
                    .insert(format!("{p}_auto_hashes"), auto.work.hash_ops);
            }
            all_pass &= parity;
            r.outputs
                .insert(format!("{p}_accept_parity"), parity.to_string());
        } else {
            r.outputs.insert(
                format!("{p}_scalar_matches"),
                "none (not a gray surface)".into(),
            );
        }
        eprintln!(
            "a{k} {:<32} winner={:<16} expect={:<16} exact={} persistent={}",
            c.class, best.name, c.expect, exact, best.persistent_bytes
        );
    }
    r.pass = all_pass;
    let path = emit_receipt(r).expect("emit");
    println!("PHASE K PASS: {all_pass}; receipt {path}");
}

/// Auto-dispatch timing helper (avoids calling `sweep_auto` twice per row).
fn auto_median_ns(asset: &Asset, range: u64) -> u64 {
    median_ns(5, || {
        let _ = sweep_auto(asset, range);
    })
}
