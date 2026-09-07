//! Phase J gate: structural reuse / sprite extraction / shared-object reuse.
//!
//! Unbakes assets with a uniform field plus one sprite (`sprite-on-field`),
//! the same sprite repeated at several translations (`sprite-repeat`, shared
//! object), and a negative control.  Receipts record winner, persistent /
//! residual / materialization-work / search axes and the exactness gate.
//!
//! Run: `cargo run --release --example phase_j_gate`

use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::evidence::receipt::{Receipt, emit_receipt};
use vole_gfx::fixed::Affine;
use vole_gfx::inverse::{self, asset::Asset};
use vole_gfx::ir::{Document, Instance, Object};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::ObservationRequest;
use vole_gfx::procedural::build;

fn placed(object: u32, order: u32, layer: u32, dx: i32, dy: i32) -> Instance {
    Instance {
        object,
        order,
        layer,
        transform: Affine::from_px_translation(dx, dy),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    }
}

fn sprite() -> Object {
    let mut data = Vec::new();
    for j in 0..8u32 {
        for i in 0..10u32 {
            let c = match (i % 3, j % 2) {
                (0, 0) => (200u8, 40u8, 40u8),
                (1, 0) => (40u8, 200u8, 40u8),
                _ => (250u8, 200u8, 40u8),
            };
            data.extend_from_slice(&[c.0, c.1, c.2, 255]);
        }
    }
    Object::Raster {
        format: ColorFormat::Rgba8,
        w: 10,
        h: 8,
        data,
    }
}

fn rasterize(d: &Document, w: u32, h: u32) -> Asset {
    vole_gfx::ir::validate::validate(d).unwrap();
    let req = ObservationRequest::full_surface(0, w, h, ColorFormat::Rgba8);
    let m = materialize_document(d, &req).unwrap();
    Asset::new(w, h, ColorFormat::Rgba8, m.output.data).unwrap()
}

struct Case {
    class: &'static str,
    asset: Asset,
}

fn field_and_sprites(at: &[(i32, i32)]) -> Asset {
    let mut d = Document::new();
    d.objects
        .push(build::constant(64, 48, Rgba::new(11, 22, 33, 255)));
    d.objects.push(sprite());
    d.instances.push(placed(0, 1, 0, 0, 0));
    for (k, &(x, y)) in at.iter().enumerate() {
        d.instances.push(placed(1, 2 + k as u32, 1, x, y));
    }
    rasterize(&d, 64, 48)
}

/// Deterministic SHA-256-derived pseudo-random RGBA raster, canonicalized
/// through the materializer (a real baked asset is a materializer output).
/// Independently generated control bytes: no detector in the evaluated set
/// matches them exactly (receipted) — the honest negative control.
fn random_asset() -> Asset {
    let mut data = Vec::new();
    for j in 0..24u32 {
        for i in 0..24u32 {
            let c =
                vole_gfx::hash::sha256(&[0x4a, (i >> 8) as u8, i as u8, (j >> 8) as u8, j as u8]);
            data.extend_from_slice(&c.0[..4]);
        }
    }
    let mut d = Document::new();
    d.objects.push(Object::Raster {
        format: ColorFormat::Rgba8,
        w: 24,
        h: 24,
        data,
    });
    d.instances.push(placed(0, 1, 0, 0, 0));
    rasterize(&d, 24, 24)
}

fn main() {
    let cases = [
        Case {
            class: "sprite-on-field",
            asset: field_and_sprites(&[(20, 16)]),
        },
        Case {
            class: "sprite-repeat-2",
            asset: field_and_sprites(&[(10, 8), (44, 34)]),
        },
        Case {
            class: "sprite-repeat-3",
            asset: field_and_sprites(&[(10, 8), (44, 34), (4, 30)]),
        },
        Case {
            class: "negative-random",
            asset: random_asset(),
        },
    ];
    let mut r = Receipt::new("phase-j-structural-reuse", "matrix");
    let mut all_pass = true;
    for (k, c) in cases.iter().enumerate() {
        let (f, best) = inverse::unbake_best(&c.asset).unwrap();
        let req = ObservationRequest::full_surface(0, c.asset.w, c.asset.h, c.asset.format);
        let m = materialize_document(&best.doc, &req).unwrap();
        let exact = m.output.data == c.asset.data;
        all_pass &= exact;
        let p = format!("a{k}_{}", c.class);
        r.outputs.insert(format!("{p}_winner"), best.name.clone());
        r.outputs.insert(format!("{p}_exact"), exact.to_string());
        r.metrics
            .insert(format!("{p}_persistent"), best.persistent_bytes);
        r.metrics
            .insert(format!("{p}_residual"), best.residual_bytes);
        r.metrics
            .insert(format!("{p}_matwork"), best.materialize_work);
        r.metrics.insert(format!("{p}_ns"), best.materialize_ns);
        r.metrics
            .insert(format!("{p}_searchwork"), best.search.total_units());
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
            "{p}: winner={} persistent={}B residual={}B exact={} frontier={}",
            best.name,
            best.persistent_bytes,
            best.residual_bytes,
            exact,
            f.candidates.len()
        ));
        eprintln!(
            "{:<20} winner={:<16} persistent={:<7} residual={:<7} exact={}",
            c.class, best.name, best.persistent_bytes, best.residual_bytes, exact
        );
    }
    r.pass = all_pass;
    let path = emit_receipt(r).expect("emit");
    println!("PHASE J PASS: {all_pass}; receipt {path}");
}
