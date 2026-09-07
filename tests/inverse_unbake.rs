//! Phase I tests: the scalar inverse procedural compiler ("unbaking").
//!
//! Every court builds a raster asset (from our own generator builders or from
//! noise), unbakes it, and asserts:
//!
//! * the frontier contains the expected exact explanation family (or the
//!   literal fallback for negative controls),
//! * every survivor materializes back to the asset byte-for-byte
//!   (residual-closed exact reconstruction),
//! * candidate documents are canonical (round-trip),
//! * the profile selection (min total bytes) prefers procedural state over
//!   the literal raster wherever an exact explanation is cheaper.

use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::inverse::{self, asset::Asset};
use vole_gfx::ir::{Document, Instance, Object};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::ObservationRequest;
use vole_gfx::procedural::build;

/// Materialize an object placed at the origin over its own extent.
fn rasterize(object: &Object, format: ColorFormat) -> Asset {
    let mut d = Document::new();
    d.objects.push(object.clone());
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: vole_gfx::fixed::Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    vole_gfx::ir::validate::validate(&d).unwrap();
    let (w, h) = match object {
        Object::Raster { w, h, .. } | Object::GeneratorField { w, h, .. } => (*w, *h),
        _ => panic!("asset object must be raster or field"),
    };
    let req = ObservationRequest::full_surface(0, w, h, format);
    let m = materialize_document(&d, &req).unwrap();
    Asset::new(w, h, format, m.output.data).unwrap()
}

fn names(f: &inverse::frontier::Frontier) -> Vec<String> {
    f.candidates.iter().map(|c| c.name.clone()).collect()
}

/// Every survivor must reproduce the asset exactly once its (possibly empty)
/// residual is applied, and its document must be canonical.
fn assert_all_survivors_exact(asset: &Asset, f: &inverse::frontier::Frontier) {
    assert!(!f.is_empty(), "literal fallback always present");
    for c in &f.candidates {
        let req = ObservationRequest::full_surface(0, asset.w, asset.h, asset.format);
        let m = materialize_document(&c.doc, &req).unwrap();
        assert_eq!(
            m.output.data, asset.data,
            "candidate {} must reproduce the asset exactly",
            c.name
        );
        // canonical round-trip of the explanation document
        let bytes = vole_gfx::ir::encode::encode(&c.doc);
        vole_gfx::ir::canonical::check_canonical(&bytes)
            .unwrap_or_else(|e| panic!("{} not canonical: {e}", c.name));
        assert_eq!(vole_gfx::ir::decode::decode(&bytes).unwrap(), c.doc);
    }
}

#[test]
fn unbakes_constant_asset() {
    let a = rasterize(
        &build::constant(32, 24, Rgba::new(7, 9, 11, 255)),
        ColorFormat::Rgba8,
    );
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(names(&f).iter().any(|n| n == "constant"));
    assert_eq!(best.name, "constant");
    assert_eq!(best.residual_bytes, 0);
    assert!(best.persistent_bytes < a.data.len() as u64);
    assert_all_survivors_exact(&a, &f);
}

#[test]
fn unbakes_checker_asset() {
    let a = rasterize(
        &build::checker(
            64,
            48,
            8,
            6,
            Rgba::new(20, 200, 20, 255),
            Rgba::new(240, 40, 240, 255),
        ),
        ColorFormat::Rgba8,
    );
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(
        names(&f).iter().any(|n| n == "periodic-checker"),
        "checker explanation expected, got {names:?}",
        names = names(&f)
    );
    assert_eq!(best.residual_bytes, 0);
    assert!(best.persistent_bytes < a.data.len() as u64);
    assert_all_survivors_exact(&a, &f);
}

#[test]
fn unbakes_palette_band_asset() {
    let a = rasterize(
        &build::palette_bands(
            60,
            20,
            5,
            vec![
                Rgba::new(10, 10, 10, 255),
                Rgba::new(120, 30, 30, 255),
                Rgba::new(30, 120, 30, 255),
            ],
        ),
        ColorFormat::Rgba8,
    );
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(
        names(&f).iter().any(|n| n == "palette-band-x"),
        "palette-band explanation expected, got {:?}",
        names(&f)
    );
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

#[test]
fn unbakes_tiled_asset() {
    // a 3x2 tile wrapped over a 15x10 extent
    let tile = vec![
        Rgba::new(200, 20, 20, 255),
        Rgba::new(20, 200, 20, 255),
        Rgba::new(20, 20, 200, 255),
        Rgba::new(200, 200, 20, 255),
        Rgba::new(200, 20, 200, 255),
        Rgba::new(20, 200, 200, 255),
    ];
    let a = rasterize(&build::tiled(15, 10, 3, 2, tile), ColorFormat::Rgba8);
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(
        names(&f).iter().any(|n| n == "tiled"),
        "tiled explanation expected, got {:?}",
        names(&f)
    );
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

#[test]
fn unbakes_horizontal_ramp_asset() {
    let a = rasterize(
        &build::linear_gradient(
            32,
            4,
            0,
            0,
            Rgba::new(0, 0, 0, 255),
            31,
            0,
            Rgba::new(255, 255, 255, 255),
        ),
        ColorFormat::Rgba8,
    );
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(
        names(&f).iter().any(|n| n == "gradient-h-ramp"),
        "gradient explanation expected, got {:?}",
        names(&f)
    );
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

#[test]
fn unbakes_bilinear_asset() {
    let a = rasterize(
        &build::bilinear_gradient(
            24,
            16,
            Rgba::new(10, 10, 200, 255),
            Rgba::new(200, 10, 10, 255),
            Rgba::new(10, 200, 10, 255),
            Rgba::new(200, 200, 10, 255),
        ),
        ColorFormat::Rgba8,
    );
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(
        names(&f).iter().any(|n| n == "gradient-bilinear"),
        "bilinear explanation expected, got {:?}",
        names(&f)
    );
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

#[test]
fn gray_asset_unbakes_in_gray_space() {
    // constant gray asset
    let mut data = vec![0u8; 40 * 40];
    data.fill(173);
    let a = Asset::new(40, 40, ColorFormat::Gray8, data).unwrap();
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert!(names(&f).iter().any(|n| n == "constant"));
    assert_eq!(best.name, "constant");
    assert_all_survivors_exact(&a, &f);

    // gray vertical bands (two gray values)
    let mut data = vec![0u8; 60 * 16];
    for j in 0..16 {
        for i in 0..60 {
            data[j * 60 + i] = if (i / 10) % 2 == 0 { 40 } else { 210 };
        }
    }
    let a = Asset::new(60, 16, ColorFormat::Gray8, data).unwrap();
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

/// Negative control: high-entropy, independently generated SHA-256 bytes
/// with no exact match in the evaluated detector universe must fall back to
/// the literal raster under the byte-min profile (its generator proposals
/// carry a near-full residual and are dominated or dominated-in-profile).
/// The bytes are canonicalized through the materializer: a real baked asset
/// is a materializer output, so fully-transparent codes canonicalize to
/// (0,0,0,0).
#[test]
fn random_noise_asset_falls_back_to_literal() {
    // deterministic SHA-256-derived pseudo-random RGBA raster (per sample)
    let mut data = Vec::new();
    for j in 0..24u32 {
        for i in 0..24u32 {
            let c =
                vole_gfx::hash::sha256(&[0x99, (i >> 8) as u8, i as u8, (j >> 8) as u8, j as u8]);
            data.extend_from_slice(&c.0[..4]);
        }
    }
    let a = rasterize(
        &Object::Raster {
            format: ColorFormat::Rgba8,
            w: 24,
            h: 24,
            data,
        },
        ColorFormat::Rgba8,
    );
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert_eq!(
        best.name, "literal",
        "byte-min profile must pick the fallback"
    );
    // every non-literal survivor (if any) must carry a large residual: its
    // explanation is mostly residual, which is honest negative evidence
    for c in &f.candidates {
        if c.name != "literal" {
            assert!(
                c.residual_bytes >= a.data.len() as u64,
                "{} should not explain random bytes with a tiny residual",
                c.name
            );
        }
    }
    assert_all_survivors_exact(&a, &f);
}

/// Phase K positive: a deterministic-field asset (gray noise from a known
/// seed) must be unbaked to the exact `seeded-field` explanation with zero
/// residual, beating the literal fallback on the byte-min profile.
#[test]
fn unbakes_seeded_field_asset() {
    for seed in [0u64, 7, 4242, 65535] {
        let a = rasterize(&build::noise_field(24, 24, seed), ColorFormat::Rgba8);
        let (f, best) = inverse::unbake_best(&a).unwrap();
        let ns = names(&f);
        assert!(
            ns.iter().any(|n| n == "seeded-field"),
            "seeded-field expected for seed {seed}, got {ns:?}"
        );
        assert_eq!(best.name, "seeded-field", "seed {seed}");
        assert_eq!(best.residual_bytes, 0, "seed {seed}");
        assert!(
            best.persistent_bytes < a.data.len() as u64 / 4,
            "seed {seed}: persistent {} vs literal {}",
            best.persistent_bytes,
            a.data.len()
        );
        assert_all_survivors_exact(&a, &f);
    }
}

/// The compiler's deterministic ordering: same asset, same frontier names.
#[test]
fn unbake_is_deterministic() {
    let a = rasterize(
        &build::checker(
            32,
            24,
            4,
            6,
            Rgba::new(10, 10, 10, 255),
            Rgba::new(250, 250, 250, 255),
        ),
        ColorFormat::Rgba8,
    );
    let f1 = inverse::unbake(&a).unwrap();
    let f2 = inverse::unbake(&a).unwrap();
    let n1: Vec<&str> = f1.candidates.iter().map(|c| c.name.as_str()).collect();
    let n2: Vec<&str> = f2.candidates.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(n1, n2);
    let b1 = f1.min_total_bytes().unwrap().name.clone();
    let b2 = f2.min_total_bytes().unwrap().name.clone();
    assert_eq!(b1, b2);
}

/// Every generator-family court doc whose output is an asset must produce an
/// explanation whose materialization equals the asset (identity court across
/// the remaining families: fractal noise and object-family/affine-reuse are
/// non-exact-by-detector in Phase K — only the deterministic gray-noise
/// family is seed-searched — but must still round-trip through the fallback).
#[test]
fn identity_roundtrip_through_fallback_for_non_detected_families() {
    // fractal: no detector explains it exactly; the fallback must round-trip
    let a = rasterize(&build::fractal(32, 32, 5, 4, 3, 32768), ColorFormat::Rgba8);
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert_eq!(best.name, "literal");
    assert_all_survivors_exact(&a, &f);
}
