//! Phase J tests: structural reuse detectors (sprite-on-field extraction,
//! shared-object sprite-repeat), composite explanations, and their negative
//! controls.

use vole_gfx::color::{ColorFormat, Rgba};
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

/// A small distinct sprite raster (10x8) with colors that never equal the
/// court field color (11,22,33).
fn sprite_object() -> Object {
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

/// Rasterize a composite doc over its declared surface (w x h at the origin).
fn rasterize_doc(d: &Document, w: u32, h: u32) -> Asset {
    vole_gfx::ir::validate::validate(d).unwrap();
    let req = ObservationRequest::full_surface(0, w, h, ColorFormat::Rgba8);
    let m = materialize_document(d, &req).unwrap();
    Asset::new(w, h, ColorFormat::Rgba8, m.output.data).unwrap()
}

fn names(f: &inverse::frontier::Frontier) -> Vec<String> {
    f.candidates.iter().map(|c| c.name.clone()).collect()
}

fn assert_all_survivors_exact(asset: &Asset, f: &inverse::frontier::Frontier) {
    assert!(!f.is_empty());
    for c in &f.candidates {
        let req = ObservationRequest::full_surface(0, asset.w, asset.h, asset.format);
        let m = materialize_document(&c.doc, &req).unwrap();
        assert_eq!(
            m.output.data, asset.data,
            "candidate {} must reproduce the asset exactly",
            c.name
        );
        let bytes = vole_gfx::ir::encode::encode(&c.doc);
        vole_gfx::ir::canonical::check_canonical(&bytes)
            .unwrap_or_else(|e| panic!("{} not canonical: {e}", c.name));
    }
}

/// Field + one sprite: the structural detector must extract the sprite as a
/// shared object over a constant field, residual-free.
#[test]
fn unbakes_sprite_on_field() {
    let mut d = Document::new();
    d.objects
        .push(build::constant(64, 48, Rgba::new(11, 22, 33, 255)));
    d.objects.push(sprite_object());
    d.instances.push(placed(0, 1, 0, 0, 0));
    d.instances.push(placed(1, 2, 1, 20, 16));
    let a = rasterize_doc(&d, 64, 48);

    let (f, best) = inverse::unbake_best(&a).unwrap();
    let ns = names(&f);
    assert!(
        ns.iter().any(|n| n == "sprite-on-field"),
        "sprite-on-field expected, got {ns:?}"
    );
    assert_eq!(best.name, "sprite-on-field", "byte-min winner");
    assert_eq!(best.residual_bytes, 0);
    // the explanation stores the crop once + the field: far below the literal
    assert!(best.persistent_bytes < a.data.len() as u64 / 2);
    assert_all_survivors_exact(&a, &f);
}

/// Field + the same sprite twice: the shared-object repeat explanation must
/// win (crop stored once, two placements, zero residual).
#[test]
fn unbakes_repeated_sprite() {
    let mut d = Document::new();
    d.objects
        .push(build::constant(64, 48, Rgba::new(11, 22, 33, 255)));
    d.objects.push(sprite_object());
    d.instances.push(placed(0, 1, 0, 0, 0));
    d.instances.push(placed(1, 2, 1, 10, 8));
    d.instances.push(placed(1, 3, 1, 44, 34));
    let a = rasterize_doc(&d, 64, 48);

    let (f, best) = inverse::unbake_best(&a).unwrap();
    let ns = names(&f);
    assert!(
        ns.iter().any(|n| n == "sprite-repeat"),
        "sprite-repeat expected, got {ns:?}"
    );
    assert_eq!(best.name, "sprite-repeat", "byte-min winner");
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

/// Gray8 composite: the crop/field split must also hold in gray code space.
#[test]
fn gray_sprite_on_field() {
    let mut d = Document::new();
    let mut data = Vec::new();
    for j in 0..40u32 {
        for i in 0..40u32 {
            data.push(if (8..28).contains(&i) && (10..22).contains(&j) {
                (i * 5 + j * 3) as u8 % 200 + 20
            } else {
                60
            });
        }
    }
    d.objects.push(Object::Raster {
        format: ColorFormat::Gray8,
        w: 40,
        h: 40,
        data,
    });
    d.instances.push(placed(0, 1, 0, 0, 0));
    vole_gfx::ir::validate::validate(&d).unwrap();
    let req = ObservationRequest::full_surface(0, 40, 40, ColorFormat::Gray8);
    let m = materialize_document(&d, &req).unwrap();
    let a = Asset::new(40, 40, ColorFormat::Gray8, m.output.data).unwrap();

    let (f, best) = inverse::unbake_best(&a).unwrap();
    let ns = names(&f);
    assert!(
        ns.iter().any(|n| n == "sprite-on-field"),
        "sprite-on-field expected in gray, got {ns:?}"
    );
    assert_eq!(best.residual_bytes, 0);
    assert_all_survivors_exact(&a, &f);
}

/// Random-bytes raster (negative control input): deterministic SHA-256-
/// derived pseudo-random RGBA bytes that no U1 generator family produces.
fn noise_asset() -> Asset {
    let mut data = Vec::new();
    for j in 0..24u32 {
        for i in 0..24u32 {
            let c =
                vole_gfx::hash::sha256(&[0x3c, (i >> 8) as u8, i as u8, (j >> 8) as u8, j as u8]);
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
    rasterize_doc(&d, 24, 24)
}

/// Negative control: no uniform field or seeded explanation exists; the
/// byte-min profile must stay with the literal fallback and no structural
/// candidate may carry a tiny residual.
#[test]
fn non_structural_random_noise_falls_back() {
    let a = noise_asset();
    let (f, best) = inverse::unbake_best(&a).unwrap();
    assert_eq!(best.name, "literal");
    for c in &f.candidates {
        if c.name.starts_with("sprite-") {
            assert!(
                c.residual_bytes >= a.data.len() as u64,
                "structural candidate {} must not explain noise with a small residual",
                c.name
            );
        }
    }
    assert_all_survivors_exact(&a, &f);
}

/// Determinism across repeated runs (composite proposals included).
#[test]
fn structural_unbake_deterministic() {
    let mut d = Document::new();
    d.objects
        .push(build::constant(48, 40, Rgba::new(11, 22, 33, 255)));
    d.objects.push(sprite_object());
    d.instances.push(placed(0, 1, 0, 0, 0));
    d.instances.push(placed(1, 2, 1, 5, 5));
    d.instances.push(placed(1, 3, 1, 30, 20));
    let a = rasterize_doc(&d, 48, 40);
    let f1 = inverse::unbake(&a).unwrap();
    let f2 = inverse::unbake(&a).unwrap();
    let n1: Vec<String> = f1.candidates.iter().map(|c| c.name.clone()).collect();
    let n2: Vec<String> = f2.candidates.iter().map(|c| c.name.clone()).collect();
    assert_eq!(n1, n2);
}
