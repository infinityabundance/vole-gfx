//! Phase B: canonical conformance vectors.
//!
//! Each vector is a deterministic scene whose full-surface materialization
//! hash is pinned below.  The scalar oracle is the only source of these
//! values; any semantic drift (arithmetic, coverage, ordering, residual
//! algebra, IR) breaks this test loudly.  The vectors are the conformance
//! target for every accelerated backend (AVX2/AVX-512/Rayon/CUDA) later.
//!
//! Regenerate (only after an intentional, receipted semantics change):
//! flip the `vector!` expectations after computing fresh hashes from the
//! scalar oracle and confirm every accelerated backend reproduces them.

mod common;

use common::*;
use vole_gfx::color::ColorFormat;
use vole_gfx::materialize::scalar::materialize_document;

fn h(doc: &vole_gfx::ir::Document, t: u64, w: u32, h: u32, format: ColorFormat) -> String {
    let m = materialize_document(doc, &surface(t, w, h, format)).expect("materialize");
    m.output.canonical_hash().to_hex()
}

macro_rules! vector {
    ($name:ident, $expected:expr, $expr:expr) => {
        #[test]
        fn $name() {
            let got = $expr;
            assert_eq!(
                got,
                $expected,
                "conformance vector {} drifted",
                stringify!($name)
            );
        }
    };
}

vector!(
    v1_fill_checker_copy,
    "6a10325504993ff5de852ee50df37ca4052e4620a4a8901f2cc8f2c05bec7ca8",
    h(&scene_v1(), 0, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v1_gray8,
    "108e61a3206271b8ee237033cb15a240cb33b912ca7c2ee40f703409f7f044d8",
    h(&scene_v1(), 0, 64, 64, ColorFormat::Gray8)
);
vector!(
    v2_palette_before,
    "3a4410accd1129ecbd8e3a3898938afbe7b95dc7d81b2aa2934e02309fdfe3f5",
    h(&scene_v2(), 0, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v2_palette_after,
    "ac872d6a9f6ed36248cf4278d4288a529d45c38c74149df6dff52672440140d7",
    h(&scene_v2(), 1000, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v2_palette_later,
    "ac872d6a9f6ed36248cf4278d4288a529d45c38c74149df6dff52672440140d7",
    h(&scene_v2(), 10_000_000, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v3_traj_t0,
    "4b89829d7f46b85a19cbaeff85436bb387f4e1a33b23c99ea544f3981ffeb13e",
    h(&scene_v3(), 0, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v3_traj_t16ms,
    "30aa19e1ee2c7d3e603df532115ed2da007b6bc89d61bab006894a00601d91de",
    h(&scene_v3(), 16_666_667, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v3_traj_t33ms,
    "0ee86e88d34235cd1007141a063f4d23a33be88ecfff3526150d29dd82c6c69a",
    h(&scene_v3(), 33_333_333, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v3_traj_irregular,
    "2699863c031949c9045773d72b1efc7244bf561445816390a418890ad0999474",
    h(&scene_v3(), 12_345_678, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v4_move_fill,
    "042efdd78dc067953634cf01590d37a3d9fd3dc6ed12e8f71e7c40123a1b82ec",
    h(&scene_v4(), 0, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v5_residuals,
    "d9593d5eb769c41d9d51866d7c699177857ab1fe34c373ad02da9c1315443daf",
    h(&scene_v5(), 5, 64, 64, ColorFormat::Rgba8)
);
vector!(
    v5_residuals_gray_skip,
    "c9ac7b0624824f844f6c7f3d50fab9741a8914e878467e8daaedca143a34d90b",
    h(&scene_v5(), 5, 64, 64, ColorFormat::Gray8)
);

#[test]
fn v1_vs_empty_differ() {
    let e = vole_gfx::ir::Document::new();
    assert_ne!(
        h(&e, 0, 64, 64, ColorFormat::Rgba8),
        h(&scene_v1(), 0, 64, 64, ColorFormat::Rgba8)
    );
}
