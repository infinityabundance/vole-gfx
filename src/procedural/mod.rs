//! Seeded procedural state (`O = Γ(U, s, θ)`), Phase H.
//!
//! A generator field object carries a bounded, versioned parameter blob that
//! *is* the generator state (`s` = seed where the family has one, `θ` = the
//! canonical parameters), evaluated as a pure function of the local pixel
//! lattice.  The seed alone is never claimed to contain arbitrary
//! information; the full representation cost is the object extent, family,
//! params (including any embedded tables/tiles/palettes), and residual.
//!
//! Family semantics, canonical parameter encodings and cost estimates are
//! normative U1 text (see docs/EXACT_SEMANTICS.md §generators and
//! docs/IR.md); the scalar oracle evaluates them; accelerated backends must
//! reproduce the oracle byte-for-byte.  No generator family is Turing
//! complete; every sample cost is bounded and versioned.

pub mod families;
pub mod mix;

pub use families::{Field, Refs, decode_field};

/// Generator family identity tags (versioned, bounded, non-Turing-complete).
pub mod family {
    pub const CONSTANT: u8 = 1;
    pub const GRADIENT: u8 = 2;
    pub const PALETTE_FIELD: u8 = 3;
    pub const PERIODIC: u8 = 4;
    pub const TILED: u8 = 5;
    pub const AFFINE_REUSE: u8 = 6;
    pub const DETERMINISTIC_FIELD: u8 = 7;
    pub const FRACTAL: u8 = 8;
    pub const SDF: u8 = 9;
    pub const OBJECT_FAMILY: u8 = 10;

    /// Is this a known (implemented) family tag?
    pub fn is_known(tag: u8) -> bool {
        matches!(
            tag,
            CONSTANT
                | GRADIENT
                | PALETTE_FIELD
                | PERIODIC
                | TILED
                | AFFINE_REUSE
                | DETERMINISTIC_FIELD
                | FRACTAL
                | SDF
                | OBJECT_FAMILY
        )
    }

    pub fn name(tag: u8) -> &'static str {
        match tag {
            CONSTANT => "constant",
            GRADIENT => "gradient",
            PALETTE_FIELD => "palette-field",
            PERIODIC => "periodic",
            TILED => "tiled",
            AFFINE_REUSE => "affine-reuse",
            DETERMINISTIC_FIELD => "deterministic-field",
            FRACTAL => "fractal",
            SDF => "sdf",
            OBJECT_FAMILY => "object-family",
            _ => "unknown",
        }
    }
}

/// Phase H status marker (see docs/IMPLEMENTATION_STATE.md).
pub const PHASE_H_STATUS: &str = "implemented";

/// Validated generator objects only.  `validate_field_object` checks the
/// family params (canonical shape, extent-relative ranges, referenced-object
/// kinds and bounds).  Blob length vs `limits.max_generator_params` is
/// checked by the caller (ir::validate).
pub fn validate_field_object(
    family: u8,
    w: u32,
    h: u32,
    params: &[u8],
    objects: &[crate::ir::Object],
) -> Result<(), crate::limits::Reject> {
    use crate::limits::Reject;
    let field = decode_field(family, params, w, h)?;
    // referenced objects must exist and be samplable content (never another
    // generator or a palette)
    for o in field.referenced_objects() {
        match objects.get(o as usize) {
            Some(crate::ir::Object::Raster { .. })
            | Some(crate::ir::Object::IndexedRaster { .. }) => {}
            _ => return Err(Reject::InvalidIndex),
        }
    }
    Ok(())
}

/// Object builders for tests and courts (canonical params by construction).
pub mod build {
    use super::families::*;
    use crate::color::Rgba;
    use crate::fixed::Affine;
    use crate::ir::Object;

    /// Wrap a decoded `Field` into an IR object of the given extent.
    pub fn field_object(w: u32, h: u32, f: &super::Field) -> Object {
        Object::GeneratorField {
            family: f.family_tag(),
            w,
            h,
            params: f.encode_params(),
        }
    }

    pub fn constant(w: u32, h: u32, color: Rgba) -> Object {
        field_object(w, h, &super::Field::Constant(constant::Params { color }))
    }

    /// Builder: two anchor pixels (integer lattice px within the extent) and
    /// their colors.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_gradient(
        w: u32,
        h: u32,
        p0x: i32,
        p0y: i32,
        c0: Rgba,
        p1x: i32,
        p1y: i32,
        c1: Rgba,
    ) -> Object {
        field_object(
            w,
            h,
            &super::Field::Gradient(gradient::Params {
                mode: gradient::Mode::Linear {
                    p0x,
                    p0y,
                    c0,
                    p1x,
                    p1y,
                    c1,
                },
            }),
        )
    }

    pub fn bilinear_gradient(w: u32, h: u32, c00: Rgba, c10: Rgba, c01: Rgba, c11: Rgba) -> Object {
        field_object(
            w,
            h,
            &super::Field::Gradient(gradient::Params {
                mode: gradient::Mode::Bilinear { c00, c10, c01, c11 },
            }),
        )
    }

    pub fn palette_field(w: u32, h: u32, entries: Vec<Rgba>, seed: u64) -> Object {
        field_object(
            w,
            h,
            &super::Field::PaletteField(palette::Params {
                mode: palette::MODE_HASH,
                period: 1,
                seed,
                entries,
            }),
        )
    }

    pub fn palette_bands(w: u32, h: u32, period: u32, entries: Vec<Rgba>) -> Object {
        field_object(
            w,
            h,
            &super::Field::PaletteField(palette::Params {
                mode: palette::MODE_BAND_X,
                period,
                seed: 0,
                entries,
            }),
        )
    }

    pub fn stripes(w: u32, h: u32, px: u32, c0: Rgba, c1: Rgba) -> Object {
        field_object(
            w,
            h,
            &super::Field::Periodic(periodic::Params {
                mode: periodic::MODE_STRIPE_X,
                px,
                py: 1,
                c0,
                c1,
            }),
        )
    }

    pub fn stripe_y(w: u32, h: u32, py: u32, c0: Rgba, c1: Rgba) -> Object {
        field_object(
            w,
            h,
            &super::Field::Periodic(periodic::Params {
                mode: periodic::MODE_STRIPE_Y,
                px: 1,
                py,
                c0,
                c1,
            }),
        )
    }

    pub fn checker(w: u32, h: u32, px: u32, py: u32, c0: Rgba, c1: Rgba) -> Object {
        field_object(
            w,
            h,
            &super::Field::Periodic(periodic::Params {
                mode: periodic::MODE_CHECKER,
                px,
                py,
                c0,
                c1,
            }),
        )
    }

    pub fn tiled(w: u32, h: u32, tw: u32, th: u32, tile: Vec<Rgba>) -> Object {
        field_object(w, h, &super::Field::Tiled(tiled::Params { tw, th, tile }))
    }

    pub fn affine_reuse(w: u32, h: u32, object: u32, affine: Affine, outside: Rgba) -> Object {
        field_object(
            w,
            h,
            &super::Field::AffineReuse(affine::Params {
                object,
                affine,
                outside,
            }),
        )
    }

    pub fn noise_field(w: u32, h: u32, seed: u64) -> Object {
        field_object(w, h, &super::Field::FieldNoise(noise::FieldParams { seed }))
    }

    pub fn fractal(
        w: u32,
        h: u32,
        seed: u64,
        base_period: u32,
        octaves: u8,
        persistence: u32,
    ) -> Object {
        field_object(
            w,
            h,
            &super::Field::Fractal(noise::FractalParams {
                seed,
                base_period,
                octaves,
                persistence,
            }),
        )
    }

    pub fn disc(w: u32, h: u32, cx: i32, cy: i32, r: i32, inside: Rgba, outside: Rgba) -> Object {
        field_object(
            w,
            h,
            &super::Field::Sdf(sdf::Params {
                shape: sdf::SHAPE_CIRCLE,
                cx,
                cy,
                p1: r,
                p2: 0,
                inside,
                outside,
            }),
        )
    }

    pub fn object_family(
        w: u32,
        h: u32,
        cell_w: u32,
        cell_h: u32,
        objects: Vec<u32>,
        background: Rgba,
    ) -> Object {
        field_object(
            w,
            h,
            &super::Field::ObjectFamily(objects::Params {
                cell_w,
                cell_h,
                mode: objects::MODE_ORDERED,
                seed: 0,
                objects,
                background,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;
    use crate::ir::Document;

    #[test]
    fn known_family_tags() {
        for t in [
            family::CONSTANT,
            family::GRADIENT,
            family::PALETTE_FIELD,
            family::PERIODIC,
            family::TILED,
            family::AFFINE_REUSE,
            family::DETERMINISTIC_FIELD,
            family::FRACTAL,
            family::SDF,
            family::OBJECT_FAMILY,
        ] {
            assert!(family::is_known(t), "tag {t}");
        }
        assert!(!family::is_known(0));
        assert!(!family::is_known(11));
        assert!(!family::is_known(255));
    }

    #[test]
    fn all_builders_produce_validated_objects() {
        use crate::color::ColorFormat;
        use crate::fixed::Affine;
        use crate::ir::Object;
        let mut d = Document::new();
        d.objects.push(crate::ir::Object::Raster {
            format: ColorFormat::Rgba8,
            w: 4,
            h: 4,
            data: vec![0x99u8; 4 * 4 * 4],
        });
        d.objects
            .push(build::constant(16, 16, Rgba::new(1, 2, 3, 255)));
        d.objects.push(build::linear_gradient(
            16,
            16,
            0,
            0,
            Rgba::OPAQUE_BLACK,
            15,
            15,
            Rgba::WHITE,
        ));
        d.objects.push(build::bilinear_gradient(
            16,
            16,
            Rgba::OPAQUE_BLACK,
            Rgba::WHITE,
            Rgba::WHITE,
            Rgba::OPAQUE_BLACK,
        ));
        d.objects.push(build::palette_field(
            16,
            16,
            vec![Rgba::OPAQUE_BLACK, Rgba::WHITE],
            3,
        ));
        d.objects.push(build::palette_bands(
            16,
            16,
            4,
            vec![Rgba::OPAQUE_BLACK, Rgba::WHITE],
        ));
        d.objects
            .push(build::stripes(16, 16, 4, Rgba::OPAQUE_BLACK, Rgba::WHITE));
        d.objects.push(build::checker(
            16,
            16,
            4,
            4,
            Rgba::OPAQUE_BLACK,
            Rgba::WHITE,
        ));
        d.objects.push(build::tiled(
            16,
            16,
            2,
            2,
            vec![
                Rgba::OPAQUE_BLACK,
                Rgba::WHITE,
                Rgba::WHITE,
                Rgba::OPAQUE_BLACK,
            ],
        ));
        d.objects.push(build::affine_reuse(
            16,
            16,
            0,
            Affine::identity(),
            Rgba::OPAQUE_BLACK,
        ));
        d.objects.push(build::noise_field(16, 16, 5));
        d.objects.push(build::fractal(16, 16, 5, 4, 3, 32768));
        d.objects.push(build::disc(
            16,
            16,
            crate::fixed::fp(8),
            crate::fixed::fp(8),
            crate::fixed::fp(6),
            Rgba::WHITE,
            Rgba::OPAQUE_BLACK,
        ));
        d.objects.push(build::object_family(
            16,
            16,
            8,
            8,
            vec![0],
            Rgba::OPAQUE_BLACK,
        ));
        for o in d.objects.iter().skip(1) {
            // raster object at index 0 is the ref target; generators 1.. validate
            if let Object::GeneratorField {
                family,
                w,
                h,
                params,
            } = o
            {
                validate_field_object(*family, *w, *h, params, &d.objects)
                    .unwrap_or_else(|e| panic!("family {} must validate: {e}", *family));
            }
        }
    }
}
