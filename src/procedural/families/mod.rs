//! Generator family modules (`O = Gamma(U, s, theta)` per family).
//!
//! Each family defines a bounded, versioned parameter set with a canonical
//! wire encoding, semantic validation, a work-unit cost estimate, and an
//! exact deterministic sampler over the object's local pixel lattice.
//! Samplers are pure functions of integer inputs — no heap, no floats, no
//! RNG state — which is what makes them portable to SIMD/CUDA later.

pub mod affine;
pub mod constant;
pub mod gradient;
pub mod noise;
pub mod objects;
pub mod palette;
pub mod periodic;
pub mod sdf;
pub mod tiled;

use crate::color::Rgba;
use crate::ir::wire::Reader;
use crate::limits::Reject;

/// Shared exact helpers re-exported to family modules.
pub(crate) use super::mix::{blend, fmix2, fmix3, gray_byte};

/// Context for generator families that reference other objects (affine
/// reuse, object families).  `sample_object` returns the referenced object's
/// value at ref-local integer pixel `(u, v)`, or `None` out of bounds.
pub struct Refs<'a> {
    pub sample_object: &'a dyn Fn(u32, i32, i32) -> Option<Rgba>,
}

/// A decoded, validated generator field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Field {
    Constant(constant::Params),
    Gradient(gradient::Params),
    PaletteField(palette::Params),
    Periodic(periodic::Params),
    Tiled(tiled::Params),
    AffineReuse(affine::Params),
    /// Deterministic hash/gray-noise field.
    FieldNoise(noise::FieldParams),
    /// Bounded value-noise fractal.
    Fractal(noise::FractalParams),
    Sdf(sdf::Params),
    ObjectFamily(objects::Params),
}

impl Field {
    /// Canonical family tag (mirrors `Object::GeneratorField.family`).
    pub fn family_tag(&self) -> u8 {
        match self {
            Field::Constant(_) => crate::procedural::family::CONSTANT,
            Field::Gradient(_) => crate::procedural::family::GRADIENT,
            Field::PaletteField(_) => crate::procedural::family::PALETTE_FIELD,
            Field::Periodic(_) => crate::procedural::family::PERIODIC,
            Field::Tiled(_) => crate::procedural::family::TILED,
            Field::AffineReuse(_) => crate::procedural::family::AFFINE_REUSE,
            Field::FieldNoise(_) => crate::procedural::family::DETERMINISTIC_FIELD,
            Field::Fractal(_) => crate::procedural::family::FRACTAL,
            Field::Sdf(_) => crate::procedural::family::SDF,
            Field::ObjectFamily(_) => crate::procedural::family::OBJECT_FAMILY,
        }
    }

    /// Canonical parameter blob (the `params` bytes of the IR object).
    pub fn encode_params(&self) -> Vec<u8> {
        let mut w = Vec::new();
        match self {
            Field::Constant(p) => constant::encode(p, &mut w),
            Field::Gradient(p) => gradient::encode(p, &mut w),
            Field::PaletteField(p) => palette::encode(p, &mut w),
            Field::Periodic(p) => periodic::encode(p, &mut w),
            Field::Tiled(p) => tiled::encode(p, &mut w),
            Field::AffineReuse(p) => affine::encode(p, &mut w),
            Field::FieldNoise(p) => noise::encode_field(p, &mut w),
            Field::Fractal(p) => noise::encode_fractal(p, &mut w),
            Field::Sdf(p) => sdf::encode(p, &mut w),
            Field::ObjectFamily(p) => objects::encode(p, &mut w),
        }
        w
    }

    /// Upper-bound work units per evaluated sample (scheduler input; all
    /// families are far below `MAX_GENERATOR_WORK_PER_SAMPLE`).
    pub fn work(&self) -> u64 {
        match self {
            Field::Constant(p) => constant::work(p),
            Field::Gradient(p) => gradient::work(p),
            Field::PaletteField(p) => palette::work(p),
            Field::Periodic(p) => periodic::work(p),
            Field::Tiled(p) => tiled::work(p),
            Field::AffineReuse(p) => affine::work(p),
            Field::FieldNoise(p) => noise::work_field(p),
            Field::Fractal(p) => noise::work_fractal(p),
            Field::Sdf(p) => sdf::work(p),
            Field::ObjectFamily(p) => objects::work(p),
        }
    }

    /// Object-table references (affine reuse / object family), for validation.
    pub fn referenced_objects(&self) -> Vec<u32> {
        match self {
            Field::AffineReuse(p) => vec![p.object],
            Field::ObjectFamily(p) => p.objects.clone(),
            _ => Vec::new(),
        }
    }

    /// Exact sample at local pixel `(i, j)` of an extent-`(w, h)` object.
    /// `i in 0..w`, `j in 0..h` is guaranteed by the caller (raster-like
    /// bounds check happens before the call).
    pub fn sample(&self, w: u32, h: u32, i: u32, j: u32, refs: &Refs<'_>) -> Rgba {
        match self {
            Field::Constant(p) => constant::sample(p, i, j),
            Field::Gradient(p) => gradient::sample(p, w, h, i, j),
            Field::PaletteField(p) => palette::sample(p, w, h, i, j),
            Field::Periodic(p) => periodic::sample(p, i, j),
            Field::Tiled(p) => tiled::sample(p, i, j),
            Field::AffineReuse(p) => affine::sample(p, i, j, refs),
            Field::FieldNoise(p) => noise::sample_field(p, i, j),
            Field::Fractal(p) => noise::sample_fractal(p, i, j),
            Field::Sdf(p) => sdf::sample(p, i, j),
            Field::ObjectFamily(p) => objects::sample(p, w, h, i, j, refs),
        }
    }
}

/// Decode + semantically validate a parameter blob for a known family.
/// `extent` is the object's declared extent (needed for extent-relative
/// parameter checks).  Returns `Reject::UnknownTag` for unknown families and
/// a deterministic rejection for malformed/non-canonical blobs.
pub fn decode_field(
    family: u8,
    blob: &[u8],
    extent_w: u32,
    extent_h: u32,
) -> Result<Field, Reject> {
    let mut r = Reader::new(blob);
    let field = match family {
        crate::procedural::family::CONSTANT => Field::Constant(constant::decode(&mut r)?),
        crate::procedural::family::GRADIENT => {
            Field::Gradient(gradient::decode(&mut r, extent_w, extent_h)?)
        }
        crate::procedural::family::PALETTE_FIELD => Field::PaletteField(palette::decode(&mut r)?),
        crate::procedural::family::PERIODIC => Field::Periodic(periodic::decode(&mut r)?),
        crate::procedural::family::TILED => Field::Tiled(tiled::decode(&mut r)?),
        crate::procedural::family::AFFINE_REUSE => Field::AffineReuse(affine::decode(&mut r)?),
        crate::procedural::family::DETERMINISTIC_FIELD => {
            Field::FieldNoise(noise::decode_field(&mut r)?)
        }
        crate::procedural::family::FRACTAL => Field::Fractal(noise::decode_fractal(&mut r)?),
        crate::procedural::family::SDF => Field::Sdf(sdf::decode(&mut r, extent_w, extent_h)?),
        crate::procedural::family::OBJECT_FAMILY => Field::ObjectFamily(objects::decode(&mut r)?),
        _ => return Err(Reject::UnknownTag),
    };
    if !r.done() {
        return Err(Reject::PayloadMismatch);
    }
    Ok(field)
}
