//! Hard resource bounds for the `vole.gfx.u1` universe.
//!
//! Every descriptor is hostile input.  These caps are enforced by the IR
//! validator (`ir::validate`) before any state is built, and again by the
//! parser so that malformed input fails closed before allocation.
//!
//! This module (and `fixed`, `color`, and `hash`) are written `no_std`-clean so
//! the CUDA-device build (`--target nvptx64-nvidia-cuda --no-default-features
//! --features cuda-device -Zbuild-std=core`) can reuse the same constants and
//! arithmetic rules as the host.

/// Identifier of the exact universe implemented by this crate.
pub const UNIVERSE_U1: &str = "vole.gfx.u1";

/// Canonical format revision within U1.  Bump only with a full semantic audit.
pub const FORMAT_VERSION: u32 = 1;

/// Canonical magic bytes for a U1 IR container: `"VOLEGFX1"`.
pub const MAGIC: &[u8; 8] = b"VOLEGFX1";

/// Maximum supported object count in one document.
pub const MAX_OBJECTS: u64 = 1 << 20;

/// Maximum serialized bytes for a single object payload.
pub const MAX_OBJECT_BYTES: u64 = 1 << 30;

/// Maximum object pixel count (any single raster object).
pub const MAX_OBJECT_PIXELS: u64 = 1 << 28;

/// Maximum instance count.
pub const MAX_INSTANCES: u64 = 1 << 22;

/// Maximum geometry elements.
pub const MAX_GEOMETRY_ELEMENTS: u64 = 1 << 22;

/// Maximum path/vector segments per object (reserved for path profile).
pub const MAX_VECTOR_SEGMENTS: u64 = 1 << 20;

/// Maximum timeline events (transitions) in one document.
pub const MAX_TRANSITIONS: u64 = 1 << 24;

/// Maximum event ops per timeline event.
pub const MAX_OPS_PER_EVENT: u64 = 1 << 16;

/// Maximum dependency depth of any materialization.
pub const MAX_DEPENDENCY_DEPTH: u32 = 8;

/// Maximum trajectory segments per trajectory.
pub const MAX_TRAJECTORY_SEGMENTS: u32 = 4096;

/// Maximum generator parameter bytes.
pub const MAX_GENERATOR_PARAMS: u64 = 1 << 16;

/// Maximum generator work units per evaluated sample.
pub const MAX_GENERATOR_WORK_PER_SAMPLE: u64 = 1 << 12;

/// Maximum residual payload bytes in one document.
pub const MAX_RESIDUAL_BYTES: u64 = 1 << 31;

/// Maximum declared raster dimension (width or height) for outputs/objects.
pub const MAX_DIMENSION: u32 = 1 << 16;

/// Maximum output sample count for a single observation request.
pub const MAX_OUTPUT_SAMPLES: u64 = 1 << 30;

/// Maximum bytes a single observation request may produce.
pub const MAX_OUTPUT_BYTES: u64 = 1 << 31;

/// Maximum palette entries.
pub const MAX_PALETTE_ENTRIES: u32 = 1 << 16;

/// Maximum layers.
pub const MAX_LAYERS: u32 = 1 << 16;

/// Maximum sparse residual scatter records.
pub const MAX_RESIDUAL_RECORDS: u64 = 1 << 28;

/// Maximum inverse-factorization tree depth.
pub const MAX_FACTOR_DEPTH: u32 = 8;

/// Maximum inverse-factorization nodes.
pub const MAX_FACTOR_NODES: u64 = 1 << 16;

/// Coordinate bound in Q16.16 fixed units (±16384 px in U1 scene space).
pub const MAX_COORD: i64 = 1 << 30;

/// Maximum scene/surface pixel coordinate magnitude (and dimension): all
/// pixel positions and extents that touch scene space are limited so that
/// sample centers `(i<<16)|0x8000` always fit `i32` with headroom.
pub const MAX_SCENE_PX: i32 = (MAX_COORD >> 16) as i32;

/// Maximum per-axis dimension of a single object raster.
pub const MAX_OBJECT_DIM: u32 = MAX_SCENE_PX as u32;

/// Instance translation bound in Q16.16 (static or trajectory): translations
/// are capped at ±2²⁹ so that, together with a linear part bounded to ±2³⁰,
/// evaluated placements never overflow `i32`.
pub const INST_TRANSLATION_LIMIT: i64 = 1 << 29;

/// Result bound of the static linear placement (bounding box of the
/// transformed object extent, static translation included): ±(MAX_COORD -
/// INST_TRANSLATION_LIMIT) guarantees runtime (linear + trajectory) totals
/// stay below `i32::MAX`.
pub const PLACEMENT_BBOX_LIMIT: i64 = MAX_COORD - INST_TRANSLATION_LIMIT;

/// Affine coefficient bound in Q16.16 (scale magnitude ≤ 16.0).
pub const MAX_AFFINE_COEFF: i64 = 1 << 20;

/// Maximum document bytes accepted by a parser.
pub const MAX_DOCUMENT_BYTES: u64 = 1 << 31;

/// Execution work budget: total primitive/sample evaluation steps per request.
pub const MAX_EXECUTION_WORK: u64 = 1 << 40;

/// Reason codes for deterministic validation failure.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reject {
    BadMagic,
    UnsupportedVersion,
    UnknownUniverse,
    UnknownTag,
    Truncated,
    CountExceedsLimit,
    BytesExceedsLimit,
    DuplicateObject,
    MissingObject,
    ObjectTooLarge,
    DimensionTooLarge,
    CoordinateOutOfRange,
    CoefficientOutOfRange,
    Degenerate,
    ReferenceCycle,
    DependencyTooDeep,
    InvalidIndex,
    PayloadMismatch,
    OutOfOrderTime,
    NonCanonicalOrder,
    SelfReference,
    ResidualRegionInvalid,
    UnsupportedProfile,
    UnsupportedColor,
}

impl Reject {
    pub fn as_str(self) -> &'static str {
        match self {
            Reject::BadMagic => "bad-magic",
            Reject::UnsupportedVersion => "unsupported-version",
            Reject::UnknownUniverse => "unknown-universe",
            Reject::UnknownTag => "unknown-tag",
            Reject::Truncated => "truncated",
            Reject::CountExceedsLimit => "count-exceeds-limit",
            Reject::BytesExceedsLimit => "bytes-exceeds-limit",
            Reject::DuplicateObject => "duplicate-object",
            Reject::MissingObject => "missing-object",
            Reject::ObjectTooLarge => "object-too-large",
            Reject::DimensionTooLarge => "dimension-too-large",
            Reject::CoordinateOutOfRange => "coordinate-out-of-range",
            Reject::CoefficientOutOfRange => "coefficient-out-of-range",
            Reject::Degenerate => "degenerate",
            Reject::ReferenceCycle => "reference-cycle",
            Reject::DependencyTooDeep => "dependency-too-deep",
            Reject::InvalidIndex => "invalid-index",
            Reject::PayloadMismatch => "payload-mismatch",
            Reject::OutOfOrderTime => "out-of-order-time",
            Reject::NonCanonicalOrder => "non-canonical-order",
            Reject::SelfReference => "self-reference",
            Reject::ResidualRegionInvalid => "residual-region-invalid",
            Reject::UnsupportedProfile => "unsupported-profile",
            Reject::UnsupportedColor => "unsupported-color",
        }
    }
}

impl core::fmt::Display for Reject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::fmt::Debug for Reject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Concrete limits object.  All defaults are the U1 constants above; an
/// embedding may tighten them (never loosen below the declared caps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Limits {
    pub max_objects: u64,
    pub max_object_bytes: u64,
    pub max_object_pixels: u64,
    pub max_instances: u64,
    pub max_geometry_elements: u64,
    pub max_vector_segments: u64,
    pub max_transitions: u64,
    pub max_ops_per_event: u64,
    pub max_dependency_depth: u32,
    pub max_trajectory_segments: u32,
    pub max_generator_params: u64,
    pub max_generator_work_per_sample: u64,
    pub max_residual_bytes: u64,
    pub max_dimension: u32,
    pub max_output_samples: u64,
    pub max_output_bytes: u64,
    pub max_palette_entries: u32,
    pub max_layers: u32,
    pub max_residual_records: u64,
    pub max_factor_depth: u32,
    pub max_factor_nodes: u64,
    pub max_document_bytes: u64,
    pub max_execution_work: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self::u1()
    }
}

impl Limits {
    /// The normative U1 default limits.
    pub const fn u1() -> Self {
        Limits {
            max_objects: MAX_OBJECTS,
            max_object_bytes: MAX_OBJECT_BYTES,
            max_object_pixels: MAX_OBJECT_PIXELS,
            max_instances: MAX_INSTANCES,
            max_geometry_elements: MAX_GEOMETRY_ELEMENTS,
            max_vector_segments: MAX_VECTOR_SEGMENTS,
            max_transitions: MAX_TRANSITIONS,
            max_ops_per_event: MAX_OPS_PER_EVENT,
            max_dependency_depth: MAX_DEPENDENCY_DEPTH,
            max_trajectory_segments: MAX_TRAJECTORY_SEGMENTS,
            max_generator_params: MAX_GENERATOR_PARAMS,
            max_generator_work_per_sample: MAX_GENERATOR_WORK_PER_SAMPLE,
            max_residual_bytes: MAX_RESIDUAL_BYTES,
            max_dimension: MAX_DIMENSION,
            max_output_samples: MAX_OUTPUT_SAMPLES,
            max_output_bytes: MAX_OUTPUT_BYTES,
            max_palette_entries: MAX_PALETTE_ENTRIES,
            max_layers: MAX_LAYERS,
            max_residual_records: MAX_RESIDUAL_RECORDS,
            max_factor_depth: MAX_FACTOR_DEPTH,
            max_factor_nodes: MAX_FACTOR_NODES,
            max_document_bytes: MAX_DOCUMENT_BYTES,
            max_execution_work: MAX_EXECUTION_WORK,
        }
    }

    /// Sanity: a tightened limit set must never exceed the normative caps.
    pub fn validate(&self) -> Result<(), &'static str> {
        let u = Limits::u1();
        #[allow(clippy::too_many_arguments)] // plain field-wise comparison
        let ok = {
            self.max_objects <= u.max_objects
                && self.max_object_bytes <= u.max_object_bytes
                && self.max_object_pixels <= u.max_object_pixels
                && self.max_instances <= u.max_instances
                && self.max_geometry_elements <= u.max_geometry_elements
                && self.max_vector_segments <= u.max_vector_segments
                && self.max_transitions <= u.max_transitions
                && self.max_ops_per_event <= u.max_ops_per_event
                && self.max_dependency_depth <= u.max_dependency_depth
                && self.max_trajectory_segments <= u.max_trajectory_segments
                && self.max_generator_params <= u.max_generator_params
                && self.max_generator_work_per_sample <= u.max_generator_work_per_sample
                && self.max_residual_bytes <= u.max_residual_bytes
                && self.max_dimension <= u.max_dimension
                && self.max_output_samples <= u.max_output_samples
                && self.max_output_bytes <= u.max_output_bytes
                && self.max_palette_entries <= u.max_palette_entries
                && self.max_layers <= u.max_layers
                && self.max_residual_records <= u.max_residual_records
                && self.max_factor_depth <= u.max_factor_depth
                && self.max_factor_nodes <= u.max_factor_nodes
                && self.max_document_bytes <= u.max_document_bytes
                && self.max_execution_work <= u.max_execution_work
        };
        if ok {
            Ok(())
        } else {
            Err("limits exceed normative U1 caps")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_pass_validation() {
        Limits::u1().validate().expect("u1 defaults valid");
    }

    #[test]
    fn tightened_limits_pass() {
        let mut l = Limits::u1();
        l.max_objects = 16;
        l.validate().expect("tightened still valid");
    }

    #[test]
    fn loosened_limits_fail() {
        let mut l = Limits::u1();
        l.max_objects = u64::MAX;
        assert!(l.validate().is_err());
    }
}
