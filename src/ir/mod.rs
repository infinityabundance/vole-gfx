//! Canonical binary IR for the `vole.gfx.u1` universe.
//!
//! The IR is an explicit, hostile-input-safe wire model — it is **not** a
//! `serde` dump of Rust structs.
//!
//! * Container: `MAGIC` (8 bytes), `FORMAT_VERSION` (u32 LE), universe string
//!   (length-prefixed), profile tag (u8).  All integers are little-endian
//!   fixed-width — canonical by construction (no alternative spellings exist).
//! * Sections appear in fixed order: objects, trajectories, instances,
//!   timeline events.  Object identity is the implicit section index; object
//!   `id` fields elsewhere refer to that index (0-based).
//! * Timeline events are strictly increasing in time; op arrays are
//!   order-significant (composition order) and kept verbatim.
//! * Malformed input fails closed with a deterministic `Reject` reason.
//! * A valid canonical byte string satisfies `decode(encode(x)) == x` and
//!   `encode(decode(b)) == b`.

pub mod canonical;
pub mod decode;
pub mod encode;
pub mod validate;
pub mod wire;

use crate::color::{ColorFormat, Rgba};
use crate::fixed::{Affine, IRect, RectF};
use crate::limits::{Limits, Reject};

/// Universe identifier carried by every container.
pub const UNIVERSE: &str = crate::limits::UNIVERSE_U1;

/// Normative profiles understood by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Profile {
    /// Exact reconstruction profile (the only implemented profile).
    Exact = 1,
}

impl Profile {
    pub fn from_tag(tag: u8) -> Option<Profile> {
        match tag {
            1 => Some(Profile::Exact),
            _ => None,
        }
    }
}

/// Object kind tags.
pub mod kind {
    pub const RASTER: u8 = 1;
    pub const PALETTE: u8 = 2;
    pub const INDEXED_RASTER: u8 = 3;
    pub const GENERATOR_FIELD: u8 = 4; // semantic support arrives with procedural phase
}

/// Op tags (state transition / structural raster operations).
pub mod op {
    pub const INST_CREATE: u8 = 1;
    pub const INST_DELETE: u8 = 2;
    pub const INST_TRANSFORM: u8 = 3;
    pub const INST_TRAJECTORY: u8 = 4;
    pub const INST_LAYER: u8 = 5;
    pub const INST_VISIBLE: u8 = 6;
    pub const INST_CLIP: u8 = 7;
    pub const PALETTE_SET: u8 = 8;
    pub const COPY_REGION: u8 = 9;
    pub const MOVE_REGION: u8 = 10;
    pub const FILL_RECT: u8 = 11;
    pub const BIND_RESIDUAL: u8 = 12; // Phase B
}

/// Trajectory kind tags.
pub mod traj {
    /// Piecewise-linear translation keyframes (exact fixed interpolation).
    pub const LINEAR_TRANSLATION: u8 = 1;
}

/// An immutable content object.  Identity = index in the document object
/// table (content-addressed by SHA-256 of its canonical encoding externally).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Object {
    /// Pixel raster.  `data` is exactly `w*h*stride` bytes, row-major.
    Raster {
        format: ColorFormat,
        w: u32,
        h: u32,
        data: Vec<u8>,
    },
    /// Color palette for indexed content.
    Palette { entries: Vec<Rgba> },
    /// Palette-indexed raster (palette object index `pal`, indices row-major).
    IndexedRaster {
        pal: u32,
        w: u32,
        h: u32,
        indices: Vec<u32>,
    },
    /// Deterministic generator field; params opaque to the IR layer and
    /// validated by the procedural module.  Grammar only until Phase H.
    GeneratorField { family: u8, params: Vec<u8> },
}

impl Object {
    pub fn kind_tag(&self) -> u8 {
        match self {
            Object::Raster { .. } => kind::RASTER,
            Object::Palette { .. } => kind::PALETTE,
            Object::IndexedRaster { .. } => kind::INDEXED_RASTER,
            Object::GeneratorField { .. } => kind::GENERATOR_FIELD,
        }
    }
    pub fn is_supported_exact(&self) -> bool {
        !matches!(self, Object::GeneratorField { .. })
    }
}

/// Piecewise-linear translation trajectory; `keys` are strictly increasing in
/// time.  Interpolation is exact fixed-point between adjacent keyframes;
/// times outside the key range clamp to the nearer end key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trajectory {
    pub kind: u8,
    pub keys: Vec<TrajKey>,
}

/// A trajectory keyframe: absolute translation at time `t` (ns).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrajKey {
    pub t: u64,
    pub tx: i32,
    pub ty: i32,
}

/// A persistent instance of an object in the composition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance {
    /// Object table index.
    pub object: u32,
    /// Insertion index (stable, distinct across the instance lifetime).
    pub order: u32,
    /// Composition layer; ascending layers draw later (on top).
    pub layer: u32,
    /// Placement transform (object-local -> scene).
    pub transform: Affine,
    /// Palette override: object table index of a `Palette`, or `None`.
    pub palette: Option<u32>,
    /// Scene-space clip; `None` = unclipped.
    pub clip: Option<RectF>,
    /// Trajectory table index + 1; 0 = static (transform used verbatim).
    pub trajectory: u32,
    pub visible: bool,
}

/// A timeline event at absolute time `t` (integer nanoseconds).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub t: u64,
    pub ops: Vec<Op>,
}

/// Structural state-transition and raster operations (ordered per event).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    InstCreate(InstCreate),
    InstDelete {
        instance: u32,
    },
    InstTransform {
        instance: u32,
        transform: Affine,
    },
    InstTrajectory {
        instance: u32,
        trajectory: u32,
    },
    InstLayer {
        instance: u32,
        layer: u32,
    },
    InstVisible {
        instance: u32,
        visible: bool,
    },
    InstClip {
        instance: u32,
        clip: Option<RectF>,
    },
    PaletteSet {
        object: u32,
        offset: u32,
        entries: Vec<Rgba>,
    },
    /// Copy `src` (scene space) to `src` translated by `(dx, dy)` using the
    /// same-time composed instance base (see docs/EXACT_SEMANTICS.md).
    CopyRegion {
        src: RectF,
        dx: i32,
        dy: i32,
    },
    /// Move: like CopyRegion but the source region is cleared afterwards.
    MoveRegion {
        src: RectF,
        dx: i32,
        dy: i32,
    },
    FillRect {
        rect: RectF,
        color: Rgba,
    },
    /// Bind a pixel-domain residual (canonical raster coordinates).
    BindResidual(ResidualBind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstCreate {
    pub object: u32,
    pub order: u32,
    pub layer: u32,
    pub transform: Affine,
    pub palette: Option<u32>,
    pub clip: Option<RectF>,
    pub trajectory: u32,
    pub visible: bool,
}

/// Residual algebra tags (explicit residual semantics, Phase B).
pub mod residual_algebra {
    /// Sparse overwrite: payload = sorted records of (x,y,value-bytes).
    pub const SPARSE_OVERWRITE: u8 = 1;
    /// XOR: payload = sorted records of (x,y,xor-bytes).
    pub const XOR: u8 = 2;
}

/// A residual binding attached to a timeline event.  Residuals are
/// pixel-domain corrections applied to the *canonical surface* after
/// procedural composition (see docs/EXACT_SEMANTICS.md §residual).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualBind {
    pub algebra: u8,
    /// Target rectangle in canonical output pixel coordinates.
    pub region: IRect,
    /// Format of the corrected surface.
    pub format: ColorFormat,
    /// Payload depends on algebra (sorted records; see residual module).
    pub payload: Vec<u8>,
}

/// Complete U1 document (the serializable "procedural program").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// Tightened limits used when this document was *encoded*.  A decoder
    /// always enforces the normative U1 caps and, when `limits` is present,
    /// the declared caps as an upper bound on what it will accept.
    pub limits: Limits,
    pub objects: Vec<Object>,
    pub trajectories: Vec<Trajectory>,
    pub instances: Vec<Instance>,
    pub events: Vec<Event>,
}

impl Document {
    pub fn new() -> Self {
        Document {
            limits: Limits::u1(),
            objects: Vec::new(),
            trajectories: Vec::new(),
            instances: Vec::new(),
            events: Vec::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.instances.is_empty() && self.events.is_empty()
    }
}

impl Default for Document {
    fn default() -> Self {
        Document::new()
    }
}

/// Convert a fixed-point scene rect bound violation into a `Reject`.
pub fn reject_rect(r: &RectF) -> Result<(), Reject> {
    if !r.valid() {
        return Err(Reject::CoordinateOutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_tags() {
        assert_eq!(
            Profile::from_tag(Profile::Exact as u8),
            Some(Profile::Exact)
        );
        assert_eq!(Profile::from_tag(0), None);
        assert_eq!(Profile::from_tag(9), None);
    }

    #[test]
    fn kind_tags_roundtrip() {
        let o = Object::Palette {
            entries: vec![Rgba::WHITE],
        };
        assert_eq!(o.kind_tag(), kind::PALETTE);
    }

    #[test]
    fn doc_default_empty() {
        assert!(Document::new().is_empty());
    }
}
