//! Seeded procedural state (`O = Γ(U, s, θ)`).
//!
//! Generator families land in PHASE H.  This module exists now so the module
//! tree and the one-crate architecture are stable; it is **not** yet a
//! normative surface (documents containing generator objects are rejected by
//! the exact validator with `UnsupportedProfile` until Phase H lands).

/// Generator family identity tags (bounded, versioned, non-Turing-complete).
pub mod family {
    // Reserved tags; semantics normative from Phase H.
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
}

/// Phase H status marker (see docs/IMPLEMENTATION_STATE.md).
pub const PHASE_H_STATUS: &str = "not-implemented";
