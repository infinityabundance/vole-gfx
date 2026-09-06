//! Universe identity for U1.
//!
//! The universe is the normative set of algorithms, tables, arithmetic
//! rules, and limits that define a conforming materialization.  This module
//! carries the machine-readable identity and supported surface; the full
//! normative text lives in `docs/UNIVERSE_U1.md`.

use crate::color::ColorFormat;
use crate::limits::{Limits, UNIVERSE_U1};

/// The versioned exact universe implemented by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Universe {
    /// `UNIVERSE_U1`
    pub name: &'static str,
    /// Normative limits of this universe.
    pub limits: Limits,
}

impl Universe {
    pub const fn u1() -> Self {
        Universe {
            name: UNIVERSE_U1,
            limits: Limits::u1(),
        }
    }

    /// Formats with normative materialization semantics in U1.
    pub fn supports_format(&self, f: ColorFormat) -> bool {
        matches!(f, ColorFormat::Gray8 | ColorFormat::Rgba8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u1_identity() {
        let u = Universe::u1();
        assert_eq!(u.name, "vole.gfx.u1");
        assert!(u.supports_format(ColorFormat::Rgba8));
        assert!(u.supports_format(ColorFormat::Gray8));
    }
}
