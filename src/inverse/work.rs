//! Deterministic search-work accounting for the inverse compiler.
//!
//! `search_work` is one of the four Pareto axes (see `frontier.rs`), so it
//! must be a faithful, host-independent count of the deterministic search
//! operations actually performed — not a flat `sample_count()` stand-in.
//! Detectors differ enormously in real cost: a constant scan is one pass over
//! the surface, while the tiled detector re-scans per candidate period and
//! structural grouping byte-compares crops against crops.
//!
//! # Boundary of the metric
//!
//! The U1 search-work metric counts **deterministic search operations only**, in
//! two phases:
//!
//! ```text
//! W_search = W_discovery + W_discrimination
//! ```
//!
//! * `W_discovery` (hypothesis discovery): operations that generate or scan
//!   candidate hypotheses — detector scans, membership probes, per-seed
//!   anchor evaluations in the seed sweep.
//! * `W_discrimination` (candidate discrimination): operations that verify or
//!   reject a specific hypothesis — full-surface survivor verification in the
//!   seed sweep, byte-exact crop grouping.
//!
//! Deliberately **outside** the metric (bounded by construction, or receipted
//! separately):
//!
//! * *preprocessing / format normalization* — deriving the gray surface from
//!   an asset (an O(w·h) clone for Gray8 or code walk for RGBA, identical for
//!   every asset of a given surface; the Phase-K detector and sweeps do not
//!   charge it);
//! * *representation construction* — lifting crops/tiles into candidate
//!   objects (bounded by content size and reflected in the candidate's
//!   persistent bytes);
//! * *wall-clock setup* — allocation, dispatch, context — receipted as time
//!   (`*_ns`), never as work units;
//! * scalar control flow and index arithmetic (`u32` width compares,
//!   `BTreeMap` bookkeeping).
//!
//! The boundary is frozen U1 text, not a per-court convenience: nobody may
//! argue later that the cost model was adjusted opportunistically to make a
//! candidate win.  A consequence of the boundary: format normalization is
//! format-symmetric in the metric (Gray8 and RGBA surfaces both pay zero
//! discovery/discrimination units for it), even though the raw walks differ
//! in bytes.
//!
//! # Frozen U1 conversion
//!
//! `SearchCounter` accumulates five disjoint operation classes; the frozen
//! conversion (documented here; never retuned per asset or per court) is
//!
//! ```text
//! search_work = pixels_read + code_compares + hash_ops
//!             + candidate_tests + crop_bytes_compared
//! ```
//!
//! Every operation is counted exactly where it occurs, at the operation's own
//! natural granularity:
//!
//! * `pixels_read` — one sample's code value materialized for a scan
//!   (`code_at` / `rgba_at` / `code_key`).  Counted per sample examined;
//!   early-exiting scans count only the samples actually examined.
//! * `code_compares` — one whole-code equality decision between two samples'
//!   code values (a row/column uniformity compare counts its `w`/`h` whole
//!   codes; the tiled period scan counts every sample pair it compares,
//!   including the failing one).
//! * `hash_ops` — one deterministic hash/seed evaluation.  The Phase-K seed
//!   sweep charges one per anchor/sample hash evaluated (exactly where
//!   executed, per backend); the Phase I/J detectors evaluate no hashes.
//! * `candidate_tests` — one full-candidate hypothesis verification.
//!   Reserved for the batched-search phases; in Phase I/J/K, candidate
//!   evaluation cost is a separate declared axis (`materialize_work`), so
//!   finalize does not charge the detector counters for it.
//! * `crop_bytes_compared` — one byte compared during byte-granular
//!   crop-content equality (structural grouping); early exit is data-exact.
//!
//! The model therefore over-counts nothing and under-counts only unmodeled
//! scalar control work; it is deterministic for a given asset and identical
//! across hosts.
//!
//! Detector attribution rule: a proposal's counter is the full deterministic
//! discovery/discrimination work of its detector (each operation increments
//! exactly once), so when a detector emits several proposals from one shared
//! scan each proposal carries the shared scan cost — a conservative
//! per-candidate charge that keeps frontier comparisons reproducible.
//!
//! Because `search_work` is a *frontier* axis, changing a detector's counting
//! changes the cost rows but never the semantics of what a proposal is;
//! exactness is still decided by byte-exact residual closure.

/// Deterministic counter of elementary search operations (see module docs).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchCounter {
    /// One sample's code value materialized for a scan.
    pub pixels_read: u64,
    /// One whole-code equality decision between two samples.
    pub code_compares: u64,
    /// One deterministic hash/seed evaluation (Phase K seed sweep: one per
    /// anchor/sample hash executed; zero in the Phase I/J detectors).
    pub hash_ops: u64,
    /// One full-candidate hypothesis verification (reserved for the
    /// batched-search phases L–M; zero through Phase K, where candidate
    /// evaluation is the separate `materialize_work` axis).
    pub candidate_tests: u64,
    /// One byte compared in a byte-granular crop equality test.
    pub crop_bytes_compared: u64,
}

impl SearchCounter {
    /// The frozen U1 search-work conversion (see module docs).
    #[inline]
    pub fn total_units(&self) -> u64 {
        self.pixels_read
            .saturating_add(self.code_compares)
            .saturating_add(self.hash_ops)
            .saturating_add(self.candidate_tests)
            .saturating_add(self.crop_bytes_compared)
    }

    #[inline]
    pub fn read_pixels(&mut self, n: u64) {
        self.pixels_read += n;
    }

    #[inline]
    pub fn compare_codes(&mut self, n: u64) {
        self.code_compares += n;
    }

    #[inline]
    pub fn eval_hashes(&mut self, n: u64) {
        self.hash_ops += n;
    }

    #[inline]
    pub fn test_candidates(&mut self, n: u64) {
        self.candidate_tests += n;
    }

    #[inline]
    pub fn compare_crop_bytes(&mut self, n: u64) {
        self.crop_bytes_compared += n;
    }

    /// A synthetic counter whose frozen total is exactly `units` (placed in
    /// `code_compares`).  Test/conformance helper: candidates whose search
    /// cost is declared outright instead of measured by a detector.
    pub fn from_total_units(units: u64) -> Self {
        SearchCounter {
            code_compares: units,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_is_the_frozen_sum() {
        let mut w = SearchCounter::default();
        w.read_pixels(10);
        w.compare_codes(4);
        w.eval_hashes(3);
        w.test_candidates(1);
        w.compare_crop_bytes(2);
        assert_eq!(w.total_units(), 20);
        assert_eq!(w.pixels_read, 10);
        assert_eq!(w.crop_bytes_compared, 2);
    }

    #[test]
    fn saturating_total_never_wraps() {
        let w = SearchCounter {
            pixels_read: u64::MAX,
            code_compares: u64::MAX,
            hash_ops: u64::MAX,
            candidate_tests: u64::MAX,
            crop_bytes_compared: u64::MAX,
        };
        assert_eq!(w.total_units(), u64::MAX);
    }

    #[test]
    fn from_total_units_recovers_the_total() {
        assert_eq!(SearchCounter::from_total_units(7).total_units(), 7);
        assert_eq!(SearchCounter::from_total_units(0).total_units(), 0);
    }

    #[test]
    fn counters_are_copy_and_deterministic() {
        let mut w = SearchCounter::default();
        w.compare_codes(3);
        let copy = w; // Copy: detectors can hand the same scan cost to several proposals
        assert_eq!(copy.total_units(), w.total_units());
        assert_eq!(copy, w);
    }
}
