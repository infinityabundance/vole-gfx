//! Candidate selection: Pareto frontier first, explicit profile second
//! (paper §Search objective).
//!
//! A candidate is dominated when another candidate is no worse on every
//! declared dimension and strictly better on at least one.  The declared
//! dimensions of Phase I are complete, **deterministic** cost rows:
//!
//! * persistent bytes (canonical IR of the procedural explanation),
//! * residual bytes (sparse-overwrite payload),
//! * materialization work (a host-independent work model: requested samples
//!   x per-sample generator/raster cost — measured wall-clock latency is
//!   reported in receipts but never decides dominance, keeping the frontier
//!   deterministic),
//! * search work (bounded detector work units, deterministic).
//!
//! The non-dominated set is kept in deterministic detector order.  A profile
//! may then pick one point from the frontier (e.g. minimum total bytes); any
//! future weighted objective must persist its lambdas in the receipt instead
//! of being tuned on the final corpus.

use super::candidate::Candidate;

/// Declared cost dimensions of a candidate (the frontier axes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Costs {
    pub persistent_bytes: u64,
    pub residual_bytes: u64,
    pub materialize_work: u64,
    pub search_work: u64,
}

impl Costs {
    pub fn of(c: &Candidate) -> Costs {
        Costs {
            persistent_bytes: c.persistent_bytes,
            residual_bytes: c.residual_bytes,
            materialize_work: c.materialize_work,
            search_work: c.search_work,
        }
    }

    pub fn total_bytes(&self) -> u64 {
        self.persistent_bytes + self.residual_bytes
    }

    /// Dominance: `self` dominates `o` iff no worse everywhere and better
    /// somewhere.  Ties (identical costs) dominate neither way.
    pub fn dominates(&self, o: &Costs) -> bool {
        let ge = self.persistent_bytes <= o.persistent_bytes
            && self.residual_bytes <= o.residual_bytes
            && self.materialize_work <= o.materialize_work
            && self.search_work <= o.search_work;
        let strictly_better = self.persistent_bytes < o.persistent_bytes
            || self.residual_bytes < o.residual_bytes
            || self.materialize_work < o.materialize_work
            || self.search_work < o.search_work;
        ge && strictly_better
    }
}

/// An immutable Pareto set over the declared cost dimensions.
#[derive(Debug, Clone, Default)]
pub struct Frontier {
    /// Non-dominated candidates in deterministic (detector insertion) order.
    pub candidates: Vec<Candidate>,
}

impl Frontier {
    pub fn insert(&mut self, c: Candidate) {
        let costs = Costs::of(&c);
        // drop anything the new candidate dominates
        self.candidates.retain(|o| !costs.dominates(&Costs::of(o)));
        // keep the new candidate unless something already in the set
        // dominates it
        let dominated = self
            .candidates
            .iter()
            .any(|o| Costs::of(o).dominates(&costs));
        if !dominated {
            self.candidates.push(c);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Profile selection: the candidate with minimum total persistent +
    /// residual bytes (ties broken by lower materialization work, then
    /// detector order).
    pub fn min_total_bytes(&self) -> Option<&Candidate> {
        self.candidates.iter().min_by(|a, b| {
            let ca = Costs::of(a);
            let cb = Costs::of(b);
            ca.total_bytes()
                .cmp(&cb.total_bytes())
                .then(ca.materialize_work.cmp(&cb.materialize_work))
        })
    }

    /// Deterministic summary rows for receipts/reports.
    pub fn rows(&self) -> Vec<(String, u64, u64, u64, u64)> {
        // (name, persistent, residual, materialize_work, search_work)
        self.candidates
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    c.persistent_bytes,
                    c.residual_bytes,
                    c.materialize_work,
                    c.search_work,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse::candidate::Candidate;
    use crate::ir::Document;

    fn c(name: &str, p: u64, r: u64, m: u64, s: u64) -> Candidate {
        Candidate {
            name: name.into(),
            doc: Document::new(),
            persistent_bytes: p,
            residual_bytes: r,
            search_work: s,
            materialize_work: m,
            materialize_ns: 0,
            output_hash: String::new(),
        }
    }

    #[test]
    fn dominance_rules() {
        let a = Costs {
            persistent_bytes: 10,
            residual_bytes: 5,
            materialize_work: 1,
            search_work: 1,
        };
        let b = Costs {
            persistent_bytes: 10,
            residual_bytes: 5,
            materialize_work: 1,
            search_work: 1,
        };
        let c = Costs {
            persistent_bytes: 20,
            residual_bytes: 5,
            materialize_work: 1,
            search_work: 1,
        };
        let d = Costs {
            persistent_bytes: 10,
            residual_bytes: 4,
            materialize_work: 1,
            search_work: 1,
        };
        assert!(!a.dominates(&b)); // identical: neither dominates
        assert!(!b.dominates(&a));
        assert!(a.dominates(&c)); // strictly better persistent
        assert!(d.dominates(&a)); // strictly better residual
        assert!(!a.dominates(&d));
    }

    #[test]
    fn frontier_keeps_non_dominated_only() {
        let mut f = Frontier::default();
        f.insert(c("big", 100, 0, 1, 1));
        f.insert(c("mid", 50, 10, 2, 2));
        f.insert(c("small", 20, 20, 3, 3)); // tradeoff: fewer persistent, more residual
        f.insert(c("dominated", 80, 5, 1, 1)); // dominated by big? no; by mid? mid has p50<80 but r10>5... keep
        assert!(f.candidates.iter().any(|x| x.name == "big"));
        assert!(f.candidates.iter().any(|x| x.name == "mid"));
        assert!(f.candidates.iter().any(|x| x.name == "small"));
        // a candidate worse on every axis is removed
        f.insert(c("worse", 200, 30, 9, 9));
        assert!(!f.candidates.iter().any(|x| x.name == "worse"));
        // min total bytes: small = 40 < mid = 60 < dominated = 85 < big = 100
        assert_eq!(f.min_total_bytes().unwrap().name, "small");
    }
}
