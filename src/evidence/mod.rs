//! Evidence system: immutable JSON receipts, environment snapshots, and the
//! claims ledger.  Every court emits receipts; the README and the evidence
//! reports must never claim more than the receipts support.

pub mod environment;
pub mod receipt;

pub use receipt::Receipt;

/// Status values of the machine-readable claims ledger (paper §Claim
/// discipline).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    PriorArt,
    Proposal,
    Hypothesis,
    Demonstrated,
    Weakened,
    Falsified,
    Unsupported,
}

/// One row of the claims ledger: a claim scoped to the exact domain of its
/// supporting receipts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Claim {
    pub id: String,
    pub status: ClaimStatus,
    pub statement: String,
    pub domain: String,
    pub receipts: Vec<String>,
}

/// The machine-readable claims ledger (`evidence/claims.json`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Ledger {
    pub claims: Vec<Claim>,
}

impl Ledger {
    pub fn load(path: &str) -> Result<Ledger, String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&s).map_err(|e| e.to_string())
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let s = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, s).map_err(|e| e.to_string())
    }
}
