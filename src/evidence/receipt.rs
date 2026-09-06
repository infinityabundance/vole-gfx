//! Immutable JSON evidence receipts.
//!
//! Every court emits one receipt with the schema below.  Receipts are never
//! overwritten: file names embed a content-derived run id plus a uniqueness
//! suffix, and the canonical hash of the receipt body is written inside it.

use super::environment::Environment;
use crate::hash::{ContentId, sha256};
use serde::Serialize;
use std::collections::BTreeMap;

/// Current evidence schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// The receipt body (flat, machine-readable; all units nanoseconds/bytes).
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Receipt {
    pub schema_version: u32,
    pub run_id: String,
    pub timestamp_unix_ms: u64,
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub command: String,
    pub court: String,
    pub universe: String,
    pub profile: String,
    pub backend: String,
    pub exact: bool,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    pub canonical_hash: Option<String>,
    pub reference_hash: Option<String>,
    pub metrics: BTreeMap<String, u64>,
    pub quality_metric: Option<String>,
    pub quality_value: Option<f64>,
    pub persistent_bytes: u64,
    pub residual_bytes: u64,
    pub samples_requested: u64,
    pub samples_evaluated: u64,
    pub pass: bool,
    pub unsupported_reason: Option<String>,
    pub notes: Vec<String>,
    pub environment: Environment,
    /// SHA-256 of the serialized body fields (excluding this field).
    pub receipt_hash: String,
}

impl Receipt {
    pub fn new(court: &str, backend: &str) -> Receipt {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Receipt {
            schema_version: SCHEMA_VERSION,
            run_id: String::new(),
            timestamp_unix_ms: now,
            git_commit: None,
            git_dirty: None,
            command: std::env::args().collect::<Vec<_>>().join(" "),
            court: court.to_string(),
            universe: crate::limits::UNIVERSE_U1.to_string(),
            profile: "exact-u1".to_string(),
            backend: backend.to_string(),
            exact: true,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            canonical_hash: None,
            reference_hash: None,
            metrics: BTreeMap::new(),
            quality_metric: None,
            quality_value: None,
            persistent_bytes: 0,
            residual_bytes: 0,
            samples_requested: 0,
            samples_evaluated: 0,
            pass: false,
            unsupported_reason: None,
            notes: Vec::new(),
            environment: Environment::snapshot(),
            receipt_hash: String::new(),
        }
    }

    /// Finalize: fill environment git fields, hash the body, and return the
    /// canonical JSON bytes.
    pub fn finalize(mut self) -> Result<Vec<u8>, String> {
        if self.git_commit.is_none() {
            self.git_commit = self.environment.git_commit.clone();
            self.git_dirty = self.environment.git_dirty;
        }
        // content-derived run id over the body
        let body = serde_json::to_vec(&self).map_err(|e| e.to_string())?;
        let body_id = sha256(&body);
        self.run_id = format!("{}-{}", self.court, &body_id.to_hex()[..16]);
        // receipt hash over everything except itself
        let mut with_id = serde_json::to_value(&self).map_err(|e| e.to_string())?;
        if let Some(obj) = with_id.as_object_mut() {
            obj.remove("receipt_hash");
        }
        let canonical = serde_json::to_vec(&with_id).map_err(|e| e.to_string())?;
        let h = sha256(&canonical);
        self.receipt_hash = h.to_hex();
        serde_json::to_vec_pretty(&self).map_err(|e| e.to_string())
    }

    /// Write the receipt to `evidence/receipts/<run_id>.json`, refusing to
    /// overwrite an existing file with different content.
    pub fn emit(&self, bytes: &[u8]) -> Result<String, String> {
        let dir = std::path::Path::new("evidence").join("receipts");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.json", self.run_id));
        if path.exists() {
            let existing = std::fs::read(&path).map_err(|e| e.to_string())?;
            if existing != bytes {
                return Err(format!(
                    "receipt collision with differing content: {}",
                    path.display()
                ));
            }
            return Ok(path.display().to_string());
        }
        std::fs::write(&path, bytes).map_err(|e| e.to_string())?;
        Ok(path.display().to_string())
    }

    /// Verify that a stored receipt parses and its hash is self-consistent.
    pub fn verify(bytes: &[u8]) -> Result<ContentId, String> {
        let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let obj = v.as_object().ok_or("not an object")?;
        let stored = obj
            .get("receipt_hash")
            .and_then(|h| h.as_str())
            .ok_or("missing receipt_hash")?;
        let mut body = v.clone();
        if let Some(o) = body.as_object_mut() {
            o.remove("receipt_hash");
        }
        let canonical = serde_json::to_vec(&body).map_err(|e| e.to_string())?;
        let h = sha256(&canonical);
        if h.to_hex() != stored {
            return Err("receipt hash mismatch".into());
        }
        Ok(h)
    }
}

/// Convenience constructor used by courts: build, finalize, emit, return path.
pub fn emit_receipt(receipt: Receipt) -> Result<String, String> {
    let bytes = receipt.finalize()?;
    let r: Receipt = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    r.emit(&bytes)
}
