//! Content identity and canonical output hashing.
//!
//! Canonical identity uses SHA-256 (`sha2`).  These functions are host-only;
//! device-side hashing (CUDA kernels) reuses the integer rules from
//! `fixed::hash64` for structural fingerprints but never defines content
//! identity.

use sha2::{Digest, Sha256};

/// Canonical SHA-256 content id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentId(pub [u8; 32]);

impl ContentId {
    pub const ZERO: ContentId = ContentId([0u8; 32]);

    /// Hex string form used in manifests and receipts.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            use core::fmt::Write;
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    pub fn from_hex(hex: &str) -> Option<ContentId> {
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, pair) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            let hi = hexval(pair[0])?;
            let lo = hexval(pair[1])?;
            out[i] = (hi << 4) | lo;
        }
        Some(ContentId(out))
    }
}

fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl core::fmt::Display for ContentId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl serde::Serialize for ContentId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for ContentId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(d)?;
        ContentId::from_hex(&hex).ok_or_else(|| serde::de::Error::custom("invalid content id"))
    }
}

/// SHA-256 of a byte slice.
pub fn sha256(data: &[u8]) -> ContentId {
    let mut h = Sha256::new();
    h.update(data);
    ContentId(h.finalize().into())
}

/// Incremental SHA-256 hasher (canonical output hashing over large buffers).
#[derive(Default)]
pub struct Hasher(Sha256);

impl Hasher {
    pub fn new() -> Self {
        Hasher(Sha256::new())
    }
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }
    pub fn update_u64_le(&mut self, v: u64) {
        self.0.update(v.to_le_bytes());
    }
    pub fn finish(self) -> ContentId {
        ContentId(self.0.finalize().into())
    }
}

/// Hash a 2D raster buffer in canonical row-major order into a `Hasher`.
pub fn hash_raster(h: &mut Hasher, buf: &[u8], stride_bytes: usize, w: usize, hh: usize) {
    for row in buf.chunks(stride_bytes).take(hh) {
        h.update(&row[..w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sha256() {
        // SHA-256("") and SHA-256("abc") published vectors.
        assert_eq!(
            sha256(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hex_roundtrip() {
        let id = sha256(b"vole-gfx");
        assert_eq!(ContentId::from_hex(&id.to_hex()), Some(id));
        assert_eq!(ContentId::from_hex("xyz"), None);
        assert_eq!(ContentId::from_hex("abc"), None);
    }
}
