//! Canonical byte writer/reader primitives.
//!
//! All integers are little-endian fixed-width (canonical by construction: no
//! alternative encodings exist, so any byte string has exactly one decoding
//! and `encode(decode(b)) == b` holds for every syntactically valid `b`).
//! Lengths/counts use the fixed widths declared here; no overlong encodings
//! are possible, and every read is bounds-checked and returns `Reject`.

use crate::limits::Reject;

/// Pushes one u8.
#[inline(always)]
pub fn put_u8(w: &mut Vec<u8>, v: u8) {
    w.push(v);
}

#[inline(always)]
pub fn put_i32(w: &mut Vec<u8>, v: i32) {
    w.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn put_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn put_bytes(w: &mut Vec<u8>, b: &[u8]) {
    w.extend_from_slice(b);
}

/// Fixed cursor reader over a byte slice.  Every method is fallible and
/// never panics.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], Reject> {
        if self.remaining() < n {
            return Err(Reject::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8, Reject> {
        Ok(self.take(1)?[0])
    }
    pub fn i32(&mut self) -> Result<i32, Reject> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, Reject> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, Reject> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], Reject> {
        self.take(n)
    }
    /// Length-prefixed byte slice (u64 LE length).
    pub fn blob(&mut self) -> Result<&'a [u8], Reject> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| Reject::BytesExceedsLimit)?;
        self.take(n)
    }
    pub fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
}

/// Error wrapper used by encode-free code paths (validation, canonicalization).
pub type WireResult<T> = Result<T, Reject>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reader_roundtrip() {
        let mut w = Vec::new();
        put_u8(&mut w, 7);
        put_i32(&mut w, -123456);
        put_u32(&mut w, 4_000_000_000);
        put_u64(&mut w, u64::MAX);
        put_bytes(&mut w, b"xyz");
        let mut r = Reader::new(&w);
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.i32().unwrap(), -123456);
        assert_eq!(r.u32().unwrap(), 4_000_000_000);
        assert_eq!(r.u64().unwrap(), u64::MAX);
        assert_eq!(r.bytes(3).unwrap(), b"xyz");
        assert!(r.done());
    }

    #[test]
    fn reader_truncation_fails_closed() {
        let b = [1u8, 2, 3];
        let mut r = Reader::new(&b);
        assert!(r.u32().is_err());
        assert!(r.u64().is_err());
        assert_eq!(r.remaining(), 3);
    }
}
