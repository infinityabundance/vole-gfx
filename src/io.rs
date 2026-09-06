//! Minimal deterministic image I/O for corpus work and visual verification.
//!
//! PPM/PGM (binary P6/P5) only: trivial, license-clean, exact.  Reading is
//! bounds-checked and fails closed; no panics on hostile input.

use crate::color::ColorFormat;
use crate::limits::Reject;

/// Write Gray8 or RGBA8 samples as binary PGM (P5) or PPM (P6).
pub fn write_pnm(
    path: &std::path::Path,
    w: u32,
    h: u32,
    format: ColorFormat,
    data: &[u8],
) -> Result<(), String> {
    let expected = w as u64 * h as u64 * format.bytes_per_sample() as u64;
    if data.len() as u64 != expected {
        return Err("pnm: data length mismatch".into());
    }
    let (magic, maxval, per) = match format {
        ColorFormat::Gray8 => ("P5", 255u32, 1usize),
        ColorFormat::Rgba8 => ("P6", 255, 3usize), // alpha dropped on write
    };
    let mut out = Vec::with_capacity(data.len() + 64);
    out.extend_from_slice(format!("{magic}\n{w} {h}\n{maxval}\n").as_bytes());
    match format {
        ColorFormat::Gray8 => out.extend_from_slice(data),
        ColorFormat::Rgba8 => {
            for px in data.chunks_exact(4) {
                out.extend_from_slice(&px[..3]);
            }
        }
    }
    let _ = per;
    std::fs::write(path, out).map_err(|e| e.to_string())
}

/// Parse a binary PGM/P5 or PPM/P6 file.  Returns `(w, h, format, samples)`.
pub fn read_pnm(bytes: &[u8]) -> Result<(u32, u32, ColorFormat, Vec<u8>), Reject> {
    let (magic, mut pos) = token(bytes, 0)?;
    let max_px = crate::limits::MAX_OBJECT_DIM as u64;
    match magic {
        b"P5" | b"P6" => {}
        _ => return Err(Reject::UnknownTag),
    }
    let w = parse_num(bytes, &mut pos)?;
    let h = parse_num(bytes, &mut pos)?;
    let maxval = parse_num(bytes, &mut pos)?;
    if maxval != 255 {
        return Err(Reject::UnsupportedProfile);
    }
    if w == 0 || h == 0 {
        return Err(Reject::Degenerate);
    }
    if w as u64 > max_px || h as u64 > max_px {
        return Err(Reject::DimensionTooLarge);
    }
    // single whitespace consumed by parse_num already; skip one optional extra
    while pos < bytes.len()
        && (bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b' ' || bytes[pos] == b'\t')
    {
        pos += 1;
    }
    let (format, bps, ch) = match magic {
        b"P5" => (ColorFormat::Gray8, 1usize, 1usize),
        _ => (ColorFormat::Rgba8, 4usize, 3usize),
    };
    let need = w as u64 * h as u64 * ch as u64;
    if (bytes.len() as u64) - (pos as u64) < need {
        return Err(Reject::Truncated);
    }
    let mut data = Vec::with_capacity((w as u64 * h as u64 * bps as u64) as usize);
    match format {
        ColorFormat::Gray8 => data.extend_from_slice(&bytes[pos..pos + need as usize]),
        ColorFormat::Rgba8 => {
            for px in bytes[pos..pos + need as usize].chunks_exact(3) {
                data.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
    }
    Ok((w, h, format, data))
}

fn token(bytes: &[u8], start: usize) -> Result<(&[u8], usize), Reject> {
    // skip whitespace/comments
    let mut i = start;
    loop {
        while i < bytes.len()
            && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n' || bytes[i] == b'\r')
        {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        break;
    }
    if i >= bytes.len() {
        return Err(Reject::Truncated);
    }
    let s = i;
    while i < bytes.len() && !matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    if i == s {
        return Err(Reject::Truncated);
    }
    Ok((&bytes[s..i], i))
}

fn parse_num(bytes: &[u8], pos: &mut usize) -> Result<u32, Reject> {
    let (tok, next) = token(bytes, *pos)?;
    *pos = next;
    let s = core::str::from_utf8(tok).map_err(|_| Reject::UnknownTag)?;
    let v: u64 = s.parse().map_err(|_| Reject::UnknownTag)?;
    if v > u32::MAX as u64 {
        return Err(Reject::DimensionTooLarge);
    }
    Ok(v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pnm_roundtrip_rgba() {
        let w = 3u32;
        let h = 2u32;
        let data: Vec<u8> = (0..w * h)
            .flat_map(|i| vec![i as u8, 200, 100, 255])
            .collect();
        let path = std::env::temp_dir().join(format!("vole_test_{}.ppm", std::process::id()));
        write_pnm(&path, w, h, ColorFormat::Rgba8, &data).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let (rw, rh, fmt, out) = read_pnm(&bytes).unwrap();
        assert_eq!((rw, rh, fmt), (w, h, ColorFormat::Rgba8));
        assert_eq!(out, data);
    }

    #[test]
    fn pnm_rejects_truncated() {
        assert_eq!(read_pnm(b"P6\n4 4\n255\nxxx"), Err(Reject::Truncated));
        assert_eq!(read_pnm(b"P6\n4 4\n255"), Err(Reject::Truncated));
        assert_eq!(read_pnm(b"P9\n1 1\n255\n0"), Err(Reject::UnknownTag));
    }
}
