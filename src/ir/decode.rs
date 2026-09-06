//! Canonical IR decoder.  Hostile-input safe: every read is bounds-checked,
//! every count is compared against the normative U1 limits before allocation,
//! and the parser never panics.  Failures are deterministic `Reject` codes.

use super::wire::Reader;
use super::{
    Document, Event, InstCreate, Instance, Object, Op, Profile, ResidualBind, TrajKey, Trajectory,
    kind, op, residual_algebra,
};
use crate::color::ColorFormat;
use crate::fixed::{Affine, IRect, RectF};
use crate::limits::Limits;
use crate::limits::{FORMAT_VERSION, MAGIC, Reject, UNIVERSE_U1};

const NO_PALETTE: u32 = u32::MAX;

/// Decode a canonical document byte string.
pub fn decode(bytes: &[u8]) -> Result<Document, Reject> {
    let mut r = Reader::new(bytes);
    // ---- header ----
    let magic = r.bytes(8)?;
    if magic != MAGIC {
        return Err(Reject::BadMagic);
    }
    let version = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(Reject::UnsupportedVersion);
    }
    let ulen = r.u32()? as usize;
    let universe = r.bytes(ulen)?;
    if universe != UNIVERSE_U1.as_bytes() {
        return Err(Reject::UnknownUniverse);
    }
    let profile = Profile::from_tag(r.u8()?).ok_or(Reject::UnsupportedProfile)?;
    if profile != Profile::Exact {
        return Err(Reject::UnsupportedProfile);
    }
    let body_len = r.u64()?;
    let body = r.bytes(usize::try_from(body_len).map_err(|_| Reject::BytesExceedsLimit)?)?;
    if !r.done() {
        return Err(Reject::PayloadMismatch);
    }
    let mut br = Reader::new(body);
    let doc = decode_body(&mut br)?;
    if !br.done() {
        return Err(Reject::PayloadMismatch);
    }
    Ok(doc)
}

fn decode_body(r: &mut Reader<'_>) -> Result<Document, Reject> {
    let limits = Limits::u1();
    // ---- objects ----
    let n_objects = checked_count(r.u64()?, limits.max_objects)?;
    let mut objects = Vec::with_capacity(n_objects);
    for _ in 0..n_objects {
        let kind_tag = r.u8()?;
        let payload_len = r.u64()?;
        let payload =
            r.bytes(usize::try_from(payload_len).map_err(|_| Reject::BytesExceedsLimit)?)?;
        let mut pr = Reader::new(payload);
        let obj = match kind_tag {
            kind::RASTER => {
                let format = ColorFormat::from_tag(pr.u8()?).ok_or(Reject::UnsupportedColor)?;
                let w = pr.u32()?;
                let h = pr.u32()?;
                let stride = format.bytes_per_sample() as u64;
                let expect = w as u64 * h as u64 * stride;
                if w as u64 > limits.max_dimension as u64 || h as u64 > limits.max_dimension as u64
                {
                    return Err(Reject::DimensionTooLarge);
                }
                if expect > limits.max_object_bytes {
                    return Err(Reject::BytesExceedsLimit);
                }
                if expect > limits.max_object_pixels * stride {
                    return Err(Reject::BytesExceedsLimit);
                }
                let n = usize::try_from(expect).map_err(|_| Reject::BytesExceedsLimit)?;
                if pr.remaining() != n {
                    return Err(Reject::PayloadMismatch);
                }
                let data = pr.bytes(n)?.to_vec();
                Object::Raster { format, w, h, data }
            }
            kind::PALETTE => {
                let count = pr.u32()?;
                if count as u64 > limits.max_palette_entries as u64 {
                    return Err(Reject::CountExceedsLimit);
                }
                let mut entries = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    let b = pr.bytes(4)?;
                    entries.push(crate::color::Rgba::from_bytes(b.try_into().unwrap()));
                }
                Object::Palette { entries }
            }
            kind::INDEXED_RASTER => {
                let pal = pr.u32()?;
                let w = pr.u32()?;
                let h = pr.u32()?;
                if w as u64 > limits.max_dimension as u64 || h as u64 > limits.max_dimension as u64
                {
                    return Err(Reject::DimensionTooLarge);
                }
                let npix = w as u64 * h as u64;
                if npix > limits.max_object_pixels {
                    return Err(Reject::BytesExceedsLimit);
                }
                if pal as u64 >= n_objects as u64 {
                    return Err(Reject::MissingObject);
                }
                let mut indices = Vec::with_capacity(npix as usize);
                for _ in 0..npix {
                    indices.push(pr.u32()?);
                }
                Object::IndexedRaster { pal, w, h, indices }
            }
            kind::GENERATOR_FIELD => {
                let family = pr.u8()?;
                let w = pr.u32()?;
                let h = pr.u32()?;
                if w as u64 > limits.max_dimension as u64 || h as u64 > limits.max_dimension as u64
                {
                    return Err(Reject::DimensionTooLarge);
                }
                let plen = pr.u32()? as usize;
                if plen as u64 > limits.max_generator_params {
                    return Err(Reject::BytesExceedsLimit);
                }
                if pr.remaining() != plen {
                    return Err(Reject::PayloadMismatch);
                }
                let params = pr.bytes(plen)?.to_vec();
                Object::GeneratorField {
                    family,
                    w,
                    h,
                    params,
                }
            }
            _ => return Err(Reject::UnknownTag),
        };
        if !pr.done() {
            return Err(Reject::PayloadMismatch);
        }
        objects.push(obj);
    }
    // ---- trajectories ----
    let n_traj = checked_count(r.u64()?, limits.max_transitions.min(1 << 20))?;
    let mut trajectories = Vec::with_capacity(n_traj);
    for _ in 0..n_traj {
        let kind = r.u8()?;
        let nkeys = r.u32()?;
        if nkeys as u64 > limits.max_trajectory_segments as u64 {
            return Err(Reject::CountExceedsLimit);
        }
        let mut keys = Vec::with_capacity(nkeys as usize);
        for _ in 0..nkeys {
            keys.push(TrajKey {
                t: r.u64()?,
                tx: r.i32()?,
                ty: r.i32()?,
            });
        }
        trajectories.push(Trajectory { kind, keys });
    }
    // ---- instances ----
    let n_inst = checked_count(r.u64()?, limits.max_instances)?;
    let mut instances = Vec::with_capacity(n_inst);
    for _ in 0..n_inst {
        let object = r.u32()?;
        let order = r.u32()?;
        let layer = r.u32()?;
        let transform = decode_affine(r)?;
        let pal = r.u32()?;
        let palette = if pal == NO_PALETTE { None } else { Some(pal) };
        let clip = decode_opt_rectf(r)?;
        let trajectory = r.u32()?;
        let visible = r.u8()? != 0;
        instances.push(Instance {
            object,
            order,
            layer,
            transform,
            palette,
            clip,
            trajectory,
            visible,
        });
    }
    // ---- events ----
    let n_events = checked_count(r.u64()?, limits.max_transitions)?;
    let mut events = Vec::with_capacity(n_events);
    for _ in 0..n_events {
        let t = r.u64()?;
        let nops = checked_count(r.u32()? as u64, limits.max_ops_per_event)?;
        let mut ops = Vec::with_capacity(nops);
        for _ in 0..nops {
            ops.push(decode_op(r)?);
        }
        events.push(Event { t, ops });
    }
    Ok(Document {
        limits,
        objects,
        trajectories,
        instances,
        events,
    })
}

fn decode_op(r: &mut Reader<'_>) -> Result<Op, Reject> {
    let tag = r.u8()?;
    Ok(match tag {
        op::INST_CREATE => {
            let object = r.u32()?;
            let order = r.u32()?;
            let layer = r.u32()?;
            let transform = decode_affine(r)?;
            let pal = r.u32()?;
            let palette = if pal == NO_PALETTE { None } else { Some(pal) };
            let clip = decode_opt_rectf(r)?;
            let trajectory = r.u32()?;
            let visible = r.u8()? != 0;
            Op::InstCreate(InstCreate {
                object,
                order,
                layer,
                transform,
                palette,
                clip,
                trajectory,
                visible,
            })
        }
        op::INST_DELETE => Op::InstDelete { instance: r.u32()? },
        op::INST_TRANSFORM => Op::InstTransform {
            instance: r.u32()?,
            transform: decode_affine(r)?,
        },
        op::INST_TRAJECTORY => Op::InstTrajectory {
            instance: r.u32()?,
            trajectory: r.u32()?,
        },
        op::INST_LAYER => Op::InstLayer {
            instance: r.u32()?,
            layer: r.u32()?,
        },
        op::INST_VISIBLE => Op::InstVisible {
            instance: r.u32()?,
            visible: r.u8()? != 0,
        },
        op::INST_CLIP => Op::InstClip {
            instance: r.u32()?,
            clip: decode_opt_rectf(r)?,
        },
        op::PALETTE_SET => {
            let object = r.u32()?;
            let offset = r.u32()?;
            let count = r.u32()?;
            if count as u64 > crate::limits::MAX_PALETTE_ENTRIES as u64 {
                return Err(Reject::CountExceedsLimit);
            }
            let mut entries = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let b = r.bytes(4)?;
                entries.push(crate::color::Rgba::from_bytes(b.try_into().unwrap()));
            }
            Op::PaletteSet {
                object,
                offset,
                entries,
            }
        }
        op::COPY_REGION => {
            let src = decode_rectf(r)?;
            let dx = r.i32()?;
            let dy = r.i32()?;
            Op::CopyRegion { src, dx, dy }
        }
        op::MOVE_REGION => {
            let src = decode_rectf(r)?;
            let dx = r.i32()?;
            let dy = r.i32()?;
            Op::MoveRegion { src, dx, dy }
        }
        op::FILL_RECT => {
            let rect = decode_rectf(r)?;
            let b = r.bytes(4)?;
            let color = crate::color::Rgba::from_bytes(b.try_into().unwrap());
            Op::FillRect { rect, color }
        }
        op::BIND_RESIDUAL => {
            let algebra = r.u8()?;
            if algebra != residual_algebra::SPARSE_OVERWRITE && algebra != residual_algebra::XOR {
                return Err(Reject::UnknownTag);
            }
            let region = decode_irect(r)?;
            let format = ColorFormat::from_tag(r.u8()?).ok_or(Reject::UnsupportedColor)?;
            let payload = r.blob()?;
            if payload.len() as u64 > crate::limits::MAX_RESIDUAL_BYTES {
                return Err(Reject::BytesExceedsLimit);
            }
            Op::BindResidual(ResidualBind {
                algebra,
                region,
                format,
                payload: payload.to_vec(),
            })
        }
        _ => return Err(Reject::UnknownTag),
    })
}

fn decode_affine(r: &mut Reader<'_>) -> Result<Affine, Reject> {
    Ok(Affine {
        a: r.i32()?,
        b: r.i32()?,
        tx: r.i32()?,
        c: r.i32()?,
        d: r.i32()?,
        ty: r.i32()?,
    })
}

fn decode_rectf(r: &mut Reader<'_>) -> Result<RectF, Reject> {
    Ok(RectF {
        x0: r.i32()?,
        y0: r.i32()?,
        x1: r.i32()?,
        y1: r.i32()?,
    })
}

fn decode_irect(r: &mut Reader<'_>) -> Result<IRect, Reject> {
    Ok(IRect {
        x0: r.i32()?,
        y0: r.i32()?,
        x1: r.i32()?,
        y1: r.i32()?,
    })
}

fn decode_opt_rectf(r: &mut Reader<'_>) -> Result<Option<RectF>, Reject> {
    match r.u8()? {
        0 => Ok(None),
        1 => Ok(Some(decode_rectf(r)?)),
        _ => Err(Reject::UnknownTag),
    }
}

fn checked_count(v: u64, cap: u64) -> Result<usize, Reject> {
    if v > cap {
        return Err(Reject::CountExceedsLimit);
    }
    usize::try_from(v).map_err(|_| Reject::CountExceedsLimit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;
    use crate::ir::encode::encode;

    #[test]
    fn decode_empty_roundtrip() {
        let d = Document::new();
        let b = encode(&d);
        assert_eq!(decode(&b).unwrap(), d);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let d = Document::new();
        let mut b = encode(&d);
        b[0] ^= 0xff;
        assert_eq!(decode(&b), Err(Reject::BadMagic));
    }

    #[test]
    fn decode_rejects_truncation() {
        let d = Document::new();
        let b = encode(&d);
        assert_eq!(decode(&b[..b.len() - 3]), Err(Reject::Truncated));
    }

    #[test]
    fn decode_rich_document() {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 2,
            h: 1,
            data: vec![255, 0, 0, 255, 0, 255, 0, 255],
        });
        d.objects.push(Object::Palette {
            entries: vec![Rgba::new(1, 2, 3, 4), Rgba::WHITE],
        });
        d.instances.push(Instance {
            object: 0,
            order: 0,
            layer: 0,
            transform: Affine::identity(),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
        d.events.push(Event {
            t: 1_000_000,
            ops: vec![Op::FillRect {
                rect: RectF::from_px(0, 0, 4, 4),
                color: Rgba::new(9, 9, 9, 255),
            }],
        });
        let b = encode(&d);
        assert_eq!(decode(&b).unwrap(), d);
    }

    #[test]
    fn decode_rejects_huge_object_count() {
        // crafted by hand: header + object count over the limit
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        b.extend_from_slice(&(UNIVERSE_U1.len() as u32).to_le_bytes());
        b.extend_from_slice(UNIVERSE_U1.as_bytes());
        b.push(Profile::Exact as u8);
        // body: n_objects > cap
        let mut body = Vec::new();
        body.extend_from_slice(&(crate::limits::MAX_OBJECTS + 1).to_le_bytes());
        b.extend_from_slice(&(body.len() as u64).to_le_bytes());
        b.extend_from_slice(&body);
        assert_eq!(decode(&b), Err(Reject::CountExceedsLimit));
    }

    #[test]
    fn decode_rejects_oversize_raster_payload() {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 65536,
            h: 65536,
            data: Vec::new(),
        });
        // encode would produce a malformed doc; instead validate rejects first
        assert!(crate::ir::validate::validate(&d).is_err());
    }
}
