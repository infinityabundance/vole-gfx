//! Canonical IR encoder.  Every field layout here is mirrored exactly by
//! `ir::decode`; see `docs/IR.md` for the normative field tables.

use super::wire::{put_bytes, put_i32, put_u8, put_u32, put_u64};
use super::{Document, InstCreate, Instance, Object, Op, Profile, op};
use crate::fixed::{Affine, IRect, RectF};
use crate::limits::{FORMAT_VERSION, MAGIC, UNIVERSE_U1};

const NO_PALETTE: u32 = u32::MAX;

/// Encode a document to its canonical byte form.
pub fn encode(doc: &Document) -> Vec<u8> {
    let mut out = Vec::new();
    // ---- header ----
    put_bytes(&mut out, MAGIC);
    put_u32(&mut out, FORMAT_VERSION);
    put_u32(&mut out, UNIVERSE_U1.len() as u32);
    put_bytes(&mut out, UNIVERSE_U1.as_bytes());
    put_u8(&mut out, Profile::Exact as u8);

    let mut body = Vec::new();
    encode_body(&mut body, doc);
    put_u64(&mut out, body.len() as u64);
    put_bytes(&mut out, &body);
    out
}

fn encode_body(w: &mut Vec<u8>, doc: &Document) {
    // ---- objects ----
    put_u64(w, doc.objects.len() as u64);
    for o in &doc.objects {
        put_u8(w, o.kind_tag());
        let mut payload = Vec::new();
        match o {
            Object::Raster {
                format,
                w: ow,
                h: oh,
                data,
            } => {
                put_u8(&mut payload, format.tag());
                put_u32(&mut payload, *ow);
                put_u32(&mut payload, *oh);
                put_bytes(&mut payload, data);
            }
            Object::Palette { entries } => {
                put_u32(&mut payload, entries.len() as u32);
                for e in entries {
                    put_bytes(&mut payload, &e.to_bytes());
                }
            }
            Object::IndexedRaster {
                pal,
                w: ow,
                h: oh,
                indices,
            } => {
                put_u32(&mut payload, *pal);
                put_u32(&mut payload, *ow);
                put_u32(&mut payload, *oh);
                for ix in indices {
                    put_u32(&mut payload, *ix);
                }
            }
            Object::GeneratorField { family, params } => {
                put_u8(&mut payload, *family);
                put_u32(&mut payload, params.len() as u32);
                put_bytes(&mut payload, params);
            }
        }
        put_u64(w, payload.len() as u64);
        put_bytes(w, &payload);
    }
    // ---- trajectories ----
    put_u64(w, doc.trajectories.len() as u64);
    for tr in &doc.trajectories {
        put_u8(w, tr.kind);
        put_u32(w, tr.keys.len() as u32);
        for k in &tr.keys {
            put_u64(w, k.t);
            put_i32(w, k.tx);
            put_i32(w, k.ty);
        }
    }
    // ---- instances ----
    put_u64(w, doc.instances.len() as u64);
    for inst in &doc.instances {
        encode_instance(w, inst);
    }
    // ---- events ----
    put_u64(w, doc.events.len() as u64);
    for ev in &doc.events {
        put_u64(w, ev.t);
        put_u32(w, ev.ops.len() as u32);
        for o in &ev.ops {
            encode_op(w, o);
        }
    }
}

fn encode_instance(w: &mut Vec<u8>, i: &Instance) {
    put_u32(w, i.object);
    put_u32(w, i.order);
    put_u32(w, i.layer);
    encode_affine(w, &i.transform);
    put_u32(w, i.palette.unwrap_or(NO_PALETTE));
    encode_opt_rectf(w, i.clip);
    put_u32(w, i.trajectory);
    put_u8(w, i.visible as u8);
}

fn encode_op(w: &mut Vec<u8>, o: &Op) {
    match o {
        Op::InstCreate(c) => {
            put_u8(w, op::INST_CREATE);
            put_u32(w, c.object);
            put_u32(w, c.order);
            put_u32(w, c.layer);
            encode_affine(w, &c.transform);
            put_u32(w, c.palette.unwrap_or(NO_PALETTE));
            encode_opt_rectf(w, c.clip);
            put_u32(w, c.trajectory);
            put_u8(w, c.visible as u8);
        }
        Op::InstDelete { instance } => {
            put_u8(w, op::INST_DELETE);
            put_u32(w, *instance);
        }
        Op::InstTransform {
            instance,
            transform,
        } => {
            put_u8(w, op::INST_TRANSFORM);
            put_u32(w, *instance);
            encode_affine(w, transform);
        }
        Op::InstTrajectory {
            instance,
            trajectory,
        } => {
            put_u8(w, op::INST_TRAJECTORY);
            put_u32(w, *instance);
            put_u32(w, *trajectory);
        }
        Op::InstLayer { instance, layer } => {
            put_u8(w, op::INST_LAYER);
            put_u32(w, *instance);
            put_u32(w, *layer);
        }
        Op::InstVisible { instance, visible } => {
            put_u8(w, op::INST_VISIBLE);
            put_u32(w, *instance);
            put_u8(w, *visible as u8);
        }
        Op::InstClip { instance, clip } => {
            put_u8(w, op::INST_CLIP);
            put_u32(w, *instance);
            encode_opt_rectf(w, *clip);
        }
        Op::PaletteSet {
            object,
            offset,
            entries,
        } => {
            put_u8(w, op::PALETTE_SET);
            put_u32(w, *object);
            put_u32(w, *offset);
            put_u32(w, entries.len() as u32);
            for e in entries {
                put_bytes(w, &e.to_bytes());
            }
        }
        Op::CopyRegion { src, dx, dy } => {
            put_u8(w, op::COPY_REGION);
            encode_rectf(w, src);
            put_i32(w, *dx);
            put_i32(w, *dy);
        }
        Op::MoveRegion { src, dx, dy } => {
            put_u8(w, op::MOVE_REGION);
            encode_rectf(w, src);
            put_i32(w, *dx);
            put_i32(w, *dy);
        }
        Op::FillRect { rect, color } => {
            put_u8(w, op::FILL_RECT);
            encode_rectf(w, rect);
            put_bytes(w, &color.to_bytes());
        }
        Op::BindResidual(rb) => {
            put_u8(w, op::BIND_RESIDUAL);
            put_u8(w, rb.algebra);
            encode_irect(w, &rb.region);
            put_u8(w, rb.format.tag());
            put_u64(w, rb.payload.len() as u64);
            put_bytes(w, &rb.payload);
        }
    }
}

fn encode_affine(w: &mut Vec<u8>, a: &Affine) {
    put_i32(w, a.a);
    put_i32(w, a.b);
    put_i32(w, a.tx);
    put_i32(w, a.c);
    put_i32(w, a.d);
    put_i32(w, a.ty);
}

fn encode_rectf(w: &mut Vec<u8>, r: &RectF) {
    put_i32(w, r.x0);
    put_i32(w, r.y0);
    put_i32(w, r.x1);
    put_i32(w, r.y1);
}

fn encode_opt_rectf(w: &mut Vec<u8>, r: Option<RectF>) {
    match r {
        None => put_u8(w, 0),
        Some(r) => {
            put_u8(w, 1);
            encode_rectf(w, &r);
        }
    }
}

fn encode_irect(w: &mut Vec<u8>, r: &IRect) {
    put_i32(w, r.x0);
    put_i32(w, r.y0);
    put_i32(w, r.x1);
    put_i32(w, r.y1);
}

/// Sanity: an `InstCreate` mirrors `Instance` field-for-field.
#[allow(dead_code)]
fn inst_create_from_instance(i: &Instance) -> InstCreate {
    InstCreate {
        object: i.object,
        order: i.order,
        layer: i.layer,
        transform: i.transform,
        palette: i.palette,
        clip: i.clip,
        trajectory: i.trajectory,
        visible: i.visible,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout_prefix() {
        let d = Document::new();
        let bytes = encode(&d);
        assert_eq!(&bytes[..8], MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            FORMAT_VERSION
        );
    }
}
