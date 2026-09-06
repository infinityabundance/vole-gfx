//! Canonical-form checking.
//!
//! Two invariants (paper §canonical IR; see `docs/IR.md`):
//!
//! 1. `decode(encode(x)) == x` — the model round-trips losslessly.
//! 2. For every *valid canonical* byte string `b`:
//!    `encode(decode(b)) == b` — there is exactly one canonical encoding of a
//!    document, and the decoder rejects every non-canonical spelling.
//!
//! Because all wire integers are fixed-width and every ordering constraint is
//! enforced in `validate`, the decoder accepts exactly the set of canonical
//! encodings; `check_canonical` proves it by re-encoding.

use super::decode::decode;
use super::encode::encode;
use super::validate::validate;
use crate::limits::Reject;

/// Verify that `bytes` is a valid canonical encoding of a valid document.
pub fn check_canonical(bytes: &[u8]) -> Result<(), Reject> {
    let doc = decode(bytes)?;
    validate(&doc)?;
    let re = encode(&doc);
    if re != bytes {
        return Err(Reject::NonCanonicalOrder);
    }
    Ok(())
}

/// The canonical encoding of the document in `bytes`, if `bytes` is a valid
/// canonical encoding (otherwise the first rejection).
pub fn canonical_form(bytes: &[u8]) -> Result<Vec<u8>, Reject> {
    check_canonical(bytes)?;
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{ColorFormat, Rgba};
    use crate::fixed::{Affine, RectF};
    use crate::ir::{Document, Event, Instance, Object, Op};

    #[test]
    fn empty_document_is_canonical() {
        let d = Document::new();
        let b = encode(&d);
        check_canonical(&b).expect("empty canonical");
    }

    #[test]
    fn rich_document_is_canonical() {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 2,
            h: 2,
            data: (0..16).map(|i| (i * 7) as u8).collect(),
        });
        d.instances.push(Instance {
            object: 0,
            order: 3,
            layer: 1,
            transform: Affine::identity(),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
        d.events.push(Event {
            t: 1_000,
            ops: vec![
                Op::FillRect {
                    rect: RectF::from_px(0, 0, 2, 2),
                    color: Rgba::WHITE,
                },
                Op::InstTransform {
                    instance: 3,
                    transform: Affine::from_px_translation(4, 5),
                },
            ],
        });
        let b = encode(&d);
        check_canonical(&b).expect("rich doc canonical");
    }

    #[test]
    fn unsorted_event_times_not_canonical() {
        let mut d = Document::new();
        d.events.push(Event { t: 2, ops: vec![] });
        d.events.push(Event { t: 1, ops: vec![] });
        let b = encode(&d);
        assert_eq!(check_canonical(&b), Err(Reject::OutOfOrderTime));
    }

    #[test]
    fn corrupted_byte_is_not_canonical() {
        let d = Document::new();
        let mut b = encode(&d);
        let n = b.len();
        b[n - 1] ^= 0x40; // flip a body byte (body len 0 => this is the len field!)
        // body_len=0 => last byte is profile? For empty doc: header ends with
        // profile then u64 body_len=0. Flipping last byte corrupts body_len.
        assert!(check_canonical(&b).is_err());
    }
}
