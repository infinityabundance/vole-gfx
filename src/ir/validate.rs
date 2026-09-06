//! Semantic validation of in-memory documents.
//!
//! The decoder enforces *syntactic* caps (counts, lengths, sizes); this module
//! enforces *semantic* rules: range checks, reference integrity, canonical
//! ordering (strictly increasing event times, unique live instance ids), and
//! profile support.  Validation is deterministic and reports the first
//! violation with a `Reject` code.

use super::{Document, Instance, Object, Op, traj};
use crate::limits::Reject;
use std::collections::HashSet;

/// Validate a document for exact-profile use.
pub fn validate(doc: &Document) -> Result<(), Reject> {
    doc.limits
        .validate()
        .map_err(|_| Reject::CountExceedsLimit)?;
    validate_objects(doc)?;
    validate_trajectories(doc)?;
    validate_instances(doc)?;
    validate_events(doc)
}

fn validate_objects(doc: &Document) -> Result<(), Reject> {
    let n = doc.objects.len() as u64;
    if n > doc.limits.max_objects {
        return Err(Reject::CountExceedsLimit);
    }
    for (i, o) in doc.objects.iter().enumerate() {
        match o {
            Object::Raster {
                format: _,
                w,
                h,
                data,
            } => {
                let expect = match o {
                    Object::Raster { format, .. } => (*w as u64)
                        .checked_mul(*h as u64)
                        .and_then(|v| v.checked_mul(format.bytes_per_sample() as u64))
                        .ok_or(Reject::BytesExceedsLimit)?,
                    _ => unreachable!(),
                };
                if *w as u64 > doc.limits.max_dimension as u64
                    || *h as u64 > doc.limits.max_dimension as u64
                {
                    return Err(Reject::DimensionTooLarge);
                }
                if *w > crate::limits::MAX_OBJECT_DIM || *h > crate::limits::MAX_OBJECT_DIM {
                    return Err(Reject::DimensionTooLarge);
                }
                if expect != data.len() as u64 {
                    return Err(Reject::PayloadMismatch);
                }
                if expect > doc.limits.max_object_bytes {
                    return Err(Reject::BytesExceedsLimit);
                }
                if (*w as u64) * (*h as u64) > doc.limits.max_object_pixels {
                    return Err(Reject::BytesExceedsLimit);
                }
            }
            Object::Palette { entries } => {
                if entries.len() as u64 > doc.limits.max_palette_entries as u64 {
                    return Err(Reject::CountExceedsLimit);
                }
            }
            Object::IndexedRaster { pal, w, h, indices } => {
                if *w as u64 > doc.limits.max_dimension as u64
                    || *h as u64 > doc.limits.max_dimension as u64
                {
                    return Err(Reject::DimensionTooLarge);
                }
                if *w > crate::limits::MAX_OBJECT_DIM || *h > crate::limits::MAX_OBJECT_DIM {
                    return Err(Reject::DimensionTooLarge);
                }
                let npix = (*w as u64) * (*h as u64);
                if npix != indices.len() as u64 {
                    return Err(Reject::PayloadMismatch);
                }
                if npix > doc.limits.max_object_pixels {
                    return Err(Reject::BytesExceedsLimit);
                }
                let pal_ref = &doc.objects[*pal as usize];
                if !matches!(pal_ref, Object::Palette { .. }) {
                    return Err(Reject::InvalidIndex);
                }
                let pal_count = match pal_ref {
                    Object::Palette { entries } => entries.len() as u64,
                    _ => unreachable!(),
                };
                for ix in indices {
                    if *ix as u64 >= pal_count {
                        return Err(Reject::InvalidIndex);
                    }
                }
                let _ = i; // index itself is the identity; nothing else to check
            }
            Object::GeneratorField {
                family,
                w,
                h,
                params,
            } => {
                if params.len() as u64 > doc.limits.max_generator_params {
                    return Err(Reject::BytesExceedsLimit);
                }
                if *w as u64 > doc.limits.max_dimension as u64
                    || *h as u64 > doc.limits.max_dimension as u64
                {
                    return Err(Reject::DimensionTooLarge);
                }
                if *w > crate::limits::MAX_OBJECT_DIM || *h > crate::limits::MAX_OBJECT_DIM {
                    return Err(Reject::DimensionTooLarge);
                }
                if !crate::procedural::family::is_known(*family) {
                    return Err(Reject::UnknownTag);
                }
                // Family parameters: canonical shape, extent-relative ranges,
                // referenced-object kinds (full procedural semantics, Phase H).
                crate::procedural::validate_field_object(*family, *w, *h, params, &doc.objects)?;
            }
        }
    }
    Ok(())
}

fn validate_trajectories(doc: &Document) -> Result<(), Reject> {
    for tr in &doc.trajectories {
        if tr.kind != traj::LINEAR_TRANSLATION {
            return Err(Reject::UnknownTag);
        }
        if tr.keys.is_empty() || tr.keys.len() as u64 > doc.limits.max_trajectory_segments as u64 {
            return Err(Reject::CountExceedsLimit);
        }
        let mut prev_t: Option<u64> = None;
        for k in &tr.keys {
            if let Some(pt) = prev_t
                && k.t <= pt
            {
                return Err(Reject::NonCanonicalOrder);
            }
            if (k.tx as i64).abs() > crate::limits::INST_TRANSLATION_LIMIT
                || (k.ty as i64).abs() > crate::limits::INST_TRANSLATION_LIMIT
            {
                return Err(Reject::CoordinateOutOfRange);
            }
            prev_t = Some(k.t);
        }
    }
    Ok(())
}

fn validate_instances(doc: &Document) -> Result<(), Reject> {
    let mut ids: HashSet<u32> = HashSet::new();
    for inst in &doc.instances {
        check_instance(doc, inst, &mut ids)?;
    }
    Ok(())
}

fn check_instance(doc: &Document, inst: &Instance, ids: &mut HashSet<u32>) -> Result<(), Reject> {
    if !ids.insert(inst.order) {
        return Err(Reject::DuplicateObject);
    }
    check_placement(doc, inst.object, inst.transform, inst.trajectory)?;
    if inst.layer as u64 > doc.limits.max_layers as u64 {
        return Err(Reject::CountExceedsLimit);
    }
    if let Some(p) = inst.palette {
        if p as usize >= doc.objects.len() {
            return Err(Reject::MissingObject);
        }
        if !matches!(doc.objects[p as usize], Object::Palette { .. }) {
            return Err(Reject::InvalidIndex);
        }
    }
    if let Some(clip) = inst.clip
        && !clip.valid()
    {
        return Err(Reject::CoordinateOutOfRange);
    }
    if inst.trajectory != 0 && inst.trajectory as usize > doc.trajectories.len() {
        return Err(Reject::MissingObject);
    }
    Ok(())
}

/// Placement overflow guard: the static bounding box of the transformed
/// object extent must stay within `PLACEMENT_BBOX_LIMIT`, and the translation
/// (static or trajectory) within `INST_TRANSLATION_LIMIT`, so runtime
/// evaluation (linear part + trajectory translation) always fits `i32` with
/// the widening arithmetic of `Affine::apply`/inverse sampling.
fn check_placement(
    doc: &Document,
    object: u32,
    transform: crate::fixed::Affine,
    trajectory: u32,
) -> Result<(), Reject> {
    if !transform.in_range() {
        return Err(Reject::CoefficientOutOfRange);
    }
    let obj = doc
        .objects
        .get(object as usize)
        .ok_or(Reject::MissingObject)?;
    match obj {
        Object::Raster { w, h, .. }
        | Object::IndexedRaster { w, h, .. }
        | Object::GeneratorField { w, h, .. } => {
            let local = crate::fixed::RectF::from_px(0, 0, *w as i32, *h as i32);
            let bbox = transform.bounds_of(local);
            if (bbox.x0 as i64).abs() > crate::limits::PLACEMENT_BBOX_LIMIT
                || (bbox.y0 as i64).abs() > crate::limits::PLACEMENT_BBOX_LIMIT
                || (bbox.x1 as i64).abs() > crate::limits::PLACEMENT_BBOX_LIMIT
                || (bbox.y1 as i64).abs() > crate::limits::PLACEMENT_BBOX_LIMIT
            {
                return Err(Reject::CoordinateOutOfRange);
            }
        }
        Object::Palette { .. } => {}
    }
    // Translation caps (static part, and any trajectory's keys).
    let cap = |v: i32| (v as i64).abs() <= crate::limits::INST_TRANSLATION_LIMIT;
    if !cap(transform.tx) || !cap(transform.ty) {
        return Err(Reject::CoordinateOutOfRange);
    }
    if trajectory != 0
        && let Some(tr) = doc.trajectories.get(trajectory as usize - 1)
    {
        for k in &tr.keys {
            if !cap(k.tx) || !cap(k.ty) {
                return Err(Reject::CoordinateOutOfRange);
            }
        }
    }
    Ok(())
}

/// Simulate the timeline and validate op references and ordering.
fn validate_events(doc: &Document) -> Result<(), Reject> {
    // live ids -> (object ref, layer) minimal tracking
    let mut live: HashSet<u32> = HashSet::new();
    for inst in &doc.instances {
        live.insert(inst.order);
    }
    let mut prev_t: Option<u64> = None;
    for ev in &doc.events {
        if let Some(pt) = prev_t
            && ev.t <= pt
        {
            return Err(Reject::OutOfOrderTime);
        }
        prev_t = Some(ev.t);
        for o in &ev.ops {
            apply_op_validate(doc, o, &mut live)?;
        }
    }
    Ok(())
}

fn apply_op_validate(doc: &Document, o: &Op, live: &mut HashSet<u32>) -> Result<(), Reject> {
    let n_obj = doc.objects.len();
    let n_traj = doc.trajectories.len();
    match o {
        Op::InstCreate(c) => {
            if !live.insert(c.order) {
                return Err(Reject::DuplicateObject);
            }
            if c.object as usize >= n_obj {
                return Err(Reject::MissingObject);
            }
            if !doc.objects[c.object as usize].is_supported_exact() {
                return Err(Reject::UnsupportedProfile);
            }
            if matches!(doc.objects[c.object as usize], Object::Palette { .. }) {
                return Err(Reject::InvalidIndex);
            }
            check_placement(doc, c.object, c.transform, c.trajectory)?;
            if c.layer as u64 > doc.limits.max_layers as u64 {
                return Err(Reject::CountExceedsLimit);
            }
            if let Some(p) = c.palette {
                if p as usize >= n_obj {
                    return Err(Reject::MissingObject);
                }
                if !matches!(doc.objects[p as usize], Object::Palette { .. }) {
                    return Err(Reject::InvalidIndex);
                }
            }
            if let Some(clip) = c.clip
                && !clip.valid()
            {
                return Err(Reject::CoordinateOutOfRange);
            }
        }
        Op::InstDelete { instance } => {
            if !live.remove(instance) {
                return Err(Reject::InvalidIndex);
            }
        }
        Op::InstTransform {
            instance,
            transform,
        } => {
            if !live.contains(instance) {
                return Err(Reject::InvalidIndex);
            }
            if !transform.in_range() {
                return Err(Reject::CoefficientOutOfRange);
            }
        }
        Op::InstTrajectory {
            instance,
            trajectory,
        } => {
            if !live.contains(instance) {
                return Err(Reject::InvalidIndex);
            }
            if *trajectory != 0 && *trajectory as usize > n_traj {
                return Err(Reject::MissingObject);
            }
        }
        Op::InstLayer { instance, layer } => {
            if !live.contains(instance) {
                return Err(Reject::InvalidIndex);
            }
            if *layer as u64 > doc.limits.max_layers as u64 {
                return Err(Reject::CountExceedsLimit);
            }
        }
        Op::InstVisible { instance, .. } => {
            if !live.contains(instance) {
                return Err(Reject::InvalidIndex);
            }
        }
        Op::InstClip { instance, clip } => {
            if !live.contains(instance) {
                return Err(Reject::InvalidIndex);
            }
            if let Some(clip) = clip
                && !clip.valid()
            {
                return Err(Reject::CoordinateOutOfRange);
            }
        }
        Op::PaletteSet {
            object,
            offset,
            entries,
        } => {
            if *object as usize >= n_obj {
                return Err(Reject::MissingObject);
            }
            let pal_count = match &doc.objects[*object as usize] {
                Object::Palette { entries } => entries.len() as u64,
                _ => return Err(Reject::InvalidIndex),
            };
            if (*offset as u64) + entries.len() as u64 > pal_count {
                return Err(Reject::InvalidIndex);
            }
        }
        Op::CopyRegion { src, dx, dy } | Op::MoveRegion { src, dx, dy } => {
            if !src.valid() || src.is_empty() {
                return Err(Reject::ResidualRegionInvalid);
            }
            // translated destination must stay inside the valid coordinate cap
            let corners = [
                src.x0 as i64 + *dx as i64,
                src.x1 as i64 + *dx as i64,
                src.y0 as i64 + *dy as i64,
                src.y1 as i64 + *dy as i64,
            ];
            for c in corners {
                if c.abs() > crate::limits::MAX_COORD {
                    return Err(Reject::CoordinateOutOfRange);
                }
            }
        }
        Op::FillRect { rect, .. } => {
            if !rect.valid() || rect.is_empty() {
                return Err(Reject::ResidualRegionInvalid);
            }
        }
        Op::BindResidual(rb) => {
            if rb.region.is_empty() {
                return Err(Reject::ResidualRegionInvalid);
            }
            if rb.region.x0 < 0 || rb.region.y0 < 0 {
                return Err(Reject::ResidualRegionInvalid);
            }
            if rb.region.x1 as u64 > doc.limits.max_dimension as u64
                || rb.region.y1 as u64 > doc.limits.max_dimension as u64
            {
                return Err(Reject::DimensionTooLarge);
            }
            if rb.region.x1 > crate::limits::MAX_SCENE_PX
                || rb.region.y1 > crate::limits::MAX_SCENE_PX
            {
                return Err(Reject::DimensionTooLarge);
            }
            if rb.payload.len() as u64 > doc.limits.max_residual_bytes {
                return Err(Reject::BytesExceedsLimit);
            }
            // payload shape validated by residual module (records sorted etc.)
            crate::residual::validate_payload(rb.algebra, rb.format, &rb.region, &rb.payload)
                .map_err(|_| Reject::PayloadMismatch)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{ColorFormat, Rgba};
    use crate::fixed::{Affine, RectF, fp};
    use crate::ir::Document;

    fn valid_doc() -> Document {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 2,
            h: 2,
            data: vec![0u8; 16],
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
        d
    }

    #[test]
    fn empty_and_simple_pass() {
        assert!(validate(&Document::new()).is_ok());
        assert!(validate(&valid_doc()).is_ok());
    }

    #[test]
    fn missing_object_rejected() {
        let mut d = valid_doc();
        d.instances[0].object = 7;
        assert_eq!(validate(&d), Err(Reject::MissingObject));
    }

    #[test]
    fn bad_affine_rejected() {
        let mut d = valid_doc();
        d.instances[0].transform = Affine {
            a: fp(17),
            ..Affine::identity()
        };
        assert_eq!(validate(&d), Err(Reject::CoefficientOutOfRange));
    }

    #[test]
    fn event_time_order_enforced() {
        let mut d = valid_doc();
        d.events.push(crate::ir::Event { t: 5, ops: vec![] });
        d.events.push(crate::ir::Event { t: 5, ops: vec![] });
        assert_eq!(validate(&d), Err(Reject::OutOfOrderTime));
    }

    #[test]
    fn op_on_missing_instance_rejected() {
        let mut d = valid_doc();
        d.events.push(crate::ir::Event {
            t: 1,
            ops: vec![Op::InstDelete { instance: 99 }],
        });
        assert_eq!(validate(&d), Err(Reject::InvalidIndex));
    }

    #[test]
    fn fill_rect_valid_ok() {
        let mut d = valid_doc();
        d.events.push(crate::ir::Event {
            t: 1,
            ops: vec![Op::FillRect {
                rect: RectF::from_px(0, 0, 4, 4),
                color: Rgba::WHITE,
            }],
        });
        assert!(validate(&d).is_ok());
    }

    #[test]
    fn duplicate_live_instance_rejected() {
        let mut d = valid_doc();
        d.events.push(crate::ir::Event {
            t: 1,
            ops: vec![Op::InstCreate(crate::ir::InstCreate {
                object: 0,
                order: 0, // conflicts with initial instance
                layer: 0,
                transform: Affine::identity(),
                palette: None,
                clip: None,
                trajectory: 0,
                visible: true,
            })],
        });
        assert_eq!(validate(&d), Err(Reject::DuplicateObject));
    }
}
