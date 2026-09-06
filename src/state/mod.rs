//! Persistent state resolution: the materializer input.
//!
//! `Scene` is the deterministic projection of a `Document` onto one explicit
//! time coordinate: the live instance set (in normative composition order),
//! the effective palettes (base entries + applied patches), and the ordered
//! structural operations (fill / copy / move / residual bindings) whose event
//! time is ≤ the request time.  Raster surfaces are never stored here — this
//! is the "no mandatory re-baking" boundary.

pub mod index;
pub mod trajectory;
pub mod transition;

use crate::color::{ColorFormat, Rgba};
use crate::fixed::{IRect, RectF, Vec2};
use crate::ir::{Document, Op};
use crate::limits::Reject;
use std::collections::HashMap;

pub use transition::{Cursor, LiveInstance};

/// A live, placed instance ready for sampling, in normative composition
/// order (layer ascending, insertion order ascending within a layer).
#[derive(Debug, Clone)]
pub struct PlacedInstance<'a> {
    pub object: u32,
    pub layer: u32,
    pub order: u32,
    /// Linear part (a,b,c,d) of the affine, Q16.16.
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub d: i32,
    /// Translation at the request time (trajectory-evaluated), Q16.16.
    pub tr: Vec2,
    /// Palette override object (must reference a Palette object).
    pub palette: Option<u32>,
    pub clip: Option<RectF>,
    /// Conservative pixel-space bounding box of the placed object (fast
    /// reject prefilter for sampling; always a superset of the true sample
    /// set).  `None` = unknown extent (no prefilter).
    pub bounds_px: Option<IRect>,
    pub doc: &'a Document,
}

impl<'a> PlacedInstance<'a> {
    /// Full placement affine at the request time.
    pub fn affine(&self) -> crate::fixed::Affine {
        crate::fixed::Affine {
            a: self.a,
            b: self.b,
            tx: self.tr.x,
            c: self.c,
            d: self.d,
            ty: self.tr.y,
        }
    }
    /// The object being placed.
    pub fn object(&self) -> &'a crate::ir::Object {
        &self.doc.objects[self.object as usize]
    }
}

/// A palette as it exists at the request time.
pub enum PaletteRef<'d, 's> {
    /// No patches applied (borrows the document object table).
    Base(&'d [Rgba]),
    /// Base entries with the request-time patches applied (borrows the scene).
    Patched(&'s Vec<Rgba>),
}

impl<'d, 's> PaletteRef<'d, 's> {
    pub fn get(&self, i: u32) -> Option<Rgba> {
        match self {
            PaletteRef::Base(b) => b.get(i as usize).copied(),
            PaletteRef::Patched(p) => p.get(i as usize).copied(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            PaletteRef::Base(b) => b.len(),
            PaletteRef::Patched(p) => p.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Ordered structural operations for the request time.  These execute after
/// the instance draws, in event order; copies read the prefix surface
/// (see docs/EXACT_SEMANTICS.md §composition).
#[derive(Debug, Clone, Copy)]
pub enum StructOp<'a> {
    FillRect {
        rect: RectF,
        color: Rgba,
    },
    CopyRegion {
        src: RectF,
        dx: i32,
        dy: i32,
    },
    MoveRegion {
        src: RectF,
        dx: i32,
        dy: i32,
    },
    BindResidual {
        algebra: u8,
        region: IRect,
        format: ColorFormat,
        payload: &'a [u8],
    },
}

/// Resolved procedural state at one time coordinate.
#[derive(Debug, Clone)]
pub struct Scene<'a> {
    /// Request time.
    pub t: u64,
    doc: &'a Document,
    /// Visible placed instances in composition order.
    pub instances: Vec<PlacedInstance<'a>>,
    /// Palette patches per palette object, in event order.
    patches: Vec<(u32, u32, &'a [Rgba])>,
    /// Palette objects with at least one patch (id -> patched entries).
    patched_cache: HashMap<u32, Vec<Rgba>>,
    /// Ordered structural operations.
    pub ops: Vec<StructOp<'a>>,
    /// Decoded generator fields, one slot per document object (None for
    /// non-generator objects).  Parsed once per resolve so per-sample
    /// evaluation never re-decodes; resolution fails closed on malformed
    /// params (deterministic `Reject`).
    fields: Vec<Option<crate::procedural::Field>>,
}

impl<'a> Scene<'a> {
    /// Resolve the document at time `t` (nanoseconds).
    pub fn resolve(doc: &'a Document, t: u64) -> Result<Scene<'a>, Reject> {
        let mut cursor = Cursor::new(doc);
        cursor.seek(t)?;
        let applied = cursor.applied_event_count();

        let mut instances = Vec::with_capacity(cursor.live().len());
        for inst in cursor.live().values() {
            if !inst.visible {
                continue;
            }
            let tr = inst.translation_at(doc, t)?;
            let bounds_px = placed_bounds_px(doc, inst.object, inst.a, inst.b, inst.c, inst.d, tr);
            instances.push(PlacedInstance {
                object: inst.object,
                layer: inst.layer,
                order: inst.order,
                a: inst.a,
                b: inst.b,
                c: inst.c,
                d: inst.d,
                tr,
                palette: inst.palette,
                clip: inst.clip,
                bounds_px,
                doc,
            });
        }
        instances.sort_by_key(|x| (x.layer, x.order));

        // Decode generator fields up front: fail closed on malformed params.
        let mut fields: Vec<Option<crate::procedural::Field>> =
            Vec::with_capacity(doc.objects.len());
        for o in &doc.objects {
            match o {
                crate::ir::Object::GeneratorField {
                    family,
                    w,
                    h,
                    params,
                } => {
                    let f = crate::procedural::decode_field(*family, params, *w, *h)
                        .map_err(|_| Reject::UnsupportedProfile)?;
                    fields.push(Some(f));
                }
                _ => fields.push(None),
            }
        }

        let mut patches: Vec<(u32, u32, &'a [Rgba])> = Vec::new();
        let mut ops: Vec<StructOp<'a>> = Vec::new();
        for ev in &doc.events[..applied] {
            for o in &ev.ops {
                match o {
                    Op::PaletteSet {
                        object,
                        offset,
                        entries,
                    } => {
                        patches.push((*object, *offset, entries));
                    }
                    Op::CopyRegion { src, dx, dy } => {
                        ops.push(StructOp::CopyRegion {
                            src: *src,
                            dx: *dx,
                            dy: *dy,
                        });
                    }
                    Op::MoveRegion { src, dx, dy } => {
                        ops.push(StructOp::MoveRegion {
                            src: *src,
                            dx: *dx,
                            dy: *dy,
                        });
                    }
                    Op::FillRect { rect, color } => {
                        ops.push(StructOp::FillRect {
                            rect: *rect,
                            color: *color,
                        });
                    }
                    Op::BindResidual(rb) => {
                        ops.push(StructOp::BindResidual {
                            algebra: rb.algebra,
                            region: rb.region,
                            format: rb.format,
                            payload: &rb.payload,
                        });
                    }
                    _ => {}
                }
            }
        }

        // Build the patched-palette cache lazily but eagerly here: only
        // palettes that received patches are rebuilt.
        let mut patched: HashMap<u32, Vec<Rgba>> = HashMap::new();
        for &(obj, _, _) in &patches {
            patched.entry(obj).or_default();
        }
        for (obj, base) in patched.iter_mut() {
            if let crate::ir::Object::Palette { entries } = &doc.objects[*obj as usize] {
                base.extend_from_slice(entries);
            }
        }
        for &(obj, offset, entries) in &patches {
            let dst = patched.get_mut(&obj).expect("cache seeded");
            let start = offset as usize;
            dst[start..start + entries.len()].copy_from_slice(entries);
        }

        Ok(Scene {
            t,
            doc,
            instances,
            patches,
            patched_cache: patched,
            ops,
            fields,
        })
    }

    /// Sample a document content object (raster / indexed raster) at its own
    /// local integer pixel `(u, v)`, honoring palette patches at this time.
    /// Returns `None` for out-of-bounds or non-content objects.  Used by the
    /// shared instance sampler and by generator families that reference
    /// objects (affine reuse / object families).
    pub fn sample_content(&self, object: u32, u: i32, v: i32) -> Option<Rgba> {
        use crate::color::ColorFormat;
        let obj = self.doc.objects.get(object as usize)?;
        match obj {
            crate::ir::Object::Raster {
                format: ColorFormat::Rgba8,
                w,
                h,
                data,
            } => {
                let (i, j) = in_bounds_local(u, v, *w, *h)?;
                let o = ((j * w) + i) as usize * 4;
                Some(Rgba::from_bytes([
                    data[o],
                    data[o + 1],
                    data[o + 2],
                    data[o + 3],
                ]))
            }
            crate::ir::Object::Raster {
                format: ColorFormat::Gray8,
                w,
                h,
                data,
            } => {
                let (i, j) = in_bounds_local(u, v, *w, *h)?;
                Some(Rgba::gray(data[(j * w + i) as usize]))
            }
            crate::ir::Object::IndexedRaster { pal, w, h, indices } => {
                let (i, j) = in_bounds_local(u, v, *w, *h)?;
                let idx = indices[(j * w + i) as usize];
                self.palette(*pal)?.get(idx)
            }
            _ => None,
        }
    }

    /// Evaluate a generator field object at its local integer pixel `(i, j)`.
    /// `object` must be a validated generator object and `(i, j)` in bounds
    /// (the raster-like bounds check happens at the caller).
    pub fn sample_field(&self, object: u32, i: u32, j: u32) -> Option<Rgba> {
        let field = self.fields.get(object as usize)?.as_ref()?;
        let obj = self.doc.objects.get(object as usize)?;
        let (w, h) = match obj {
            crate::ir::Object::GeneratorField { w, h, .. } => (*w, *h),
            _ => return None,
        };
        if i >= w || j >= h {
            return None;
        }
        let refs = crate::procedural::Refs {
            sample_object: &|o, u, v| self.sample_content(o, u, v),
        };
        Some(field.sample(w, h, i, j, &refs))
    }

    /// Decoded generator field of `object` (None for non-generators).
    pub fn field_of(&self, object: u32) -> Option<&crate::procedural::Field> {
        self.fields.get(object as usize)?.as_ref()
    }

    /// Effective palette contents for a palette object id at this time.
    pub fn palette(&self, object: u32) -> Option<PaletteRef<'a, '_>> {
        let base = match &self.doc.objects.get(object as usize)? {
            crate::ir::Object::Palette { entries } => entries,
            _ => return None,
        };
        if let Some(p) = self.patched_cache.get(&object) {
            Some(PaletteRef::Patched(p))
        } else {
            Some(PaletteRef::Base(base))
        }
    }

    pub fn doc(&self) -> &'a Document {
        self.doc
    }

    /// Number of palette patch operations in scope (accounting metric).
    pub fn palette_patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Total live instance count (visible or not).
    pub fn live_instance_count(&self) -> usize {
        self.instances.len()
    }
}

/// Conservative pixel bounds of an object placed with the given runtime
/// affine (trajectory translation included).
pub fn placed_bounds_px(
    doc: &Document,
    object: u32,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    tr: Vec2,
) -> Option<IRect> {
    let obj = doc.objects.get(object as usize)?;
    let (w, h) = match obj {
        crate::ir::Object::Raster { w, h, .. }
        | crate::ir::Object::IndexedRaster { w, h, .. }
        | crate::ir::Object::GeneratorField { w, h, .. } => (*w, *h),
        _ => return None,
    };
    let aff = crate::fixed::Affine {
        a,
        b,
        tx: tr.x,
        c,
        d,
        ty: tr.y,
    };
    let local = crate::fixed::RectF::from_px(0, 0, w as i32, h as i32);
    Some(aff.bounds_of(local).px_bounds())
}

/// Integer bounds check helper for local content sampling.
fn in_bounds_local(u: i32, v: i32, w: u32, h: u32) -> Option<(u32, u32)> {
    if u >= 0 && v >= 0 && u < w as i32 && v < h as i32 {
        Some((u as u32, v as u32))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorFormat;
    use crate::fixed::Affine;
    use crate::ir::{Event, Instance, Object, Op};

    fn doc_two_instances() -> Document {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 1,
            h: 1,
            data: vec![1, 2, 3, 255],
        });
        for (order, layer, tx) in [(1u32, 0u32, 0i32), (2, 1, 5)] {
            d.instances.push(Instance {
                object: 0,
                order,
                layer,
                transform: Affine::from_px_translation(tx, 0),
                palette: None,
                clip: None,
                trajectory: 0,
                visible: true,
            });
        }
        d
    }

    #[test]
    fn composition_order_by_layer_then_order() {
        let d = doc_two_instances();
        let s = Scene::resolve(&d, 0).unwrap();
        let layers: Vec<u32> = s.instances.iter().map(|i| i.layer).collect();
        assert_eq!(layers, vec![0, 1]);
    }

    #[test]
    fn invisible_instances_excluded() {
        let mut d = doc_two_instances();
        d.events.push(Event {
            t: 1,
            ops: vec![Op::InstVisible {
                instance: 2,
                visible: false,
            }],
        });
        let s = Scene::resolve(&d, 5).unwrap();
        assert_eq!(s.instances.len(), 1);
        assert_eq!(s.instances[0].order, 1);
    }

    #[test]
    fn palette_patch_applies_in_order() {
        let mut d = Document::new();
        d.objects.push(crate::ir::Object::Palette {
            entries: vec![Rgba::OPAQUE_BLACK, Rgba::WHITE],
        });
        d.objects.push(crate::ir::Object::IndexedRaster {
            pal: 0,
            w: 1,
            h: 1,
            indices: vec![0],
        });
        d.instances.push(Instance {
            object: 1,
            order: 1,
            layer: 0,
            transform: Affine::identity(),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
        // event 1: palette[1] = red
        d.events.push(Event {
            t: 10,
            ops: vec![Op::PaletteSet {
                object: 0,
                offset: 1,
                entries: vec![Rgba::new(255, 0, 0, 255)],
            }],
        });
        let s = Scene::resolve(&d, 0).unwrap();
        assert_eq!(s.palette(0).unwrap().get(1), Some(Rgba::WHITE));
        let s = Scene::resolve(&d, 10).unwrap();
        assert_eq!(
            s.palette(0).unwrap().get(1),
            Some(Rgba::new(255, 0, 0, 255))
        );
    }
}
