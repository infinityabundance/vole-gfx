//! Timeline transitions: `G' = Phi(U, G, delta)`.
//!
//! A `Cursor` walks the document timeline in event order and applies the
//! transition operators (create/delete instances, retarget transforms,
//! trajectories, palettes, clipping, visibility) to a live instance table.
//! The cursor is the incremental form of the state-transition function; a
//! request at an earlier time than the cursor's position rebuilds from the
//! document start (correctness first; checkpoint caching arrives with the
//! spatial-index phase).

use super::trajectory::eval as eval_traj;
use crate::fixed::Vec2;
use crate::ir::{Document, Instance, Op};
use crate::limits::Reject;
use std::collections::HashMap;

/// Mutable live instance during transition application.
#[derive(Debug, Clone)]
pub struct LiveInstance {
    pub object: u32,
    pub layer: u32,
    pub order: u32,
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub d: i32,
    /// Static translation (used when no trajectory is attached).
    pub tx: i32,
    pub ty: i32,
    pub palette: Option<u32>,
    pub clip: Option<crate::fixed::RectF>,
    /// 1-based trajectory index; 0 = static.
    pub trajectory: u32,
    pub visible: bool,
}

impl LiveInstance {
    pub fn from_ir(inst: &Instance) -> LiveInstance {
        LiveInstance {
            object: inst.object,
            layer: inst.layer,
            order: inst.order,
            a: inst.transform.a,
            b: inst.transform.b,
            c: inst.transform.c,
            d: inst.transform.d,
            tx: inst.transform.tx,
            ty: inst.transform.ty,
            palette: inst.palette,
            clip: inst.clip,
            trajectory: inst.trajectory,
            visible: inst.visible,
        }
    }

    /// Effective translation at time `t` (trajectory overrides static).
    pub fn translation_at(&self, doc: &Document, t: u64) -> Result<Vec2, Reject> {
        if self.trajectory == 0 {
            return Ok(Vec2::new(self.tx, self.ty));
        }
        let tr = doc
            .trajectories
            .get(self.trajectory as usize - 1)
            .ok_or(Reject::MissingObject)?;
        eval_traj(tr, t)
    }
}

/// Incremental timeline cursor.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    doc: &'a Document,
    /// Live instances keyed by instance id (their `order`).
    live: HashMap<u32, LiveInstance>,
    /// Event index applied so far (`applied` events consumed; next = index).
    applied: usize,
    /// Time of the state currently held.
    pub t: u64,
}

impl<'a> Cursor<'a> {
    pub fn new(doc: &'a Document) -> Cursor<'a> {
        let live = doc
            .instances
            .iter()
            .map(|i| (i.order, LiveInstance::from_ir(i)))
            .collect();
        Cursor {
            doc,
            live,
            applied: 0,
            t: 0,
        }
    }

    pub fn doc(&self) -> &'a Document {
        self.doc
    }

    pub fn live(&self) -> &HashMap<u32, LiveInstance> {
        &self.live
    }

    /// Advance (or rebuild) so the state reflects time `t`.
    pub fn seek(&mut self, t: u64) -> Result<(), Reject> {
        if t < self.t {
            *self = Cursor::new(self.doc);
        }
        while self.applied < self.doc.events.len() && self.doc.events[self.applied].t <= t {
            let ev = &self.doc.events[self.applied];
            for o in &ev.ops {
                self.apply_op(o)?;
            }
            self.applied += 1;
        }
        self.t = t;
        Ok(())
    }

    fn apply_op(&mut self, o: &Op) -> Result<(), Reject> {
        match o {
            Op::InstCreate(c) => {
                let inst = LiveInstance {
                    object: c.object,
                    layer: c.layer,
                    order: c.order,
                    a: c.transform.a,
                    b: c.transform.b,
                    c: c.transform.c,
                    d: c.transform.d,
                    tx: c.transform.tx,
                    ty: c.transform.ty,
                    palette: c.palette,
                    clip: c.clip,
                    trajectory: c.trajectory,
                    visible: c.visible,
                };
                if self.live.insert(c.order, inst).is_some() {
                    return Err(Reject::DuplicateObject);
                }
            }
            Op::InstDelete { instance } => {
                self.live.remove(instance).ok_or(Reject::InvalidIndex)?;
            }
            Op::InstTransform {
                instance,
                transform,
            } => {
                let i = self.live.get_mut(instance).ok_or(Reject::InvalidIndex)?;
                i.a = transform.a;
                i.b = transform.b;
                i.c = transform.c;
                i.d = transform.d;
                i.tx = transform.tx;
                i.ty = transform.ty;
                i.trajectory = 0;
            }
            Op::InstTrajectory {
                instance,
                trajectory,
            } => {
                let i = self.live.get_mut(instance).ok_or(Reject::InvalidIndex)?;
                i.trajectory = *trajectory;
            }
            Op::InstLayer { instance, layer } => {
                let i = self.live.get_mut(instance).ok_or(Reject::InvalidIndex)?;
                i.layer = *layer;
            }
            Op::InstVisible { instance, visible } => {
                let i = self.live.get_mut(instance).ok_or(Reject::InvalidIndex)?;
                i.visible = *visible;
            }
            Op::InstClip { instance, clip } => {
                let i = self.live.get_mut(instance).ok_or(Reject::InvalidIndex)?;
                i.clip = *clip;
            }
            // Structural / residual ops are *not* state transitions; the
            // materializer collects them separately in event order.
            Op::PaletteSet { .. }
            | Op::CopyRegion { .. }
            | Op::MoveRegion { .. }
            | Op::FillRect { .. }
            | Op::BindResidual(_) => {}
        }
        Ok(())
    }

    /// Op application state is only for instance transitions; structural ops
    /// never appear here.
    pub fn applied_event_count(&self) -> usize {
        self.applied
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{ColorFormat, Rgba};
    use crate::fixed::{Affine, RectF};
    use crate::ir::{Document, Event, InstCreate, Instance, Object};

    fn doc_with_inst() -> Document {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 1,
            h: 1,
            data: vec![255, 0, 0, 255],
        });
        d.instances.push(Instance {
            object: 0,
            order: 1,
            layer: 0,
            transform: Affine::from_px_translation(10, 10),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
        d
    }

    #[test]
    fn seek_applies_events_incrementally() {
        let mut d = doc_with_inst();
        d.events.push(Event {
            t: 100,
            ops: vec![Op::InstTransform {
                instance: 1,
                transform: Affine::from_px_translation(20, 20),
            }],
        });
        let mut cur = Cursor::new(&d);
        assert_eq!(cur.live[&1].tx, Affine::from_px_translation(10, 10).tx);
        cur.seek(50).unwrap();
        assert_eq!(cur.live[&1].tx, Affine::from_px_translation(10, 10).tx);
        cur.seek(100).unwrap();
        assert_eq!(cur.live[&1].tx, Affine::from_px_translation(20, 20).tx);
        cur.seek(500).unwrap();
        assert_eq!(cur.live[&1].tx, Affine::from_px_translation(20, 20).tx);
    }

    #[test]
    fn seek_backwards_rebuilds() {
        let mut d = doc_with_inst();
        d.events.push(Event {
            t: 100,
            ops: vec![Op::InstTransform {
                instance: 1,
                transform: Affine::from_px_translation(20, 20),
            }],
        });
        let mut cur = Cursor::new(&d);
        cur.seek(100).unwrap();
        assert_eq!(cur.live[&1].tx, Affine::from_px_translation(20, 20).tx);
        cur.seek(0).unwrap();
        assert_eq!(cur.live[&1].tx, Affine::from_px_translation(10, 10).tx);
    }

    #[test]
    fn create_and_delete_lifecycle() {
        let mut d = doc_with_inst();
        d.events.push(Event {
            t: 10,
            ops: vec![Op::InstCreate(InstCreate {
                object: 0,
                order: 2,
                layer: 1,
                transform: Affine::identity(),
                palette: None,
                clip: None,
                trajectory: 0,
                visible: true,
            })],
        });
        d.events.push(Event {
            t: 20,
            ops: vec![Op::InstDelete { instance: 2 }],
        });
        let mut cur = Cursor::new(&d);
        cur.seek(10).unwrap();
        assert!(cur.live.contains_key(&2));
        cur.seek(20).unwrap();
        assert!(!cur.live.contains_key(&2));
    }

    #[test]
    fn trajectory_override_and_clear() {
        let mut d = doc_with_inst();
        d.trajectories.push(crate::ir::Trajectory {
            kind: crate::ir::traj::LINEAR_TRANSLATION,
            keys: vec![
                crate::ir::TrajKey {
                    t: 0,
                    tx: crate::fixed::fp(0),
                    ty: 0,
                },
                crate::ir::TrajKey {
                    t: 1000,
                    tx: crate::fixed::fp(500),
                    ty: 0,
                },
            ],
        });
        let mut cur = Cursor::new(&d);
        cur.seek(0).unwrap();
        cur.apply_op(&Op::InstTrajectory {
            instance: 1,
            trajectory: 1,
        })
        .unwrap();
        let at = cur.live[&1].translation_at(&d, 500).unwrap();
        assert_eq!(at.x, crate::fixed::fp(250));
    }

    #[test]
    fn rect_op_is_not_a_transition() {
        let mut d = doc_with_inst();
        d.events.push(Event {
            t: 1,
            ops: vec![Op::FillRect {
                rect: RectF::from_px(0, 0, 1, 1),
                color: Rgba::WHITE,
            }],
        });
        let mut cur = Cursor::new(&d);
        cur.seek(1).unwrap();
        assert_eq!(cur.live.len(), 1);
    }
}
