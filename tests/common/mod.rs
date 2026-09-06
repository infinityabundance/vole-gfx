//! Shared helpers for integration tests (not a test binary itself).

#![allow(dead_code)]

use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::fixed::{Affine, IRect, RectF, fp};
use vole_gfx::ir::{Document, Event, Instance, Object, Op};

/// Deterministic split-mix64 RNG for property-style tests.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
    pub fn u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

/// A deterministic RGBA checker field object (w×h).
pub fn checker(w: u32, h: u32, a: Rgba, b: Rgba) -> Object {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for j in 0..h {
        for i in 0..w {
            let c = if (i + j) % 2 == 0 { a } else { b };
            data.extend_from_slice(&c.to_bytes());
        }
    }
    Object::Raster {
        format: ColorFormat::Rgba8,
        w,
        h,
        data,
    }
}

/// A horizontal ramp field object (w×h) of `steps` color bands.
pub fn ramp(w: u32, h: u32, steps: u32) -> Object {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _j in 0..h {
        for i in 0..w {
            let k = ((i as u64 * steps as u64) / w as u64).min(steps as u64 - 1) as u32;
            let v = (255 * k / (steps - 1)) as u8;
            data.extend_from_slice(&[v, 255 - v, 128, 255]);
        }
    }
    Object::Raster {
        format: ColorFormat::Rgba8,
        w,
        h,
        data,
    }
}

pub fn placed(object: u32, order: u32, layer: u32, tx_px: i32, ty_px: i32) -> Instance {
    Instance {
        object,
        order,
        layer,
        transform: Affine::from_px_translation(tx_px, ty_px),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    }
}

/// V1: fill + checkers + copies on a 64x64 surface.
pub fn scene_v1() -> Document {
    let mut d = Document::new();
    d.objects.push(checker(
        8,
        8,
        Rgba::new(200, 30, 30, 255),
        Rgba::new(30, 30, 200, 255),
    ));
    d.objects.push(ramp(64, 4, 8));
    d.instances.push(placed(0, 1, 0, 0, 0));
    d.instances.push(placed(1, 2, 1, 0, 30));
    d.events.push(Event {
        t: 0,
        ops: vec![
            Op::FillRect {
                rect: RectF::from_px(0, 0, 64, 64),
                color: Rgba::new(10, 20, 30, 255),
            },
            Op::CopyRegion {
                src: RectF::from_px(0, 0, 16, 16),
                dx: fp(32),
                dy: fp(32),
            },
        ],
    });
    d
}

/// V2: indexed/palette + palette animation + clip.
pub fn scene_v2() -> Document {
    let mut d = Document::new();
    d.objects.push(Object::Palette {
        entries: vec![Rgba::OPAQUE_BLACK, Rgba::WHITE, Rgba::new(255, 0, 0, 255)],
    });
    let mut indices = Vec::new();
    for j in 0..32u32 {
        for i in 0..32u32 {
            indices.push((i / 4 + j / 4) % 3);
        }
    }
    d.objects.push(Object::IndexedRaster {
        pal: 0,
        w: 32,
        h: 32,
        indices,
    });
    d.instances.push(Instance {
        object: 1,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: Some(RectF::from_px(4, 4, 28, 28)),
        trajectory: 0,
        visible: true,
    });
    // palette animation: entry 2 becomes green at t=1000
    d.events.push(Event {
        t: 1000,
        ops: vec![Op::PaletteSet {
            object: 0,
            offset: 2,
            entries: vec![Rgba::new(0, 255, 0, 255)],
        }],
    });
    d
}

/// V3: a moving instance on a trajectory (cadence-independent times).
pub fn scene_v3() -> Document {
    let mut d = Document::new();
    d.objects.push(checker(
        4,
        4,
        Rgba::new(0, 255, 0, 255),
        Rgba::new(255, 0, 255, 255),
    ));
    d.trajectories.push(vole_gfx::ir::Trajectory {
        kind: vole_gfx::ir::traj::LINEAR_TRANSLATION,
        keys: vec![
            vole_gfx::ir::TrajKey { t: 0, tx: 0, ty: 0 },
            vole_gfx::ir::TrajKey {
                t: 33_333_333,
                tx: fp(60),
                ty: fp(20),
            },
        ],
    });
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 1,
        visible: true,
    });
    d
}

/// V4: move region + fill ordering semantics.
pub fn scene_v4() -> Document {
    let mut d = Document::new();
    d.objects.push(checker(
        8,
        8,
        Rgba::new(255, 255, 0, 255),
        Rgba::new(0, 0, 0, 255),
    ));
    d.instances.push(placed(0, 1, 0, 2, 2));
    d.events.push(Event {
        t: 0,
        ops: vec![
            Op::FillRect {
                rect: RectF::from_px(0, 0, 64, 64),
                color: Rgba::new(40, 40, 40, 255),
            },
            Op::MoveRegion {
                src: RectF::from_px(0, 0, 16, 16),
                dx: fp(24),
                dy: fp(24),
            },
        ],
    });
    d
}

/// V5: residual closure, both algebras, corner records.
pub fn scene_v5() -> Document {
    let mut d = Document::new();
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 64, 64),
            color: Rgba::new(7, 7, 7, 255),
        }],
    });
    let mut ow = Vec::new();
    ow.extend_from_slice(&2u64.to_le_bytes());
    vole_gfx::residual::push_record(&mut ow, 3, 3, &[1, 2, 3, 4]);
    vole_gfx::residual::push_record(&mut ow, 60, 60, &[250, 250, 250, 250]);
    let mut xr = Vec::new();
    xr.extend_from_slice(&2u64.to_le_bytes());
    vole_gfx::residual::push_record(&mut xr, 3, 3, &[254, 253, 252, 0]);
    vole_gfx::residual::push_record(&mut xr, 61, 61, &[5, 5, 5, 5]);
    d.events.push(Event {
        t: 5,
        ops: vec![
            Op::BindResidual(vole_gfx::ir::ResidualBind {
                algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
                region: IRect::new(0, 0, 64, 64),
                format: ColorFormat::Rgba8,
                payload: ow,
            }),
            Op::BindResidual(vole_gfx::ir::ResidualBind {
                algebra: vole_gfx::ir::residual_algebra::XOR,
                region: IRect::new(0, 0, 64, 64),
                format: ColorFormat::Rgba8,
                payload: xr,
            }),
        ],
    });
    d
}

pub fn surface(
    t: u64,
    w: u32,
    h: u32,
    format: ColorFormat,
) -> vole_gfx::observation::ObservationRequest {
    vole_gfx::observation::ObservationRequest::full_surface(t, w, h, format)
}
