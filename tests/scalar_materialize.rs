//! Phase A/B conformance: scalar materialization invariants.
//!
//! The scalar backend is the semantic oracle; these tests pin down
//! sample-center coverage, layer order, clipping, partial-domain
//! equivalence, trajectory evaluation at explicit times, structural
//! copy/move operations, and residual closure.

use vole_gfx::color::{ColorFormat, Rgba};
use vole_gfx::fixed::{Affine, RectF};
use vole_gfx::ir::{Document, Event, InstCreate, Object, Op};
use vole_gfx::materialize::scalar::materialize_document;
use vole_gfx::observation::{Domain, ObservationRequest};

/// Red 2x1 raster object helper.
fn red_2x1() -> Object {
    Object::Raster {
        format: ColorFormat::Rgba8,
        w: 2,
        h: 1,
        data: vec![255, 0, 0, 255, 200, 0, 0, 255],
    }
}

fn request(t: u64, domain: Domain, format: ColorFormat) -> ObservationRequest {
    ObservationRequest {
        time_ns: t,
        view: Default::default(),
        domain,
        sampling: Default::default(),
        format,
    }
}

fn full(t: u64, w: u32, h: u32, format: ColorFormat) -> ObservationRequest {
    request(t, Domain::FullSurface { w, h }, format)
}

#[test]
fn fill_rect_covers_sample_centers() {
    let mut d = Document::new();
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 4, 4),
            color: Rgba::new(10, 20, 30, 255),
        }],
    });
    let m = materialize_document(&d, &full(0, 4, 4, ColorFormat::Rgba8)).unwrap();
    let px = |x: usize, y: usize| {
        let o = (y * 4 + x) * 4;
        Rgba::from_bytes(m.output.data[o..o + 4].try_into().unwrap())
    };
    assert_eq!(px(0, 0), Rgba::new(10, 20, 30, 255));
    assert_eq!(px(3, 3), Rgba::new(10, 20, 30, 255));
    // outside the fill
    let d2 = Document::new();
    let m2 = materialize_document(&d2, &full(0, 4, 4, ColorFormat::Rgba8)).unwrap();
    assert_eq!(m2.output.data, vec![0u8; 4 * 16]);
}

#[test]
fn fill_rect_subpixel_boundary() {
    // rect [1.0, 3.0) in float px with half-open sample-center coverage:
    // covers pixels whose centers (x+0.5) are in [1,3) => pixels 1..=2? no:
    // center 0.5 < 1 => out; 1.5, 2.5 in; 3.5 >= 3 out => pixels 1,2.
    let mut d = Document::new();
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(1, 1, 3, 3),
            color: Rgba::new(255, 255, 255, 255),
        }],
    });
    let m = materialize_document(&d, &full(0, 4, 4, ColorFormat::Rgba8)).unwrap();
    for y in 0..4u32 {
        for x in 0..4u32 {
            let o = ((y * 4 + x) * 4) as usize;
            let filled = m.output.data[o] == 255;
            let expect = (1..3).contains(&x) && (1..3).contains(&y);
            assert_eq!(filled, expect, "pixel ({x},{y})");
        }
    }
}

#[test]
fn instance_raster_draw_and_layer_order() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    d.objects.push(Object::Raster {
        format: ColorFormat::Rgba8,
        w: 1,
        h: 1,
        data: vec![0, 255, 0, 255],
    });
    // layer 0: red 2px object at x=0; layer 1: green 1px at x=0 -> green wins at x=0.
    d.instances.push(vole_gfx::ir::Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    d.instances.push(vole_gfx::ir::Instance {
        object: 1,
        order: 2,
        layer: 1,
        transform: Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    let m = materialize_document(&d, &full(0, 3, 1, ColorFormat::Rgba8)).unwrap();
    let px = |x: usize| Rgba::from_bytes(m.output.data[x * 4..x * 4 + 4].try_into().unwrap());
    assert_eq!(px(0), Rgba::new(0, 255, 0, 255), "top layer at x=0");
    assert_eq!(px(1), Rgba::new(200, 0, 0, 255), "red second px untouched");
    assert_eq!(px(2), Rgba::TRANSPARENT, "outside objects transparent");
}

#[test]
fn clip_limits_instance() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    d.instances.push(vole_gfx::ir::Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: Some(RectF::from_px(0, 0, 1, 1)),
        trajectory: 0,
        visible: true,
    });
    let m = materialize_document(&d, &full(0, 2, 1, ColorFormat::Rgba8)).unwrap();
    assert_eq!(m.output.data[0], 255);
    assert_eq!(m.output.data[4], 0, "clipped pixel stays transparent");
}

#[test]
fn partial_domain_matches_full_surface_crop() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    for (i, x) in [0i32, 5, -3].into_iter().enumerate() {
        d.instances.push(vole_gfx::ir::Instance {
            object: 0,
            order: i as u32,
            layer: 0,
            transform: Affine::from_px_translation(x, 0),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        });
    }
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(-10, -10, 10, 10),
            color: Rgba::new(9, 9, 9, 255),
        }],
    });
    let m = materialize_document(&d, &full(0, 16, 16, ColorFormat::Rgba8)).unwrap();
    // crop region (4,4,8x8) from the full surface
    let part = materialize_document(
        &d,
        &request(
            0,
            Domain::Rectangle {
                x0: 4,
                y0: 4,
                w: 8,
                h: 8,
            },
            ColorFormat::Rgba8,
        ),
    )
    .unwrap();
    for y in 0..8usize {
        for x in 0..8usize {
            let fo = ((y + 4) * 16 + (x + 4)) * 4;
            let po = (y * 8 + x) * 4;
            assert_eq!(
                &m.output.data[fo..fo + 4],
                &part.output.data[po..po + 4],
                "crop mismatch at ({x},{y})"
            );
        }
    }
}

#[test]
fn trajectory_moves_object_over_time() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    d.trajectories.push(vole_gfx::ir::Trajectory {
        kind: vole_gfx::ir::traj::LINEAR_TRANSLATION,
        keys: vec![
            vole_gfx::ir::TrajKey { t: 0, tx: 0, ty: 0 },
            vole_gfx::ir::TrajKey {
                t: 1_000,
                tx: vole_gfx::fixed::fp(10),
                ty: 0,
            },
        ],
    });
    d.instances.push(vole_gfx::ir::Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 1,
        visible: true,
    });
    let at = |t: u64| materialize_document(&d, &full(t, 16, 1, ColorFormat::Rgba8)).unwrap();
    let t0 = at(0);
    let t500 = at(500);
    let t1000 = at(1_000);
    // t0: red at x=0,1
    assert_eq!(t0.output.data[0], 255);
    assert_eq!(t0.output.data[4], 200);
    assert_eq!(t0.output.data[8], 0);
    // t1000: shifted by 10px
    assert_eq!(t1000.output.data[40], 255, "at x=10");
    assert_eq!(t1000.output.data[0], 0);
    // t500: half-shift => exact 5px (fixed interpolation, no rounding)
    assert_eq!(t500.output.data[20], 255, "at x=5");
    // arbitrary times: identical surface content, only translated
    assert_eq!(
        &t1000.output.data[40..48],
        &[255, 0, 0, 255, 200, 0, 0, 255]
    );
}

#[test]
fn copy_region_duplicates_content() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    d.instances.push(vole_gfx::ir::Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    // copy scene rect (0,0,2,1) to (4,0,6,1)
    d.events.push(Event {
        t: 0,
        ops: vec![Op::CopyRegion {
            src: RectF::from_px(0, 0, 2, 1),
            dx: vole_gfx::fixed::fp(4),
            dy: 0,
        }],
    });
    let m = materialize_document(&d, &full(0, 8, 1, ColorFormat::Rgba8)).unwrap();
    let px = |x: usize| Rgba::from_bytes(m.output.data[x * 4..x * 4 + 4].try_into().unwrap());
    assert_eq!(px(0), Rgba::new(255, 0, 0, 255));
    assert_eq!(px(4), Rgba::new(255, 0, 0, 255), "copied");
    assert_eq!(px(5), Rgba::new(200, 0, 0, 255), "copied second pixel");
}

#[test]
fn gray8_output_luma() {
    let mut d = Document::new();
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 2, 1),
            color: Rgba::new(255, 0, 0, 255),
        }],
    });
    let m = materialize_document(&d, &full(0, 2, 1, ColorFormat::Gray8)).unwrap();
    // both pixels filled: luma(255,0,0) = round((299*255)/1000) = round(76.245) = 76
    assert_eq!(m.output.data[0], 76);
    assert_eq!(m.output.data[1], 76);
}

#[test]
fn sparse_residual_overwrite_and_xor() {
    use vole_gfx::ir::residual_algebra;
    use vole_gfx::residual::push_record;
    let mut d = Document::new();
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 4, 1),
            color: Rgba::new(100, 100, 100, 255),
        }],
    });
    // overwrite pixel 0 to white; xor pixel 1 by (0, 10, 0, 0)
    let mut ow = Vec::new();
    ow.extend_from_slice(&1u64.to_le_bytes());
    push_record(&mut ow, 0, 0, &[255, 255, 255, 255]);
    let mut xr = Vec::new();
    xr.extend_from_slice(&1u64.to_le_bytes());
    push_record(&mut xr, 1, 0, &[0, 10, 0, 0]);
    d.events.push(Event {
        t: 1,
        ops: vec![
            Op::BindResidual(vole_gfx::ir::ResidualBind {
                algebra: residual_algebra::SPARSE_OVERWRITE,
                region: vole_gfx::fixed::IRect::new(0, 0, 4, 1),
                format: ColorFormat::Rgba8,
                payload: ow,
            }),
            Op::BindResidual(vole_gfx::ir::ResidualBind {
                algebra: residual_algebra::XOR,
                region: vole_gfx::fixed::IRect::new(0, 0, 4, 1),
                format: ColorFormat::Rgba8,
                payload: xr,
            }),
        ],
    });
    let m = materialize_document(&d, &full(1, 4, 1, ColorFormat::Rgba8)).unwrap();
    let px = |x: usize| Rgba::from_bytes(m.output.data[x * 4..x * 4 + 4].try_into().unwrap());
    assert_eq!(px(0), Rgba::new(255, 255, 255, 255), "overwritten");
    assert_eq!(px(1), Rgba::new(100, 110, 100, 255), "xor applied");
    assert_eq!(px(2), Rgba::new(100, 100, 100, 255), "untouched");
    // before binding time, no residual applied
    let m0 = materialize_document(&d, &full(0, 4, 1, ColorFormat::Rgba8)).unwrap();
    assert_eq!(m0.output.data[0], 100);
}

#[test]
fn output_hashes_differ_when_content_differs() {
    let mut d = Document::new();
    d.events.push(Event {
        t: 0,
        ops: vec![Op::FillRect {
            rect: RectF::from_px(0, 0, 8, 8),
            color: Rgba::new(1, 1, 1, 255),
        }],
    });
    let a = materialize_document(&d, &full(0, 8, 8, ColorFormat::Rgba8)).unwrap();
    let mut d2 = d.clone();
    d2.events[0].ops[0] = Op::FillRect {
        rect: RectF::from_px(0, 0, 8, 8),
        color: Rgba::new(2, 1, 1, 255),
    };
    let b = materialize_document(&d2, &full(0, 8, 8, ColorFormat::Rgba8)).unwrap();
    assert_ne!(a.output.canonical_hash(), b.output.canonical_hash());
    let m = materialize_document(&d2, &full(0, 8, 8, ColorFormat::Rgba8)).unwrap();
    assert_eq!(
        b.output.canonical_hash(),
        m.output.canonical_hash(),
        "deterministic"
    );
}

#[test]
fn move_region_shifts_and_clears() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    d.instances.push(vole_gfx::ir::Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    d.events.push(Event {
        t: 0,
        ops: vec![Op::MoveRegion {
            src: RectF::from_px(0, 0, 2, 1),
            dx: vole_gfx::fixed::fp(3),
            dy: 0,
        }],
    });
    let m = materialize_document(&d, &full(0, 6, 1, ColorFormat::Rgba8)).unwrap();
    let px = |x: usize| Rgba::from_bytes(m.output.data[x * 4..x * 4 + 4].try_into().unwrap());
    assert_eq!(px(3), Rgba::new(255, 0, 0, 255), "moved to x=3");
    assert_eq!(px(0), Rgba::TRANSPARENT, "source cleared to base");
}

#[test]
fn background_over_zero_alpha_semantics() {
    // RGBA surface starts transparent: an empty doc yields transparent pixels
    // but Gray8 yields 0 (luma over black).
    let d = Document::new();
    let rgba = materialize_document(&d, &full(0, 2, 2, ColorFormat::Rgba8)).unwrap();
    assert_eq!(rgba.output.data, vec![0u8; 16]);
    let gray = materialize_document(&d, &full(0, 2, 2, ColorFormat::Gray8)).unwrap();
    assert_eq!(gray.output.data, vec![0u8; 4]);
}

#[test]
fn instance_lifecycle_ops_materialize_correctly() {
    let mut d = Document::new();
    d.objects.push(red_2x1());
    // create at t=10, move at t=20, delete at t=30
    d.events.push(Event {
        t: 10,
        ops: vec![Op::InstCreate(InstCreate {
            object: 0,
            order: 7,
            layer: 0,
            transform: Affine::identity(),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        })],
    });
    d.events.push(Event {
        t: 20,
        ops: vec![Op::InstTransform {
            instance: 7,
            transform: Affine::from_px_translation(4, 0),
        }],
    });
    d.events.push(Event {
        t: 30,
        ops: vec![Op::InstDelete { instance: 7 }],
    });
    let red_at = |t: u64| {
        let m = materialize_document(&d, &full(t, 8, 1, ColorFormat::Rgba8)).unwrap();
        m.output.data[0] == 255
    };
    assert!(!red_at(0), "not yet created");
    assert!(red_at(10), "created at t=10");
    let m20 = materialize_document(&d, &full(20, 8, 1, ColorFormat::Rgba8)).unwrap();
    assert_eq!(m20.output.data[16], 255, "moved to x=4 at t=20");
    assert!(!red_at(40), "deleted at t=30");
}
