//! Security / adversarial tests (paper §Security and Bounded Execution).
//!
//! Rules under test: no parser panic on any input; malformed input fails
//! closed deterministically; huge declared counts and lengths rejected before
//! allocation; cyclic/invalid references rejected; decoding is total.

mod common;

use common::*;
use vole_gfx::color::Rgba;
use vole_gfx::ir::decode::decode;
use vole_gfx::ir::{Document, Event, Op};

/// Decode must never panic; result may be Ok or any Err.
fn must_not_panic(bytes: &[u8]) {
    let _ = decode(bytes);
}

#[test]
fn every_single_byte_truncation_is_safe() {
    let doc = scene_v1();
    let full = vole_gfx::ir::encode::encode(&doc);
    for cut in 0..=full.len() {
        must_not_panic(&full[..cut]);
    }
}

#[test]
fn single_bit_flips_are_safe() {
    let doc = scene_v5(); // includes residuals + all op kinds
    let full = vole_gfx::ir::encode::encode(&doc);
    let mut rng = Rng::new(42);
    for _ in 0..4000 {
        let mut b = full.clone();
        let pos = rng.below(b.len() as u32) as usize;
        let bit = 1u8 << rng.below(8);
        b[pos] ^= bit;
        must_not_panic(&b);
    }
}

#[test]
fn random_bytes_are_safe() {
    let mut rng = Rng::new(0xC0FF_EE00_2026_0906);
    for len in [0u32, 1, 2, 3, 7, 8, 9, 16, 31, 64, 128, 512] {
        for _ in 0..200 {
            let mut b = vec![0u8; len as usize];
            for byte in b.iter_mut() {
                *byte = rng.u8();
            }
            must_not_panic(&b);
        }
    }
}

#[test]
fn random_mutations_never_panic_through_full_pipeline() {
    // decode+validate+materialize on mutated documents; any failure must be a
    // deterministic Err, never a panic.
    let doc = scene_v1();
    let full = vole_gfx::ir::encode::encode(&doc);
    let mut rng = Rng::new(7);
    let mut ok_count = 0u32;
    for _ in 0..1500 {
        let mut b = full.clone();
        for _ in 0..1 + rng.below(3) {
            let pos = rng.below(b.len() as u32) as usize;
            b[pos] = rng.u8();
        }
        if let Ok(d) = decode(&b)
            && vole_gfx::ir::validate::validate(&d).is_ok() {
                // hostile but valid: materialization must still be total
                let req = surface(0, 8, 8, vole_gfx::color::ColorFormat::Rgba8);
                let _ = vole_gfx::materialize::scalar::materialize_document(&d, &req);
                ok_count += 1;
            }
    }
    // a good share of single/double byte mutations of a small doc stay valid
    assert!(ok_count > 0);
}

#[test]
fn huge_declared_counts_rejected() {
    // object count beyond cap
    let mut head = Vec::new();
    head.extend_from_slice(vole_gfx::limits::MAGIC);
    head.extend_from_slice(&vole_gfx::limits::FORMAT_VERSION.to_le_bytes());
    head.extend_from_slice(&(vole_gfx::limits::UNIVERSE_U1.len() as u32).to_le_bytes());
    head.extend_from_slice(vole_gfx::limits::UNIVERSE_U1.as_bytes());
    head.push(1); // profile exact
    let mut body = Vec::new();
    body.extend_from_slice(&(vole_gfx::limits::MAX_OBJECTS + 1).to_le_bytes());
    head.extend_from_slice(&(body.len() as u64).to_le_bytes());
    head.extend_from_slice(&body);
    assert!(decode(&head).is_err());

    // instance count beyond cap
    let mut body = Vec::new();
    body.extend_from_slice(&0u64.to_le_bytes()); // objects
    body.extend_from_slice(&0u64.to_le_bytes()); // trajectories
    body.extend_from_slice(&(vole_gfx::limits::MAX_INSTANCES + 1).to_le_bytes());
    head.clear();
    head.extend_from_slice(vole_gfx::limits::MAGIC);
    head.extend_from_slice(&vole_gfx::limits::FORMAT_VERSION.to_le_bytes());
    head.extend_from_slice(&(vole_gfx::limits::UNIVERSE_U1.len() as u32).to_le_bytes());
    head.extend_from_slice(vole_gfx::limits::UNIVERSE_U1.as_bytes());
    head.push(1);
    head.extend_from_slice(&(body.len() as u64).to_le_bytes());
    head.extend_from_slice(&body);
    assert!(decode(&head).is_err());
}

#[test]
fn semantic_adversarial_cases_fail_closed() {
    use vole_gfx::ir::validate::validate;
    use vole_gfx::ir::{InstCreate, Instance};
    use vole_gfx::limits::Reject;

    // op referencing deleted instance (out-of-order delete before create)
    let mut d = Document::new();
    d.objects
        .push(checker(2, 2, Rgba::WHITE, Rgba::OPAQUE_BLACK));
    d.events.push(Event {
        t: 1,
        ops: vec![Op::InstDelete { instance: 99 }],
    });
    assert_eq!(validate(&d), Err(Reject::InvalidIndex));

    // duplicate live id via InstCreate
    let mut d = Document::new();
    d.objects
        .push(checker(2, 2, Rgba::WHITE, Rgba::OPAQUE_BLACK));
    d.instances.push(Instance {
        object: 0,
        order: 5,
        layer: 0,
        transform: Default::default(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    });
    d.events.push(Event {
        t: 1,
        ops: vec![Op::InstCreate(InstCreate {
            object: 0,
            order: 5,
            layer: 0,
            transform: Default::default(),
            palette: None,
            clip: None,
            trajectory: 0,
            visible: true,
        })],
    });
    assert_eq!(validate(&d), Err(Reject::DuplicateObject));

    // self-referential trajectory index beyond table
    let mut d = Document::new();
    d.objects
        .push(checker(2, 2, Rgba::WHITE, Rgba::OPAQUE_BLACK));
    d.instances.push(Instance {
        object: 0,
        order: 1,
        layer: 0,
        transform: Default::default(),
        palette: None,
        clip: None,
        trajectory: 1, // none exist
        visible: true,
    });
    assert!(validate(&d).is_err());

    // residual payload records outside binding region
    let mut d = Document::new();
    let mut p = Vec::new();
    p.extend_from_slice(&1u64.to_le_bytes());
    vole_gfx::residual::push_record(&mut p, 500, 500, &[1, 2, 3, 4]);
    d.events.push(Event {
        t: 1,
        ops: vec![Op::BindResidual(vole_gfx::ir::ResidualBind {
            algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
            region: vole_gfx::fixed::IRect::new(0, 0, 4, 4),
            format: vole_gfx::color::ColorFormat::Rgba8,
            payload: p,
        })],
    });
    assert!(validate(&d).is_err());

    // negative residual region is rejected
    let mut d = Document::new();
    d.events.push(Event {
        t: 1,
        ops: vec![Op::BindResidual(vole_gfx::ir::ResidualBind {
            algebra: vole_gfx::ir::residual_algebra::SPARSE_OVERWRITE,
            region: vole_gfx::fixed::IRect::new(-1, 0, 4, 4),
            format: vole_gfx::color::ColorFormat::Rgba8,
            payload: vec![],
        })],
    });
    assert!(validate(&d).is_err());
}

#[test]
fn deep_copy_chain_fails_closed() {
    // a copy chain deeper than the hard limit must error, not hang
    let mut d = Document::new();
    d.events.push(Event {
        t: 1,
        ops: vec![Op::FillRect {
            rect: vole_gfx::fixed::RectF::from_px(0, 0, 2, 2),
            color: Rgba::new(9, 9, 9, 255),
        }],
    });
    // chain: every copy targets the same pixel as its source (self-covering
    // stack); the evaluator must fail closed, not hang or explode.
    for k in 0..600u32 {
        d.events.push(Event {
            t: 2 + k as u64,
            ops: vec![Op::CopyRegion {
                src: vole_gfx::fixed::RectF::from_px(0, 0, 1, 1),
                dx: 0,
                dy: 0,
            }],
        });
    }
    let req = surface(10_000, 8, 8, vole_gfx::color::ColorFormat::Rgba8);
    let res = vole_gfx::materialize::scalar::materialize_document(&d, &req);
    // must either terminate (ok) or fail closed with a bounded-execution error
    match res {
        Ok(_) => {}
        Err(vole_gfx::limits::Reject::DependencyTooDeep)
        | Err(vole_gfx::limits::Reject::ExecutionBudgetExceeded) => {}
        Err(e) => panic!("unexpected error {e}"),
    }
}
