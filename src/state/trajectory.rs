//! Trajectory evaluation: `x(t)` is authoritative; states are never baked per
//! observation cadence (paper §Parametric dynamics).

use crate::fixed::Vec2;
use crate::ir::Trajectory;
use crate::limits::Reject;

/// Exact floor division of a signed numerator by a positive denominator.
#[inline(always)]
pub fn floor_div_pos(a: i64, b: u64) -> i64 {
    let b = b as i64;
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 { q - 1 } else { q }
}

/// Evaluate a translation trajectory at absolute time `t` (ns).
///
/// Semantics (U1 exact):
/// * `t` before the first key or after the last key clamps to that key.
/// * Between keys `k` and `k+1` the translation is the exact rational linear
///   interpolant `p_k + (p_{k+1}-p_k)·(t-t_k)/(t_{k+1}-t_k)`, with the
///   quotient taken by `floor_div_pos` (floor toward −∞).
pub fn eval(traj: &Trajectory, t: u64) -> Result<Vec2, Reject> {
    if traj.kind != crate::ir::traj::LINEAR_TRANSLATION {
        return Err(Reject::UnknownTag);
    }
    let keys = &traj.keys;
    debug_assert!(!keys.is_empty());
    if t <= keys[0].t {
        return Ok(Vec2::new(keys[0].tx, keys[0].ty));
    }
    let last = keys.len() - 1;
    if t >= keys[last].t {
        return Ok(Vec2::new(keys[last].tx, keys[last].ty));
    }
    // find segment: keys[k].t <= t < keys[k+1].t
    let k = (0..last)
        .find(|&i| t < keys[i + 1].t)
        .expect("t within range");
    let (k0, k1) = (&keys[k], &keys[k + 1]);
    let dt = k1.t - k0.t;
    let dd = t - k0.t;
    let dx = k1.tx as i64 - k0.tx as i64;
    let dy = k1.ty as i64 - k0.ty as i64;
    let x = k0.tx as i64 + floor_div_pos(dx * dd as i64, dt);
    let y = k0.ty as i64 + floor_div_pos(dy * dd as i64, dt);
    Ok(Vec2::new(x as i32, y as i32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::fp;
    use crate::ir::{TrajKey, Trajectory};

    fn lin(keys: Vec<TrajKey>) -> Trajectory {
        Trajectory {
            kind: crate::ir::traj::LINEAR_TRANSLATION,
            keys,
        }
    }

    #[test]
    fn clamps_before_and_after() {
        let tr = lin(vec![
            TrajKey {
                t: 100,
                tx: fp(0),
                ty: fp(0),
            },
            TrajKey {
                t: 200,
                tx: fp(10),
                ty: fp(-10),
            },
        ]);
        assert_eq!(eval(&tr, 0).unwrap(), Vec2::new(fp(0), fp(0)));
        assert_eq!(eval(&tr, 100).unwrap(), Vec2::new(fp(0), fp(0)));
        assert_eq!(eval(&tr, 9999).unwrap(), Vec2::new(fp(10), fp(-10)));
    }

    #[test]
    fn midpoint_interpolation() {
        let tr = lin(vec![
            TrajKey {
                t: 0,
                tx: fp(0),
                ty: fp(0),
            },
            TrajKey {
                t: 1_000,
                tx: fp(100),
                ty: fp(-200),
            },
        ]);
        let v = eval(&tr, 500).unwrap();
        assert_eq!(v, Vec2::new(fp(50), fp(-100)));
    }

    #[test]
    fn negative_direction_floor_rounding() {
        // from 0 to -100 px over 3 steps: at dd=1 the fixed-space quotient is
        // floor((-100<<16)/3) = floor(-2184533.33) = -2184534.
        let tr = lin(vec![
            TrajKey {
                t: 0,
                tx: fp(0),
                ty: fp(0),
            },
            TrajKey {
                t: 3,
                tx: fp(-100),
                ty: fp(0),
            },
        ]);
        let v = eval(&tr, 1).unwrap();
        assert_eq!(v.x, floor_div_pos(-(100i64 << 16), 3) as i32);
        assert_eq!(v.x, -2_184_534);
    }

    #[test]
    fn floor_div_exact() {
        assert_eq!(floor_div_pos(5, 3), 1);
        assert_eq!(floor_div_pos(-5, 3), -2);
        assert_eq!(floor_div_pos(-6, 3), -2);
        assert_eq!(floor_div_pos(0, 3), 0);
    }
}
