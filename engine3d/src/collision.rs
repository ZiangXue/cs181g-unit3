use std::{cmp::max, usize};

use cgmath::Vector3;

use crate::geom::*;

const SAMPLE_DENSITY: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
pub struct Contact<T: Copy> {
    pub a: T,
    pub b: T,
    pub mtv: Vec3,
}

pub fn restitute_dyn_stat<S1: Shape, S2: Shape>(
    ashapes: &mut [S1],
    avels: &mut [Vec3],
    bshapes: &[S2],
    contacts: &mut [Contact<usize>],
) where
    S1: Collide<S2>,
{
    contacts.sort_unstable_by(|a, b| b.mtv.magnitude2().partial_cmp(&a.mtv.magnitude2()).unwrap());
    for c in contacts.iter() {
        let a = c.a;
        let b = c.b;
        // Are they still touching?  This way we don't need to track disps or anything
        // at the expense of some extra collision checks
        if let Some(disp) = ashapes[a].disp(&bshapes[b]) {
            // We can imagine we're instantaneously applying a
            // velocity change to pop the object just above the floor.
            ashapes[a].translate(disp);
            // It feels a little weird to be adding displacement (in
            // units) to velocity (in units/frame), but we'll roll
            // with it.  We're not exactly modeling a normal force
            // here but it's something like that.
            avels[a] += disp;
        }
    }
}

pub fn restitute_dyn_dyn<S1: Shape, S2: Shape>(
    ashapes: &mut Vec<S1>,
    avels: &mut Vec<Vec3>,
    ahps: &mut Vec<usize>,
    bshapes: &mut Vec<S2>,
    bvels: &mut Vec<Vec3>,
    bhps: &mut Vec<usize>,
    contacts: &mut Vec<Contact<usize>>,
)
where
    S1: Collide<S2>,
{
    contacts.sort_unstable_by(|a, b| b.mtv.magnitude2().partial_cmp(&a.mtv.magnitude2()).unwrap());
    // That can bump into each other in perfectly elastic collisions!
    for c in contacts.iter() {
        let a = c.a;
        let b = c.b;
        // Just split the difference.  In crowded situations this will
        // cause issues, but those will always be hard to solve with
        // this kind of technique.
        if let Some(disp) = ashapes[a].disp(&bshapes[b]) {
            ashapes[a].translate(-disp / 2.0);
            avels[a] -= disp / 2.0;
            bshapes[b].translate(disp / 2.0);
            bvels[b] += disp / 2.0;
            if ahps[a] >= 1 {
                ahps[a] -= 1;
            } else {
                ahps[a] = 0;
            }
            if bhps[b] >= 1 {
                bhps[b] -= 1;
            } else {
                bhps[b] = 0;
            }
        }
    }
}

pub fn restitute_dyns<S1: Shape>(
    ashapes: &mut [S1],
    avels: &mut [Vec3],
    ahps: &mut [usize],
    contacts: &mut [Contact<usize>],
) where
    S1: Collide<S1>,
{
    contacts.sort_unstable_by(|a, b| b.mtv.magnitude2().partial_cmp(&a.mtv.magnitude2()).unwrap());
    // That can bump into each other in perfectly elastic collisions!
    for c in contacts.iter() {
        let a = c.a;
        let b = c.b;
        // Just split the difference.  In crowded situations this will
        // cause issues, but those will always be hard to solve with
        // this kind of technique.
        if let Some(disp) = ashapes[a].disp(&ashapes[b]) {
            /* ashapes[a].translate(-disp / 2.0);
            avels[a] -= disp / 2.0;
            ashapes[b].translate(disp / 2.0);
            avels[b] += disp / 2.0; */
            let direction = direction(ashapes[a].pos(), ashapes[b].pos());
            let (a_gain, b_gain) = vel_distribute(
                avels[a],
                avels[b],
                ashapes[a].mass(SAMPLE_DENSITY),
                ashapes[b].mass(SAMPLE_DENSITY),
                direction,
            );
            //hit object should gain speed along hit direction:
            //hitting object should lose speed and change to cut direction.
            avels[a] += a_gain;
            avels[b] += b_gain;
            ashapes[a].translate(-disp / 2.0);
            ashapes[b].translate(disp / 2.0);
            if ahps[a] >= 1 {
                ahps[a] -= 1;
            } else {
                ahps[a] = 0;
            }
            if ahps[b] >= 1 {
                ahps[b] -= 1;
            } else {
                ahps[b] = 0;
            }
        }
    }
}

pub fn gather_contacts_ab<S1: Shape, S2: Shape>(a: &[S1], b: &[S2], into: &mut Vec<Contact<usize>>)
where
    S1: Collide<S2>,
{
    for (ai, a) in a.iter().enumerate() {
        for (bi, b) in b.iter().enumerate() {
            if let Some(disp) = a.disp(b) {
                into.push(Contact {
                    a: ai,
                    b: bi,
                    mtv: disp,
                });
            }
        }
    }
}

pub fn gather_contacts_aa<S1: Shape>(ss: &[S1], into: &mut Vec<Contact<usize>>)
where
    S1: Collide<S1>,
{
    for (ai, a) in ss.iter().enumerate() {
        for (bi, b) in ss[(ai + 1)..].iter().enumerate() {
            let bi = ai + 1 + bi;
            if let Some(disp) = a.disp(b) {
                into.push(Contact {
                    a: ai,
                    b: bi,
                    mtv: disp,
                });
            }
        }
    }
}

// return a unit vector pointing from pos 1 to pos 2, i.e. contact normal
fn direction(pos_from: Vec3, pos_to: Vec3) -> Vector3<f32> {
    (pos_to - pos_from).normalize()
}

// distribute velocity of objects with impulse
fn vel_distribute(v_1: Vec3, v_2: Vec3, m_1: f32, m_2: f32, direction: Vec3) -> (Vec3, Vec3) {
    let momentum_a = v_1 * m_1;
    let ma_dir_norm = (momentum_a.dot(direction) as f32).abs() / norm(direction);
    let momentum_b = v_2 * m_2;
    let mb_dir_norm = (momentum_b.dot(direction) as f32).abs() / norm(direction);
    let sum_impulse = (ma_dir_norm + mb_dir_norm) * direction;
    let a_gain = sum_impulse * -1.0 * (m_2 / (m_1 + m_2)) / m_1;
    let b_gain = sum_impulse * (m_1 / (m_1 + m_2)) / m_2;
    return (a_gain, b_gain);
}

// return norm of a Vec3
fn norm(v_1: Vec3) -> f32 {
    (v_1.dot(v_1) as f32).sqrt()
}