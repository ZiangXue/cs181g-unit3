#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(register_attr, lang_items),
    register_attr(spirv)
)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

#[cfg(not(target_arch = "spirv"))]
#[macro_use]
pub extern crate spirv_std_macros;
use glam::{Vec2, Vec3, Vec4};
use spirv_std::{discard, Image2d, Sampler};

#[spirv(fragment)]
pub fn main_fs(
    v_tex_coords: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] t_diffuse: &Image2d,
    #[spirv(descriptor_set = 0, binding = 1)] s_diffuse: &Sampler,
    output: &mut Vec4,
) {
    let texel: Vec4 = t_diffuse.sample(*s_diffuse, v_tex_coords);
    if texel.w < 0.5 {
        discard();
    }
    *output = texel;
}

#[spirv(vertex)]
#[deny(clippy::too_many_arguments)]
pub fn main_vs(
    a_position: Vec3,
    a_tex_coords: Vec2,
    pos_offset: Vec3,
    a_pos_scale: Vec2,
    a_tex_offset: Vec2,
    a_tex_scale: Vec2,
    #[spirv(uniform, descriptor_set = 1, binding = 0)] camera_pos: &Vec2,
    #[spirv(position)] out_pos: &mut Vec4,
    v_tex_coords: &mut Vec2,
) {
    *v_tex_coords = (a_tex_coords * a_tex_scale) + a_tex_offset;
    *out_pos =
        (a_position * a_pos_scale.extend(1.0) + pos_offset - camera_pos.extend(0.0)).extend(1.0);
}
