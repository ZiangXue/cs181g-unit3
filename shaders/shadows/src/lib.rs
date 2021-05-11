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
#[allow(unused_imports)]
use glam::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
#[allow(unused_imports)]
use spirv_std::{Image2d, Sampler, discard,};

#[derive(Copy, Clone)]
#[repr(C)]
pub struct Uniforms {
    u_view_position: Vec4, // unused
    u_view: Mat4,
    u_proj: Mat4,
}


#[derive(Copy, Clone)]
#[repr(C)]
pub struct Light {
    pos: Vec4,
    color: Vec4,
    light_space_mat: Mat4,
}

#[allow(unused_variables)]
#[spirv(fragment)]
pub fn main_fs(
) { 
    // Intentionally empty
}

#[allow(unused_variables)]
#[spirv(vertex)]
pub fn main_vs(
    a_position: Vec3,
    a_tex_coords: Vec2,
    a_normal: Vec3,
    a_tangent: Vec4,
    a_bitangent: Vec4,
    bone_ids: u32,
    bone_weights: Vec4,
    model_matrix_0: Vec4,
    model_matrix_1: Vec4,
    model_matrix_2: Vec4,
    model_matrix_3: Vec4,
    normal_matrix_0: Vec4,
    normal_matrix_1: Vec4,
    normal_matrix_2: Vec4,
    normal_matrix_3: Vec4,
    #[spirv(uniform, descriptor_set = 1, binding = 0)] uniforms: &Uniforms,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] lights: &[Light],
    #[spirv(position)] out_pos: &mut Vec4,
) {
    let model_matrix = Mat4::from_cols(
            model_matrix_0,
            model_matrix_1,
            model_matrix_2,
            model_matrix_3,
    );
    
    let model_space = model_matrix * a_position.extend(1.0);
    *out_pos = lights[0].light_space_mat * model_space;
}
