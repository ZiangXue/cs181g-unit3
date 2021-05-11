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
use glam::Vec3Swizzles;
#[allow(unused_imports)]
use glam::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
#[allow(unused_imports)]
use spirv_std::{Image2d, Sampler, discard,};
/* Frag Shader
#version 450

layout(location=0) in vec2 v_tex_coords;
layout(location=1) in vec3 v_normal;
layout(location=2) in vec3 v_position;

layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;
layout(set=1, binding=0)
uniform Uniforms {
    vec3 u_view_position; // unused
    mat4 u_view;
    mat4 u_proj;
};

struct Light {
  vec4 pos;
  vec4 color;
  // vec4 dir;
};

layout(set=2, binding=0)
uniform Lights {
    Light lights[10];
};
layout(set=2, binding=1)
uniform LightsAmbient {
    float ambient;
};
*/

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
    v_tex_coords: Vec2,
    v_position: Vec3,
    v_light_space: Vec4,
    v_tangent_matrix: Mat3,
    #[spirv(descriptor_set = 0, binding = 0)] t_diffuse: &Image2d,
    #[spirv(descriptor_set = 0, binding = 1)] t_normal: &Image2d,
    #[spirv(descriptor_set = 0, binding = 2)] s_diffuse: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3)] s_normal: &Sampler,
    #[spirv(uniform, descriptor_set = 1, binding = 0)] uniforms: &Uniforms,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] lights: &[Light],
    #[spirv(uniform, descriptor_set = 2, binding = 1)] ambient: &f32,
    #[spirv(descriptor_set = 2, binding = 2)] shadow_maps: &Image2d,
    output: &mut Vec4,
) { 
    let object_normal: Vec4 = t_normal.sample(*s_normal, v_tex_coords);
    let normal = (v_tangent_matrix * (object_normal.xyz() * 2.0 - Vec3::splat(1.0))).normalize();
    let object_color: Vec4 = t_diffuse.sample(*s_diffuse, v_tex_coords);
    let view_dir = (uniforms.u_view_position.xyz() - v_position).normalize();

    let mut result_strength = Vec3::splat(*ambient);
    for i in 0..10 {
        // Point-light specific; change if directional lights, spotlights are used
        // to branch on e.g. position.w == 0 (directional) or direction.w == 0 (point) or else spot
        let light_color = lights[i].color.xyz();
        let light_position = lights[i].pos.xyz();
        let light_dir = (light_position - v_position).normalize();
        let diffuse_strength = normal.dot(light_dir).max(0.0);
        let diffuse_color = light_color * diffuse_strength;
        let half_dir = (view_dir + light_dir).normalize();
        let specular_strength = normal.dot(half_dir).max(0.0);
        let specular_color = specular_strength * light_color;
        let shadow = shadow_calculation(v_light_space, &shadow_maps, s_diffuse);
        result_strength += diffuse_color + specular_color;
    }
    if object_color.w < 0.1 {
        discard();
    }
    let strength_r = discretize(result_strength.x);
    let strength_g = discretize(result_strength.y);
    let strength_b = discretize(result_strength.z);
    let strength: Vec3 = [strength_r, strength_g, strength_b].into();
    *output = (strength * object_color.xyz()).extend(object_color.w);
}

#[allow(dead_code)]
fn shadow_calculation(light_space: Vec4, shadow_map: &Image2d, sampler: &Sampler) -> f32
{
    // perform perspective divide
    let proj_coords = light_space.xyz() / light_space.w;
    // transform to [0,1] range
    let proj_coords = proj_coords * 0.5 + Vec3::splat(0.5);
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    let closest_depth: Vec4 = shadow_map.sample(*sampler, proj_coords.xy());
    // get depth of current fragment from light's perspective
    let current_depth = proj_coords.z;
    // check whether current frag pos is in shadow
    let shadow = (current_depth > closest_depth.x)  as u32 as f32;

    return shadow;
}  

fn discretize(intensity: f32) -> f32 {
    match intensity {
        strength if (strength <= 0.0) => 0.0,
        strength if (strength <= 0.25) => 0.25,
        strength if (strength <= 0.75) => 0.75,
        _ => 1.0
    }
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
    v_tex_coords: &mut Vec2,
    v_position: &mut Vec3,
    v_light_space: &mut Vec4,
    v_tangent_matrix: &mut Mat3,
) {
    let model_matrix = Mat4::from_cols(
            model_matrix_0,
            model_matrix_1,
            model_matrix_2,
            model_matrix_3,
    );
    let normal_matrix = Mat4::from_cols(
        normal_matrix_0,
        normal_matrix_1,
        normal_matrix_2,
        normal_matrix_3,
    );
    let normal = (normal_matrix * a_normal.extend(1.0)).normalize();
    let tangent = (normal_matrix * a_tangent.xyz().extend(1.0)).normalize();
    let bitangent = (normal_matrix * a_bitangent.xyz().extend(1.0)).normalize();
    *v_tangent_matrix = Mat3::from_cols(
            tangent.xyz(),
            bitangent.xyz(),
            normal.xyz(),
        ).transpose();
    *v_tex_coords = a_tex_coords;
    let model_space = model_matrix * a_position.extend(1.0);
    *v_position = model_space.xyz();
    *out_pos = uniforms.u_proj * uniforms.u_view * model_space;
    *v_light_space = lights[0].light_space_mat * model_space;
    for i in 0..10 {
        //v_frag_pos[i] = lights[i].light_space_mat * (*out_pos);
    }
}
