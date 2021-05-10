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
}

#[allow(unused_variables)]
#[spirv(fragment)]
pub fn main_fs(
    v_tex_coords: Vec2,
    v_position: Vec3,
    v_tangent_matrix: Mat3,
    #[spirv(descriptor_set = 0, binding = 0)] t_diffuse: &Image2d,
    #[spirv(descriptor_set = 0, binding = 1)] t_normal: &Image2d,
    #[spirv(descriptor_set = 0, binding = 2)] s_diffuse: &Sampler,
    #[spirv(descriptor_set = 0, binding = 3)] s_normal: &Sampler,
    #[spirv(uniform, descriptor_set = 1, binding = 0)] uniforms: &Uniforms,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] lights: &[Light],
    #[spirv(uniform, descriptor_set = 2, binding = 1)] ambient: &f32,
    output: &mut Vec4,
) { 
    let object_normal: Vec4 = t_normal.sample(*s_normal, v_tex_coords);
    let normal = (v_tangent_matrix * (object_normal.xyz() * 2.0 - Vec3::splat(1.0))).normalize();
    let object_color: Vec4 = t_diffuse.sample(*s_diffuse, v_tex_coords);
    let view_dir = (uniforms.u_view_position.xyz() - v_position).normalize();

    let mut result = (*ambient) * object_color.xyz();
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
        result += (diffuse_color + specular_color) * object_color.xyz();
    }
    if object_color.w < 0.1 {
        discard();
    }
    *output = result.extend(object_color.w);
}

/* Vert shader
    #version 450

    layout(location=0) in vec3 a_position;
    layout(location=1) in vec2 a_tex_coords;
    layout(location=2) in vec3 a_normal;

    layout(location=0) out vec2 v_tex_coords;
    layout(location=1) out vec3 v_normal;
    layout(location=2) out vec3 v_position;

    layout(location=5) in vec4 model_matrix_0;
    layout(location=6) in vec4 model_matrix_1;
    layout(location=7) in vec4 model_matrix_2;
    layout(location=8) in vec4 model_matrix_3;

    layout(set=1, binding=0)
    uniform Uniforms {
        vec4 u_view_pos;
        mat4 u_view;
        mat4 u_proj;
    };
*/

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
    #[spirv(position)] out_pos: &mut Vec4,
    v_tex_coords: &mut Vec2,
    v_position: &mut Vec3,
    v_tangent_matrix: &mut Mat3,

) {
    // *v_tex_coords = (a_tex_coords * a_tex_scale) + a_tex_offset;
    // *out_pos = (a_position  * a_pos_scale.extend(1.0) + pos_offset - camera_pos.extend(0.0)).extend(1.0);
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
    let normal = (normal_matrix * a_normal.extend(0.0)).normalize();
    let tangent = (normal_matrix * a_tangent.xyz().extend(0.0)).normalize();
    let bitangent = (normal_matrix * a_bitangent.xyz().extend(0.0)).normalize();
    *v_tangent_matrix = Mat3::from_cols(
            tangent.xyz(),
            bitangent.xyz(),
            normal.xyz(),
        ).transpose();
    *v_tex_coords = a_tex_coords;
    let model_space = model_matrix * a_position.extend(1.0);
    *v_position = model_space.xyz();
    *out_pos = uniforms.u_proj * uniforms.u_view * model_space;
}

// /**
// * Built in Mat4 inverse currently broken in spirv
// **/
// #[inline(always)]
// fn mat4_inverse(matrix: Mat4) -> Mat4 {
//     let determinant = matrix.determinant();
//     let array: [[f32; 4];4] = matrix.to_cols_array_2d();
//     let mut new_array: [[f32; 4]; 4] = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],];
//     for j in 0..4 {
//         for i in 0..4 {
//             let mut temp = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
//             let mut place = 0;
//             for new_j in 0..4 {
//                 for new_i in 0..4 {
//                     if (new_j == j) || (new_i == i) {
//                         continue;
//                     }
//                     temp[place/3][place%3] = array[new_j][new_i];
//                     place += 1;
//                 }
//             }
//             let temp_mat = Mat2::from_cols(
//                 Vec2::new(temp[0][0], temp[0][1]),
//                 Vec2::new(temp[1][0], temp[1][1]),
//             );
//             new_array[j][i] = (-1i32).pow(i as u32 + j as u32) as f32 * temp_mat.determinant();
//         }
//     }
//     return (1.0 / determinant) * Mat4::from_cols(
//         Vec4::new(new_array[0][0], new_array[0][1], new_array[0][2], new_array[0][3]),
//         Vec4::new(new_array[1][0], new_array[1][1], new_array[1][2], new_array[1][3]),
//         Vec4::new(new_array[2][0], new_array[2][1], new_array[2][2], new_array[2][3]),
//         Vec4::new(new_array[3][0], new_array[3][1], new_array[3][2], new_array[3][3]),
//     ).transpose();
// }

// #[cfg(test)]
// mod tests {
//     // Note this useful idiom: importing names from outer (for mod tests) scope.
//     use super::*;

//     #[test]
//     fn test_invert() {
//         let test_mat: Mat4 = Mat4::from_cols_array(&[2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 5.0]).transpose();
//         let reference = test_mat.inverse();
//         let implement = mat4_inverse(test_mat);
//         assert_eq!(reference, implement);
//     }

//     #[test]
//     fn test_invert_mult() {
//         let test_mat: Mat4 = Mat4::from_cols_array(&[2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 5.0]).transpose();
//         let test_vec: Vec3 = [0.0, 1.0, 0.0].into();
//         let reference = test_mat.inverse();
//         let implement = mat4_inverse(test_mat);
//         assert_eq!(reference * test_vec, implement * test_vec);
//     }
// }
