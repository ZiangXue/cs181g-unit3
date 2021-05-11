use spirv_builder::SpirvBuilder;
use std::error::Error;

fn build_shader(path_to_create: &str) -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed={}/src/lib.rs", path_to_create);
    SpirvBuilder::new(path_to_create, "spirv-unknown-vulkan1.1").build()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    build_shader("../shaders/models")?;
    build_shader("../shaders/shadows")?;
    build_shader("../shaders/sky-shader")?;
    Ok(())
}
