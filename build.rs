extern crate cmake;
use cmake::Config;

fn main() {
    // Run the CMake build process
    let dst = Config::new("libkokkos_bridge").build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=kokkos_bridge");
}
