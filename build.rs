extern crate cmake;
use cmake::Config;

fn main() {
    // Run the CMake build process
    let dst = Config::new("libkokkos_bridge").build();

    //println!("cargo:rustc-link-arg=-mmacosx-version-min=11.0");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=kokkoscore");
    println!("cargo:rustc-link-search=native={}/build/out/lib", dst.display());
    println!("cargo:rustc-link-lib=static=kokkos_bridge");

    //println!("cargo:rustc-link-search=native=/usr/local/lib/");
    //println!("cargo:rustc-link-lib=static=kokkoscore");
    //println!("cargo:rustc-link-lib=static=kokkoskernels");
    //println!("cargo:rustc-link-lib=static=omp"); // OpenMP runtime
}
