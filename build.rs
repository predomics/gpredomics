// build.rs
use std::env;
use std::path::PathBuf;

fn main() {
    // 1) Tell cargo to look for Kokkos include/lib if needed
    //    Suppose we installed Kokkos in /usr/local/kokkos as an example.
    //    Or, you might have environment variables for these.
    let kokkos_include = "/usr/local/kokkos/include";
    let kokkos_lib = "/usr/local/kokkos/lib";

    // 2) Compile our C++ code using `cc::Build`.
    let mut build = cc::Build::new();
    build
        .cpp(true)                 // We're compiling C++.
        .file("c_src/kokkos_bridge.cpp")
        .flag("-std=c++17")       // Or c++14/20, depending on your Kokkos build
        .include(kokkos_include)  // Let the compiler find <Kokkos_Core.hpp>, etc.
        .warnings(false);         // Optional: silence warnings

    // If you need special flags for Apple Clang or find you need to link e.g. -lc++:
    // build.flag("-stdlib=libc++");

    // 3) Finalize and compile
    build.compile("kokkos_bridge"); 
    // This produces a libkokkos_bridge.a in <target>/build/ folder.

    // 4) Instruct cargo about the Kokkos library path:
    println!("cargo:rustc-link-search=native={}", kokkos_lib);
    // Link to Kokkos libraries, e.g. "kokkoscore", "kokkoscontainers", etc.
    // The actual libs you need depend on how Kokkos was built.
    println!("cargo:rustc-link-lib=static=kokkoscore");  
    // If your Kokkos build outputs libs named differently, adapt accordingly.
    
    // If Kokkos depends on extra libs, link them as well (like pthread).
    println!("cargo:rustc-link-lib=dylib=c++"); // On macOS, might be "c++", "stdc++", etc.

    // If we store these in environment variables, we can read them with env::var().
}
