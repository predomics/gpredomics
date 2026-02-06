use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GPREDOMICS_GIT_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");

    if let Ok(sha) = env::var("GPREDOMICS_GIT_SHA") {
        if !sha.trim().is_empty() {
            println!("cargo:rustc-env=GPREDOMICS_GIT_SHA={}", sha.trim());
        }
        return;
    }

    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let sha = String::from_utf8_lossy(&output.stdout);
            let sha = sha.trim();
            if !sha.is_empty() {
                println!("cargo:rustc-env=GPREDOMICS_GIT_SHA={}", sha);
            }
        }
    }
}
