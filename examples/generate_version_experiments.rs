use gpredomics::param::Param;
/// Example: Generate versioned experiment files for serialization compatibility testing
///
/// This example runs various combinations of GA, BEAM and MCMC algorithms with different parameters
/// (with/without CV, with/without random sampling) and serializes only the final generation
/// to test forward/backward compatibility of the serialization format.
/// Ideally, it should be run each time a new update is developed, so that the results can ultimately
/// be compared with previous ones.
///
/// Run with: cargo run --example generate_version_experiments --release
//use gpredomics::{run_ga, run_beam, run_mcmc};
use gpredomics::run;
use log::{info, warn};
use std::fs;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn main() {
    // Initialize logging using flexi_logger (already a dependency)
    flexi_logger::Logger::try_with_env_or_str("info")
        .unwrap()
        .start()
        .unwrap();

    let version = env!("CARGO_PKG_VERSION");
    let git_hash = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.chars().take(7).collect::<String>())
        .unwrap_or_else(|| "unknown".to_string());
    let gpredomics_version = format!("{}_{}", version, git_hash);

    info!("=== Generating Version Experiments ===");
    info!("Gpredomics version: {}", gpredomics_version);

    // Create version directory for this version
    let version_dir = format!("tests/consistency/v{}", gpredomics_version);
    if let Err(e) = fs::create_dir_all(&version_dir) {
        panic!("Failed to create version directory: {}", e);
    }

    // Use default parameters to avoid dependency on param.yaml
    let mut base_param = Param::default();

    // Configure data paths with Qin2014 datasets
    base_param.data.X = "samples/Qin2014/Xtrain.tsv".to_string();
    base_param.data.y = "samples/Qin2014/Ytrain.tsv".to_string();
    base_param.data.Xtest = "samples/Qin2014/Xtest.tsv".to_string();
    base_param.data.ytest = "samples/Qin2014/Ytest.tsv".to_string();
    base_param.data.features_in_rows = true;
    base_param.data.classes = vec![
        "healthy".to_string(),
        "cirrhosis".to_string(),
        "unknown".to_string(),
    ];

    // Reduce computational load for faster example execution
    base_param.general.thread_number = 4;
    base_param.general.language = "bin,ter".to_string();
    base_param.general.data_type = "raw,log".to_string();
    base_param.ga.population_size = 500;
    base_param.ga.max_epochs = 10;
    base_param.ga.min_epochs = 5;
    base_param.beam.k_start = 2;
    base_param.beam.k_stop = 10;
    base_param.cv.outer_folds = 3;
    base_param.cv.inner_folds = 3;
    base_param.mcmc.n_iter = 5000;
    base_param.mcmc.n_burn = 2500;
    base_param.general.keep_trace = false; // Only keep final generation
    base_param.general.gpu = false; // Disable GPU for compatibility

    let running = Arc::new(AtomicBool::new(true));

    // ========================================
    // GA Experiments
    // ========================================
    info!("\n--- Generating GA Experiments ---");

    // 1. GA without CV
    {
        info!("1/9: GA without CV");
        let mut param = base_param.clone();
        param.general.algo = "ga".to_string();
        param.general.cv = false;
        param.ga.random_sampling_pct = 0.0;
        param.cv.overfit_penalty = 0.0;
        param.general.save_exp = format!("{}/ga_no_cv.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save GA no-CV experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // 2. GA with outer CV
    {
        info!("2/9: GA with outer CV");
        let mut param = base_param.clone();
        param.general.algo = "ga".to_string();
        param.general.cv = true;
        param.ga.random_sampling_pct = 0.0;
        param.cv.overfit_penalty = 0.0;
        param.general.save_exp = format!("{}/ga_with_outer_cv.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save GA with outer-CV experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // 3. GA with inner CV (overfit penalty)
    {
        info!("3/9: GA with inner CV (overfit penalty)");
        let mut param = base_param.clone();
        param.general.algo = "ga".to_string();
        param.general.cv = false;
        param.ga.random_sampling_pct = 0.0;
        param.cv.overfit_penalty = 0.5;
        param.cv.inner_folds = 3;
        param.general.save_exp = format!("{}/ga_with_inner_cv.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save GA with inner-CV experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // 4. GA with random sampling (no CV)
    {
        info!("4/9: GA with random sampling");
        let mut param = base_param.clone();
        param.general.algo = "ga".to_string();
        param.general.cv = false;
        param.ga.random_sampling_pct = 50.0;
        param.ga.random_sampling_epochs = 2;
        param.cv.overfit_penalty = 0.0;
        param.general.save_exp = format!("{}/ga_random_sampling.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save GA random-sampling experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // ========================================
    // BEAM Experiments
    // ========================================
    info!("\n--- Generating BEAM Experiments ---");

    // 5. BEAM without CV
    {
        info!("5/9: BEAM without CV");
        let mut param = base_param.clone();
        param.general.algo = "beam".to_string();
        param.general.cv = false;
        param.general.save_exp = format!("{}/beam_no_cv.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save BEAM no-CV experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // 6. BEAM with CV
    {
        info!("6/9: BEAM with CV");
        let mut param = base_param.clone();
        param.general.algo = "beam".to_string();
        param.general.cv = true;
        param.general.save_exp = format!("{}/beam_with_cv.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save BEAM with-CV experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // 7. BEAM ParallelForward method
    {
        info!("7/9: BEAM ParallelForward");
        let mut param = base_param.clone();
        param.general.algo = "beam".to_string();
        param.general.cv = false;
        param.beam.method = gpredomics::beam::BeamMethod::ParallelForward;
        param.general.save_exp = format!("{}/beam_method_2.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save BEAM ParallelForward experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // ========================================
    // MCMC Experiments
    // ========================================
    info!("\n--- Generating MCMC Experiments ---");

    // 8. MCMC without SBS
    {
        info!("8/9: MCMC without SBS");
        let mut param = base_param.clone();
        param.general.algo = "mcmc".to_string();
        param.general.cv = false;
        param.general.data_type = "raw".to_string(); // MCMC requires single data type
        param.mcmc.n_iter = 100;
        param.mcmc.n_burn = 50;
        param.mcmc.nmin = 0; // Disable SBS
        param.general.save_exp = format!("{}/mcmc_no_sbs.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save MCMC no-SBS experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    // 9. MCMC with SBS
    {
        info!("9/9: MCMC with SBS");
        let mut param = base_param.clone();
        param.general.algo = "mcmc".to_string();
        param.general.cv = false;
        param.general.data_type = "raw".to_string(); // MCMC requires single data type
        param.mcmc.nmin = 480;
        param.mcmc.n_iter = 100;
        param.mcmc.n_burn = 50;
        param.general.save_exp = format!("{}/mcmc_with_sbs.mp", version_dir);

        let exp = run(&param, running.clone());
        if let Err(e) = exp.save_auto(&param.general.save_exp) {
            warn!("Failed to save MCMC with-SBS experiment: {}", e);
        } else {
            info!("Saved: {}", param.general.save_exp);
        }
    }

    info!("\n=== Generation Complete ===");
    info!("Experiments saved in: {}", version_dir);
    info!("\nTo test compatibility, run:");
    info!("  cargo test --test test_consistency -- --no-capture --test-threads=1");
}
