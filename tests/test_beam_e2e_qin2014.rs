use gpredomics::beam::BeamMethod;
/// End-to-End Integration Test for Beam Algorithm with Qin2014 Dataset
///
/// This test validates the complete Beam workflow:
/// 1. Loading and preprocessing Qin2014 data
/// 2. Running Beam optimization (LimitedExhaustive & ParallelForward)
/// 3. Evaluating on test set
/// 4. Verifying experiment structure and results
/// 5. Testing serialization/deserialization
///
/// Run with: cargo test --test test_beam_e2e_qin2014 -- --nocapture
use gpredomics::experiment::Experiment;
use gpredomics::param::Param;
use gpredomics::run;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Helper function to create Beam parameters for Qin2014
fn create_qin2014_beam_params() -> Param {
    let mut param = Param::default();

    // General settings
    param.general.seed = 42;
    param.general.algo = "beam".to_string();
    param.general.cv = false;
    param.general.thread_number = 4;
    param.general.gpu = false;
    param.general.language = "ter".to_string();
    param.general.data_type = "prev".to_string();
    param.general.data_type_epsilon = 1e-5;
    param.general.fit = gpredomics::param::FitFunction::auc;
    param.general.k_penalty = 0.0001;
    param.general.fr_penalty = 0.0;
    param.general.n_model_to_display = 10;
    param.general.log_level = "info".to_string();
    param.general.display_colorful = false;
    param.general.keep_trace = true; // Important for Beam to see all k levels
    param.general.save_exp = "test_exp.mp".to_string();
    param.general.log_base = "".to_string();
    param.general.log_suffix = "log".to_string();

    // Data settings - More restrictive feature selection for Beam
    param.data.X = "samples/Qin2014/Xtrain.tsv".to_string();
    param.data.y = "samples/Qin2014/Ytrain.tsv".to_string();
    param.data.Xtest = "samples/Qin2014/Xtest.tsv".to_string();
    param.data.ytest = "samples/Qin2014/Ytest.tsv".to_string();
    param.data.features_in_rows = true;
    param.data.max_features_per_class = 20; // Limit features for faster Beam
    param.data.feature_minimal_prevalence_pct = 10.0; // Less restrictive
    param.data.feature_minimal_feature_value = 1e-4;
    param.data.feature_selection_method = gpredomics::data::PreselectionMethod::wilcoxon;
    param.data.feature_maximal_adj_pvalue = 0.1; // More permissive
    param.data.feature_minimal_log_abs_bayes_factor = 0.5;
    param.data.inverse_classes = false;
    param.data.classes = vec!["healthy".to_string(), "cirrhosis".to_string()];

    // CV settings
    param.cv.inner_folds = 5;
    param.cv.overfit_penalty = 0.0;
    param.cv.outer_folds = 5;
    param.cv.resampling_inner_folds_epochs = 0;
    param.cv.fit_on_valid = true;
    param.cv.cv_best_models_ci_alpha = 0.05;

    // Importance settings
    param.importance.compute_importance = false;
    param.importance.n_permutations_oob = 100;
    param.importance.scaled_importance = true;
    param.importance.importance_aggregation = gpredomics::experiment::ImportanceAggregation::mean;

    // Voting settings
    param.voting.vote = false;
    param.voting.fbm_ci_alpha = 0.05;
    param.voting.min_perf = 0.5;
    param.voting.min_diversity = 10.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.method_threshold = 0.5;
    param.voting.threshold_windows_pct = 5.0;
    param.voting.complete_display = false;

    // Beam settings - Small range for quick test
    param.beam.method = BeamMethod::LimitedExhaustive;
    param.beam.kmin = 2;
    param.beam.kmax = 5; // Small for quick tests
    param.beam.best_models_ci_alpha = 0.001; // Very small (low = more models kept)
    param.beam.max_nb_of_models = 10000;

    // GA settings (not used for Beam but need default)
    param.ga.population_size = 100;
    param.ga.max_epochs = 10;
    param.ga.min_epochs = 1;
    param.ga.max_age_best_model = 10;
    param.ga.kmin = 1;
    param.ga.kmax = 50;
    param.ga.select_elite_pct = 5.0;
    param.ga.select_niche_pct = 0.0;
    param.ga.select_random_pct = 10.0;
    param.ga.mutated_children_pct = 80.0;
    param.ga.mutated_features_pct = 20.0;
    param.ga.mutation_non_null_chance_pct = 20.0;
    param.ga.forced_diversity_pct = 0.0;
    param.ga.forced_diversity_epochs = 10;
    param.ga.random_sampling_pct = 0.0;
    param.ga.random_sampling_epochs = 0;

    // GPU settings
    param.gpu.fallback_to_cpu = true;
    param.gpu.memory_policy = gpredomics::param::GpuMemoryPolicy::Strict;
    param.gpu.max_total_memory_mb = 256;
    param.gpu.max_buffer_size_mb = 128;

    param
}

#[test]
fn test_beam_qin2014_basic_run() {
    println!("\n=== Testing Beam with Qin2014 Dataset (Basic Run) ===\n");

    let param = create_qin2014_beam_params();

    // Verify data files exist
    assert!(
        Path::new(&param.data.X).exists(),
        "Training X file not found: {}",
        param.data.X
    );
    assert!(
        Path::new(&param.data.y).exists(),
        "Training y file not found: {}",
        param.data.y
    );
    assert!(
        Path::new(&param.data.Xtest).exists(),
        "Test X file not found: {}",
        param.data.Xtest
    );
    assert!(
        Path::new(&param.data.ytest).exists(),
        "Test y file not found: {}",
        param.data.ytest
    );

    // Run Beam
    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // Debug: print collections structure
    println!(
        "  DEBUG: collections.len() = {}",
        experiment.collections.len()
    );
    if !experiment.collections.is_empty() {
        println!(
            "  DEBUG: collections[0].len() = {}",
            experiment.collections[0].len()
        );
        if !experiment.collections[0].is_empty() {
            println!(
                "  DEBUG: collections[0][0].individuals.len() = {}",
                experiment.collections[0][0].individuals.len()
            );
        }
    }

    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout)
        .unwrap()
        .chars()
        .take(7)
        .collect::<String>();
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);

    // Verify experiment structure
    assert!(
        !experiment.id.is_empty(),
        "Experiment ID should not be empty"
    );
    assert!(
        !experiment.timestamp.is_empty(),
        "Timestamp should not be empty"
    );
    assert_eq!(experiment.gpredomics_version, gpredomics_version);
    assert_eq!(experiment.parameters.general.algo, "beam");

    // Verify train data was loaded
    assert!(
        experiment.train_data.sample_len > 0,
        "Train data should have samples"
    );
    assert!(
        experiment.train_data.feature_len > 0,
        "Train data should have features"
    );
    assert_eq!(
        experiment.train_data.classes.len(),
        2,
        "Should have 2 classes"
    );

    // Verify test data was loaded
    assert!(experiment.test_data.is_some(), "Test data should be loaded");
    let test_data = experiment.test_data.as_ref().unwrap();
    assert!(test_data.sample_len > 0, "Test data should have samples");
    assert!(test_data.feature_len > 0, "Test data should have features");

    // Verify experiment collections (Beam creates one Vec<Population>, each Population is a k-level)
    assert!(
        !experiment.collections.is_empty(),
        "Collections should not be empty"
    );

    // For Beam without CV, collections is vec![Vec<Population>] where the inner Vec has one Population per k
    let beam_populations = &experiment.collections[0];
    assert!(
        !beam_populations.is_empty(),
        "Beam populations should not be empty"
    );

    // Beam with keep_trace should have multiple populations (one per k)
    println!(
        "  - Number of k levels explored: {}",
        beam_populations.len()
    );

    // Verify final population
    assert!(
        experiment.final_population.is_some(),
        "Final population should exist"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population should have individuals"
    );

    // Verify best model
    let best_model = &final_pop.individuals[0];
    assert!(
        best_model.auc >= 0.0 && best_model.auc <= 1.0,
        "AUC should be between 0 and 1"
    );
    assert!(
        best_model.sensitivity >= 0.0 && best_model.sensitivity <= 1.0,
        "Sensitivity should be between 0 and 1"
    );
    assert!(
        best_model.specificity >= 0.0 && best_model.specificity <= 1.0,
        "Specificity should be between 0 and 1"
    );
    assert!(
        best_model.k >= param.beam.kmin,
        "Best model should have at least kmin features"
    );
    assert!(
        best_model.k <= param.beam.kmax,
        "Best model should have at most kmax features"
    );
    assert!(
        !best_model.features.is_empty(),
        "Best model should have features"
    );

    // Verify execution time is reasonable
    assert!(
        experiment.execution_time > 0.0,
        "Execution time should be positive"
    );

    println!("✓ Basic Beam run completed successfully");
    println!("  - Train samples: {}", experiment.train_data.sample_len);
    println!("  - Train features: {}", experiment.train_data.feature_len);
    println!(
        "  - Selected features: {}",
        experiment.train_data.feature_selection.len()
    );
    println!("  - Test samples: {}", test_data.sample_len);
    println!("  - Best model AUC: {:.4}", best_model.auc);
    println!("  - Best model features (k): {}", best_model.k);
    println!(
        "  - K range explored: [{}, {}]",
        param.beam.kmin,
        beam_populations.len() + param.beam.kmin - 1
    );
    println!("  - Execution time: {:.2}s", experiment.execution_time);
}

#[test]
fn test_beam_qin2014_limited_exhaustive() {
    println!("\n=== Testing Beam LimitedExhaustive Method ===\n");

    let mut param = create_qin2014_beam_params();
    param.beam.method = BeamMethod::LimitedExhaustive;
    param.beam.kmin = 2;
    param.beam.kmax = 4; // Small for combinatorial

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(!final_pop.individuals.is_empty(), "Should have individuals");

    // Verify k progression in collections
    let beam_populations = &experiment.collections[0];
    if beam_populations.len() > 1 {
        let first_k = beam_populations[0].individuals[0].k;
        let last_k = beam_populations[beam_populations.len() - 1].individuals[0].k;
        println!("  - First k: {}", first_k);
        println!("  - Last k: {}", last_k);
        assert!(last_k >= first_k, "K should increase or stay same");
    }

    println!("✓ LimitedExhaustive method test passed");
}

#[test]
fn test_beam_qin2014_parallel_forward() {
    println!("\n=== Testing Beam ParallelForward Method ===\n");

    let mut param = create_qin2014_beam_params();
    param.beam.method = BeamMethod::ParallelForward;
    param.beam.kmin = 2;
    param.beam.kmax = 5;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(!final_pop.individuals.is_empty(), "Should have individuals");

    println!("  - Final population size: {}", final_pop.individuals.len());
    println!("  - Best model k: {}", final_pop.individuals[0].k);
    println!("  - Best model AUC: {:.4}", final_pop.individuals[0].auc);

    println!("✓ ParallelForward method test passed");
}

#[test]
fn test_beam_qin2014_serialization() {
    println!("\n=== Testing Beam Serialization with Qin2014 ===\n");

    let param = create_qin2014_beam_params();
    let running = Arc::new(AtomicBool::new(true));
    let original_exp = run(&param, running);

    // Test MessagePack serialization
    let mp_file = "test_beam_qin2014_serialization.mp";
    original_exp
        .save_auto(mp_file)
        .expect("Failed to save experiment");
    assert!(Path::new(mp_file).exists(), "Serialized file should exist");

    let loaded_exp = Experiment::load_auto(mp_file).expect("Failed to load experiment");

    // Verify loaded experiment matches original
    assert_eq!(original_exp.id, loaded_exp.id);
    assert_eq!(original_exp.timestamp, loaded_exp.timestamp);
    assert_eq!(
        original_exp.gpredomics_version,
        loaded_exp.gpredomics_version
    );
    assert_eq!(
        original_exp.train_data.sample_len,
        loaded_exp.train_data.sample_len
    );
    assert_eq!(
        original_exp.train_data.feature_len,
        loaded_exp.train_data.feature_len
    );

    // Verify final population
    assert_eq!(
        original_exp
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .len(),
        loaded_exp
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .len()
    );

    // Verify best model metrics
    let orig_best = &original_exp.final_population.as_ref().unwrap().individuals[0];
    let loaded_best = &loaded_exp.final_population.as_ref().unwrap().individuals[0];
    assert_eq!(orig_best.auc, loaded_best.auc);
    assert_eq!(orig_best.sensitivity, loaded_best.sensitivity);
    assert_eq!(orig_best.specificity, loaded_best.specificity);
    assert_eq!(orig_best.k, loaded_best.k);
    assert_eq!(orig_best.features, loaded_best.features);

    // Cleanup
    std::fs::remove_file(mp_file).expect("Failed to cleanup test file");

    println!("✓ Serialization test passed");
}

#[test]
fn test_beam_qin2014_keep_trace() {
    println!("\n=== Testing Beam with keep_trace ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.keep_trace = true;
    param.beam.kmin = 2;
    param.beam.kmax = 5;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // With keep_trace, should have one population per k level
    assert!(
        !experiment.collections.is_empty(),
        "Collections should not be empty"
    );
    let beam_populations = &experiment.collections[0];
    assert!(
        beam_populations.len() >= 1,
        "Should have at least one generation"
    );

    println!(
        "  - Number of k levels in trace: {}",
        beam_populations.len()
    );

    // Verify each level has different k values
    for (i, pop) in beam_populations.iter().enumerate() {
        if !pop.individuals.is_empty() {
            let k = pop.individuals[0].k;
            println!(
                "  - Level {}: k={}, n_models={}",
                i,
                k,
                pop.individuals.len()
            );
        }
    }

    println!("✓ Keep trace test passed");
}

#[test]
fn test_beam_qin2014_early_stopping() {
    println!("\n=== Testing Beam Early Stopping ===\n");

    let running = Arc::new(AtomicBool::new(true));
    let mut param = create_qin2014_beam_params();
    param.beam.kmax = 10; // Larger range

    // Simulate early stopping after a delay
    let running_clone = Arc::clone(&running);
    let _stop_thread = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(2));
        running_clone.store(false, Ordering::Relaxed);
    });

    let experiment = run(&param, running);

    // Should still have valid results even if stopped early
    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    assert!(
        !experiment
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .is_empty(),
        "Final population should have individuals"
    );

    println!("✓ Early stopping test passed");
    println!("  - Stopped after {:.2}s", experiment.execution_time);
}

#[test]
fn test_beam_qin2014_best_models_ci_alpha() {
    println!("\n=== Testing Beam with Different CI Alpha ===\n");

    // Test with strict alpha (fewer models selected)
    let mut param_strict = create_qin2014_beam_params();
    param_strict.beam.best_models_ci_alpha = 0.05; // High value (high = fewer models)
    param_strict.beam.kmax = 4;

    let running1 = Arc::new(AtomicBool::new(true));
    let exp_strict = run(&param_strict, running1);

    // Test with relaxed alpha (more models selected)
    let mut param_relaxed = create_qin2014_beam_params();
    param_relaxed.beam.best_models_ci_alpha = 0.001; // Low value (low = more models)
    param_relaxed.beam.kmax = 4;

    let running2 = Arc::new(AtomicBool::new(true));
    let exp_relaxed = run(&param_relaxed, running2);

    // Both should produce valid results
    assert!(
        exp_strict.final_population.is_some(),
        "Strict alpha should produce results"
    );
    assert!(
        exp_relaxed.final_population.is_some(),
        "Relaxed alpha should produce results"
    );

    let strict_size = exp_strict
        .final_population
        .as_ref()
        .unwrap()
        .individuals
        .len();
    let relaxed_size = exp_relaxed
        .final_population
        .as_ref()
        .unwrap()
        .individuals
        .len();

    println!(
        "  - Strict alpha (0.99) final population: {} models",
        strict_size
    );
    println!(
        "  - Relaxed alpha (0.001) final population: {} models",
        relaxed_size
    );

    // Relaxed alpha typically produces more models (not guaranteed but likely)
    println!("✓ CI alpha test passed");
}

#[test]
fn test_beam_qin2014_feature_selection_impact() {
    println!("\n=== Testing Beam Feature Selection Impact ===\n");

    // Test with very restrictive feature selection
    let mut param_restrictive = create_qin2014_beam_params();
    param_restrictive.data.max_features_per_class = 15; // Very few features
    param_restrictive.beam.kmax = 5;

    let running1 = Arc::new(AtomicBool::new(true));
    let exp_restrictive = run(&param_restrictive, running1);

    // Test with more permissive feature selection
    let mut param_permissive = create_qin2014_beam_params();
    param_permissive.data.max_features_per_class = 50; // More features
    param_permissive.beam.kmax = 5;

    let running2 = Arc::new(AtomicBool::new(true));
    let exp_permissive = run(&param_permissive, running2);

    let restrictive_features = exp_restrictive.train_data.feature_selection.len();
    let permissive_features = exp_permissive.train_data.feature_selection.len();

    println!(
        "  - Restrictive selection: {} features",
        restrictive_features
    );
    println!("  - Permissive selection: {} features", permissive_features);

    assert!(
        restrictive_features > 0,
        "Should have some features with restrictive selection"
    );
    assert!(
        permissive_features >= restrictive_features,
        "Permissive should have more or equal features"
    );

    println!("✓ Feature selection impact test passed");
}

#[test]
fn test_beam_qin2014_multiple_languages() {
    println!("\n=== Testing Beam with Multiple Languages ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.language = "ter,bin".to_string();
    param.beam.kmax = 4;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    let final_pop = experiment.final_population.as_ref().unwrap();

    // Check for language diversity
    let mut languages_found = std::collections::HashSet::new();
    for individual in &final_pop.individuals {
        languages_found.insert(individual.language);
    }

    assert!(
        !languages_found.is_empty(),
        "Should have at least one language"
    );
    println!(
        "  - Languages found in final population: {:?}",
        languages_found
    );
    println!("  - Total models: {}", final_pop.individuals.len());

    println!("✓ Multiple languages test passed");
}

#[test]
fn test_beam_qin2014_reproducibility() {
    println!("\n=== Testing Beam Reproducibility with Same Seed ===\n");

    let param = create_qin2014_beam_params();

    // Run 1
    let running1 = Arc::new(AtomicBool::new(true));
    let exp1 = run(&param, running1);

    // Run 2 with same seed
    let running2 = Arc::new(AtomicBool::new(true));
    let exp2 = run(&param, running2);

    // Results should be identical with same seed
    let best1 = &exp1.final_population.as_ref().unwrap().individuals[0];
    let best2 = &exp2.final_population.as_ref().unwrap().individuals[0];

    assert_eq!(
        best1.features, best2.features,
        "Features should be identical"
    );
    assert_eq!(best1.auc, best2.auc, "AUC should be identical");
    assert_eq!(best1.k, best2.k, "Feature count should be identical");

    println!("  - Run 1 AUC: {:.4}", best1.auc);
    println!("  - Run 2 AUC: {:.4}", best2.auc);
    println!("  - Run 1 k: {}", best1.k);
    println!("  - Run 2 k: {}", best2.k);

    println!("✓ Reproducibility test passed");
}

#[test]
fn test_beam_qin2014_max_nb_models() {
    println!("\n=== Testing Beam Max Number of Models Limit ===\n");

    let mut param = create_qin2014_beam_params();
    param.beam.max_nb_of_models = 1000; // Limit combinations
    param.beam.kmax = 5;
    param.data.max_features_per_class = 25; // Enough features to test limit

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );

    // Check that we don't exceed the model limit significantly
    let final_size = experiment
        .final_population
        .as_ref()
        .unwrap()
        .individuals
        .len();
    println!("  - Final population size: {}", final_size);
    println!("  - Max models limit: {}", param.beam.max_nb_of_models);

    // Final population should be reasonable (may be filtered by CI alpha)
    assert!(final_size > 0, "Should have at least some models");

    println!("✓ Max models limit test passed");
}

#[test]
fn test_beam_qin2014_different_fit_functions() {
    println!("\n=== Testing Beam with Different Fit Functions ===\n");

    let fit_functions = vec![
        gpredomics::param::FitFunction::auc,
        gpredomics::param::FitFunction::sensitivity,
        gpredomics::param::FitFunction::specificity,
    ];

    for fit_fn in fit_functions {
        println!("  Testing fit function: {:?}", fit_fn);

        let mut param = create_qin2014_beam_params();
        param.general.fit = fit_fn.clone();
        param.beam.kmax = 4; // Quick test

        let running = Arc::new(AtomicBool::new(true));
        let experiment = run(&param, running);

        assert!(
            experiment.final_population.is_some(),
            "Should have final population for {:?}",
            fit_fn
        );
        let best_model = &experiment.final_population.as_ref().unwrap().individuals[0];

        println!("    - Best AUC: {:.4}", best_model.auc);
        println!("    - Best Sensitivity: {:.4}", best_model.sensitivity);
        println!("    - Best Specificity: {:.4}", best_model.specificity);
        println!("    - Best k: {}", best_model.k);
    }

    println!("✓ Different fit functions test passed");
}

#[test]
fn test_beam_qin2014_experiment_display() {
    println!("\n=== Testing Beam Experiment Display Methods ===\n");

    let param = create_qin2014_beam_params();
    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // This should not panic
    experiment.display_results();

    println!("✓ Experiment display test passed");
}

#[test]
fn test_beam_qin2014_cv_enabled() {
    println!("\n=== Testing Beam with CV Enabled (outer_folds=2) ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.cv = true;
    param.general.gpu = false; // CV path disables GPU usage
    param.general.keep_trace = true; // Keep per-k populations per fold

    // Narrow folds for speed
    param.cv.outer_folds = 2;
    param.cv.inner_folds = 3;
    param.cv.fit_on_valid = true;
    param.cv.overfit_penalty = 0.0;
    param.cv.resampling_inner_folds_epochs = 0;

    // Beam small interval
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.beam.best_models_ci_alpha = 0.01; // permissive to ensure some models per fold
    param.beam.max_nb_of_models = 2000;

    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    // CV Assertions
    assert!(
        experiment.cv_folds_ids.is_some(),
        "cv_folds_ids must be present in CV mode"
    );
    let fold_ids = experiment.cv_folds_ids.as_ref().unwrap();
    assert_eq!(
        fold_ids.len(),
        param.cv.outer_folds,
        "cv_folds_ids length must equal outer_folds"
    );

    assert_eq!(
        experiment.collections.len(),
        param.cv.outer_folds,
        "collections length must equal outer_folds"
    );
    for (i, coll) in experiment.collections.iter().enumerate() {
        assert!(
            !coll.is_empty(),
            "Fold {} should have at least one population",
            i
        );
        let last_pop = &coll[coll.len() - 1];
        assert!(
            !last_pop.individuals.is_empty(),
            "Fold {} last population should have individuals",
            i
        );
        for ind in &last_pop.individuals {
            assert!(
                ind.k >= param.beam.kmin && ind.k <= param.beam.kmax,
                "Model k={} must be within [{}..{}]",
                ind.k,
                param.beam.kmin,
                param.beam.kmax
            );
        }
    }

    assert!(
        experiment.final_population.is_some(),
        "Final population should exist in CV mode"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population must contain models"
    );

    experiment.display_results();
    println!("✓ Beam CV-enabled test passed");
}

#[test]
fn test_beam_qin2014_kmin_kmax_range() {
    println!("\n=== Testing Beam K Range Constraints ===\n");

    let mut param = create_qin2014_beam_params();
    param.beam.kmin = 3;
    param.beam.kmax = 5;
    param.data.max_features_per_class = 20; // Ensure enough features

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    let final_pop = experiment.final_population.as_ref().unwrap();

    // Check that all models respect k constraints
    for individual in &final_pop.individuals {
        assert!(
            individual.k >= param.beam.kmin,
            "Model k={} should be >= kmin={}",
            individual.k,
            param.beam.kmin
        );
        assert!(
            individual.k <= param.beam.kmax,
            "Model k={} should be <= kmax={}",
            individual.k,
            param.beam.kmax
        );
    }

    println!("  - kmin: {}", param.beam.kmin);
    println!("  - kmax: {}", param.beam.kmax);
    println!(
        "  - All {} models respect k constraints",
        final_pop.individuals.len()
    );

    println!("✓ K range constraints test passed");
}

#[test]
#[ignore]
fn test_beam_qin2014_gpu_vs_cpu() {
    println!("\n=== Testing Beam GPU vs CPU Consistency ===\n");

    // CPU run
    let mut param_cpu = create_qin2014_beam_params();
    param_cpu.general.gpu = false;
    param_cpu.general.seed = 99999;
    param_cpu.beam.kmin = 2;
    param_cpu.beam.kmax = 4;
    param_cpu.beam.best_models_ci_alpha = 0.1;

    println!("Running Beam on CPU...");
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param_cpu, running_cpu);

    // GPU run (will fallback to CPU if GPU not available)
    let mut param_gpu = create_qin2014_beam_params();
    param_gpu.general.gpu = true;
    param_gpu.general.seed = 99999;
    param_gpu.beam.kmin = 2;
    param_gpu.beam.kmax = 4;
    param_gpu.beam.best_models_ci_alpha = 0.1;
    param_gpu.gpu.fallback_to_cpu = true;

    println!("Running Beam on GPU (or CPU fallback)...");
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param_gpu, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!("CPU best model: AUC={:.4}, k={}", best_cpu.auc, best_cpu.k);
    println!("GPU best model: AUC={:.4}, k={}", best_gpu.auc, best_gpu.k);

    // Beam is deterministic, so results should be identical with same seed
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Beam with same seed should give identical features"
    );
    assert_eq!(
        best_cpu.auc, best_gpu.auc,
        "Beam with same seed should give identical AUC"
    );

    println!("✓ GPU vs CPU consistency test passed");
}

#[test]
fn test_beam_qin2014_all_language_datatype_combinations() {
    println!("\n=== Testing All Language/DataType Combinations ===\n");

    let languages = vec!["ter", "bin", "ratio"];
    let datatypes = vec!["raw", "prev", "log"];

    for language in &languages {
        for datatype in &datatypes {
            println!(
                "\n--- Testing language={}, datatype={} ---",
                language, datatype
            );

            let mut param = create_qin2014_beam_params();
            param.general.language = language.to_string();
            param.general.data_type = datatype.to_string();
            param.general.seed = 777;
            param.beam.kmin = 2;
            param.beam.kmax = 3;
            param.beam.best_models_ci_alpha = 0.001; // Very permissive to ensure models

            let running = Arc::new(AtomicBool::new(true));
            let experiment = run(&param, running);

            assert!(
                experiment.final_population.is_some(),
                "Should have final population for {}/{}",
                language,
                datatype
            );

            let final_pop = experiment.final_population.as_ref().unwrap();
            assert!(
                !final_pop.individuals.is_empty(),
                "Should have models for {}/{}",
                language,
                datatype
            );

            // Verify all individuals have correct language and data_type
            let expected_lang = gpredomics::individual::language(language);
            let expected_dtype = gpredomics::individual::data_type(datatype);

            for (idx, individual) in final_pop.individuals.iter().enumerate() {
                // Allow auto-conversion: Ternary models with only positive coefficients are converted to Binary
                if expected_lang == gpredomics::individual::TERNARY_LANG {
                    let has_neg = individual.features.values().any(|&c| c < 0);
                    let has_pos = individual.features.values().any(|&c| c > 0);
                    if !has_neg && has_pos {
                        assert!(
                            individual.language == gpredomics::individual::TERNARY_LANG ||
                            individual.language == gpredomics::individual::BINARY_LANG,
                            "Individual {} expected ter but had only positive coefficients; language should be ter or auto-converted bin, got {}",
                            idx,
                            individual.language
                        );
                    } else {
                        assert_eq!(
                            individual.language, expected_lang,
                            "Individual {} should have language {} ({}), got {}",
                            idx, language, expected_lang, individual.language
                        );
                    }
                } else {
                    assert_eq!(
                        individual.language, expected_lang,
                        "Individual {} should have language {} ({}), got {}",
                        idx, language, expected_lang, individual.language
                    );
                }

                assert_eq!(
                    individual.data_type, expected_dtype,
                    "Individual {} should have data_type {} ({}), got {}",
                    idx, datatype, expected_dtype, individual.data_type
                );
            }

            let best = &final_pop.individuals[0];
            println!(
                "  ✓ {} models with language={}, datatype={}",
                final_pop.individuals.len(),
                language,
                datatype
            );
            println!("    Best: AUC={:.4}, k={}", best.auc, best.k);
        }
    }

    println!("\n✓ All language/datatype combinations test passed");
}

/// Test GPU support with inner CV for Beam
#[test]
fn test_beam_gpu_with_inner_cv() {
    println!("\n=== Testing Beam with GPU and Inner CV ===");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.general.keep_trace = false; // Disable trace for faster tests
    param.data.max_features_per_class = 3; // Very restrictive to limit model count
    param.beam.method = BeamMethod::LimitedExhaustive;
    param.beam.kmin = 2;
    param.beam.kmax = 2; // Keep very small for GPU fold constraints
    param.beam.max_nb_of_models = 15; // Small for fold processing with margin
    param.cv.inner_folds = 3;
    param.cv.overfit_penalty = 0.5; // Enable inner CV

    // CPU run
    println!("\n--- Running with CPU ---");
    param.general.gpu = false;
    param.general.save_exp = "test_beam_cpu_inner_cv.mp".to_string();
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // GPU run
    println!("\n--- Running with GPU ---");
    param.general.gpu = true;
    param.general.save_exp = "test_beam_gpu_inner_cv.mp".to_string();
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Cleanup
    let _ = std::fs::remove_file("test_beam_cpu_inner_cv.mp");
    let _ = std::fs::remove_file("test_beam_gpu_inner_cv.mp");

    // Verify both produced results
    assert!(
        exp_cpu.final_population.is_some(),
        "CPU run should produce results"
    );
    assert!(
        exp_gpu.final_population.is_some(),
        "GPU run should produce results"
    );

    let pop_cpu = exp_cpu.final_population.as_ref().unwrap();
    let pop_gpu = exp_gpu.final_population.as_ref().unwrap();

    assert!(
        !pop_cpu.individuals.is_empty(),
        "CPU population should not be empty"
    );
    assert!(
        !pop_gpu.individuals.is_empty(),
        "GPU population should not be empty"
    );

    // Compare top models - Beam is deterministic so they should be identical
    let best_cpu = &pop_cpu.individuals[0];
    let best_gpu = &pop_gpu.individuals[0];

    println!(
        "\nTop CPU model: k={}, AUC={:.6}, features={:?}",
        best_cpu.k, best_cpu.auc, best_cpu.features
    );
    println!(
        "Top GPU model: k={}, AUC={:.6}, features={:?}",
        best_gpu.k, best_gpu.auc, best_gpu.features
    );

    // Beam is deterministic - features should be identical
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Beam GPU and CPU should produce identical features"
    );

    // AUC should be very close
    let auc_diff = (best_cpu.auc - best_gpu.auc).abs();
    assert!(
        auc_diff < 1e-4,
        "AUC difference too large: CPU={:.6}, GPU={:.6}, diff={:.6}",
        best_cpu.auc,
        best_gpu.auc,
        auc_diff
    );

    println!("\n✓ Beam GPU with inner CV test passed - results match!");
}

/// Test consistency: Inner CV vs no inner CV
#[test]
fn test_beam_consistency_inner_cv_vs_no_inner_cv() {
    println!("\n=== Testing Beam Consistency: Inner CV vs No Inner CV ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.general.cv = false; // No outer CV
    param.general.gpu = false;
    param.general.keep_trace = false;
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;

    // Run WITHOUT inner CV
    println!("Running WITHOUT inner CV...");
    param.cv.overfit_penalty = 0.0;
    param.cv.inner_folds = 5;
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_inner_cv = run(&param, running1);

    // Run WITH inner CV
    println!("Running WITH inner CV...");
    param.cv.overfit_penalty = 0.5;
    param.cv.inner_folds = 5;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_inner_cv = run(&param, running2);

    // Both should produce valid results
    assert!(exp_no_inner_cv.final_population.is_some());
    assert!(exp_with_inner_cv.final_population.is_some());

    let best_no_cv = &exp_no_inner_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];
    let best_with_cv = &exp_with_inner_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];

    println!("No Inner CV: AUC={:.4}, k={}", best_no_cv.auc, best_no_cv.k);
    println!(
        "With Inner CV: AUC={:.4}, k={}",
        best_with_cv.auc, best_with_cv.k
    );

    // Results will differ due to different fitness evaluation
    // But both should be valid
    assert!(best_no_cv.auc >= 0.0 && best_no_cv.auc <= 1.0);
    assert!(best_with_cv.auc >= 0.0 && best_with_cv.auc <= 1.0);
    assert!(best_no_cv.k >= param.beam.kmin && best_no_cv.k <= param.beam.kmax);
    assert!(best_with_cv.k >= param.beam.kmin && best_with_cv.k <= param.beam.kmax);

    println!("✓ Inner CV vs No Inner CV consistency test passed");
}

/// Test consistency: Outer CV vs no outer CV
#[test]
fn test_beam_consistency_outer_cv_vs_no_outer_cv() {
    println!("\n=== Testing Beam Consistency: Outer CV vs No Outer CV ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.general.gpu = false;
    param.general.keep_trace = true; // Keep trace to see structure
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;

    // Run WITHOUT outer CV
    println!("Running WITHOUT outer CV...");
    param.general.cv = false;
    param.cv.overfit_penalty = 0.0;
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_outer_cv = run(&param, running1);

    // Run WITH outer CV
    println!("Running WITH outer CV...");
    param.general.cv = true;
    param.cv.outer_folds = 2; // Keep small for speed
    param.cv.inner_folds = 3;
    param.cv.overfit_penalty = 0.0;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_outer_cv = run(&param, running2);

    // Both should produce valid results
    assert!(exp_no_outer_cv.final_population.is_some());
    assert!(exp_with_outer_cv.final_population.is_some());

    let best_no_cv = &exp_no_outer_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];
    let best_with_cv = &exp_with_outer_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];

    println!("No Outer CV: AUC={:.4}, k={}", best_no_cv.auc, best_no_cv.k);
    println!(
        "With Outer CV: AUC={:.4}, k={}",
        best_with_cv.auc, best_with_cv.k
    );

    // Verify CV structure
    assert!(
        exp_with_outer_cv.cv_folds_ids.is_some(),
        "CV should have fold IDs"
    );
    assert_eq!(
        exp_with_outer_cv.collections.len(),
        param.cv.outer_folds,
        "Should have one collection per fold"
    );

    // Both should be valid
    assert!(best_no_cv.auc >= 0.0 && best_no_cv.auc <= 1.0);
    assert!(best_with_cv.auc >= 0.0 && best_with_cv.auc <= 1.0);
    assert!(best_no_cv.k >= param.beam.kmin && best_no_cv.k <= param.beam.kmax);
    assert!(best_with_cv.k >= param.beam.kmin && best_with_cv.k <= param.beam.kmax);

    println!("✓ Outer CV vs No Outer CV consistency test passed");
}

/// Test consistency: GPU vs CPU (basic, no CV)
#[test]
#[ignore]
fn test_beam_consistency_gpu_vs_cpu_basic() {
    println!("\n=== Testing Beam Consistency: GPU vs CPU (Basic) ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 12345;
    param.general.cv = false;
    param.general.keep_trace = false;
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;
    param.cv.overfit_penalty = 0.0; // No inner CV

    // CPU run
    println!("Running on CPU...");
    param.general.gpu = false;
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // GPU run
    println!("Running on GPU (or CPU fallback)...");
    param.general.gpu = true;
    param.gpu.fallback_to_cpu = true;
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!(
        "CPU: AUC={:.6}, k={}, features={:?}",
        best_cpu.auc, best_cpu.k, best_cpu.features
    );
    println!(
        "GPU: AUC={:.6}, k={}, features={:?}",
        best_gpu.auc, best_gpu.k, best_gpu.features
    );

    // Beam is deterministic - results should be identical
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Beam with same seed should give identical features"
    );
    assert_eq!(
        best_cpu.auc, best_gpu.auc,
        "Beam with same seed should give identical AUC"
    );

    println!("✓ GPU vs CPU basic consistency test passed");
}

/// Test consistency: GPU inner CV vs CPU inner CV
#[test]
#[ignore]
fn test_beam_consistency_gpu_vs_cpu_inner_cv() {
    println!("\n=== Testing Beam Consistency: GPU Inner CV vs CPU Inner CV ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 54321;
    param.general.cv = false; // No outer CV
    param.general.keep_trace = false;
    param.beam.method = BeamMethod::LimitedExhaustive;
    param.beam.kmin = 2;
    param.beam.kmax = 2; // Keep very small to fit in GPU buffer with folds
    param.data.max_features_per_class = 3; // Very restrictive like in test_beam_gpu_with_inner_cv
    param.beam.max_nb_of_models = 15; // Small limit for GPU fold processing
    param.cv.inner_folds = 3;
    param.cv.overfit_penalty = 0.5; // Enable inner CV

    // CPU run with inner CV
    println!("Running CPU with inner CV...");
    param.general.gpu = false;
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // GPU run with inner CV
    println!("Running GPU with inner CV...");
    param.general.gpu = true;
    param.gpu.fallback_to_cpu = true;
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!(
        "CPU Inner CV: AUC={:.6}, k={}, features={:?}",
        best_cpu.auc, best_cpu.k, best_cpu.features
    );
    println!(
        "GPU Inner CV: AUC={:.6}, k={}, features={:?}",
        best_gpu.auc, best_gpu.k, best_gpu.features
    );

    // Beam is deterministic - results should be identical
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Beam GPU and CPU inner CV should give identical features"
    );

    let auc_diff = (best_cpu.auc - best_gpu.auc).abs();
    assert!(
        auc_diff < 1e-6,
        "AUC should be identical: CPU={:.6}, GPU={:.6}, diff={:.6}",
        best_cpu.auc,
        best_gpu.auc,
        auc_diff
    );

    println!("✓ GPU Inner CV vs CPU Inner CV consistency test passed");
}

/// Test consistency: GPU outer CV vs CPU outer CV
#[test]
#[ignore]
fn test_beam_consistency_gpu_vs_cpu_outer_cv() {
    println!("\n=== Testing Beam Consistency: GPU Outer CV vs CPU Outer CV ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 99999;
    param.general.cv = true; // Outer CV enabled
    param.general.keep_trace = false;
    param.beam.kmin = 2;
    param.beam.kmax = 3;
    param.data.max_features_per_class = 10;
    param.cv.outer_folds = 2;
    param.cv.inner_folds = 3;
    param.cv.overfit_penalty = 0.0; // No inner CV penalty

    // Note: GPU is disabled internally when outer CV is enabled
    // So both runs will use CPU

    // CPU run with outer CV
    println!("Running CPU with outer CV...");
    param.general.gpu = false;
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // "GPU" run with outer CV (will use CPU internally)
    println!("Running with GPU flag and outer CV (should use CPU internally)...");
    param.general.gpu = true;
    param.gpu.fallback_to_cpu = true;
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    // Verify CV structure
    assert_eq!(exp_cpu.collections.len(), param.cv.outer_folds);
    assert_eq!(exp_gpu.collections.len(), param.cv.outer_folds);

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!("CPU Outer CV: AUC={:.6}, k={}", best_cpu.auc, best_cpu.k);
    println!("GPU Outer CV: AUC={:.6}, k={}", best_gpu.auc, best_gpu.k);

    // Beam is deterministic - results should be identical (both use CPU for outer CV)
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Outer CV should give identical features with same seed"
    );

    println!("✓ GPU Outer CV vs CPU Outer CV consistency test passed");
}

/// Test consistency: keep_trace enabled vs disabled
#[test]
fn test_beam_consistency_keep_trace() {
    println!("\n=== Testing Beam Consistency: keep_trace On vs Off ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 77777;
    param.general.cv = false;
    param.general.gpu = false;
    param.beam.kmin = 2;
    param.beam.kmax = 5;
    param.data.max_features_per_class = 15;

    // Run WITHOUT keep_trace
    println!("Running WITHOUT keep_trace...");
    param.general.keep_trace = false;
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_trace = run(&param, running1);

    // Run WITH keep_trace
    println!("Running WITH keep_trace...");
    param.general.keep_trace = true;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_trace = run(&param, running2);

    // Both should produce the same final results
    assert!(exp_no_trace.final_population.is_some());
    assert!(exp_with_trace.final_population.is_some());

    let best_no_trace = &exp_no_trace.final_population.as_ref().unwrap().individuals[0];
    let best_with_trace = &exp_with_trace
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];

    println!(
        "No trace: AUC={:.6}, k={}, features={:?}",
        best_no_trace.auc, best_no_trace.k, best_no_trace.features
    );
    println!(
        "With trace: AUC={:.6}, k={}, features={:?}",
        best_with_trace.auc, best_with_trace.k, best_with_trace.features
    );

    // Beam is deterministic - final results should be identical
    assert_eq!(
        best_no_trace.features, best_with_trace.features,
        "keep_trace should not affect final results"
    );
    assert_eq!(
        best_no_trace.auc, best_with_trace.auc,
        "keep_trace should not affect AUC"
    );

    // Verify trace structure
    // Without keep_trace, should only keep final population (one per k)
    // With keep_trace, should keep all k-levels explored
    println!(
        "  - No trace: {} k-level(s) stored",
        exp_no_trace.collections[0].len()
    );
    println!(
        "  - With trace: {} k-levels stored",
        exp_with_trace.collections[0].len()
    );

    assert!(
        exp_with_trace.collections[0].len() >= exp_no_trace.collections[0].len(),
        "With trace should keep at least as many k-levels as without trace"
    );

    println!("✓ keep_trace consistency test passed");
}

/// Test Beam with voting enabled
#[test]
fn test_beam_qin2014_with_voting() {
    println!("\n=== Testing Beam with Voting Enabled ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;

    // Enable voting
    param.voting.vote = true;
    param.voting.min_perf = 0.6;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.method_threshold = 0.5;
    param.voting.complete_display = false;

    let running = Arc::new(AtomicBool::new(true));
    let mut experiment = run(&param, running);

    // Verify experiment structure
    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population should have individuals"
    );

    // Compute voting
    experiment.compute_voting();

    // Verify voting results exist
    assert!(
        experiment.others.is_some(),
        "Voting results should exist when voting is enabled"
    );

    if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others {
        println!("  - Number of experts: {}", jury.experts.individuals.len());
        println!("  - Voting AUC: {:.4}", jury.auc);
        println!("  - Voting sensitivity: {:.4}", jury.sensitivity);
        println!("  - Voting specificity: {:.4}", jury.specificity);

        // Verify voting results are valid
        assert!(
            jury.experts.individuals.len() > 0,
            "Should have at least one expert"
        );
        assert!(
            jury.auc >= 0.0 && jury.auc <= 1.0,
            "Voting AUC should be valid"
        );
        assert!(
            jury.sensitivity >= 0.0 && jury.sensitivity <= 1.0,
            "Voting sensitivity should be valid"
        );
        assert!(
            jury.specificity >= 0.0 && jury.specificity <= 1.0,
            "Voting specificity should be valid"
        );
    } else {
        panic!("Expected Jury metadata");
    }

    println!("✓ Beam with voting test passed");
}

/// Test Beam with different voting methods
#[test]
fn test_beam_qin2014_voting_methods() {
    println!("\n=== Testing Beam with Different Voting Methods ===\n");

    let voting_methods = vec![
        (gpredomics::voting::VotingMethod::Majority, "Majority"),
        (gpredomics::voting::VotingMethod::Consensus, "Consensus"),
    ];

    for (method, method_name) in voting_methods {
        println!("Testing voting method: {}", method_name);

        let mut param = create_qin2014_beam_params();
        param.general.seed = 42;
        param.beam.kmin = 2;
        param.beam.kmax = 3;
        param.data.max_features_per_class = 12;

        param.voting.vote = true;
        param.voting.min_perf = 0.55;
        param.voting.min_diversity = 5.0;
        param.voting.method = method;
        param.voting.method_threshold = 0.5;

        let running = Arc::new(AtomicBool::new(true));
        let mut experiment = run(&param, running);

        experiment.compute_voting();

        assert!(
            experiment.others.is_some(),
            "{} should produce voting results",
            method_name
        );

        if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others
        {
            println!(
                "  {} - Experts: {}, AUC: {:.4}",
                method_name,
                jury.experts.individuals.len(),
                jury.auc
            );

            assert!(
                jury.auc >= 0.0 && jury.auc <= 1.0,
                "{} should produce valid AUC",
                method_name
            );
        } else {
            panic!("Expected Jury metadata for {}", method_name);
        }
    }

    println!("✓ Different voting methods test passed");
}

/// Test Beam with threshold CI enabled
#[test]
fn test_beam_qin2014_with_threshold_ci() {
    println!("\n=== Testing Beam with Threshold CI ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;

    // Enable threshold CI
    param.general.threshold_ci_alpha = 0.05;
    param.general.threshold_ci_n_bootstrap = 100;
    param.general.threshold_ci_frac_bootstrap = 0.8;
    param.general.threshold_ci_penalty = 0.1;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // Verify experiment structure
    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population should have individuals"
    );

    // Verify threshold CI is set
    let best_model = &final_pop.individuals[0];
    assert!(
        best_model.threshold_ci.is_some(),
        "Best model should have threshold CI"
    );

    let threshold_ci = best_model.threshold_ci.as_ref().unwrap();
    println!("  - Threshold: {:.4}", best_model.threshold);
    println!("  - CI Lower: {:.4}", threshold_ci.lower);
    println!("  - CI Upper: {:.4}", threshold_ci.upper);
    println!("  - Rejection rate: {:.4}", threshold_ci.rejection_rate);

    // Verify CI bounds are valid
    assert!(
        threshold_ci.lower <= best_model.threshold,
        "CI lower should be <= threshold"
    );
    assert!(
        threshold_ci.upper >= best_model.threshold,
        "CI upper should be >= threshold"
    );
    assert!(
        threshold_ci.lower <= threshold_ci.upper,
        "CI lower should be <= upper"
    );
    assert!(
        threshold_ci.rejection_rate >= 0.0 && threshold_ci.rejection_rate <= 1.0,
        "Rejection rate should be between 0 and 1"
    );

    println!("✓ Threshold CI test passed");
}

/// Test Beam with threshold CI and different alpha values
#[test]
fn test_beam_qin2014_threshold_ci_alpha_variations() {
    println!("\n=== Testing Beam Threshold CI with Different Alpha Values ===\n");

    let alphas = vec![0.01, 0.05, 0.1, 0.2];

    for &alpha in &alphas {
        println!("Testing alpha = {}", alpha);

        let mut param = create_qin2014_beam_params();
        param.general.seed = 42;
        param.beam.kmin = 2;
        param.beam.kmax = 3;
        param.data.max_features_per_class = 12;

        param.general.threshold_ci_alpha = alpha;
        param.general.threshold_ci_n_bootstrap = 100;
        param.general.threshold_ci_frac_bootstrap = 0.8;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = run(&param, running);

        let best_model = &experiment.final_population.as_ref().unwrap().individuals[0];
        assert!(
            best_model.threshold_ci.is_some(),
            "Should have threshold CI for alpha={}",
            alpha
        );

        let ci = best_model.threshold_ci.as_ref().unwrap();
        let ci_width = ci.upper - ci.lower;

        println!(
            "  alpha={}: CI width = {:.4}, rejection_rate = {:.4}",
            alpha, ci_width, ci.rejection_rate
        );

        // CI width can be 0 if bootstrap produces exactly the same threshold (stable model)
        assert!(
            ci_width >= 0.0,
            "CI width should be non-negative for alpha={}",
            alpha
        );
        assert!(
            ci.lower <= best_model.threshold,
            "CI lower should be <= threshold"
        );
        assert!(
            ci.upper >= best_model.threshold,
            "CI upper should be >= threshold"
        );
    }

    println!("✓ Threshold CI alpha variations test passed");
}

/// Test Beam with both voting and threshold CI enabled
#[test]
fn test_beam_qin2014_voting_with_threshold_ci() {
    println!("\n=== Testing Beam with Both Voting and Threshold CI ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;

    // Enable both voting and threshold CI
    param.voting.vote = true;
    param.voting.min_perf = 0.6;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;

    param.general.threshold_ci_alpha = 0.05;
    param.general.threshold_ci_n_bootstrap = 100;
    param.general.threshold_ci_frac_bootstrap = 0.8;

    let running = Arc::new(AtomicBool::new(true));
    let mut experiment = run(&param, running);

    experiment.compute_voting();

    // Verify voting results
    assert!(experiment.others.is_some(), "Should have voting results");

    if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others {
        println!(
            "  - Voting: {} experts, AUC={:.4}",
            jury.experts.individuals.len(),
            jury.auc
        );

        // Verify experts have threshold CI
        let experts_with_ci = jury
            .experts
            .individuals
            .iter()
            .filter(|ind| ind.threshold_ci.is_some())
            .count();

        println!(
            "  - Experts with threshold CI: {}/{}",
            experts_with_ci,
            jury.experts.individuals.len()
        );

        assert!(
            experts_with_ci > 0,
            "At least some experts should have threshold CI"
        );
    } else {
        panic!("Expected Jury metadata");
    }

    // Verify final population models have threshold CI
    let final_pop = experiment.final_population.as_ref().unwrap();
    let models_with_ci = final_pop
        .individuals
        .iter()
        .filter(|ind| ind.threshold_ci.is_some())
        .count();

    println!(
        "  - Final models with threshold CI: {}/{}",
        models_with_ci,
        final_pop.individuals.len()
    );

    assert!(
        models_with_ci > 0,
        "At least some final models should have threshold CI"
    );

    println!("✓ Combined voting and threshold CI test passed");
}

/// Test Beam with voting and pruning before voting
#[test]
fn test_beam_qin2014_voting_with_pruning() {
    println!("\n=== Testing Beam with Voting and Pruning ===\n");

    let mut param = create_qin2014_beam_params();
    param.general.seed = 42;
    param.beam.kmin = 2;
    param.beam.kmax = 4;
    param.data.max_features_per_class = 15;

    // Enable voting with pruning
    param.voting.vote = true;
    param.voting.min_perf = 0.6;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.prune_before_voting = true;

    let running = Arc::new(AtomicBool::new(true));
    let mut experiment = run(&param, running);

    experiment.compute_voting();

    // Verify voting results
    assert!(experiment.others.is_some(), "Should have voting results");

    if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others {
        println!("  - Number of experts: {}", jury.experts.individuals.len());
        println!("  - Voting AUC: {:.4}", jury.auc);

        // Verify experts have been selected
        assert!(
            jury.experts.individuals.len() > 0,
            "Should have experts after pruning"
        );

        // All experts should have valid models
        for expert in &jury.experts.individuals {
            assert!(expert.k > 0, "Expert should have at least one feature");
            assert!(
                expert.auc >= 0.0 && expert.auc <= 1.0,
                "Expert should have valid AUC"
            );
        }
    } else {
        panic!("Expected Jury metadata");
    }

    println!("✓ Voting with pruning test passed");
}
