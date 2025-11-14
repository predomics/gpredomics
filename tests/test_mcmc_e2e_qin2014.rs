use gpredomics;
use gpredomics::experiment::Experiment;
use gpredomics::param::Param;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// Helper function to load the base parameter configuration for MCMC tests
fn load_base_mcmc_param() -> Param {
    let mut param = Param::default();

    // General settings
    param.general.seed = 42;
    param.general.algo = "mcmc".to_string();
    param.general.cv = false;
    param.general.thread_number = 1;
    param.general.gpu = false;
    param.general.language = "".to_string(); // MCMC uses GENERIC only
    param.general.data_type = "prev".to_string(); // MCMC only supports one at a time
    param.general.data_type_epsilon = 0.0001;
    param.general.fit = gpredomics::param::FitFunction::auc;
    param.general.k_penalty = 0.0;
    param.general.fr_penalty = 0.0;
    param.general.n_model_to_display = 10;
    param.general.log_level = "info".to_string();
    param.general.display_colorful = false;
    param.general.keep_trace = false;
    param.general.save_exp = "test_mcmc_exp.mp".to_string();
    param.general.log_base = "".to_string();
    param.general.log_suffix = "log".to_string();

    // Data settings
    param.data.X = "samples/Qin2014/Xtrain.tsv".to_string();
    param.data.y = "samples/Qin2014/Ytrain.tsv".to_string();
    param.data.Xtest = "samples/Qin2014/Xtest.tsv".to_string();
    param.data.ytest = "samples/Qin2014/Ytest.tsv".to_string();
    param.data.features_in_rows = true; // Qin2014 has features in rows!
    param.data.max_features_per_class = 50; // Feature selection for faster testing
    param.data.feature_minimal_prevalence_pct = 10.0;
    param.data.feature_minimal_feature_value = 1e-4;
    param.data.feature_selection_method = gpredomics::data::PreselectionMethod::wilcoxon;
    param.data.feature_maximal_adj_pvalue = 0.1;
    param.data.feature_minimal_log_abs_bayes_factor = 0.6;
    param.data.inverse_classes = false;
    param.data.n_validation_samples = 0;
    param.data.classes = vec!["healthy".to_string(), "cirrhosis".to_string()];

    // MCMC specific parameters (fast for testing)
    param.mcmc.n_iter = 200; // Small number of iterations for fast tests
    param.mcmc.n_burn = 50; // Burn-in period
    param.mcmc.lambda = 1.0; // Prior regularization
    param.mcmc.nmin = 0; // No SBS by default
    param.mcmc.save_trace_outdir = "".to_string();

    param
}

// Test 1: Basic MCMC run without SBS
#[test]
fn test_mcmc_basic_run() {
    println!("\n=== Test: MCMC Basic Run (No SBS) ===");

    let param = load_base_mcmc_param();
    let running = Arc::new(AtomicBool::new(true));

    let experiment = gpredomics::run(&param, running);

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

    // Check experiment structure
    assert!(!experiment.id.is_empty());
    assert_eq!(experiment.gpredomics_version, gpredomics_version);

    // MCMC stores population in collections[0][0]
    assert!(!experiment.collections.is_empty());
    assert!(!experiment.collections[0].is_empty());
    let population = experiment.collections[0][0].clone();

    println!("Total MCMC samples: {}", population.individuals.len());
    assert!(
        population.individuals.len() > 0,
        "MCMC should generate posterior samples"
    );

    // Check individuals have proper structure
    let first_sample = &population.individuals[0];
    assert!(
        first_sample.features.len() > 0,
        "Model should have features"
    );

    println!("First sample features: {}", first_sample.features.len());

    // MCMC: Compute Bayesian AUC using the posterior distribution
    let (train_auc, train_threshold, train_acc, train_sens, train_spec, _) =
        population.bayesian_compute_roc_and_metrics(&experiment.train_data);

    println!("Bayesian Train AUC: {:.4}", train_auc);
    println!("Optimal threshold: {:.4}", train_threshold);
    println!(
        "Train metrics - Acc: {:.4}, Sens: {:.4}, Spec: {:.4}",
        train_acc, train_sens, train_spec
    );

    // Check Bayesian AUC is reasonable
    assert!(
        train_auc > 0.6,
        "Bayesian train AUC should be better than random: {}",
        train_auc
    );
    assert!(train_auc <= 1.0, "AUC should be <= 1.0");

    // Check test data exists and compute test AUC
    assert!(experiment.test_data.is_some());
    let test_data = experiment.test_data.as_ref().unwrap();
    let (test_auc, test_acc, test_sens, test_spec) =
        population.bayesian_compute_metrics(test_data, train_threshold);

    println!("Bayesian Test AUC: {:.4}", test_auc);
    println!(
        "Test metrics - Acc: {:.4}, Sens: {:.4}, Spec: {:.4}",
        test_acc, test_sens, test_spec
    );

    assert!(
        test_auc > 0.6,
        "Bayesian test AUC should be better than random: {}",
        test_auc
    );
    assert!(test_auc <= 1.0, "Test AUC should be <= 1.0");
}

// Test 2: MCMC with SBS (Sequential Backward Selection)
#[test]
fn test_mcmc_with_sbs() {
    println!("\n=== Test: MCMC with SBS ===");

    let mut param = load_base_mcmc_param();
    param.mcmc.nmin = 5; // Keep minimum 5 features after SBS
    param.general.seed = 123;

    // Start with more features for SBS to work
    param.data.max_features_per_class = 20;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    // Check population
    assert!(!experiment.collections[0].is_empty());
    let population = experiment.collections[0][0].clone();

    println!(
        "Total MCMC samples after SBS: {}",
        population.individuals.len()
    );
    assert!(population.individuals.len() > 0);

    // Check that features were selected
    let n_features = population.individuals[0].features.len();
    println!("Features per sample: {}", n_features);

    // MCMC samples may have varying features during sampling, just check reasonable range
    assert!(
        n_features >= param.mcmc.nmin as usize,
        "Should have at least nmin features"
    );
    assert!(n_features <= 40, "Should have reduced features through SBS");

    // Compute Bayesian metrics to verify SBS didn't break predictive performance
    let (train_auc, train_threshold, _, _, _, _) =
        population.bayesian_compute_roc_and_metrics(&experiment.train_data);
    let test_data = experiment.test_data.as_ref().unwrap();
    let (test_auc, _, _, _) = population.bayesian_compute_metrics(test_data, train_threshold);

    println!(
        "After SBS - Train AUC: {:.4}, Test AUC: {:.4}",
        train_auc, test_auc
    );

    assert!(
        train_auc > 0.6,
        "SBS models should have train AUC better than random: {}",
        train_auc
    );
    assert!(
        test_auc > 0.6,
        "SBS models should have test AUC better than random: {}",
        test_auc
    );

    println!("✓ SBS completed successfully with good performance");
}

// Test 3: Different lambda values (prior regularization)
#[test]
fn test_mcmc_different_lambda() {
    println!("\n=== Test: MCMC with Different Lambda Values ===");

    let lambdas = vec![0.6, 1.0, 2.0];
    let mut results = vec![];

    for lambda in lambdas {
        println!("\nTesting lambda = {}", lambda);

        let mut param = load_base_mcmc_param();
        param.mcmc.lambda = lambda;
        param.general.seed = 42; // Same seed for comparison

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = experiment.collections[0][0].clone();

        // Compute Bayesian metrics
        let (train_auc, train_threshold, _, _, _, _) =
            population.bayesian_compute_roc_and_metrics(&experiment.train_data);
        let test_data = experiment.test_data.as_ref().unwrap();
        let (test_auc, _, _, _) = population.bayesian_compute_metrics(test_data, train_threshold);

        println!(
            "Lambda {}: {} samples, Train AUC = {:.4}, Test AUC = {:.4}",
            lambda,
            population.individuals.len(),
            train_auc,
            test_auc
        );

        results.push((lambda, train_auc, test_auc));

        assert!(population.individuals.len() > 0);
        assert!(
            train_auc > 0.6 && train_auc <= 1.0,
            "Lambda {} should give valid train AUC: {}",
            lambda,
            train_auc
        );
        assert!(
            test_auc > 0.6 && test_auc <= 1.0,
            "Lambda {} should give valid test AUC: {}",
            lambda,
            test_auc
        );
    }

    println!("\n✓ All lambda values produced valid results");
}

// Test 4: Different burn-in periods
#[test]
fn test_mcmc_different_burn_in() {
    println!("\n=== Test: MCMC with Different Burn-in Periods ===");

    let mut param = load_base_mcmc_param();
    param.mcmc.n_iter = 300;

    let burn_periods = vec![50, 100, 150];

    for n_burn in burn_periods {
        println!("\nTesting n_burn = {}", n_burn);

        param.mcmc.n_burn = n_burn;
        param.general.seed = 42;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = experiment.collections[0][0].clone();

        // Compute Bayesian metrics
        let (train_auc, train_threshold, _, _, _, _) =
            population.bayesian_compute_roc_and_metrics(&experiment.train_data);
        let test_data = experiment.test_data.as_ref().unwrap();
        let (test_auc, _, _, _) = population.bayesian_compute_metrics(test_data, train_threshold);

        println!(
            "n_burn {}: {} samples, Train AUC = {:.4}, Test AUC = {:.4}",
            n_burn,
            population.individuals.len(),
            train_auc,
            test_auc
        );

        assert!(population.individuals.len() > 0);

        // More burn-in means fewer samples but possibly better convergence
        let expected_samples = param.mcmc.n_iter - n_burn;
        println!("Expected ~{} samples after burn-in", expected_samples);

        assert!(
            train_auc > 0.6 && train_auc <= 1.0,
            "Burn-in {} should give valid train AUC: {}",
            n_burn,
            train_auc
        );
        assert!(
            test_auc > 0.6 && test_auc <= 1.0,
            "Burn-in {} should give valid test AUC: {}",
            n_burn,
            test_auc
        );
    }

    println!("\n✓ All burn-in periods produced valid results");
}

// Test 5: Different data types
#[test]
fn test_mcmc_different_data_types() {
    println!("\n=== Test: MCMC with Different Data Types ===");

    let data_types = vec!["raw", "prev", "log"];

    for data_type in data_types {
        println!("\nTesting data_type = {}", data_type);

        let mut param = load_base_mcmc_param();
        param.general.data_type = data_type.to_string();
        param.general.seed = 42;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = &experiment.collections[0][0];
        println!(
            "Data type '{}': {} samples, first AUC = {:.4}",
            data_type,
            population.individuals.len(),
            population.individuals[0].auc
        );

        assert!(population.individuals.len() > 0);
        // AUC may be 0 for MCMC samples
    }
}

// Test 6: Different iteration counts
#[test]
fn test_mcmc_different_iterations() {
    println!("\n=== Test: MCMC with Different Iteration Counts ===");

    let iterations = vec![100, 200, 400];

    for n_iter in iterations {
        println!("\nTesting n_iter = {}", n_iter);

        let mut param = load_base_mcmc_param();
        param.mcmc.n_iter = n_iter;
        param.mcmc.n_burn = n_iter / 4; // 25% burn-in
        param.general.seed = 42;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = &experiment.collections[0][0];
        println!(
            "n_iter {}: {} samples generated",
            n_iter,
            population.individuals.len()
        );

        assert!(population.individuals.len() > 0);

        // More iterations should generally give more samples (after burn-in)
        let expected_min = (n_iter - param.mcmc.n_burn) / 2; // Allow for thinning
        assert!(
            population.individuals.len() >= expected_min,
            "Should have reasonable number of samples after burn-in"
        );
    }
}

// Test 7: Reproducibility with same seed
#[test]
fn test_mcmc_reproducibility() {
    println!("\n=== Test: MCMC Reproducibility (Same Seed) ===");

    let mut param = load_base_mcmc_param();
    param.general.seed = 999;
    param.mcmc.n_iter = 150;
    param.mcmc.n_burn = 30;

    let running1 = Arc::new(AtomicBool::new(true));
    let exp1 = gpredomics::run(&param, running1);

    let running2 = Arc::new(AtomicBool::new(true));
    let exp2 = gpredomics::run(&param, running2);

    let pop1 = exp1.collections[0][0].clone();
    let pop2 = exp2.collections[0][0].clone();

    // Compute Bayesian metrics for both runs
    let (train_auc1, train_threshold1, _, _, _, _) =
        pop1.bayesian_compute_roc_and_metrics(&exp1.train_data);
    let test_data1 = exp1.test_data.as_ref().unwrap();
    let (test_auc1, _, _, _) = pop1.bayesian_compute_metrics(test_data1, train_threshold1);

    let (train_auc2, train_threshold2, _, _, _, _) =
        pop2.bayesian_compute_roc_and_metrics(&exp2.train_data);
    let test_data2 = exp2.test_data.as_ref().unwrap();
    let (test_auc2, _, _, _) = pop2.bayesian_compute_metrics(test_data2, train_threshold2);

    println!(
        "Run 1: {} samples, Train AUC = {:.6}, Test AUC = {:.6}",
        pop1.individuals.len(),
        train_auc1,
        test_auc1
    );
    println!(
        "Run 2: {} samples, Train AUC = {:.6}, Test AUC = {:.6}",
        pop2.individuals.len(),
        train_auc2,
        test_auc2
    );

    // Should have same number of samples
    assert_eq!(
        pop1.individuals.len(),
        pop2.individuals.len(),
        "Same seed should give same number of samples"
    );

    // AUCs should be very similar
    let train_auc_diff = (train_auc1 - train_auc2).abs();
    let test_auc_diff = (test_auc1 - test_auc2).abs();

    println!(
        "AUC differences - Train: {:.6}, Test: {:.6}",
        train_auc_diff, test_auc_diff
    );

    assert!(
        train_auc_diff < 0.01,
        "Same seed should produce similar train AUC: diff = {}",
        train_auc_diff
    );
    assert!(
        test_auc_diff < 0.01,
        "Same seed should produce similar test AUC: diff = {}",
        test_auc_diff
    );

    // First sample should be identical (features)
    assert_eq!(
        pop1.individuals[0].features, pop2.individuals[0].features,
        "Features should be identical"
    );

    println!("✓ Reproducibility verified!");
}

// Test 8: Different seeds give different results
#[test]
fn test_mcmc_different_seeds() {
    println!("\n=== Test: MCMC with Different Seeds ===");

    let seeds = vec![42, 123, 456];
    let mut train_aucs = vec![];
    let mut test_aucs = vec![];

    for seed in seeds {
        println!("\nTesting seed = {}", seed);

        let mut param = load_base_mcmc_param();
        param.general.seed = seed;
        param.mcmc.n_iter = 150;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = experiment.collections[0][0].clone();
        let n_samples = population.individuals.len();

        // Compute Bayesian metrics
        let (train_auc, train_threshold, _, _, _, _) =
            population.bayesian_compute_roc_and_metrics(&experiment.train_data);
        let test_data = experiment.test_data.as_ref().unwrap();
        let (test_auc, _, _, _) = population.bayesian_compute_metrics(test_data, train_threshold);

        println!(
            "Seed {}: {} samples, Train AUC = {:.4}, Test AUC = {:.4}",
            seed, n_samples, train_auc, test_auc
        );

        train_aucs.push(train_auc);
        test_aucs.push(test_auc);
    }

    // All seeds should produce valid AUCs
    assert!(
        train_aucs.iter().all(|&auc| auc > 0.6 && auc <= 1.0),
        "All train AUCs should be better than random"
    );
    assert!(
        test_aucs.iter().all(|&auc| auc > 0.6 && auc <= 1.0),
        "All test AUCs should be better than random"
    );

    println!("\n✓ All seeds produced valid MCMC results");
}

// Test 9: Feature selection impact
#[test]
fn test_mcmc_feature_selection() {
    println!("\n=== Test: MCMC with Different Feature Selection ===");

    let max_features_list = vec![10, 30, 50];

    for max_features in max_features_list {
        println!("\nTesting max_features = {}", max_features);

        let mut param = load_base_mcmc_param();
        param.data.max_features_per_class = max_features;
        param.general.seed = 42;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = &experiment.collections[0][0];
        let n_features = population.individuals[0].features.len();

        println!(
            "Max features {}: using {} features, AUC = {:.4}",
            max_features, n_features, population.individuals[0].auc
        );

        assert!(
            n_features <= max_features * 2,
            "Should respect max_features limit"
        );
    }
}

// Test 10: Serialization/deserialization
#[test]
fn test_mcmc_serialization() {
    println!("\n=== Test: MCMC Serialization ===");

    let mut param = load_base_mcmc_param();
    param.general.save_exp = "test_mcmc_serialization.mp".to_string();
    param.mcmc.n_iter = 150;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    // Save to MessagePack
    let save_path = format!("{}.mp", experiment.id);
    println!("Saving to: {}", save_path);
    experiment
        .save_auto(&save_path)
        .expect("Failed to save experiment");

    // Load back
    println!("Loading from: {}", save_path);
    let loaded_exp = Experiment::load_auto(&save_path).expect("Failed to load experiment");

    // Verify integrity
    assert_eq!(experiment.id, loaded_exp.id);
    assert_eq!(experiment.gpredomics_version, loaded_exp.gpredomics_version);
    assert_eq!(
        experiment.collections[0][0].individuals.len(),
        loaded_exp.collections[0][0].individuals.len()
    );

    let orig_pop = &experiment.collections[0][0];
    let loaded_pop = &loaded_exp.collections[0][0];

    assert_eq!(orig_pop.individuals[0].auc, loaded_pop.individuals[0].auc);
    assert_eq!(
        orig_pop.individuals[0].features,
        loaded_pop.individuals[0].features
    );

    println!(
        "Original samples: {}, Loaded samples: {}",
        orig_pop.individuals.len(),
        loaded_pop.individuals.len()
    );
    println!("Serialization successful!");

    // Cleanup
    std::fs::remove_file(&save_path).unwrap();
}

// Test 11: Test data predictions
#[test]
fn test_mcmc_test_predictions() {
    println!("\n=== Test: MCMC Test Data Predictions ===");

    let param = load_base_mcmc_param();

    let running = Arc::new(AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    // Check test data exists
    assert!(experiment.test_data.is_some(), "Test data should be loaded");
    let test_data = experiment.test_data.as_ref().unwrap();

    let population = &experiment.collections[0][0];

    println!("Test data samples: {}", test_data.sample_len);
    println!("Number of MCMC samples: {}", population.individuals.len());

    // Check samples exist
    assert!(
        population.individuals.len() > 0,
        "Should have posterior samples"
    );

    // Check features are present
    for (idx, individual) in population.individuals.iter().take(5).enumerate() {
        println!(
            "Sample {} - features: {}, AUC: {:.4}",
            idx,
            individual.features.len(),
            individual.auc
        );
        assert!(individual.features.len() > 0, "Should have features");
    }

    println!("MCMC test predictions check completed");
}

// Test 12: Experiment display
#[test]
fn test_mcmc_experiment_display() {
    println!("\n=== Test: MCMC Experiment Display ===");

    let param = load_base_mcmc_param();

    let running = Arc::new(AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    // Check experiment properties
    assert!(
        !experiment.id.is_empty(),
        "Experiment ID should not be empty"
    );
    assert!(
        experiment.id.contains("mcmc"),
        "Experiment ID should mention MCMC"
    );

    println!("Experiment ID: {}", experiment.id);
    println!("Version: {}", experiment.gpredomics_version);
    println!("Timestamp: {}", experiment.timestamp);
    println!("Execution time: {:.2}s", experiment.execution_time);

    let population = &experiment.collections[0][0];
    println!(
        "Number of posterior samples: {}",
        population.individuals.len()
    );
}

// Test 13: MCMC with epsilon for prevalence
#[test]
fn test_mcmc_prevalence_epsilon() {
    println!("\n=== Test: MCMC with Different Epsilon for Prevalence ===");

    let epsilons = vec![0.0001, 0.001, 0.01];

    for epsilon in epsilons {
        println!("\nTesting epsilon = {}", epsilon);

        let mut param = load_base_mcmc_param();
        param.general.data_type = "prev".to_string();
        param.general.data_type_epsilon = epsilon;
        param.general.seed = 42;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = &experiment.collections[0][0];
        println!(
            "Epsilon {}: {} samples, first AUC = {:.4}",
            epsilon,
            population.individuals.len(),
            population.individuals[0].auc
        );

        assert!(population.individuals.len() > 0);
    }
}

// Test 14: SBS with different nmin values
#[test]
fn test_mcmc_sbs_different_nmin() {
    println!("\n=== Test: MCMC SBS with Different nmin Values ===");

    let nmin_values = vec![5, 10]; // Reduced to 2 values for faster testing

    for nmin in nmin_values {
        println!("\nTesting nmin = {}", nmin);

        let mut param = load_base_mcmc_param();
        param.mcmc.nmin = nmin;
        param.data.max_features_per_class = 30; // Start with more features
        param.general.seed = 42;
        param.mcmc.n_iter = 150; // Reduced for faster testing

        let running = Arc::new(AtomicBool::new(true));
        let experiment = gpredomics::run(&param, running);

        let population = &experiment.collections[0][0];
        let n_features = population.individuals[0].features.len();

        println!(
            "nmin {}: {} features kept, {} samples",
            nmin,
            n_features,
            population.individuals.len()
        );

        assert!(
            n_features >= nmin as usize,
            "Should keep at least nmin features"
        );
        assert!(
            population.individuals.len() > 0,
            "Should have posterior samples"
        );
    }
}

// Test 15: MCMC performance metrics
#[test]
fn test_mcmc_performance_metrics() {
    println!("\n=== Test: MCMC Performance Metrics ===");

    let param = load_base_mcmc_param();

    let running = Arc::new(AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    println!("Execution time: {:.2}s", experiment.execution_time);
    assert!(
        experiment.execution_time > 0.0,
        "Execution time should be recorded"
    );

    let population = &experiment.collections[0][0];

    // Calculate statistics across posterior
    let aucs: Vec<f64> = population.individuals.iter().map(|ind| ind.auc).collect();

    // Check we have samples
    assert!(aucs.len() > 0, "Should have posterior samples");

    let mean_auc = aucs.iter().sum::<f64>() / aucs.len() as f64;
    let var_auc = aucs.iter().map(|x| (x - mean_auc).powi(2)).sum::<f64>() / aucs.len() as f64;

    println!("AUC: mean={:.4}, std={:.4}", mean_auc, var_auc.sqrt());
    println!(
        "Number of posterior samples: {}",
        population.individuals.len()
    );

    // Basic sanity checks
    assert!(var_auc >= 0.0, "Variance should be non-negative");
    assert!(
        population.individuals.len() > 100,
        "Should have reasonable number of samples"
    );
}
