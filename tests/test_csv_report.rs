/// Integration tests for CSV performance report export
///
/// Tests validate:
/// 1. CSV report generation with correct header and structure
/// 2. All metrics (f1, mcc, ppv, npv, g_mean) are computed even when fit=auc
/// 3. Best model, FBM, and jury rows are present when expected
/// 4. Parameter columns are correctly populated
/// 5. Append mode works when header matches
///
/// Run with: cargo test --test test_csv_report -- --nocapture
use gpredomics::beam::BeamMethod;
use gpredomics::param::Param;
use gpredomics::run;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Helper to create Beam parameters for Qin2014 with csv_report enabled
fn create_csv_test_params() -> Param {
    let mut param = Param::default();

    param.general.seed = 42;
    param.general.algo = "beam".to_string();
    param.general.cv = false;
    param.general.thread_number = 2;
    param.general.gpu = false;
    param.general.language = "binary".to_string();
    param.general.data_type = "raw".to_string();
    param.general.fit = gpredomics::param::FitFunction::auc;
    param.general.k_penalty = 0.0;
    param.general.display_colorful = false;
    param.general.keep_trace = true;
    param.general.save_exp = "".to_string();
    param.general.log_base = "".to_string();
    param.general.csv_report = true;

    param.data.X = "samples/Qin2014/Xtrain.tsv".to_string();
    param.data.y = "samples/Qin2014/Ytrain.tsv".to_string();
    param.data.Xtest = "samples/Qin2014/Xtest.tsv".to_string();
    param.data.ytest = "samples/Qin2014/Ytest.tsv".to_string();
    param.data.features_in_rows = true;
    param.data.feature_minimal_prevalence_pct = 10.0;
    param.data.feature_maximal_adj_pvalue = 0.001;

    param.beam.method = BeamMethod::LimitedExhaustive;
    param.beam.k_start = 1;
    param.beam.k_stop = 3;
    param.beam.max_nb_of_models = 10000;

    param.voting.vote = false;

    param
}

/// Helper to create params with voting enabled
fn create_csv_test_params_with_voting() -> Param {
    let mut param = create_csv_test_params();
    param.voting.vote = true;
    param.voting.min_perf = 0.5;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.method_threshold = 0.5;
    param
}

#[test]
fn test_csv_report_has_correct_header() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_header.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();

    // Header + best_model + fbm = at least 3 lines
    assert!(
        lines.len() >= 3,
        "Expected at least 3 lines (header + best_model + fbm), got {}",
        lines.len()
    );

    let header = lines[0];
    // Check key header columns exist
    assert!(header.starts_with("section,"));
    assert!(header.contains("train_f1"));
    assert!(header.contains("train_mcc"));
    assert!(header.contains("train_ppv"));
    assert!(header.contains("train_npv"));
    assert!(header.contains("train_g_mean"));
    assert!(header.contains("test_f1"));
    assert!(header.contains("test_mcc"));
    assert!(header.contains("test_ppv"));
    assert!(header.contains("test_npv"));
    assert!(header.contains("test_g_mean"));
    // Check parameter columns exist (not extra_params)
    assert!(header.contains("holdout_ratio"));
    assert!(header.contains("beam_method"));
    assert!(header.contains("ga_pop_size"));
    assert!(header.contains("cv_enabled"));
    assert!(header.contains("vote_enabled"));
    assert!(
        !header.contains("extra_params"),
        "extra_params should be split into individual columns"
    );

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_no_na_metrics_with_fit_auc() {
    let param = create_csv_test_params();
    assert!(
        matches!(param.general.fit, gpredomics::param::FitFunction::auc),
        "Test requires fit=auc"
    );

    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_no_na.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let header_cols: Vec<&str> = lines[0].split(',').collect();

    // Check best_model row (line 1)
    let best_cols: Vec<&str> = lines[1].split(',').collect();
    assert_eq!(best_cols[0], "best_model");

    // Find indices for metrics that used to be NA
    let metric_names = [
        "train_f1",
        "train_mcc",
        "train_ppv",
        "train_npv",
        "train_g_mean",
        "test_f1",
        "test_mcc",
        "test_ppv",
        "test_npv",
        "test_g_mean",
    ];
    for name in &metric_names {
        let idx = header_cols
            .iter()
            .position(|&h| h == *name)
            .unwrap_or_else(|| panic!("Column {} not found in header", name));
        assert_ne!(
            best_cols[idx], "NA",
            "Column {} should not be NA when csv_report forces metric computation",
            name
        );
    }

    // Check fbm row (line 2)
    let fbm_cols: Vec<&str> = lines[2].split(',').collect();
    assert_eq!(fbm_cols[0], "fbm");
    for name in &metric_names {
        let idx = header_cols.iter().position(|&h| h == *name).unwrap();
        assert_ne!(fbm_cols[idx], "NA", "FBM column {} should not be NA", name);
    }

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_best_model_and_fbm_sections() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_sections.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();

    let sections: Vec<&str> = lines[1..]
        .iter()
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert!(
        sections.contains(&"best_model"),
        "Missing best_model section"
    );
    assert!(sections.contains(&"fbm"), "Missing fbm section");
    // No jury since voting is disabled
    assert!(
        !sections.contains(&"jury"),
        "jury should not be present when voting is disabled"
    );

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_model_k_present() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_model_k.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let header_cols: Vec<&str> = lines[0].split(',').collect();
    let k_idx = header_cols
        .iter()
        .position(|&h| h == "model_k")
        .expect("model_k column not found");

    // Best model k should be a positive integer
    let best_cols: Vec<&str> = lines[1].split(',').collect();
    let k_val: f64 = best_cols[k_idx].parse().expect("model_k should be numeric");
    assert!(k_val >= 1.0, "Best model k should be >= 1, got {}", k_val);

    // FBM k should be a positive average
    let fbm_cols: Vec<&str> = lines[2].split(',').collect();
    let k_avg: f64 = fbm_cols[k_idx]
        .parse()
        .expect("FBM model_k should be numeric");
    assert!(k_avg >= 1.0, "FBM average k should be >= 1, got {}", k_avg);

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_column_count_consistency() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_col_count.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let header_count = lines[0].split(',').count();

    for (i, line) in lines[1..].iter().enumerate() {
        let col_count = line.split(',').count();
        assert_eq!(
            col_count,
            header_count,
            "Row {} has {} columns but header has {}",
            i + 1,
            col_count,
            header_count
        );
    }

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_append_mode() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_append.csv";
    // Write once
    exp.export_csv_report(csv_path)
        .expect("First CSV export failed");
    let content1 = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines1 = content1.lines().count();

    // Write again - should append
    exp.export_csv_report(csv_path)
        .expect("Second CSV export failed");
    let content2 = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines2 = content2.lines().count();

    // Second write should have added data rows (minus header)
    let data_rows_first = lines1 - 1; // minus header
    assert_eq!(
        lines2,
        lines1 + data_rows_first,
        "Append mode should add {} data rows, got {} total (was {})",
        data_rows_first,
        lines2,
        lines1
    );

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_with_voting_has_jury_section() {
    let param = create_csv_test_params_with_voting();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_jury.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let header_cols: Vec<&str> = lines[0].split(',').collect();

    let sections: Vec<&str> = lines[1..]
        .iter()
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert!(
        sections.contains(&"best_model"),
        "Missing best_model section"
    );
    assert!(sections.contains(&"fbm"), "Missing fbm section");
    assert!(
        sections.contains(&"jury"),
        "Missing jury section when voting is enabled"
    );

    // Jury row should also have computed metrics (no NA for f1/mcc/ppv/npv/g_mean)
    let jury_line = lines[1..]
        .iter()
        .find(|l| l.starts_with("jury"))
        .expect("jury line not found");
    let jury_cols: Vec<&str> = jury_line.split(',').collect();

    let metric_names = [
        "train_f1",
        "train_mcc",
        "train_ppv",
        "train_npv",
        "train_g_mean",
    ];
    for name in &metric_names {
        let idx = header_cols
            .iter()
            .position(|&h| h == *name)
            .unwrap_or_else(|| panic!("Column {} not found", name));
        assert_ne!(
            jury_cols[idx], "NA",
            "Jury column {} should not be NA",
            name
        );
    }

    // Check header column count matches jury row
    let header_count = header_cols.len();
    let jury_count = jury_cols.len();
    assert_eq!(
        jury_count, header_count,
        "Jury row has {} columns but header has {}",
        jury_count, header_count
    );

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_parameter_columns() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_params.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let header_cols: Vec<&str> = lines[0].split(',').collect();
    let best_cols: Vec<&str> = lines[1].split(',').collect();

    // Check algorithm column
    let algo_idx = header_cols.iter().position(|&h| h == "algorithm").unwrap();
    assert_eq!(best_cols[algo_idx], "beam");

    // Check beam parameters reflect input
    let beam_method_idx = header_cols
        .iter()
        .position(|&h| h == "beam_method")
        .unwrap();
    assert_eq!(best_cols[beam_method_idx], "LimitedExhaustive");

    let beam_k_start_idx = header_cols
        .iter()
        .position(|&h| h == "beam_k_start")
        .unwrap();
    assert_eq!(best_cols[beam_k_start_idx], "1");

    let beam_k_stop_idx = header_cols
        .iter()
        .position(|&h| h == "beam_k_stop")
        .unwrap();
    assert_eq!(best_cols[beam_k_stop_idx], "3");

    // Check voting disabled
    let vote_idx = header_cols
        .iter()
        .position(|&h| h == "vote_enabled")
        .unwrap();
    assert_eq!(best_cols[vote_idx], "false");

    // Check cv disabled
    let cv_idx = header_cols.iter().position(|&h| h == "cv_enabled").unwrap();
    assert_eq!(best_cols[cv_idx], "false");

    std::fs::remove_file(csv_path).ok();
}

#[test]
fn test_csv_report_metrics_are_valid_numbers() {
    let param = create_csv_test_params();
    let running = Arc::new(AtomicBool::new(true));
    let exp = run(&param, running);

    let csv_path = "/tmp/test_csv_valid_numbers.csv";
    exp.export_csv_report(csv_path).expect("CSV export failed");

    let content = std::fs::read_to_string(csv_path).expect("Failed to read CSV");
    let lines: Vec<&str> = content.lines().collect();
    let header_cols: Vec<&str> = lines[0].split(',').collect();

    let numeric_metrics = [
        "train_auc",
        "train_accuracy",
        "train_sensitivity",
        "train_specificity",
        "train_f1",
        "train_mcc",
        "train_ppv",
        "train_npv",
        "train_g_mean",
        "test_auc",
        "test_accuracy",
        "test_sensitivity",
        "test_specificity",
        "test_f1",
        "test_mcc",
        "test_ppv",
        "test_npv",
        "test_g_mean",
    ];

    let best_cols: Vec<&str> = lines[1].split(',').collect();
    for name in &numeric_metrics {
        let idx = header_cols
            .iter()
            .position(|&h| h == *name)
            .unwrap_or_else(|| panic!("Column {} not found", name));
        let val: f64 = best_cols[idx].parse().unwrap_or_else(|_| {
            panic!(
                "Column {} value '{}' is not a valid number",
                name, best_cols[idx]
            )
        });
        assert!(
            val.is_finite(),
            "Column {} should be a finite number, got {}",
            name,
            val
        );
    }

    std::fs::remove_file(csv_path).ok();
}
