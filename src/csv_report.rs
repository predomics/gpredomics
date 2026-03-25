//! CSV performance report export for gpredomics experiments.
//!
//! Generates a CSV file with up to three rows per experiment:
//! - **best_model**: metrics for the #1 ranked individual
//! - **fbm**: averaged metrics across the Family of Best Models
//! - **jury**: voting ensemble metrics (only when voting is enabled)
//!
//! All classification metrics (AUC, F1, MCC, PPV, NPV, G-mean) are always computed
//! regardless of the fit function used during training. Parameters are exported as
//! individual named columns rather than a single string.
//!
//! Activated via CLI (`--csv-report`) or YAML (`general.csv_report: true`).
//! Output filename: `<timestamp>_csvr.csv`. Supports append mode.

use crate::data::Data;
use crate::experiment::{Experiment, ExperimentMetadata};
use crate::individual::Individual;
use crate::param::Param;
use crate::utils::{
    compute_auc_from_value, compute_metrics_from_classes, compute_metrics_from_value, mean_and_std,
};
use log::info;
use std::error::Error;
use std::io::Write;

/// CSV header columns shared by all report sections
const CSV_HEADER: &str = "\
section,\
experiment_id,\
timestamp,\
gpredomics_version,\
algorithm,\
language,\
data_type,\
fit_function,\
n_features_total,\
n_features_selected,\
n_train_samples,\
n_test_samples,\
is_cv,\
cv_folds,\
seed,\
k_penalty,\
bias_penalty,\
threshold_ci_penalty,\
threshold_ci_alpha,\
user_penalties_weight,\
model_k,\
train_auc,\
train_fit,\
train_accuracy,\
train_sensitivity,\
train_specificity,\
train_f1,\
train_mcc,\
train_ppv,\
train_npv,\
train_g_mean,\
train_rejection_rate,\
test_auc,\
test_accuracy,\
test_sensitivity,\
test_specificity,\
test_f1,\
test_mcc,\
test_ppv,\
test_npv,\
test_g_mean,\
test_rejection_rate,\
n_models,\
execution_time,\
holdout_ratio,\
fr_penalty,\
feature_selection_method,\
max_features_per_class,\
min_prevalence_pct,\
max_adj_pvalue,\
ga_pop_size,\
ga_max_epochs,\
ga_min_epochs,\
ga_k_min,\
ga_k_max,\
ga_elite_pct,\
ga_random_pct,\
ga_mut_children_pct,\
ga_mut_features_pct,\
beam_method,\
beam_k_start,\
beam_k_stop,\
beam_best_models_criterion,\
beam_max_nb_of_models,\
mcmc_n_iter,\
mcmc_n_burn,\
mcmc_lambda,\
mcmc_nmin,\
cv_enabled,\
cv_inner_folds,\
cv_outer_folds,\
cv_fbm_ci_method,\
cv_best_models_ci_alpha,\
vote_enabled,\
vote_method,\
vote_threshold,\
vote_min_perf,\
vote_min_diversity";

fn fmt_f64(v: f64) -> String {
    format!("{:.6}", v)
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.6}", x),
        None => "NA".to_string(),
    }
}

/// Force-compute all additional metrics for an individual on given data (train or test).
/// Always computes mcc, f1, npv, ppv, g_mean regardless of the fit function used during training.
fn compute_full_metrics(
    ind: &Individual,
    d: &Data,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let value = ind.evaluate(d);
    let auc = ind.compute_new_auc(d);
    let (acc, sens, spec, rejection, add) = compute_metrics_from_value(
        &value,
        &d.y,
        ind.cls.threshold,
        ind.cls.threshold_ci.as_ref().map(|ci| [ci.lower, ci.upper]),
        [true; 5],
    );
    (
        auc,
        acc,
        sens,
        spec,
        add.f1_score.unwrap_or(f64::NAN),
        add.mcc.unwrap_or(f64::NAN),
        add.ppv.unwrap_or(f64::NAN),
        add.npv.unwrap_or(f64::NAN),
        add.g_mean.unwrap_or(f64::NAN),
        rejection,
    )
}

/// Force-compute all additional train metrics for an individual from its stored train predictions.
/// Since the individual already has train auc/fit/accuracy/sensitivity/specificity,
/// we only need to recompute the additional metrics by re-evaluating on train data.
fn compute_train_additional(ind: &Individual, d: &Data) -> (f64, f64, f64, f64, f64) {
    let value = ind.evaluate(d);
    let (_, _, _, _, add) = compute_metrics_from_value(
        &value,
        &d.y,
        ind.cls.threshold,
        ind.cls.threshold_ci.as_ref().map(|ci| [ci.lower, ci.upper]),
        [true; 5],
    );
    (
        add.f1_score.unwrap_or(f64::NAN),
        add.mcc.unwrap_or(f64::NAN),
        add.ppv.unwrap_or(f64::NAN),
        add.npv.unwrap_or(f64::NAN),
        add.g_mean.unwrap_or(f64::NAN),
    )
}

/// Builds the common prefix columns from Experiment metadata (18 columns)
fn common_prefix(exp: &Experiment) -> Vec<String> {
    let p = &exp.parameters;
    let is_cv = exp.cv_folds_ids.is_some();
    let cv_folds = if is_cv {
        exp.cv_folds_ids.as_ref().unwrap().len()
    } else {
        0
    };
    let n_test = exp.test_data.as_ref().map(|d| d.sample_len).unwrap_or(0);

    vec![
        exp.id.clone(),
        exp.timestamp.clone(),
        exp.gpredomics_version.clone(),
        p.general.algo.clone(),
        p.general.language.clone(),
        p.general.data_type.clone(),
        format!("{:?}", p.general.fit),
        exp.train_data.feature_len.to_string(),
        exp.train_data.feature_selection.len().to_string(),
        exp.train_data.sample_len.to_string(),
        n_test.to_string(),
        is_cv.to_string(),
        cv_folds.to_string(),
        p.general.seed.to_string(),
        fmt_f64(p.general.k_penalty),
        fmt_f64(p.general.bias_penalty),
        fmt_f64(p.general.threshold_ci_penalty),
        fmt_f64(p.general.threshold_ci_alpha),
        fmt_f64(p.general.user_penalties_weight),
    ]
}

/// Builds individual named parameter columns (replaces the old extra_params quoted string)
fn param_columns(p: &Param) -> Vec<String> {
    vec![
        // Data params
        fmt_f64(p.data.holdout_ratio),
        fmt_f64(p.general.fr_penalty),
        format!("{:?}", p.data.feature_selection_method),
        p.data.max_features_per_class.to_string(),
        fmt_f64(p.data.feature_minimal_prevalence_pct),
        fmt_f64(p.data.feature_maximal_adj_pvalue),
        // GA params
        p.ga.population_size.to_string(),
        p.ga.max_epochs.to_string(),
        p.ga.min_epochs.to_string(),
        p.ga.k_min.to_string(),
        p.ga.k_max.to_string(),
        fmt_f64(p.ga.select_elite_pct),
        fmt_f64(p.ga.select_random_pct),
        fmt_f64(p.ga.mutated_children_pct),
        fmt_f64(p.ga.mutated_features_pct),
        // Beam params
        format!("{:?}", p.beam.method),
        p.beam.k_start.to_string(),
        p.beam.k_stop.to_string(),
        fmt_f64(p.beam.best_models_criterion),
        p.beam.max_nb_of_models.to_string(),
        // MCMC params
        p.mcmc.n_iter.to_string(),
        p.mcmc.n_burn.to_string(),
        fmt_f64(p.mcmc.lambda),
        p.mcmc.nmin.to_string(),
        // CV params
        p.general.cv.to_string(),
        p.cv.inner_folds.to_string(),
        p.cv.outer_folds.to_string(),
        format!("{:?}", p.cv.cv_fbm_ci_method),
        fmt_f64(p.cv.cv_best_models_ci_alpha),
        // Voting params
        p.voting.vote.to_string(),
        format!("{:?}", p.voting.method),
        fmt_f64(p.voting.method_threshold),
        fmt_f64(p.voting.min_perf),
        fmt_f64(p.voting.min_diversity),
    ]
}

/// Assembles a full CSV row
fn build_row(
    section: &str,
    prefix: &[String],
    metrics: &[String],
    n_models: usize,
    exec_time: f64,
    params: &[String],
) -> String {
    let mut cols = Vec::with_capacity(80);
    cols.push(section.to_string());
    cols.extend_from_slice(prefix);
    cols.extend_from_slice(metrics);
    cols.push(n_models.to_string());
    cols.push(fmt_f64(exec_time));
    cols.extend_from_slice(params);
    cols.join(",")
}

/// Builds metrics columns for a single individual (22 columns: model_k through test_rejection_rate)
/// Forces computation of all additional metrics (f1, mcc, ppv, npv, g_mean).
fn individual_metrics_cols(
    ind: &Individual,
    train_data: &Data,
    test_data: Option<&Data>,
) -> Vec<String> {
    let train_rejection = ind
        .cls
        .threshold_ci
        .as_ref()
        .map(|ci| ci.rejection_rate)
        .unwrap_or(0.0);

    // Force-compute train additional metrics
    let (train_f1, train_mcc, train_ppv, train_npv, train_gmean) =
        compute_train_additional(ind, train_data);

    // Force-compute test metrics with all additional metrics
    let (
        test_auc,
        test_acc,
        test_sens,
        test_spec,
        test_f1,
        test_mcc,
        test_ppv,
        test_npv,
        test_gmean,
        test_rej,
    ) = if let Some(td) = test_data {
        let r = compute_full_metrics(ind, td);
        (
            Some(r.0),
            Some(r.1),
            Some(r.2),
            Some(r.3),
            Some(r.4),
            Some(r.5),
            Some(r.6),
            Some(r.7),
            Some(r.8),
            Some(r.9),
        )
    } else {
        (None, None, None, None, None, None, None, None, None, None)
    };

    vec![
        ind.k.to_string(),
        fmt_f64(ind.cls.auc),
        fmt_f64(ind.fit),
        fmt_f64(ind.cls.accuracy),
        fmt_f64(ind.cls.sensitivity),
        fmt_f64(ind.cls.specificity),
        fmt_f64(train_f1),
        fmt_f64(train_mcc),
        fmt_f64(train_ppv),
        fmt_f64(train_npv),
        fmt_f64(train_gmean),
        fmt_f64(train_rejection),
        fmt_opt(test_auc),
        fmt_opt(test_acc),
        fmt_opt(test_sens),
        fmt_opt(test_spec),
        fmt_opt(test_f1),
        fmt_opt(test_mcc),
        fmt_opt(test_ppv),
        fmt_opt(test_npv),
        fmt_opt(test_gmean),
        fmt_opt(test_rej),
    ]
}

/// Exports the best model (rank #1) performance as a CSV row
pub fn export_best_model_csv(
    exp: &Experiment,
    writer: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    let final_pop = exp
        .final_population
        .as_ref()
        .expect("No final population available for CSV report");

    if final_pop.individuals.is_empty() {
        return Ok(());
    }

    let best = &final_pop.individuals[0];
    let prefix = common_prefix(exp);
    let metrics = individual_metrics_cols(best, &exp.train_data, exp.test_data.as_ref());
    let params = param_columns(&exp.parameters);

    writeln!(
        writer,
        "{}",
        build_row(
            "best_model",
            &prefix,
            &metrics,
            1,
            exp.execution_time,
            &params
        )
    )?;
    Ok(())
}

/// Exports the FBM (Family of Best Models) averaged performance as a CSV row
pub fn export_fbm_csv(exp: &Experiment, writer: &mut impl Write) -> Result<(), Box<dyn Error>> {
    let final_pop = exp
        .final_population
        .as_ref()
        .expect("No final population available for CSV report");

    let fbm = final_pop.select_best_population(0.05);

    if fbm.individuals.is_empty() {
        return Ok(());
    }

    let n = fbm.individuals.len();

    // Train metrics - force-compute all additional metrics
    let train_aucs: Vec<f64> = fbm.individuals.iter().map(|i| i.cls.auc).collect();
    let train_fits: Vec<f64> = fbm.individuals.iter().map(|i| i.fit).collect();
    let train_accs: Vec<f64> = fbm.individuals.iter().map(|i| i.cls.accuracy).collect();
    let train_sens: Vec<f64> = fbm.individuals.iter().map(|i| i.cls.sensitivity).collect();
    let train_specs: Vec<f64> = fbm.individuals.iter().map(|i| i.cls.specificity).collect();
    let train_ks: Vec<f64> = fbm.individuals.iter().map(|i| i.k as f64).collect();
    let train_rejs: Vec<f64> = fbm
        .individuals
        .iter()
        .map(|i| {
            i.cls
                .threshold_ci
                .as_ref()
                .map(|ci| ci.rejection_rate)
                .unwrap_or(0.0)
        })
        .collect();

    // Force-compute additional train metrics for each individual
    let mut train_f1s = Vec::with_capacity(n);
    let mut train_mccs = Vec::with_capacity(n);
    let mut train_ppvs = Vec::with_capacity(n);
    let mut train_npvs = Vec::with_capacity(n);
    let mut train_gmeans = Vec::with_capacity(n);

    for ind in &fbm.individuals {
        let (f1, mcc, ppv, npv, gmean) = compute_train_additional(ind, &exp.train_data);
        train_f1s.push(f1);
        train_mccs.push(mcc);
        train_ppvs.push(ppv);
        train_npvs.push(npv);
        train_gmeans.push(gmean);
    }

    let (auc_m, _) = mean_and_std(&train_aucs);
    let (fit_m, _) = mean_and_std(&train_fits);
    let (acc_m, _) = mean_and_std(&train_accs);
    let (sens_m, _) = mean_and_std(&train_sens);
    let (spec_m, _) = mean_and_std(&train_specs);
    let (k_m, _) = mean_and_std(&train_ks);
    let (rej_m, _) = mean_and_std(&train_rejs);
    let (f1_m, _) = mean_and_std(&train_f1s);
    let (mcc_m, _) = mean_and_std(&train_mccs);
    let (ppv_m, _) = mean_and_std(&train_ppvs);
    let (npv_m, _) = mean_and_std(&train_npvs);
    let (gmean_m, _) = mean_and_std(&train_gmeans);

    // Test metrics - force-compute all additional metrics
    let (
        test_auc_m,
        test_acc_m,
        test_sens_m,
        test_spec_m,
        test_f1_m,
        test_mcc_m,
        test_ppv_m,
        test_npv_m,
        test_gmean_m,
        test_rej_m,
    ) = if let Some(ref td) = exp.test_data {
        let mut t_aucs = Vec::with_capacity(n);
        let mut t_accs = Vec::with_capacity(n);
        let mut t_senss = Vec::with_capacity(n);
        let mut t_specss = Vec::with_capacity(n);
        let mut t_f1s = Vec::with_capacity(n);
        let mut t_mccs = Vec::with_capacity(n);
        let mut t_ppvs = Vec::with_capacity(n);
        let mut t_npvs = Vec::with_capacity(n);
        let mut t_gmeans = Vec::with_capacity(n);
        let mut t_rejs = Vec::with_capacity(n);

        for ind in &fbm.individuals {
            let (auc, acc, sens, spec, f1, mcc, ppv, npv, gmean, rej) =
                compute_full_metrics(ind, td);
            t_aucs.push(auc);
            t_accs.push(acc);
            t_senss.push(sens);
            t_specss.push(spec);
            t_f1s.push(f1);
            t_mccs.push(mcc);
            t_ppvs.push(ppv);
            t_npvs.push(npv);
            t_gmeans.push(gmean);
            t_rejs.push(rej);
        }

        (
            Some(mean_and_std(&t_aucs).0),
            Some(mean_and_std(&t_accs).0),
            Some(mean_and_std(&t_senss).0),
            Some(mean_and_std(&t_specss).0),
            Some(mean_and_std(&t_f1s).0),
            Some(mean_and_std(&t_mccs).0),
            Some(mean_and_std(&t_ppvs).0),
            Some(mean_and_std(&t_npvs).0),
            Some(mean_and_std(&t_gmeans).0),
            Some(mean_and_std(&t_rejs).0),
        )
    } else {
        (None, None, None, None, None, None, None, None, None, None)
    };

    let prefix = common_prefix(exp);
    let params = param_columns(&exp.parameters);

    let metrics = vec![
        fmt_f64(k_m),
        fmt_f64(auc_m),
        fmt_f64(fit_m),
        fmt_f64(acc_m),
        fmt_f64(sens_m),
        fmt_f64(spec_m),
        fmt_f64(f1_m),
        fmt_f64(mcc_m),
        fmt_f64(ppv_m),
        fmt_f64(npv_m),
        fmt_f64(gmean_m),
        fmt_f64(rej_m),
        fmt_opt(test_auc_m),
        fmt_opt(test_acc_m),
        fmt_opt(test_sens_m),
        fmt_opt(test_spec_m),
        fmt_opt(test_f1_m),
        fmt_opt(test_mcc_m),
        fmt_opt(test_ppv_m),
        fmt_opt(test_npv_m),
        fmt_opt(test_gmean_m),
        fmt_opt(test_rej_m),
    ];

    writeln!(
        writer,
        "{}",
        build_row("fbm", &prefix, &metrics, n, exp.execution_time, &params)
    )?;
    Ok(())
}

/// Exports the Jury (voting ensemble) performance as a CSV row
pub fn export_jury_csv(exp: &Experiment, writer: &mut impl Write) -> Result<(), Box<dyn Error>> {
    let jury = match &exp.others {
        Some(ExperimentMetadata::Jury { jury }) => jury,
        _ => return Ok(()),
    };

    let n_experts = jury.experts.individuals.len();
    let avg_k = if n_experts > 0 {
        let ks: Vec<f64> = jury
            .experts
            .individuals
            .iter()
            .map(|i| i.k as f64)
            .collect();
        mean_and_std(&ks).0
    } else {
        0.0
    };

    // Force-compute train additional metrics for the jury
    // Jury train metrics come from its stored confusion-matrix-level values
    // We need to recompute using the jury's predict on train data
    let (train_f1, train_mcc, train_ppv, train_npv, train_gmean) = {
        let (pred_classes, _scores) = jury.predict(&exp.train_data);
        let filtered: Vec<(u8, f64)> = pred_classes
            .iter()
            .zip(exp.train_data.y.iter())
            .filter(|(&p, _)| p != 2)
            .map(|(&p, &y)| (p, y))
            .collect();
        if !filtered.is_empty() {
            let (preds, trues): (Vec<u8>, Vec<f64>) = filtered.into_iter().unzip();
            let (_, _, _, add) = compute_metrics_from_classes(&preds, &trues, [true; 5]);
            (
                add.f1_score.unwrap_or(f64::NAN),
                add.mcc.unwrap_or(f64::NAN),
                add.ppv.unwrap_or(f64::NAN),
                add.npv.unwrap_or(f64::NAN),
                add.g_mean.unwrap_or(f64::NAN),
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        }
    };

    // Force-compute test metrics with all additional metrics
    let (
        test_auc,
        test_acc,
        test_sens,
        test_spec,
        test_f1,
        test_mcc,
        test_ppv,
        test_npv,
        test_gmean,
        test_rej,
    ) = if let Some(ref td) = exp.test_data {
        let (pred_classes, scores) = jury.predict(td);
        let filtered: Vec<(f64, u8, f64)> = scores
            .iter()
            .zip(pred_classes.iter())
            .zip(td.y.iter())
            .filter_map(|((&score, &pred_class), &true_class)| {
                if score >= 0.0 && score <= 1.0 && pred_class != 2 {
                    Some((score, pred_class, true_class))
                } else {
                    None
                }
            })
            .collect();

        let rejection_rate =
            pred_classes.iter().filter(|&&c| c == 2).count() as f64 / pred_classes.len() as f64;

        if !filtered.is_empty() {
            let (scores_f, preds_f, trues_f): (Vec<f64>, Vec<u8>, Vec<f64>) =
                filtered.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut s, mut p, mut t), (sc, pr, tr)| {
                        s.push(sc);
                        p.push(pr);
                        t.push(tr);
                        (s, p, t)
                    },
                );
            let auc = compute_auc_from_value(&scores_f, &trues_f);
            let (acc, sens, spec, add) =
                compute_metrics_from_classes(&preds_f, &trues_f, [true; 5]);
            (
                Some(auc),
                Some(acc),
                Some(sens),
                Some(spec),
                Some(add.f1_score.unwrap_or(f64::NAN)),
                Some(add.mcc.unwrap_or(f64::NAN)),
                Some(add.ppv.unwrap_or(f64::NAN)),
                Some(add.npv.unwrap_or(f64::NAN)),
                Some(add.g_mean.unwrap_or(f64::NAN)),
                Some(rejection_rate),
            )
        } else {
            (
                Some(0.5),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(f64::NAN),
                Some(f64::NAN),
                Some(f64::NAN),
                Some(f64::NAN),
                Some(f64::NAN),
                Some(rejection_rate),
            )
        }
    } else {
        (None, None, None, None, None, None, None, None, None, None)
    };

    let prefix = common_prefix(exp);
    let params = param_columns(&exp.parameters);

    let metrics = vec![
        fmt_f64(avg_k),
        fmt_f64(jury.auc),
        "NA".to_string(),
        fmt_f64(jury.accuracy),
        fmt_f64(jury.sensitivity),
        fmt_f64(jury.specificity),
        fmt_f64(train_f1),
        fmt_f64(train_mcc),
        fmt_f64(train_ppv),
        fmt_f64(train_npv),
        fmt_f64(train_gmean),
        fmt_f64(jury.rejection_rate),
        fmt_opt(test_auc),
        fmt_opt(test_acc),
        fmt_opt(test_sens),
        fmt_opt(test_spec),
        fmt_opt(test_f1),
        fmt_opt(test_mcc),
        fmt_opt(test_ppv),
        fmt_opt(test_npv),
        fmt_opt(test_gmean),
        fmt_opt(test_rej),
    ];

    writeln!(
        writer,
        "{}",
        build_row(
            "jury",
            &prefix,
            &metrics,
            n_experts,
            exp.execution_time,
            &params
        )
    )?;
    Ok(())
}

impl Experiment {
    /// Exports a CSV performance report containing up to 3 rows:
    /// 1. best_model - the #1 ranked individual
    /// 2. fbm - the Family of Best Models (averaged metrics)
    /// 3. jury - the voting ensemble (if voting was enabled)
    ///
    /// The file is created (or appended to if it already exists with a matching header).
    pub fn export_csv_report(&self, path: &str) -> Result<(), Box<dyn Error>> {
        use std::fs::OpenOptions;
        use std::path::Path;

        let file_exists = Path::new(path).exists();
        let header_matches = if file_exists {
            let first_line = std::fs::read_to_string(path)?
                .lines()
                .next()
                .unwrap_or("")
                .to_string();
            first_line == CSV_HEADER
        } else {
            false
        };

        let mut file = if file_exists && header_matches {
            info!("Appending to existing CSV report: {}", path);
            OpenOptions::new().append(true).open(path)?
        } else {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            writeln!(f, "{}", CSV_HEADER)?;
            f
        };

        export_best_model_csv(self, &mut file)?;
        export_fbm_csv(self, &mut file)?;
        export_jury_csv(self, &mut file)?;

        info!("CSV performance report written to {}", path);
        Ok(())
    }
}
