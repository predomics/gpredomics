#![allow(unused_variables)]
use polars::prelude::*;
use std::fs::File;
use std::iter::zip;
use rand::Rng;
use rand::prelude::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
// use rand::{Rng, seq::IteratorRandom};
// use rand_distr::{Normal, Distribution};
use statrs::function::logistic::logistic;
use statrs::function::erf::{erf, erf_inv};
use std::process;
use argmin::{
    core::{CostFunction, Error as argmin_Error, Executor},
    solver::brent::BrentOpt,
};
use ndarray::{Array1,Array2};

const BIG_NUMBER: f64 = 100.0;

//-----------------------------------------------------------------------------
fn log_logistic(x: f64) -> f64 {
    if x >= - BIG_NUMBER {
        logistic(x).ln()
    } else {
        x + logistic(-BIG_NUMBER).ln() + BIG_NUMBER
    }
}fn vector_i32_to_f64(x: &Vec<i32>) -> Vec<f64> {
    let y: Vec<_> = x.into_iter().map(|&v| v as f64).collect();
    y
}

// vector operations
// fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
//     a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
// }

fn vector_squared(a: &Vec<f64>) -> Vec<f64> {
    a.iter().map(|x| x * x).collect()
}

fn sum_vectors(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn divide_vector_by_number(a: &Vec<f64>, b: f64) -> Vec<f64> {
    a.iter().map(|x| x / b).collect()
}

fn transpose_vec_arr(a: &Vec<[f64;3]>) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut a_t = Vec::new(); 
    for j in 0..m {
        let v = (0..n).map(|i| a[i][j]).collect();
        a_t.push(v);
    }
    a_t
}
fn transpose_vec_vec_i32(a: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let n = a.len();
    let m = a[0].len();
    let mut a_t = Vec::new(); 
    for j in 0..m {
        let v = (0..n).map(|i| a[i][j]).collect();
        a_t.push(v);
    }
    a_t
}
//-----------------------------------------------------------------------------

// sampling from truncated normal distributions
fn truncnorm_pos(mu: f64, scale: f64, rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    // truncated normal positive random number 
    let erf0 = erf(- mu / (2_f64.sqrt() * scale));
    let u: f64 = rng.gen_range(0.0..1.0);
    mu + 2_f64.sqrt() * scale * erf_inv(u * (1.0 - erf0) + erf0)
}

fn truncnorm_neg(mu: f64, scale: f64, rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    // truncated normal negative random number
    let erf0 = erf(- mu / (2_f64.sqrt() * scale));
    let u: f64 = rng.gen_range(0.0..1.0);
    mu + 2_f64.sqrt() * scale * erf_inv(u * (1.0 + erf0) - 1.0)
}

fn random_normal(mu: f64, scale: f64, rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    // normal random number
    let u: f64 = rng.gen_range(0.0..1.0);
    mu + 2_f64.sqrt() * scale * erf_inv(2.0 * u - 1.0)
}

//-----------------------------------------------------------------------------
// Structure for holding data for MCMC
pub struct BayesPred {
    x: Array2<f64>,
    y: Array1<f64>,
    features: Column,
    lmbd: f64,
    n_samples: usize,
    n_features: usize,
}

impl BayesPred {
    pub fn new(df_x: &DataFrame, df_y: &DataFrame, lmbd: f64) -> BayesPred {

        let y: Vec<f64> = df_y.column("status").unwrap().cast(&DataType::Float64).unwrap()
        .f64().unwrap().to_vec_null_aware().left().unwrap();
        let y = Array1::from_vec(y);

        BayesPred {
            // x: x_presence,
            x: df_x.drop("msp_name").unwrap().to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap(),
            y: y,
            features: df_x["msp_name"].clone(),
            lmbd: lmbd,
            n_samples: df_x.width() - 1,
            n_features: df_x.height(),
        }
    }

    fn biomarker_components(&self, model: &Vec<i32>) -> Vec<Vec<f64>> {
        let mut z = Vec::with_capacity(self.n_samples);
        // let mut z = Array2::<f64>::zeros((self.n_samples, 3));
        for x_smpl in self.x.columns() {
            let mut sig1: f64 = 0.0;
            let mut sig2: f64 = 0.0;
            for (v, m) in x_smpl.iter().zip(model.iter()) {
                match *m {
                    1 => sig1 += v,
                    -1 => sig2 += v,
                    _ => (),
                }
            }
            z.push(vec![sig1, sig2, 1.]);
        }
        z
    }
    
    fn update_biomarker_components(&self, z: &Vec<Vec<f64>>, k: usize, old_value: i32, new_value: i32) -> Vec<Vec<f64>> {
        let mut z_new = z.clone();
        for (i_smpl, x_smpl) in self.x.columns().into_iter().enumerate() {
            match (old_value, new_value) {
                (0, 1) => z_new[i_smpl][0] += x_smpl[k],
                (0, -1) => z_new[i_smpl][1] += x_smpl[k],
                (1, 0) => z_new[i_smpl][0] -= x_smpl[k],
                (-1, 0) => z_new[i_smpl][1] -= x_smpl[k],
                (-1, 1) => {z_new[i_smpl][0] += x_smpl[k]; z_new[i_smpl][1] -= x_smpl[k]},
                (1, -1) => {z_new[i_smpl][0] -= x_smpl[k]; z_new[i_smpl][1] += x_smpl[k]},
                _ => (),
            }
        }
        z_new
    }

    fn log_posterior(&self, beta : &[f64; 3], z: &Vec<Vec<f64>>) -> f64 {
        let mut log_likelihood= 0_f64;
        for (z_sample, &y_sample) in zip(z, &self.y) {
            // let value = dot_product(z_sample, beta);
            let value = z_sample[0] * beta[0] + z_sample[1] * beta[1] + z_sample[2] * beta[2];
            if y_sample == 1.0 {
                log_likelihood += log_logistic(value)
            }
            else {
                log_likelihood += log_logistic(-value)
            }
            if log_logistic(value).is_infinite() || log_logistic(-value).is_infinite() {
                println!("v={}, logistic(x)={}, logistic(-x)={}", value, logistic(value), logistic(-value));
                println!("{}, {}, {}", value, log_logistic(value), log_logistic(-value));
                process::exit(1)
            };
        }
        let log_prior: f64 = - self.lmbd * beta.iter().map(|v| v*v).sum::<f64>();
        log_likelihood + log_prior
    }

    fn sigma_i(&self, beta : &[f64; 3], z: &Vec<Vec<f64>>, i: usize) -> f64 {
        let mut cov_inv= 0_f64;
        for (z_sample, &y_sample) in zip(z, &self.y) {
            // let value = dot_product(z_sample, &beta);
            let value = z_sample[0] * beta[0] + z_sample[1] * beta[1] + z_sample[2] * beta[2];
            cov_inv += logistic(value) * logistic(-value) * z_sample[i].powf(2_f64)
    
        }
        1_f64 / (cov_inv + self.lmbd).sqrt()
    }
}

struct NegLogPostToMinimize<'a> {
    bp: &'a BayesPred,
    i: usize,
    beta: &'a [f64; 3],
    z: &'a Vec<Vec<f64>>,
}

impl CostFunction for NegLogPostToMinimize<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, beta_i: &Self::Param) -> Result<Self::Output, argmin_Error> {
        let mut new_beta = self.beta.clone();
        new_beta[self.i] = *beta_i;
        Ok(- self.bp.log_posterior(&new_beta, &self.z))
    }
}


//-----------------------------------------------------------------------------
//structure for MCMC book-keeping

pub struct ResultMCMC {
    n_features: usize,
    beta_mean: [f64; 3],
    beta_var: [f64; 3],
    model_mean: Vec<f64>,
    model_var: Vec<f64>,
    log_post_mean: f64,
    log_post_var: f64,
    pub post_mean: f64,
    post_var: f64,
    pub p_sig1: Vec<f64>,
    pub p_sig2: Vec<f64>,
    pub p_sig0: Vec<f64>,
    log_post_trace: Vec<f64>,
    beta_trace: Vec<[f64;3]>,
    model_trace: Vec<Vec<i32>>,
    keep_trace: bool,
}

impl ResultMCMC {
    fn new(n_features: usize, keep_trace: bool) -> ResultMCMC {
        ResultMCMC{
            n_features: n_features,
            beta_mean: [0_f64, 0_f64, 0_f64],
            beta_var: [0_f64, 0_f64, 0_f64],
            model_mean: vec![0.0; n_features],
            model_var: vec![0.0; n_features],
            log_post_mean: 0_f64,
            log_post_var: 0_f64,
            post_mean: 0_f64,
            post_var: 0_f64,
            p_sig1: vec![0.0; n_features],
            p_sig2: vec![0.0; n_features],
            p_sig0: vec![0.0; n_features],
            log_post_trace: Vec::new(),
            beta_trace: Vec::new(),
            model_trace: Vec::new(),
            keep_trace: keep_trace,        }
    }

    fn update(&mut self, beta: &[f64; 3], model: &Vec<i32>, log_post: f64) {
        self.beta_mean = [self.beta_mean[0] + beta[0], self.beta_mean[1] + beta[1], self.beta_mean[2] + beta[2]];
        self.beta_var = [self.beta_var[0] + beta[0].powf(2.), self.beta_var[1] + beta[1].powf(2.), self.beta_var[2] + beta[2].powf(2.)];

        let model_f64 = vector_i32_to_f64(model);
        self.model_mean = sum_vectors(&self.model_mean, &model_f64);
        let model_squared = vector_squared(&model_f64);
        self.model_var = sum_vectors(&self.model_var, &model_squared);
        self.log_post_mean += log_post;
        self.log_post_var += log_post.powf(2.);
        self.post_mean += log_post.exp();
        self.post_var += (2.0 * log_post).exp();
        let sig1 = model.iter().map(|&m| (m as f64) * (m as f64 + 1.0) / 2.0).collect();
        let sig2 = model.iter().map(|&m| (m as f64) * (m as f64 - 1.0) / 2.0).collect();
        self.p_sig1 = sum_vectors(&self.p_sig1, &sig1);
        self.p_sig2 = sum_vectors(&self.p_sig2, &sig2);

        if self.keep_trace {
            self.beta_trace.push(beta.clone());
            self.model_trace.push(model.clone());
            self.log_post_trace.push(log_post);
        }
    }

    fn finalize(&mut self, n_iter: usize, n_burn: usize) {
        let n_mean = ((self.n_features + 3) * (n_iter - n_burn)) as f64;
        // self.beta_mean = divide_vector_by_number(&self.beta_mean, n_mean);
        self.beta_mean = self.beta_mean.iter().map(|v| v / n_mean).collect::<Vec<f64>>().try_into().unwrap();
        // self.beta_mean = [self.beta_mean[0] / n_mean, self.beta_mean[1] / n_mean, self.beta_mean[2] / n_mean];
        self.beta_var = self.beta_var.iter().zip(self.beta_mean.iter()).map(
            |(v, m)| (v - m.powf(2.) * n_mean) /(n_mean - 1.0)
        ).collect::<Vec<f64>>().try_into().unwrap();

        self.model_mean = divide_vector_by_number(&self.model_mean, n_mean);
        self.model_var = self.model_var.iter().zip(self.model_mean.iter()).map(
            |(v, m)| (v - m.powf(2.) * n_mean) /(n_mean - 1.0)
        ).collect();

        self.log_post_mean /= n_mean;
        // self.log_post_var = (self.log_post_var - self.log_post_mean.powf(2.) * n_mean) / (n_mean - 1.0);
        self.log_post_var = (self.log_post_var - self.log_post_mean.powf(2.) * n_mean) / (n_mean - 1.0);

        self.post_mean /= n_mean;
        self.post_var = (self.post_var - self.post_mean.powf(2.) * n_mean) / (n_mean - 1.0);

        self.p_sig1 = divide_vector_by_number(&self.p_sig1, n_mean);
        self.p_sig2 = divide_vector_by_number(&self.p_sig2, n_mean);
        self.p_sig0 = self.p_sig1.iter()
            .zip(self.p_sig2.iter())
            .map(|(&x, &y)| 1.0 - x - y).collect();
    }

    pub fn save_trace(&self, bp: &BayesPred, outdir: &str) {
        // Saving SIG probabilities
        // let p_sig0: Vec<_> = self.p_sig1.iter()
        //     .zip(self.p_sig2.iter())
        //     .map(|(&x, &y)| 1.0 - x - y).collect();
        let c1 = Column::new("SIG1".into(), &self.p_sig1);
        let c2 = Column::new("SIG0".into(), &self.p_sig0);
        let c3 = Column::new("SIG2".into(), &self.p_sig2);
        let mut p_mean = DataFrame::new(vec![bp.features.clone(), c1, c2, c3]).unwrap();

        let p_mean_path = outdir.to_owned() + "P_mean.tsv";
        let mut file = File::create(&p_mean_path).expect("could not create file");
        CsvWriter::new(&mut file)
            .include_header(true)
            .with_separator(b'\t')
            .finish(&mut p_mean).expect("couldn't write to file");

        // Saving means and variances
        //betas dataframe
        let c1 = Column::new("parameter".into(), &vec!["a", "b", "c"]);
        let c2 = Column::new("mean".into(), &self.beta_mean);
        let c3 = Column::new("variance".into(), &self.beta_var);
        let df_beta = DataFrame::new(vec![c1, c2, c3]).unwrap();

        //models dataframe
        let c2 = Column::new("mean".into(), &self.model_mean);
        let c3 = Column::new("variance".into(), &self.model_var);
        let mut df_model = DataFrame::new(vec![bp.features.clone(), c2, c3]).unwrap();
        let _ = df_model.rename("msp_name", "parameter".into());

        //posterior dataframe
        let df_post: DataFrame = df!("parameter" => ["logPost", "Post"],
                                    "mean" => [self.log_post_mean, self.post_mean],
                                    "variance" => [self.log_post_var, self.post_var]).unwrap();

        let mut df = df_beta.vstack(&df_model)
            .expect("Couldn't merge DataFrames")
            .vstack(&df_post)
            .expect("Couldn't merge DataFrames");

        let means_vars_path = outdir.to_owned() + "means_vars.tsv";
        let mut file = File::create(&means_vars_path).expect("could not create file");
        CsvWriter::new(&mut file)
            .include_header(true)
            .with_separator(b'\t')
            .finish(&mut df).expect("couldn't write to file");
    
        // Saving trace
        if self.keep_trace {
            //saving log Posterior trace
            let mut df: DataFrame = df!("logPost" => &self.log_post_trace).expect("Couldn't create DataFrame");
            let log_post_path = outdir.to_owned() + "logPost.tsv";
            let mut file = File::create(&log_post_path).expect("could not create file");
            CsvWriter::new(&mut file)
                .include_header(true)
                .with_separator(b'\t')
                .finish(&mut df).expect("couldn't write to file");

            // saving betas trace
            let beta_trace = transpose_vec_arr(&self.beta_trace);
            let names= vec!["a", "b", "c"];
            let mut df = Vec::new();
            for (n, col) in names.into_iter().zip(beta_trace.iter()) {
                let c = Column::new(n.into(), &col);
                df.push(c);
            }
            let mut df = DataFrame::new(df).unwrap();
            let beta_trace_path = outdir.to_owned() + "betas.tsv";
            let mut file = File::create(&beta_trace_path).expect("could not create file");
            CsvWriter::new(&mut file)
                .include_header(true)
                .with_separator(b'\t')
                .finish(&mut df).expect("couldn't write to file");

            //saving model trace
            let model_trace = transpose_vec_vec_i32(&self.model_trace);
            let names: Vec<_> = bp.features.str().into_iter().collect();
            let mut df = Vec::new();
            for (n, col) in names[0].into_iter().zip(model_trace.iter()) {
                let c = Column::new(n.unwrap().into(), &col);
                df.push(c);
            }
            let mut df = DataFrame::new(df).unwrap();
            let model_trace_path = outdir.to_owned() + "models.tsv";
            let mut file = File::create(&model_trace_path).expect("could not create file");
            CsvWriter::new(&mut file)
                .include_header(true)
                .with_separator(b'\t')
                .finish(&mut df).expect("couldn't write to file");

        }
    }
}


//-----------------------------------------------------------------------------
// Principal MCMC code

pub fn run_mcmc(bp: &BayesPred, n_iter: usize, n_burn: usize, keep_trace: bool, seed: u64) -> ResultMCMC {

    // initializing variables
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vals = [1, 0, -1];
    let nvals = 3;
    let mut model: Vec<_> = (0..bp.n_features).map(|_| *vals.choose(&mut rng).unwrap()).collect();
    let mut new_model_k: i32;
    // let mut model = Array1::from_vec(model);
    let mut beta = [1_f64, -1_f64, 1_f64];
    let nbeta: usize = 3;
    let mut z = bp.biomarker_components(&model);
    let mut log_post: f64 = 0.;// log_posterior(&bp, &beta, &z);

    // initialize results stack
    let mut res_mcmc = ResultMCMC::new(bp.n_features, keep_trace);

    // running mcmc iterations
    let solver = BrentOpt::new(-10_f64.powf(4_f64), 10_f64.powf(4_f64));
    for n in 0..n_iter {
        // sampling continuous parameters one by one
        for i in 0..nbeta {
            let cost = NegLogPostToMinimize {
                bp: bp,
                i: i,
                beta: &beta,
                z: &z,
            };
        
            let res = Executor::new(cost, solver.clone())
                .configure(|state| state.max_iters(100))
                .run()
                .unwrap();

            beta[i] = res.state.param.unwrap();
            let scale_i = bp.sigma_i(&beta, &z, i);

            if beta[i].abs() / scale_i > 10_f64 {
                // beta[i] = Normal::new(beta[i], scale_i).unwrap().sample(&mut rng)
                beta[i] = random_normal(beta[i], scale_i, &mut rng)
            } else {
                match i {
                    0_usize => beta[i] = truncnorm_pos(beta[i], scale_i, &mut rng),
                    1_usize => beta[i] = truncnorm_neg(beta[i], scale_i, &mut rng),
                    // 2_usize => beta[i] = Normal::new(beta[i], scale_i).unwrap().sample(&mut rng),
                    2_usize => beta[i] = random_normal(beta[i], scale_i, &mut rng),
                    _ => beta[i] = 0.0,
                }
            }

            //update values
            log_post = bp.log_posterior(&beta, &z);
            if n > n_burn {
                res_mcmc.update(&beta, &model, log_post)
            };
        }

        // sampling discrete features one by one
        for k in 0..bp.n_features {
            let new_model_k = (&model[k] + [2, 3].choose(&mut rng).unwrap()) % &nvals - 1;
            let z_new = bp.update_biomarker_components(&z, k, model[k], new_model_k);
            let log_post_new = bp.log_posterior(&beta, &z_new);
            let diff_log_post = log_post_new - log_post;
            let u: f64 = rng.gen_range(0.0..1.0);
            if diff_log_post > 0.0 || u < diff_log_post.exp() {
                model[k] = new_model_k;
                log_post = log_post_new.clone();
                z = z_new.clone();
            }
            if n > n_burn {
                res_mcmc.update(&beta, &model, log_post)
            };
        }
    }

    res_mcmc.finalize(n_iter, n_burn);
    res_mcmc
}
