#![allow(non_snake_case)]

pub mod beam;
pub mod data;
mod utils;
pub mod individual;
pub mod param;
pub mod population;
mod ga;
mod cv;
pub mod gpu;
mod bayesian_mcmc;

pub use beam::run_beam;
use data::Data;
use individual::Individual;
use population::Population;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
use param::Param;

use log::{debug, info, warn, error};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

use polars::prelude::*;
use DataType::UInt32;
// use serde_derive::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use std::ops::Not;


/// a very basic use
pub fn basic_test(param: &Param) {
    info!("                          BASIC TEST\n-----------------------------------------------------");
    // define some data
    let mut my_data = Data::new();
    my_data.X.insert((0,0), 0.1);
    my_data.X.insert((0,1), 0.2);
    my_data.X.insert((0,2), 0.3);
    my_data.X.insert((2,0), 0.9);
    my_data.X.insert((2,1), 0.8);
    my_data.X.insert((2,2), 0.7);    
    my_data.feature_len = 3;
    my_data.sample_len = 3;    my_data.samples = string_vec! ["a","b","c"];
    my_data.features = string_vec! ["msp1","msp2","msp3"];
    my_data.y = vec! [0,1,1];
    my_data.feature_len = 3;
    my_data.sample_len = 3;
    info!("{:?}", my_data);

    // create a model
    let mut my_individual = Individual::new();
    my_individual.features.insert(0, 1);
    my_individual.features.insert(2, -1);
    my_individual.compute_hash();
    info!("my individual: {:?}",my_individual.features);
    info!("my individual hash: {}",my_individual.hash);
    info!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual AUC: {:?}",my_individual.compute_auc(&my_data));

    let mut my_individual2 = Individual::new();
    my_individual2.features.insert(0, 1);
    my_individual2.features.insert(1, -1);
    my_individual2.compute_hash();
    info!("my individual2 {:?}",my_individual2.features);
    info!("my individual2 hash: {}",my_individual2.hash);
    info!("my individual2 evaluation: {:?}",my_individual2.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual2 AUC: {:?}",my_individual2.compute_auc(&my_data));


    let mut data2=Data::new();
    let _ = data2.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let parent1 = Individual::random(&data2, &mut rng);
    let parent2 = Individual::random(&data2, &mut rng);
    let mut parents=population::Population::new();
    parents.individuals.push(parent1);
    parents.individuals.push(parent2);

    let children_number = param.ga.population_size as usize-parents.individuals.len();
    let mut children = ga::cross_over(&parents, data2.feature_len, children_number, &mut rng);
    for (i,individual) in parents.individuals.iter().enumerate() { info!("Parent #{}: {:?}",i,individual); }
    for (i,individual) in children.individuals.iter().enumerate() { info!("Child #{}: {:?}",i,individual); }
    let feature_selection:Vec<usize> = (0..data2.feature_len).collect();
    ga::mutate(&mut children, param, &feature_selection, &mut rng);
    children.compute_hash();
    let clone_number = children.remove_clone();
    if clone_number>0 { warn!("There were {} clone(s)",clone_number); }
    for (i,individual) in children.individuals.iter().enumerate() { info!("Mutated Child #{}: {:?}",i,individual); }    

}

/// a more elaborate use with random models
pub fn random_run(param: &Param) {
    info!("                          RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for _ in 0..1000 {
        let mut my_individual = Individual::random(&my_data, &mut rng);


        let auc = my_individual.compute_auc(&my_data);
        if auc>auc_max {auc_max=auc;best_individual=my_individual;}
    }
    warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// a more elaborate use with random models
pub fn gpu_random_run(param: &Param) {
    info!("                          GPU RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("Selecting features...");
    my_data.select_features(param);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    //let mut auc_max = 0.0;
    //let mut best_individual: Individual = Individual::new();
    let nb_individuals: usize = 1000;


    let assay = gpu::GpuAssay::new(&my_data.X, &my_data.feature_selection, my_data.sample_len, nb_individuals as usize);

    let mut individuals:Vec<Individual> = (0..nb_individuals).map(|i| {Individual::random(&my_data, &mut rng)}).collect();
    individuals = individuals.into_iter()
        .map(|i| {
            // we filter random features for selected features, not efficient but we do not care
            let mut new = Individual::new();
            new.features = i.features.into_iter()
                    .filter(|(i,_f)| {my_data.feature_selection.contains(i)})
                    .collect();
            new.data_type = *vec![individual::RAW_TYPE, individual::LOG_TYPE, individual::PREVALENCE_TYPE].choose(&mut rng).unwrap();
            new.language = *vec![individual::TERNARY_LANG, individual::RATIO_LANG].choose(&mut rng).unwrap();
            new
        }).collect();
        

    for _ in 0..1000 {
        
        //println!("First individual {:?}", individuals[0]);

        let scores = assay.compute_scores(&individuals, param.general.data_type_epsilon as f32);

        println!("First scores {:?}", &scores[0..4]);
        //println!("Last scores {:?}", &scores[scores.len()-4..scores.len()]);

    }

    //let auc = my_individual.compute_auc(&my_data);
    //warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}

/// the Genetic Algorithm test
pub fn run_ga(param: &Param, running: Arc<AtomicBool>) -> (Vec<Population>,Data,Data) {
    info!("                          GA TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    my_data.set_classes(param.data.classes.clone());
    info!("{:?}", my_data); 
    let has_auc = true;

    let (mut run_test_data, run_data): (Option<Data>,Option<Data>) = if param.general.overfit_penalty>0.0 {
        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
        let mut cv=cv::CV::new(&my_data, param.cv.fold_number, &mut rng);
        (Some(cv.folds.remove(0)),Some(cv.datasets.remove(0)))
    } else { (None,None) };

    let mut populations = if let Some(mut this_data)=run_data {
        ga::ga(&mut this_data, &mut run_test_data, &param, running)
    } else {
        ga::ga(&mut my_data, &mut run_test_data, &param, running)
    };
    
    //if param.general.overfit_penalty == 0.0 
    //    {   match param.general.fit.to_lowercase().as_str() {
    //            "auc" => {
    //                info!("Fitting by AUC with k penalty {}",param.general.k_penalty);
    //                ga::ga(&mut my_data,&param,running, 
    //                    |p: &mut Population,d: &Data| { 
    //                    p.auc_fit(d, param.general.k_penalty, param.general.thread_number); 
    //                } )
    //            },
    //            "specificity"|"sensitivity" => {
    //                info!("Fitting by objective {} with k penalty {}",param.general.fit,param.general.k_penalty);
    //                let fpr_penalty = if param.general.fit.to_lowercase().as_str()=="specificity" {1.0} else {param.general.fr_penalty}; 
    //                let fnr_penalty = if param.general.fit.to_lowercase().as_str()=="sensitivity" {1.0} else {param.general.fr_penalty}; 
    //                info!("FPR penalty {}  |  FNR penalty {}",fpr_penalty,fnr_penalty);
    //                has_auc = false;
    //                ga::ga(&mut my_data,&param,running, 
    //                    |p: &mut Population,d: &Data| { 
    //                        p.objective_fit(d, fpr_penalty,fnr_penalty,param.general.k_penalty,
    //                            param.general.thread_number); 
    //                    } )
    //            },
    //            other => { error!("Unrecognised fit {}",other); panic!("Unrecognised fit {}",other)}
    //        }
    //    } else {
    //        let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    //        let cv=cv::CV::new(&my_data, param.cv.fold_number, &mut rng);
//
    //        match param.general.fit.to_lowercase().as_str() {
    //            "auc" => {
    //                info!("Fitting by AUC with k penalty {} and overfit penalty {}",param.general.k_penalty, param.general.overfit_penalty);
    //                ga::ga(&mut cv.datasets[0].clone(),&param,running, 
    //                |p: &mut Population,d: &Data| { 
    //                    p.auc_nooverfit_fit(d,
    //                        param.general.k_penalty, &cv.folds[0].clone(), param.general.overfit_penalty,
    //                        param.general.thread_number); } )
    //            },
    //            "specificity"|"sensitivity" => {
    //                info!("Fitting by objective {} with k penalty {} and overfit penalty {}",param.general.fit,param.general.k_penalty, param.general.overfit_penalty);
    //                let fpr_penalty = if param.general.fit.to_lowercase().as_str()=="specificity" {1.0} else {param.general.fr_penalty}; 
    //                let fnr_penalty = if param.general.fit.to_lowercase().as_str()=="sensitivity" {1.0} else {param.general.fr_penalty}; 
    //                info!("FPR penalty {}  |  FNR penalty {}",fpr_penalty,fnr_penalty);
    //                has_auc = false;
    //                ga::ga(&mut cv.datasets[0].clone(),&param,running, 
    //                    |p: &mut Population,d: &Data| { 
    //                        p.objective_nooverfit_fit(d, fpr_penalty,fnr_penalty,param.general.k_penalty,
    //                            &cv.folds[0].clone(), param.general.overfit_penalty, param.general.thread_number); 
    //                    } )
    //            },
    //            other => { error!("Unrecognised fit {}",other); panic!("Unrecognised fit {}",other)}
    //        }
    //    };
    let generations = populations.len();
    let population= &mut populations[generations-1];

    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        test_data.set_classes(param.data.classes.clone());
        
        debug!("Length of population {}",population.individuals.len());

        let nb_model_to_test = if param.general.nb_best_model_to_test>0 {param.general.nb_best_model_to_test as usize} else {population.individuals.len()};
        debug!("Testing {} models",nb_model_to_test);

        // Prepare the evaluation pool
        let pool = ThreadPoolBuilder::new()
            .num_threads(param.general.thread_number)
            .build()
            .expect("Failed to build thread pool");

        // Compute the final metrics
        pool.install(|| {
            let results: Vec<String> = population.individuals[..nb_model_to_test]
                .par_iter_mut()
                .enumerate()
                .map(|(i, individual)| {
                    let mut auc = individual.auc;
                    let test_auc = individual.compute_auc(&test_data);

                    if has_auc {
                        individual.auc = auc;
                        let (threshold, accuracy, sensitivity, specificity) =
                            individual.compute_threshold_and_metrics(&my_data);
                        individual.threshold = threshold;
                        individual.accuracy = accuracy;
                        individual.sensitivity = sensitivity;
                        individual.specificity = specificity;
                    } else {
                        auc = individual.compute_auc(&my_data);
                    }

                    let (tp, fp, tn, fn_count) = individual.calculate_confusion_matrix(&test_data);
                    format!(
                        "Model #{} [k={}] [gen:{}] threshold {:.3} : AUC {:.3}/{:.3} |  accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3} \n   < {:?} >",
                        i + 1,
                        individual.k,
                        individual.epoch,
                        individual.threshold,
                        test_auc,
                        auc,
                        (tp + tn) as f64 / (fp + tp + fn_count + tn) as f64,
                        individual.accuracy,
                        if tp + fn_count > 0 {
                            tp as f64 / (tp + fn_count) as f64
                        } else {
                            0.0
                        },
                        individual.sensitivity,
                        if tn + fp > 0 {
                            tn as f64 / (tn + fp) as f64
                        } else {
                            0.0
                        },
                        individual.specificity,
                        individual
                    )
                })
                .collect();

            // Output results in order
            for result in results {
                info!("{}", result);
            }
        });
    }
    else {
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            if has_auc {
                (individual.threshold, individual.accuracy, individual.sensitivity, individual.specificity) = 
                    individual.compute_threshold_and_metrics(&test_data);
            } else {
                individual.compute_auc(&my_data);
            }
            info!("Model #{} [k={}] [gen:{}]: train AUC {:.3}",i+1,individual.k,individual.epoch,individual.auc);
        }    
    }

    (populations,my_data,test_data) 

}


/// the Genetic Algorithm test with Crossval (not useful but test CV)
pub fn gacv_run(param: &Param, running: Arc<AtomicBool>) -> (cv::CV,Data,Data) {
    info!("                          GA CV TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    
    let _ = my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    my_data.set_classes(param.data.classes.clone());
    info!("{:?}", my_data); 

    let mut crossval = cv::CV::new(&my_data, 10, &mut rng);

    let mut cv_param = param.clone();
    cv_param.general.thread_number = 1;

    let results=crossval.pass(|d: &mut Data,p: &Param,r: Arc<AtomicBool>| 
        { ga::ga(d,&mut None,p,r) }, &cv_param, param.general.thread_number, running);
    
    let mut test_data=Data::new();
    if param.data.Xtest.len()>0 {
        let _ = test_data.load_data(&param.data.Xtest, &param.data.ytest);
        test_data.set_classes(param.data.classes.clone());
        
        for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            let holdout_auc=best_model.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = 
                best_model.compute_threshold_and_metrics(&test_data);
            info!("Model #{} [gen:{}] [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
                        i+1,best_model.epoch,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
            info!("Features importance on train+test... ");
            info!("{}",
                best_model.compute_oob_feature_importance(&my_data, param.ga.feature_importance_permutations,&mut rng)
                    .into_iter()
                    .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
            info!("Features importance on holdout... ");
            info!("{}",
                best_model.compute_oob_feature_importance(&test_data, param.ga.feature_importance_permutations,&mut rng)
                    .into_iter()
                    .map(|feature_importance| { format!("[{:.4}]",feature_importance) })
                    .collect::<Vec<String>>()
                    .join(" ")
            );
        }    
    }
    else {
        for (i,(best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            warn!("Model #{} [gen:{}] [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.epoch,best_model.k,train_auc,test_auc,best_model);
        }    
    }

    (crossval,my_data,test_data) 

}

//Code de Vadim


// #[derive(Serialize, Deserialize, Debug)]
// struct ParamsMCMC {
//     path_to_x: String,
//     path_to_y: String,
//     outdir: String,
//     n_iter: usize,
//     n_burn: usize,
//     lmbd: f64,
//     nmin: u32,
//     language: String
// }

struct SequentialBackwardSelection {
    x: DataFrame, 
    y: DataFrame, 
    outdir: String,
    n_iter: usize,
    n_burn: usize,
    lmbd: f64,
    nmin: u32,
    nsigs: f64
}
 
impl SequentialBackwardSelection {
    fn run_sbs(&self, seed: u64) -> DataFrame {
        let mut x_train = self.x.clone();
        let nmax = self.x.height() as u32;
        let mut post_mean = Vec::new();
        let mut msp_names = Vec::new();
        let mut msp_to_drop: String;
        for n in (self.nmin..=nmax).rev() {
            println!("\nn = {}, (#MSPs, #samples) = {:?}", n, x_train.shape());

            let now = Instant::now();
            let bp = bayesian_mcmc::BayesPred::new(&x_train, &self.y, self.lmbd); //initializing mcmc data structure        
            let res = bayesian_mcmc::run_mcmc(&bp, self.n_iter, self.n_burn, false, seed);

            // let outdir_n = self.outdir.clone().to_owned() + &format!("n{}/", n);
            // fs::create_dir_all(&outdir_n).expect("Failed to create output directory");
            // let p_mean = res.save_trace(&bp,&outdir_n);
            let elapsed = now.elapsed();
            println!("Elapsed: {:.2?}", elapsed);

            post_mean.push(res.post_mean);

            // index of feature with greatest SIG0
            let idx: Option<usize> = res.p_sig0
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index);
            let Some(idx) = idx else {panic!("Note Some()")};
            let AnyValue::String(v1) = x_train["msp_name"].get(idx).unwrap() else {panic!("Not AnyValue()")};
            msp_to_drop = v1.into();

            // let max_row = p_mean
            //     .lazy()
            //     .filter(col("SIG0").eq(max("SIG0")))
            //     .collect()
            //     .unwrap();
            // let v = max_row.column("msp_name").unwrap().get(0).unwrap();
            // let Some(v1) = v.get_str() else {panic!("Not Some()")};
            // msp_to_drop = v1.into();
            let s = Series::new("a".into(), [msp_to_drop.clone()]);
            let mask = is_in(x_train["msp_name"].as_series().unwrap(), &s).unwrap().not();
            x_train = x_train.filter(&mask).unwrap();
            println!("dropping {:?}", msp_to_drop);
            msp_names.push(msp_to_drop)
        }

        let nn: Vec<u32> = (self.nmin..=nmax).rev().collect::<Vec<_>>();
        let log_post_mean: Vec<f64> = post_mean.iter().map(|v| v.log10()).collect();
        let log_evidence: Vec<f64> = (self.nmin..=nmax).rev().
            zip(log_post_mean.iter()).
            map(|v| v.1 - (v.0 as f64) * self.nsigs.log10()).collect();

        let c1 = Column::new("nfeat".into(), nn);
        let c2 = Column::new("Posterior mean".into(), &post_mean);
        let c3 = Column::new("Log Posterior mean".into(), &log_post_mean);
        let c4 = Column::new("Log Evidence".into(), &log_evidence);
        let c5 = Column::new("MSP to drop".into(), &msp_names);
        let df_post = DataFrame::new(vec![c1, c2, c3, c4, c5]).unwrap();
        df_post
    }

    fn best_full_trace(&self, df_post: &DataFrame, seed: u64) {
        //find best model
        let best_evidence = df_post.clone()
            .lazy()
            .filter(col("Log Evidence").eq(max("Log Evidence")))
            .collect()
            .unwrap();

        let n_best = if let AnyValue::UInt32(b) = best_evidence.column("nfeat").unwrap().get(0).unwrap() {b} else {panic!("OOPS!!!")};

        //compile list of msps to drop
        let df_tmp =df_post
            .clone()
            .lazy()
            .filter(col("nfeat").gt(n_best))
            .collect()
            .unwrap();

        let msps_to_drop = df_tmp
            .column("MSP to drop")
            .unwrap()
            .as_materialized_series()
            .clone();

        let mask = is_in(self.x["msp_name"].as_series().unwrap(), &msps_to_drop).unwrap().not();
        let x_train = self.x.filter(&mask).unwrap();

        println!("\nBest model: n = {}, x_train.shape = {:?}", n_best, x_train.shape());
        let outdir_best = self.outdir.clone().to_owned() + "best_full_trace/";
        fs::create_dir_all(&outdir_best).expect("Failed to create output directory");

        let now = Instant::now();
        let bp = bayesian_mcmc::BayesPred::new(&x_train, &self.y, self.lmbd); //initializing mcmc data structure        
        let res = bayesian_mcmc::run_mcmc(&bp, self.n_iter, self.n_burn, true, seed);
        res.save_trace(&bp,&outdir_best);
        let elapsed = now.elapsed();
        println!("Elapsed: {:.2?}", elapsed);
    }
}

pub fn mcmc(param: &Param, running: Arc<AtomicBool>) {
    //Reading parameters from a json
    // let filepath = "./src/params.json";
    // let params_str = fs::read_to_string(filepath).expect("Unable to read a file");
    // let params: ParamsMCMC = serde_json::from_str(&params_str).expect("Couldn't parse JSON");
    // println!("{:?}", params);

    // Preparing training data
    let parse = CsvParseOptions::default().with_separator(b'\t'); // set tub as delimiter

    // read abundance data from path
    let mut x = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(parse.clone())
        .try_into_reader_with_file_path(Some(param.data.X.clone().into()))
        .unwrap()
        .finish()
        .unwrap();

    let old_name = &x.get_column_names_owned()[0]; // rename first column
    let _ = x.rename(old_name, "msp_name".into());

    // read response variable values
    let myschema = Schema::from_iter(vec![
        Field::new("Sample".into(), DataType::String),
        Field::new("status".into(), DataType::Float64)
    ]);
    let y0 = CsvReadOptions::default()
        .with_skip_lines(1)
        .with_has_header(false)
        .with_schema(Arc::new(myschema).into())
        .with_parse_options(parse)
        .try_into_reader_with_file_path(Some(param.data.y.clone().into()))
        .unwrap()
        .finish()
        .unwrap();

    let y = DataFrame::new(vec![y0["Sample"].clone(), y0.column("status").unwrap().cast(&UInt32).unwrap()]).unwrap();

    // ------------------------------------------------------------------------
    //Sequential backward elimination
    let sbs = SequentialBackwardSelection {
        x: x.clone(), 
        y: y.clone(),
        outdir: "".to_string(), 
        n_iter: param.mcmc.n_iter, 
        n_burn: param.mcmc.n_burn, 
        lmbd: param.mcmc.lmbd,
        nmin: param.mcmc.nmin,
        nsigs: 3.0
    };
    let mut df_post = sbs.run_sbs(param.general.seed);
    println!("Posterior summary: {:?}", df_post);
    
    let df_post_path = "Post_mean.tsv";
    let mut file = fs::File::create(&df_post_path).expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_post).expect("couldn't write to file");

    sbs.best_full_trace(&df_post, param.general.seed)
}

