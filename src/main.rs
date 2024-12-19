mod data;
mod utils;
mod individual;
mod param;
mod population;
mod ga;
mod cv;
mod hyper;

use data::Data;
use individual::Individual;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
use param::Param;
use std::process;
use flexi_logger::{Logger, WriteMode, FileSpec};
use chrono::Local;
use log::{info, warn, error};

/// a very basic use
fn basic_test(param: &Param) {
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
    info!("my individual: {:?}",my_individual.features);
    info!("my individual evaluation: {:?}",my_individual.evaluate(&my_data));
    // shoud display 1.0 (the AUC is 1.0)
    info!("my individual AUC: {:?}",my_individual.compute_auc(&my_data));


    let mut data2=Data::new();
    data2.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    let parent1 = Individual::random(&data2, &mut rng);
    let parent2 = Individual::random(&data2, &mut rng);
    let mut parents=population::Population::new();
    parents.individuals.push(parent1);
    parents.individuals.push(parent2);
    let mut children = ga::cross_over(&parents, &param, data2.feature_len, &mut rng);
    for (i,individual) in parents.individuals.iter().enumerate() { info!("Parent #{}: {:?}",i,individual); }
    for (i,individual) in children.individuals.iter().enumerate() { info!("Child #{}: {:?}",i,individual); }
    let feature_selection:Vec<usize> = (0..data2.feature_len).collect();
    ga::mutate(&mut children, param, &feature_selection, &mut rng);
    for (i,individual) in children.individuals.iter().enumerate() { info!("Mutated Child #{}: {:?}",i,individual); }    

}

/// a more elaborate use with random models
fn random_run(param: &Param) {
    info!("                          RANDOM TEST\n-----------------------------------------------------");
    // use some data
    let mut my_data = Data::new();
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut auc_max = 0.0;
    let mut best_individual: Individual = Individual::new();
    for i in 0..10000 {
        let mut my_individual = Individual::random(&my_data, &mut rng);

        let auc = my_individual.compute_auc(&my_data);
        if (auc>auc_max) {auc_max=auc;best_individual=my_individual;}
    }
    warn!("AUC max: {} model: {:?}",auc_max, best_individual.features);
}


/// the Genetic Algorithm test
fn ga_run(param: &Param) {
    info!("                          GA TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("{:?}", my_data); 
    
    let mut populations = ga::ga(&mut my_data,&param);

    let mut population=populations.pop().unwrap();

    if param.data.Xtest.len()>0 {
        let mut test_data=Data::new();
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let auc=individual.auc;
            let test_auc=individual.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = individual.compute_threshold_and_metrics(&test_data);
            info!("Model #{} [k={}] [gen:{}]: train AUC {}  | test AUC {} | threshold {} | accuracy {} | sensitivity {} | specificity {} | {:?}",
                        i+1,individual.k,individual.n,auc,test_auc,threshold,accuracy,sensitivity,specificity,individual);
        }    
    }
    else {
        for (i,individual) in population.individuals[..10].iter_mut().enumerate() {
            let auc=individual.auc;
            info!("Model #{} [k={}] [gen:{}]: train AUC {}",i+1,individual.k,individual.n,auc);
        }    
    }


}


/// the Genetic Algorithm test with Crossval (not useful but test CV)
fn gacv_run(param: &Param) {
    info!("                          GA CV TEST\n-----------------------------------------------------");
    let mut my_data = Data::new();
    let mut rng = ChaCha8Rng::seed_from_u64(param.general.seed);
    
    my_data.load_data(param.data.X.as_str(),param.data.y.as_str());
    info!("{:?}", my_data); 

    let mut crossval = cv::CV::new(&my_data, 10, &mut rng);
    let results=crossval.pass(ga::ga, &param, param.general.thread_number);
    
    if param.data.Xtest.len()>0 {
        let mut test_data=Data::new();
        test_data.load_data(&param.data.Xtest, &param.data.ytest);
        
        for (i,(mut best_model, train_auc, test_auc)) in results.into_iter().enumerate() {
            let holdout_auc=best_model.compute_auc(&test_data);
            let (threshold, accuracy, sensitivity, specificity) = best_model.compute_threshold_and_metrics(&test_data);
            info!("Model #{} [gen:{}] [k={}]: train AUC {:.3}  | test AUC {:.3} | holdout AUC {:.3} | threshold {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3} | {:?}",
                        i+1,best_model.n,best_model.k,train_auc,test_auc,holdout_auc,threshold,accuracy,sensitivity,specificity,best_model);
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
            warn!("Model #{} [gen:{}] [k={}]: train AUC {:.3} | test AUC {:.3} | {:?}",i+1,best_model.n,best_model.k,train_auc,test_auc,best_model);
        }    
    }


}

/// custom format for logs
fn custom_format(
    w: &mut dyn std::io::Write,
    now: &mut flexi_logger::DeferredNow,
    record: &log::Record,
) -> std::io::Result<()> {
    write!(
        w,
        "{} [{}] {}",
        now.now().format("%Y-%m-%d %H:%M:%S"), // Format timestamp
        record.level(),
        record.args()
    )
}

fn main() {
    let param= param::get("param.yaml".to_string()).unwrap();
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    // Initialize the logger
    if param.general.log_base.len()>0 {
        Logger::try_with_str(&param.general.log_level) // Set log level (e.g., "info")
            .unwrap()
            .log_to_file(
                FileSpec::default()
                    .basename(&param.general.log_base) // Logs go into the "logs" directory
                    .suffix(&param.general.log_suffix)     // Use the ".log" file extension
                    .discriminant(&timestamp), // Add timestamp to each log file
            )
            .write_mode(WriteMode::BufferAndFlush) // Control file write buffering
            .format_for_files(custom_format) // Custom format for the log file
            .format_for_stderr(custom_format) // Same format for the console
            .start()
            .unwrap_or_else(|e| panic!("Logger initialization failed with {}", e));
    }
    else {
        Logger::try_with_str(&param.general.log_level) // Set the log level (e.g., "info")
            .unwrap()
            .write_mode(WriteMode::BufferAndFlush) // Use buffering for smoother output
            .start() // Start the logger
            .unwrap_or_else(|e| panic!("Logger initialization failed with {}", e));
    }

    info!("param.yaml");
    info!("{:?}", &param);

    match param.general.algo.as_str() {
        "basic" => basic_test(&param),
        "random" => random_run(&param),
        "ga" => ga_run(&param),
        "ga+cv" => gacv_run(&param),
        other => { error!("ERROR! No such algorithm {}", other);  process::exit(1); }
    }
    //basic_test();
}

