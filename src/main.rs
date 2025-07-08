use log::{info, error};
use gpredomics::{param, run_ga, run_cv, run_beam, run_mcmc};
use flexi_logger::{Logger, WriteMode, FileSpec};
use chrono::Local;
use gpredomics::experiment::Experiment;
use std::env;

use std::thread;
use signal_hook::{iterator::Signals, consts::signal::*};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use signal_hook::flag;

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
    let version = env!("CARGO_PKG_VERSION");

    let param = param::get("param.yaml".to_string()).unwrap();

    if param.general.overfit_penalty != 0.0 {
        error!("overfit_penalty parameter is deprecated, please set it to 0.");
        panic!("overfit_penalty parameter is deprecated, please set it to 0.");
    }

    let args: Vec<String> = env::args().collect();
    
    // CLI first
    let load_path = if let Some(load_arg) = parse_load_argument(&args) {
        Some(load_arg)
    } else {
        None
    };
    
    // Load experiment if args, else launch new experiment
    if let Some(path) = load_path {
        match Experiment::load_auto(&path) {
            Ok(experiment) => {
                info!("Loading experiment from: {}", path);
                experiment.display_results();
                return; // Sortie directe après affichage
            }
            Err(e) => {
                error!("Loading failed: {}", e);
                eprintln!("Error while loading experiment: {}", e);
                return;
            }
        }
    }

    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();

    // Initialize the logger
    let logger = if param.general.log_base.len()>0 {
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
            .unwrap_or_else(|e| panic!("Logger initialization failed with {}", e))
    }
    else {
        Logger::try_with_str(&param.general.log_level) // Set the log level (e.g., "info")
            .unwrap()
            .write_mode(WriteMode::BufferAndFlush) // Use buffering for smoother output
            .start() // Start the logger
            .unwrap_or_else(|e| panic!("Logger initialization failed with {}", e))
    };

    info!("GPREDOMICS v{}", version);
    info!("Loading param.yaml");
    info!("\x1b[2;97m{:?}\x1b[0m", &param);

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    let running_clone_for_signal = Arc::clone(&running);
    let mut signals = Signals::new(&[SIGTERM, SIGHUP]).expect("Failed to set up signal handler");

    // Register signal handler with original Arc
    flag::register(SIGHUP, Arc::clone(&running)).expect("Failed to register SIGHUP handler");
    info!("Signal handler thread started. Send `kill -1 {}` to stop the application.", std::process::id());
    info!("Signal registration state is {}", running.load(Ordering::Relaxed));

    let thread_param = param.clone();
    let start = std::time::Instant::now();

    let handle = thread::spawn(move || {
        if thread_param.general.cv {
            run_cv(&thread_param, running_clone)
        } else {
            match thread_param.general.algo.as_str() {
                "ga" => run_ga(&thread_param, running_clone),
                "beam" => run_beam(&thread_param, running_clone),
                "mcmc" => run_mcmc(&thread_param, running_clone),
                other => {
                    panic!("ERROR! No such algorithm {}", other);
                }
            }
        }
    });

    let (collection, data_train, data_test) = handle.join().expect("Thread panicked!");
    let exec_time = start.elapsed().as_secs_f64();
    
    let exp = Experiment {
        id: "Test".to_string(),
        timestamp: timestamp,
        algorithm: param.general.algo.clone(),

        train_data: data_train,
        test_data: Some(data_test),

        final_population: collection.last().cloned(),
        collection: if param.general.keep_trace { Some(collection) } else { None },
        
        importance_collection: None,
        execution_time: exec_time,

        cv_data: None,
        parameters: param.clone()
    };

    if param.general.save_exp != "".to_string() {
        exp.save_auto(&param.general.save_exp).expect("Error while exporting experiment");
    }
    

    // Main thread now monitors the signal
    let _signal_thread = thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGTERM | SIGHUP => {
                    info!("Received signal: {}", sig);
                    running_clone_for_signal.store(false, Ordering::Relaxed);
                    break;
                }
                _ => {}
            }
        }
    });

    logger.flush();
}

// Load previous experiment
fn parse_load_argument(args: &[String]) -> Option<String> {
    for i in 0..args.len() {
        match args[i].as_str() {
            "--load" | "-l" => {
                if i + 1 < args.len() {
                    return Some(args[i + 1].clone());
                }
            }
            arg if arg.starts_with("--load=") => {
                return Some(arg.strip_prefix("--load=").unwrap().to_string());
            }
            _ => {}
        }
    }
    None
}
