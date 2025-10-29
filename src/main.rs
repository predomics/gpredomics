use log::{info, error};
use gpredomics::{param, run_ga, run_beam, run_mcmc};
use flexi_logger::{Logger, WriteMode, FileSpec};
use chrono::Local;
use gpredomics::experiment::Experiment;
use clap::Parser;
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

    // if param.cv.overfit_penalty != 0.0 {
    //     error!("overfit_penalty parameter is deprecated, please set it to 0.");
    //     panic!("overfit_penalty parameter is deprecated, please set it to 0.");
    // }

    let args = Cli::parse();

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

    let param = param::get("param.yaml".to_string()).unwrap();

    if let Some(experiment_path) = args.load {
        match Experiment::load_auto(&experiment_path) {
            Ok(mut experiment) => {
                info!("Loading experiment from: {}", experiment_path);
                
                if args.evaluate {
                    // Evaluation mode
                    let x_path = args.x_test.expect("X test path required for evaluation");
                    let y_path = args.y_test.expect("Y test path required for evaluation");
                    
                    info!("Evaluating on new dataset: {} | {}", x_path, y_path);
                    
                    experiment.evaluate_on_new_dataset(&x_path, &y_path);
                } else {
                    // Display mode
                    experiment.display_results();
                }
                return;
            }
            Err(e) => {
                error!("Failed to load experiment: {}", e);
                return;
            }
        }
    }

    info!("GPREDOMICS v{}", version);
    info!("Loading param.yaml");
    info!("\x1b[2;97m{:?}\x1b[0m", &param);

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    let running_clone_for_signal = Arc::clone(&running);
    let mut signals = Signals::new(&[SIGTERM, SIGHUP, SIGINT]).expect("Failed to set up signal handler");

    rayon::ThreadPoolBuilder::new()
    .num_threads(param.general.thread_number as usize)
    .build_global()
    .expect("Rayon global pool already set");

    // Main thread now monitors the signal
    let mut soft_kill = false;
    let _signal_thread = thread::spawn(move || {
        for sig in signals.forever() {
            if soft_kill == false {
                match sig {
                    SIGTERM | SIGHUP | SIGINT => {
                        info!("Received signal: {}", sig);
                        info!("Algorithm will be stopped at next iteration and results returned... You can force the kill by pressing Ctrl+C again.");
                        soft_kill = true;
                        running_clone_for_signal.store(false, Ordering::Relaxed);
                    }
                    _ => {}
                }
            } else {
                std::process::exit(1);
            }
        }
    });

    // Register signal handler with original Arc
    flag::register(SIGHUP, Arc::clone(&running)).expect("Failed to register SIGHUP handler");
    info!("Signal handler thread started. Send `kill -1 {}` to stop the application.", std::process::id());
    info!("Signal registration state is {}", running.load(Ordering::Relaxed));

    let thread_param = param.clone();
    let handle = thread::spawn(move || {
        match thread_param.general.algo.as_str() {
            "ga" => run_ga(&thread_param, running_clone),
            "beam" => run_beam(&thread_param, running_clone),
            "mcmc" => run_mcmc(&thread_param, running_clone),
            other => {
                panic!("ERROR! No such algorithm {}", other);
            }
        }
    });

    let mut exp = handle.join().expect("Thread panicked!");

    if param.importance.compute_importance {
        info!("Computing feature importance...");
        let start_importance = std::time::Instant::now();
        
        exp.compute_importance();
        
        let importance_time = start_importance.elapsed().as_secs_f64();
        info!("Importance calculation completed in {:.2}s", importance_time);
    } else {
        info!("Skipping importance calculation (disabled in parameters)");
    }

    // Voting
    if param.voting.vote {
        exp.compute_voting();
    } else {
        info!("Voting stage ignored (disabled in parameters)");
    }

    if param.general.save_exp != "".to_string() {
        info!("Saving experiment...");
        exp.save_auto(format!("{}_{}", timestamp, &param.general.save_exp)).expect("Error while exporting experiment");
    }

    logger.flush();
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Load experiment from file
    #[arg(short, long)]
    load: Option<String>,
    
    /// Evaluate loaded experiment on new dataset
    #[arg(long, requires = "load")]
    evaluate: bool,
    
    /// X test data path (required if --evaluate is used)
    #[arg(long, required_if_eq("evaluate", "true"))]
    x_test: Option<String>,
    
    /// y test data path (required if --evaluate is used)
    #[arg(long, required_if_eq("evaluate", "true"))]
    y_test: Option<String>,
}
