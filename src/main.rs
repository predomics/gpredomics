use log::{info, error};
use gpredomics::{param, run_ga, run_cv, run_beam, run_mcmc};
use flexi_logger::{Logger, WriteMode, FileSpec};
use chrono::Local;

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

    let param= param::get("param.yaml".to_string()).unwrap();

    if param.general.overfit_penalty != 0.0 {
        error!("overfit_penalty parameter is deprecated, please set it to 0.");
        panic!("overfit_penalty parameter is deprecated, please set it to 0.");
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
    let mut signals = Signals::new(&[SIGTERM, SIGHUP]).expect("Failed to set up signal handler");

    // Register signal handler in this thread
    flag::register(SIGHUP, Arc::clone(&running_clone)).expect("Failed to register SIGHUP handler");
    info!("Signal handler thread started. Send `kill -1 {}` to stop the application.", std::process::id());
    info!("Signal registration state is {}",running_clone.load(Ordering::Relaxed));

    let sub_thread = thread::spawn(move || {
        let _result = if param.general.cv {
            run_cv(&param, running)
        } else {
            match param.general.algo.as_str() {
                "ga" => run_ga(&param, running),
                "beam" => run_beam(&param, running),
                "mcmc"  => run_mcmc(&param, running),
                other => {
                    error!("ERROR! No such algorithm {}", other);
                    panic!("ERROR! No such algorithm {}", other);
                }
            }
        };
    });

    // Main thread now monitors the signal
    let _signal_thread = thread::spawn(move || {
        for sig in signals.forever() {
            match sig {
                SIGTERM | SIGHUP => {
                    info!("Received signal: {}", sig);
                    running_clone.store(false, Ordering::Relaxed);
                    break;
                }
                _ => {}
            }
        }
    });

    sub_thread.join().expect("Computation thread panicked");

    logger.flush();
}

