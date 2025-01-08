use log::{debug, info, warn, error};
use gpredomics::{param, basic_test, random_run, ga_run, gacv_run};
use std::process;
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
    let param= param::get("param.yaml".to_string()).unwrap();
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

    info!("param.yaml");
    info!("{:?}", &param);

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    let mut signals = Signals::new(&[SIGTERM, SIGHUP]).expect("Failed to set up signal handler");

    // Register signal handler in this thread
    flag::register(SIGHUP, Arc::clone(&running_clone)).expect("Failed to register SIGHUP handler");
    info!("Signal handler thread started. Send `kill -1 {}` to stop the application.", std::process::id());
    info!("Signal registration state is {}",running_clone.load(Ordering::Relaxed));

    let sub_thread = thread::spawn(move || {
        match param.general.algo.as_str() {
            "basic" => basic_test(&param),
            "random" => random_run(&param),
            "ga"|"ga2"|"ga_no_overfit"|"ga2_no_overfit" => { ga_run(&param, running); },
            "ga+cv"|"ga2+cv" => { gacv_run(&param, running); },
            other => { error!("ERROR! No such algorithm {}", other);  process::exit(1); }
        } 
    });

    // Main thread now monitors the signal
    let signal_thread = thread::spawn(move || {
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

