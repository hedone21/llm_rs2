use clap::Parser;
use llm_manager::config::Config;

#[derive(Parser)]
#[command(
    about = "LLM Resource Manager — monitors system resources and emits signals to LLM engine"
)]
struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long, default_value = "/etc/llm-manager/config.toml")]
    config: std::path::PathBuf,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let config = if args.config.exists() {
        log::info!("Loading config from {}", args.config.display());
        Config::from_file(&args.config)?
    } else {
        log::info!("Config not found, using defaults");
        Config::default()
    };

    log::info!(
        "LLM Manager starting (poll_interval={}ms)",
        config.monitor.poll_interval_ms
    );

    // TODO: Initialize collectors, policy engine, emitter, and run main loop
    log::warn!("Main loop not yet implemented");
    println!("llm_manager: architecture established, awaiting collector/emitter implementation");

    Ok(())
}
