//! Command-line interface module for WhisperNow
//! 
//! This module provides a comprehensive CLI interface for the WhisperNow application,
//! supporting both interactive and batch processing modes with full configuration control.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::{sleep, timeout};
use tokio::signal;
use chrono::Utc;

use crate::{
    WhisperApp, AppState, AppEvent, AppConfig, ModelConfig, AudioConfig, VadConfig,
    RecordingState, TranscriptionState, AudioRecording, TranscriptionResult,
    copy_to_clipboard, format_duration, format_file_size, app_name, app_version
};

/// WhisperNow CLI Application
#[derive(Parser)]
#[command(name = app_name())]
#[command(version = app_version())]
#[command(about = "High-performance voice transcription tool")]
#[command(long_about = "WhisperNow is a fast, native voice transcription application using Whisper models")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Whisper model to use
    #[arg(short, long, default_value = "distil-small.en")]
    pub model: String,

    /// Compute device (cpu, cuda, metal)
    #[arg(short, long, default_value = "cpu")]
    pub device: String,

    /// Audio sample rate in Hz
    #[arg(short = 'r', long, default_value = "16000")]
    pub sample_rate: u32,

    /// Number of audio channels
    #[arg(short, long, default_value = "1")]
    pub channels: u16,

    /// Recording directory
    #[arg(short = 'o', long)]
    pub output_dir: Option<PathBuf>,

    /// Enable Voice Activity Detection
    #[arg(long, default_value = "true")]
    pub vad: bool,

    /// Disable clipboard integration
    #[arg(long)]
    pub no_clipboard: bool,

    /// Verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (suppress non-essential output)
    #[arg(short, long)]
    pub quiet: bool,

    /// Configuration file path
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Beam size for transcription
    #[arg(short, long, default_value = "2")]
    pub beam_size: usize,
}

/// Available CLI subcommands
#[derive(Subcommand)]
pub enum Commands {
    /// Interactive recording mode (default)
    Record {
        /// Automatically start recording on launch
        #[arg(short, long)]
        auto_start: bool,
        
        /// Maximum recording duration in seconds
        #[arg(short, long)]
        max_duration: Option<u64>,
        
        /// Number of recordings to make before exiting
        #[arg(short, long)]
        count: Option<usize>,
    },

    /// Transcribe existing audio files
    Transcribe {
        /// Audio file paths to transcribe
        files: Vec<PathBuf>,
        
        /// Output format
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
        
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// List available models
    Models {
        /// Show detailed model information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Download models
    Download {
        /// Model name to download
        model: String,
        
        /// Download directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Test audio devices
    Test {
        /// Test recording for specified duration
        #[arg(short, long, default_value = "3")]
        duration: u64,
    },
}

/// Configuration subcommands
#[derive(Subcommand)]
pub enum ConfigAction {
    /// Show current configuration
    Show,
    /// Edit configuration interactively
    Edit,
    /// Reset to defaults
    Reset,
    /// Export configuration
    Export { path: PathBuf },
    /// Import configuration
    Import { path: PathBuf },
}

/// Output format options
#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Text,
    Json,
    Srt,
    Vtt,
}

/// CLI application state
pub struct CliApp {
    app: WhisperApp,
    args: Cli,
    event_receiver: tokio::sync::broadcast::Receiver<AppEvent>,
}

/// Terminal colors and formatting
mod colors {
    pub const RED: &str = "\x1b[91m";
    pub const GREEN: &str = "\x1b[92m";
    pub const YELLOW: &str = "\x1b[93m";
    pub const BLUE: &str = "\x1b[94m";
    pub const MAGENTA: &str = "\x1b[95m";
    pub const CYAN: &str = "\x1b[96m";
    pub const WHITE: &str = "\x1b[97m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const RESET: &str = "\x1b[0m";
    
    pub fn red(text: &str) -> String {
        format!("{}{}{}", RED, text, RESET)
    }
    
    pub fn green(text: &str) -> String {
        format!("{}{}{}", GREEN, text, RESET)
    }
    
    pub fn yellow(text: &str) -> String {
        format!("{}{}{}", YELLOW, text, RESET)
    }
    
    pub fn blue(text: &str) -> String {
        format!("{}{}{}", BLUE, text, RESET)
    }
    
    pub fn bold(text: &str) -> String {
        format!("{}{}{}", BOLD, text, RESET)
    }
    
    pub fn dim(text: &str) -> String {
        format!("{}{}{}", DIM, text, RESET)
    }
}

impl CliApp {
    /// Create new CLI application
    pub async fn new(args: Cli) -> Result<Self> {
        // Build configuration from CLI args
        let config = Self::build_config(&args).await?;
        
        // Create WhisperApp with config
        let app = WhisperApp::new(config).await
            .context("Failed to create WhisperApp")?;
        
        let event_receiver = app.state.subscribe_events();
        
        Ok(Self {
            app,
            args,
            event_receiver,
        })
    }

    /// Build app configuration from CLI arguments
    async fn build_config(args: &Cli) -> Result<AppConfig> {
        let mut config = if let Some(ref config_path) = args.config {
            // Load from config file
            let config_manager = crate::config::ConfigManager::new()?;
            config_manager.import_config(config_path).await
                .context("Failed to load configuration file")?
        } else {
            AppConfig::default()
        };

        // Override with CLI arguments
        config.model.name = args.model.clone();
        config.model.device = args.device.clone();
        config.model.beam_size = args.beam_size;
        config.audio.sample_rate = args.sample_rate;
        config.audio.channels = args.channels;
        config.vad.enabled = args.vad;
        config.app.auto_copy_to_clipboard = !args.no_clipboard;

        if let Some(ref output_dir) = args.output_dir {
            config.audio.recording_path = output_dir.clone();
        }

        Ok(config)
    }

    /// Run the CLI application
    pub async fn run(&mut self) -> Result<()> {
        // Initialize components
        self.app.init_audio().await
            .context("Failed to initialize audio")?;
        self.app.init_whisper().await
            .context("Failed to initialize Whisper")?;

        // Handle subcommands
        match &self.args.command {
            Some(Commands::Record { auto_start, max_duration, count }) => {
                self.run_interactive_mode(*auto_start, *max_duration, *count).await
            },
            Some(Commands::Transcribe { files, format, output }) => {
                self.run_batch_transcribe(files, format, output).await
            },
            Some(Commands::Models { detailed }) => {
                self.list_models(*detailed).await
            },
            Some(Commands::Download { model, output }) => {
                self.download_model(model, output).await
            },
            Some(Commands::Config { action }) => {
                self.handle_config_command(action).await
            },
            Some(Commands::Test { duration }) => {
                self.test_audio_devices(*duration).await
            },
            None => {
                // Default: interactive recording mode
                self.run_interactive_mode(false, None, None).await
            }
        }
    }

    /// Run interactive recording mode
    async fn run_interactive_mode(
        &mut self,
        auto_start: bool,
        max_duration: Option<u64>,
        count: Option<usize>,
    ) -> Result<()> {
        if !self.args.quiet {
            self.print_welcome();
        }

        // Load model
        let config = self.app.state.get_config();
        if let Some(ref engine) = self.app.whisper_engine {
            self.print_status(&format!("Loading model '{}'...", config.model.name));
            let start_time = std::time::Instant::now();
            
            engine.load_model(&config.model).await
                .context("Failed to load Whisper model")?;
            
            let elapsed = start_time.elapsed();
            self.print_success(&format!("Model loaded in {}", format_duration(elapsed.as_secs_f64())));
            
            // Start processing loop
            let engine_clone = engine.clone();
            tokio::spawn(async move {
                if let Err(e) = engine_clone.start_processing_loop().await {
                    eprintln!("Transcription processing error: {}", e);
                }
            });
        }

        let mut recordings_made = 0;
        let target_count = count.unwrap_or(usize::MAX);

        // Auto-start if requested
        if auto_start {
            self.start_recording().await?;
        }

        loop {
            if recordings_made >= target_count {
                break;
            }

            match self.app.state.get_recording_state() {
                RecordingState::Idle => {
                    if !auto_start || recordings_made > 0 {
                        self.prompt_for_recording().await?;
                    }
                },
                RecordingState::Recording => {
                    self.handle_recording_session(max_duration).await?;
                    recordings_made += 1;
                },
                RecordingState::Stopping => {
                    sleep(Duration::from_millis(100)).await;
                },
                RecordingState::Error(ref error) => {
                    self.print_error(&format!("Recording error: {}", error));
                    break;
                }
            }

            // Handle events
            self.handle_events().await;
        }

        Ok(())
    }

    /// Print welcome message
    fn print_welcome(&self) {
        println!("{}", colors::bold(&format!("🎙️  {} v{}", app_name(), app_version())));
        println!("{}", colors::dim("High-performance voice transcription"));
        println!();
        
        let config = self.app.state.get_config();
        println!("Configuration:");
        println!("  Model: {}", colors::cyan(&config.model.name));
        println!("  Device: {}", colors::cyan(&config.model.device));
        println!("  Sample Rate: {} Hz", config.audio.sample_rate);
        println!("  Channels: {}", config.audio.channels);
        println!("  VAD: {}", if config.vad.enabled { colors::green("enabled") } else { colors::red("disabled") });
        println!();
    }

    /// Prompt user for recording action
    async fn prompt_for_recording(&self) -> Result<()> {
        print!("{}", colors::yellow("Press Enter to start recording, 'q' to quit: "));
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        match input.trim().to_lowercase().as_str() {
            "q" | "quit" | "exit" => {
                println!("Goodbye!");
                std::process::exit(0);
            },
            _ => {
                self.start_recording().await?;
            }
        }

        Ok(())
    }

    /// Start recording
    async fn start_recording(&self) -> Result<()> {
        if let Some(ref recorder) = self.app.audio_recorder {
            self.print_status("Starting recording...");
            recorder.start_recording().await
                .context("Failed to start recording")?;
            self.print_info(&colors::red("🔴 Recording...\n⚠️ Press Enter to stop"));
        }
        Ok(())
    }

    /// Handle active recording session
    async fn handle_recording_session(&self, max_duration: Option<u64>) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        loop {
            // Check for stop input
            if self.check_for_stop_input().await? {
                break;
            }

            // Check max duration
            if let Some(max_dur) = max_duration {
                if start_time.elapsed().as_secs() >= max_dur {
                    self.print_info("Maximum duration reached, stopping recording");
                    break;
                }
            }

            // Show recording progress
            if !self.args.quiet {
                let elapsed = start_time.elapsed().as_secs_f64();
                print!("\r🔴 Recording... {:.1}s", elapsed);
                io::stdout().flush()?;
            }

            sleep(Duration::from_millis(100)).await;
        }

        // Stop recording
        self.stop_recording().await?;
        Ok(())
    }

    /// Check for user input to stop recording
    async fn check_for_stop_input(&self) -> Result<bool> {
        // Non-blocking input check using tokio
        let input_task = tokio::task::spawn_blocking(|| {
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)
        });

        match timeout(Duration::from_millis(10), input_task).await {
            Ok(Ok(_)) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Stop recording
    async fn stop_recording(&self) -> Result<()> {
        if let Some(ref recorder) = self.app.audio_recorder {
            self.print_status("Stopping recording...");
            
            if let Some(recording) = recorder.stop_recording().await
                .context("Failed to stop recording")? 
            {
                self.print_success(&format!(
                    "Recording saved: {} ({})", 
                    recording.file_path.file_name().unwrap().to_string_lossy(),
                    format_duration(recording.duration_seconds)
                ));

                // Queue for transcription
                if let Some(ref engine) = self.app.whisper_engine {
                    engine.queue_transcription(recording).await
                        .context("Failed to queue transcription")?;
                }
            }
        }
        Ok(())
    }

    /// Handle application events
    async fn handle_events(&mut self) {
        while let Ok(event) = self.event_receiver.try_recv() {
            match event {
                AppEvent::TranscriptionStarted(_) => {
                    self.print_status("Transcribing...");
                },
                AppEvent::TranscriptionCompleted(result) => {
                    self.handle_transcription_result(result).await;
                },
                AppEvent::TranscriptionError(_, error) => {
                    self.print_error(&format!("Transcription failed: {}", error));
                },
                AppEvent::ModelLoaded(model_name) => {
                    self.print_success(&format!("Model '{}' loaded", model_name));
                },
                _ => {}
            }
        }
    }

    /// Handle completed transcription
    async fn handle_transcription_result(&self, result: TranscriptionResult) {
        println!();
        self.print_success(&format!("Transcription completed in {}", 
                                   format_duration(result.processing_time_ms as f64 / 1000.0)));
        
        // Display transcription
        self.display_transcription(&result);
        
        // Copy to clipboard if enabled
        if self.app.state.get_config().app.auto_copy_to_clipboard {
            if let Err(e) = copy_to_clipboard(&result.text).await {
                self.print_warning(&format!("Failed to copy to clipboard: {}", e));
            } else {
                
                self.print_info("📋 Copied to clipboard");
            }
        }
        
        println!();
    }

    /// Display transcription result
    fn display_transcription(&self, result: &TranscriptionResult) {
        let border = colors::green(&format!("+{}+", "-".repeat(50)));
        println!("{}", border);
        
        // Metadata
        if self.args.verbose > 0 {
            println!("{}", colors::dim(&format!("Model: {} | Confidence: {:.1}% | Language: {}", 
                     result.model_name, result.confidence * 100.0, result.language)));
        }
        
        // Text content
        println!("{}", result.text);
        println!("{}", border);
    }

    /// Run batch transcription mode
    async fn run_batch_transcribe(
        &mut self,
        files: &[PathBuf],
        format: &OutputFormat,
        output: &Option<PathBuf>,
    ) -> Result<()> {
        if files.is_empty() {
            return Err(anyhow::anyhow!("No files specified for transcription"));
        }

        // Load model
        let config = self.app.state.get_config();
        if let Some(ref engine) = self.app.whisper_engine {
            self.print_status(&format!("Loading model '{}'...", config.model.name));
            engine.load_model(&config.model).await
                .context("Failed to load model")?;
            
            // Start processing loop
            let engine_clone = engine.clone();
            tokio::spawn(async move {
                if let Err(e) = engine_clone.start_processing_loop().await {
                    eprintln!("Processing error: {}", e);
                }
            });
        }

        let mut results = Vec::new();

        for file_path in files {
            if !file_path.exists() {
                self.print_warning(&format!("File not found: {}", file_path.display()));
                continue;
            }

            self.print_status(&format!("Transcribing: {}", file_path.display()));

            // Create a dummy recording for the file
            let recording = AudioRecording {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                file_path: file_path.clone(),
                duration_seconds: 0.0, // Will be calculated during processing
                file_size_bytes: std::fs::metadata(file_path)?.len(),
                sample_rate: config.audio.sample_rate,
                channels: config.audio.channels,
            };

            // Queue for transcription and wait for result
            if let Some(ref engine) = self.app.whisper_engine {
                engine.queue_transcription(recording.clone()).await?;
                
                // Wait for transcription result
                let mut event_receiver = self.app.state.subscribe_events();
                while let Ok(event) = event_receiver.recv().await {
                    if let AppEvent::TranscriptionCompleted(result) = event {
                        if result.recording_id == recording.id {
                            results.push(result);
                            self.print_success(&format!("Completed: {}", file_path.display()));
                            break;
                        }
                    }
                }
            }
        }

        // Output results
        self.output_batch_results(&results, format, output).await?;

        Ok(())
    }

    /// Output batch transcription results
    async fn output_batch_results(
        &self,
        results: &[TranscriptionResult],
        format: &OutputFormat,
        output: &Option<PathBuf>,
    ) -> Result<()> {
        let content = match format {
            OutputFormat::Text => {
                results.iter()
                    .map(|r| r.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n\n")
            },
            OutputFormat::Json => {
                serde_json::to_string_pretty(results)?
            },
            OutputFormat::Srt => {
                self.format_srt(results)
            },
            OutputFormat::Vtt => {
                self.format_vtt(results)
            },
        };

        if let Some(output_path) = output {
            tokio::fs::write(output_path, &content).await
                .context("Failed to write output file")?;
            self.print_success(&format!("Results written to: {}", output_path.display()));
        } else {
            println!("{}", content);
        }

        Ok(())
    }

    /// Format results as SRT subtitles
    fn format_srt(&self, results: &[TranscriptionResult]) -> String {
        let mut output = String::new();
        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!("{}\n", i + 1));
            output.push_str("00:00:00,000 --> 00:00:10,000\n"); // Placeholder timing
            output.push_str(&result.text);
            output.push_str("\n\n");
        }
        output
    }

    /// Format results as WebVTT
    fn format_vtt(&self, results: &[TranscriptionResult]) -> String {
        let mut output = String::from("WEBVTT\n\n");
        for result in results {
            output.push_str("00:00:00.000 --> 00:00:10.000\n");
            output.push_str(&result.text);
            output.push_str("\n\n");
        }
        output
    }

    /// List available models
    async fn list_models(&self, detailed: bool) -> Result<()> {
        let models = crate::whisper::WhisperEngine::list_available_models();
        
        println!("{}", colors::bold("Available Whisper Models:"));
        println!();
        
        for model in models {
            if detailed {
                println!("{}", colors::cyan(&format!("📦 {}", model.name)));
                println!("   Description: {}", model.description);
                println!("   Size: {}", model.size);
                println!();
            } else {
                println!("  {} - {} ({})", 
                        colors::cyan(&model.name), 
                        model.description, 
                        model.size);
            }
        }

        Ok(())
    }

    /// Download model
    async fn download_model(&mut self, model: &str, _output: &Option<PathBuf>) -> Result<()> {
        self.print_status(&format!("Downloading model '{}'...", model));
        
        // This would integrate with the actual model download functionality
        // For now, just show a placeholder message
        self.print_info("Model download functionality not yet implemented");
        
        Ok(())
    }

    /// Handle configuration commands
    async fn handle_config_command(&self, action: &ConfigAction) -> Result<()> {
        let config_manager = crate::config::ConfigManager::new()?;
        
        match action {
            ConfigAction::Show => {
                let config = self.app.state.get_config();
                println!("{}", colors::bold("Current Configuration:"));
                println!("{}", serde_json::to_string_pretty(&config)?);
            },
            ConfigAction::Edit => {
                self.print_info("Interactive configuration editing not yet implemented");
            },
            ConfigAction::Reset => {
                config_manager.reset_to_defaults().await?;
                self.print_success("Configuration reset to defaults");
            },
            ConfigAction::Export { path } => {
                let config = self.app.state.get_config();
                config_manager.export_config(&config, path).await?;
                self.print_success(&format!("Configuration exported to: {}", path.display()));
            },
            ConfigAction::Import { path } => {
                let config = config_manager.import_config(path).await?;
                self.app.state.update_config(config)?;
                self.print_success(&format!("Configuration imported from: {}", path.display()));
            },
        }
        
        Ok(())
    }

    /// Test audio devices
    async fn test_audio_devices(&mut self, duration: u64) -> Result<()> {
        self.print_status("Testing audio devices...");
        
        if let Some(ref recorder) = self.app.audio_recorder {
            let devices = recorder.list_input_devices()?;
            
            println!("{}", colors::bold("Available Audio Devices:"));
            for (i, device) in devices.iter().enumerate() {
                let marker = if device.is_default { " (default)" } else { "" };
                println!("  {}: {}{}", i + 1, colors::cyan(&device.name), marker);
                
                if self.args.verbose > 0 {
                    for config in &device.supported_configs {
                        println!("    Channels: {}, Sample Rate: {}-{} Hz", 
                                config.channels,
                                config.sample_rate_range.0,
                                config.sample_rate_range.1);
                    }
                }
            }
            
            println!();
            self.print_status(&format!("Recording test for {} seconds...", duration));
            
            // Start test recording
            recorder.start_recording().await?;
            sleep(Duration::from_secs(duration)).await;
            
            if let Some(recording) = recorder.stop_recording().await? {
                self.print_success(&format!(
                    "Test recording successful: {} ({}, {})",
                    format_duration(recording.duration_seconds),
                    format_file_size(recording.file_size_bytes),
                    recording.file_path.display()
                ));
                
                // Clean up test file
                if let Err(e) = std::fs::remove_file(&recording.file_path) {
                    self.print_warning(&format!("Failed to clean up test file: {}", e));
                }
            }
        } else {
            return Err(anyhow::anyhow!("Audio recorder not initialized"));
        }
        
        Ok(())
    }

    // Print utility methods
    fn print_status(&self, message: &str) {
        if !self.args.quiet {
            println!("{} {}", colors::blue("ℹ"), message);
        }
    }

    fn print_success(&self, message: &str) {
        if !self.args.quiet {
            println!("{} {}", colors::green("✓"), message);
        }
    }

    fn print_error(&self, message: &str) {
        eprintln!("{} {}", colors::red("✗"), message);
    }

    fn print_info(&self, message: &str) {
        if !self.args.quiet {
            println!("{} {}", colors::cyan("→"), message);
        }
    }

    fn print_warning(&self, message: &str) {
        if !self.args.quiet {
            println!("{} {}", colors::yellow("⚠"), message);
        }
    }
}

/// Run the CLI application
pub async fn run_cli_app(app: WhisperApp) -> Result<()> {
    // Parse command line arguments
    let args = Cli::parse();
    
    // Initialize logging based on verbosity
    let log_level = match args.verbose {
        0 => "warn",
        1 => "info", 
        2 => "debug",
        _ => "trace",
    };
    
    if !args.quiet {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::new(log_level))
            .init();
    }

    // Create CLI app and run
    let mut cli_app = CliApp::new(args).await
        .context("Failed to initialize CLI application")?;
    
    // Handle Ctrl+C gracefully
    let state_clone = cli_app.app.state.clone();
    tokio::spawn(async move {
        if let Ok(()) = signal::ctrl_c().await {
            println!("\n{}", colors::yellow("Received interrupt signal, shutting down..."));
            let _ = state_clone.shutdown().await;
            std::process::exit(0);
        }
    });
    
    cli_app.run().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_cli_arg_parsing() {
        let args = Cli::try_parse_from([
            "whisper-now",
            "--model", "large-v3",
            "--device", "cuda", 
            "--sample-rate", "22050",
            "--channels", "2",
            "--verbose"
        ]).unwrap();
        
        assert_eq!(args.model, "large-v3");
        assert_eq!(args.device, "cuda");
        assert_eq!(args.sample_rate, 22050);
        assert_eq!(args.channels, 2);
        assert_eq!(args.verbose, 1);
    }
    
    #[test]
    fn test_output_format_parsing() {
        use clap::ValueEnum;
        assert!(OutputFormat::from_str("text", true).is_ok());
        assert!(OutputFormat::from_str("json", true).is_ok());
        assert!(OutputFormat::from_str("srt", true).is_ok());
        assert!(OutputFormat::from_str("vtt", true).is_ok());
    }
    
    #[tokio::test]
    async fn test_config_building() {
        let args = Cli {
            command: None,
            model: "small.en".to_string(),
            device: "cpu".to_string(),
            sample_rate: 16000,
            channels: 1,
            output_dir: None,
            vad: true,
            no_clipboard: false,
            verbose: 0,
            quiet: false,
            config: None,
            beam_size: 2,
        };
        
        let config = CliApp::build_config(&args).await.unwrap();
        assert_eq!(config.model.name, "small.en");
        assert_eq!(config.audio.sample_rate, 16000);
        assert!(config.vad.enabled);
        assert!(config.app.auto_copy_to_clipboard);
    }
}