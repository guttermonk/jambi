//! Jambi - High-Performance Voice Transcription
//!
//! A blazing-fast voice transcription application built with Rust,
//! providing superior performance over traditional Python implementations.

use anyhow::{Context, Result};
use serde::{Serialize, Deserialize};

use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;

use std::time::Duration;
use tokio::time::timeout;

use tracing::{info, warn, error};

mod audio;
mod whisper;
mod config;
mod vosk_engine;

use audio::{AudioRecorder, AudioConfig};
use vosk_engine::{VoskEngine, VoskConfig, VoskModel};
// Whisper imports removed - using Vosk instead

/// Jambi command-line interface
#[derive(Parser)]
#[command(
    name = "jambi",
    version,
    about = "High-performance voice transcription tool",
    long_about = "A blazing-fast voice transcription application built with Rust, \
                  providing superior performance over traditional Python implementations."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Whisper model to use
    #[arg(short, long)]
    model: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Record audio and transcribe in real-time
    Record {
        /// Start recording immediately
        #[arg(long)]
        auto_start: bool,

        /// Maximum recording duration in seconds
        #[arg(long, default_value = "300")]
        max_duration: u64,

        /// Output directory for recordings
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Enable live transcription (show text as you speak)
        #[arg(long)]
        live: bool,
    },

    /// Transcribe existing audio files
    Transcribe {
        /// Audio files to transcribe
        files: Vec<PathBuf>,

        /// Output format (text, json, srt)
        #[arg(long, default_value = "text")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Copy result to clipboard
        #[arg(long)]
        clipboard: bool,
    },

    /// List available Whisper models
    Models {
        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },

    /// Test audio recording setup
    Test {
        /// Test duration in seconds
        #[arg(long, default_value = "5")]
        duration: u64,
    },

    /// Download and cache a model
    Download {
        /// Model name to download
        model: String,
    },

    /// Test clipboard functionality
    TestClipboard {
        /// Text to copy to clipboard
        #[arg(default_value = "Hello from Jambi! Clipboard test successful.")]
        text: String,
    },
}

/// Application state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub audio: AudioConfig,
    pub vosk: VoskConfig,
    #[serde(default = "default_auto_copy")]
    pub auto_copy: bool,
    #[serde(default = "default_keep_recordings")]
    pub keep_recordings: bool,
}

fn default_auto_copy() -> bool {
    true
}

fn default_keep_recordings() -> bool {
    false
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            vosk: VoskConfig::default(),
            auto_copy: true,
            keep_recordings: false,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose)?;

    if cli.verbose {
        info!("🎙️ Jambi - High-Performance Voice Transcription");
        info!("Version: {}", env!("CARGO_PKG_VERSION"));
    }

    // Load configuration
    let config = load_config(cli.config.as_deref())?;

    // Override model if specified
    let vosk_config = config.vosk.clone();
    if let Some(_model_name) = &cli.model {
        // For now, model override via CLI is not supported for Vosk
        // You can specify the model in the config file
        eprintln!("Note: Model override via CLI not yet implemented for Vosk. Use config file instead.");
    }

    // Execute command
    match cli.command {
        Some(Commands::Record { auto_start, max_duration, output, live }) => {
            let mut audio_config = config.audio.clone();
            if let Some(output_dir) = output {
                audio_config.output_dir = output_dir;
            }
            audio_config.max_duration = Some(max_duration);
            audio_config.verbose = cli.verbose;

            let mut vosk_config = vosk_config;
            vosk_config.verbose = cli.verbose;

            run_recording_session(audio_config, vosk_config, auto_start, live, config.auto_copy, cli.verbose).await
        }
        Some(Commands::Transcribe { files, format, output, clipboard }) => {
            run_transcription(files, vosk_config, format, output, clipboard || config.auto_copy).await
        }
        Some(Commands::Models { detailed }) => {
            list_models(detailed).await
        }
        Some(Commands::Test { duration }) => {
            let mut audio_config = config.audio;
            audio_config.verbose = cli.verbose;
            test_audio_setup(audio_config, duration).await
        }
        Some(Commands::Download { model }) => {
            download_vosk_model(&model).await
        }
        Some(Commands::TestClipboard { text }) => {
            test_clipboard_functionality(&text).await
        }
        _ => {
            // Default: interactive recording mode
            let auto_copy = config.auto_copy;
            run_interactive_mode(config, auto_copy, cli.verbose).await
        }
    }
}

/// Initialize logging based on verbosity level
fn init_logging(verbose: bool) -> Result<()> {
    let level = if verbose {
        "jambi=debug,info"
    } else {
        "jambi=warn,error"
    };

    tracing_subscriber::fmt()
        .with_env_filter(level)
        .with_target(false)
        .init();

    Ok(())
}

/// Load configuration from file or use defaults
fn load_config(config_path: Option<&std::path::Path>) -> Result<AppConfig> {
    if let Some(path) = config_path {
        // Load and parse the TOML file
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        let config: AppConfig = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        
        info!("Loaded configuration from: {}", path.display());
        Ok(config)
    } else {
        // Use defaults if no config file specified
        Ok(AppConfig::default())
    }
}



/// Run interactive recording mode
async fn run_interactive_mode(config: AppConfig, auto_copy: bool, verbose: bool) -> Result<()> {
    println!("🎙️  Jambi - Interactive Voice Transcription");
    println!("============================================");
    println!("Model: {}", config.vosk.model);
    println!("Sample Rate: {}Hz, Channels: {}", 
             config.audio.sample_rate, config.audio.channels);
    println!("Output Directory: {}", config.audio.output_dir.display());
    println!();

    let mut audio_config = config.audio.clone();
    audio_config.verbose = verbose;
    let mut recorder = AudioRecorder::new(audio_config)?;
    
    let mut vosk_config = config.vosk;
    vosk_config.verbose = verbose;
    let mut vosk_engine = VoskEngine::new(vosk_config)?;
    
    // Load the whisper model
    println!("🔄 Loading Vosk model...");
    if let Err(e) = vosk_engine.load_model().await {
        println!("⚠️  Model loading failed: {}. Will download on first use.", e);
    }

    // Test audio setup
    println!("🔍 Checking audio devices...");
    match recorder.list_devices() {
        Ok(devices) if !devices.is_empty() => {
            println!("✅ Found {} audio devices", devices.len());
            if devices.len() <= 3 {
                for device in &devices {
                    println!("   • {}", device);
                }
            }
        }
        Ok(_) => println!("⚠️  No audio devices found"),
        Err(e) => println!("⚠️  Error listing devices: {}", e),
    }
    println!();

    loop {
        println!("Choose an action:");
        println!("  [R] Record and transcribe");
        println!("  [T] Transcribe existing file");
        println!("  [M] Switch model");
        println!("  [Q] Quit");
        print!("Enter choice (R/T/M/Q): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        match input.trim().to_uppercase().as_str() {
            "R" | "RECORD" => {
                if let Err(e) = record_and_transcribe(&mut recorder, &mut vosk_engine, auto_copy).await {
                    error!("Recording failed: {}", e);
                    println!("❌ Recording failed: {}", e);
                }
            }
            "T" | "TRANSCRIBE" => {
                if let Err(e) = transcribe_file_interactive(&mut vosk_engine, auto_copy).await {
                    error!("Transcription failed: {}", e);
                    println!("❌ Transcription failed: {}", e);
                }
            }
            "M" | "MODEL" => {
                if let Err(e) = switch_model_interactive(&mut vosk_engine).await {
                    error!("Model switch failed: {}", e);
                    println!("❌ Model switch failed: {}", e);
                }
            }
            "Q" | "QUIT" => {
                println!("👋 Goodbye!");
                break;
            }
            _ => {
                println!("Invalid choice. Please enter R, T, M, or Q.");
            }
        }
        println!();
    }

    Ok(())
}

/// Record audio and transcribe it
async fn record_and_transcribe(
    recorder: &mut AudioRecorder,
    vosk_engine: &mut VoskEngine,
    auto_copy: bool,
) -> Result<()> {
    let recording_info = recorder.record_audio().await?;

    println!("✅ Recording completed:");
    println!("   Duration: {}", audio::format_duration(recording_info.duration));
    println!("   File: {}", recording_info.file_path.display());
    println!("   Size: {}", audio::format_file_size(recording_info.file_size));

    println!("🧠 Transcribing audio...");
    let result = vosk_engine.transcribe_file(&recording_info.file_path).await?;

    println!("📝 Transcription completed successfully");
    println!();
    println!("┌{}┐", "─".repeat(60));
    println!("│ {:^58} │", "TRANSCRIPTION RESULT");
    println!("├{}┤", "─".repeat(60));
    for line in result.text.lines() {
        println!("│ {:<58} │", truncate_string(line, 58));
    }
    println!("└{}┘", "─".repeat(60));
    println!();

    if auto_copy && !result.text.is_empty() {
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            copy_to_clipboard(&result.text)
        ).await {
            Ok(Ok(_)) => println!("\n📋 Copied to clipboard"),
            Ok(Err(e)) => {
                warn!("Failed to copy to clipboard: {}", e);
                println!("⚠️  Failed to copy to clipboard: {}", e);
                if e.to_string().contains("wl-copy") {
                    println!("   Install wl-clipboard: sudo apt install wl-clipboard");
                } else if e.to_string().contains("xclip") {
                    println!("   Install xclip: sudo apt install xclip");
                }
            }
            Err(_) => {
                warn!("Clipboard operation timed out");
                println!("⚠️  Clipboard operation timed out - clipboard tools may not be installed");
            }
        }
    }

    Ok(())
}

/// Transcribe an existing file interactively
async fn transcribe_file_interactive(vosk_engine: &mut VoskEngine, auto_copy: bool) -> Result<()> {
    print!("Enter path to audio file: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let file_path = input.trim();

    if file_path.is_empty() {
        println!("No file path provided");
        return Ok(());
    }

    let path = PathBuf::from(file_path);
    if !path.exists() {
        println!("❌ File not found: {}", path.display());
        return Ok(());
    }

    println!("🧠 Transcribing: {}", path.display());
    let result = vosk_engine.transcribe_file(&path).await?;

    println!("📝 Transcription completed successfully");
    println!();
    println!("{}", result.text);
    println!();

    if auto_copy && !result.text.is_empty() {
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            copy_to_clipboard(&result.text)
        ).await {
            Ok(Ok(_)) => println!("\n📋 Copied to clipboard"),
            Ok(Err(e)) => {
                warn!("Failed to copy to clipboard: {}", e);
                println!("⚠️  Failed to copy to clipboard: {}", e);
            }
            Err(_) => {
                warn!("Clipboard operation timed out");
                println!("⚠️  Clipboard operation timed out");
            }
        }
    }

    Ok(())
}

/// Switch model interactively
async fn switch_model_interactive(vosk_engine: &mut VoskEngine) -> Result<()> {
    println!("Available models:");
    let models = VoskEngine::available_models();
    for (i, model) in models.iter().enumerate() {
        println!("  {}. {} - {} ({}MB)", 
                 i + 1, model, model.description(), model.size_mb());
    }

    print!("Select model (1-{}): ", models.len());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if let Ok(choice) = input.trim().parse::<usize>() {
        if choice > 0 && choice <= models.len() {
            let selected_model = models[choice - 1];
            let mut config = vosk_engine.config.clone();
            config.model = selected_model;
            vosk_engine.config = config;
            vosk_engine.model = None; // Reset model to force reload
            
            // Reload the model
            println!("Loading new model: {}", selected_model);
            vosk_engine.load_model().await?;
            
            println!("✅ Switched to model: {}", selected_model);
            return Ok(());
        }
    }

    println!("❌ Invalid selection");
    Ok(())
}

/// Run a recording session
async fn run_recording_session(
    audio_config: AudioConfig,
    vosk_config: VoskConfig,
    auto_start: bool,
    live: bool,
    auto_copy: bool,
    verbose: bool,
) -> Result<()> {
    let mut recorder = AudioRecorder::new(audio_config)?;
    let mut vosk_engine = VoskEngine::new(vosk_config)?;
    
    // Load the vosk model
    if let Err(e) = vosk_engine.load_model().await {
        if verbose {
            eprintln!("⚠️  Model loading failed: {}. Will download on first use.", e);
        }
    }

    loop {
        if !auto_start {
            print!("🚦 Press Enter to start recording...");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
        }

        // Check if live transcription is enabled
        // Perform recording/transcription based on mode
        let result = if live {
            // Use live transcription mode
            println!("🎤 Live transcription enabled & now listening...");
            
            let live_result = vosk_engine.transcribe_live().await?;
            
            println!("\n📝 Final Transcription:");
            println!("{}", live_result.text);
            
            live_result
        } else {
            // Use the new record_audio function that handles Enter-to-stop
            let recording_info = recorder.record_audio().await?;

            println!("⏱️ Recording completed: {}", audio::format_duration(recording_info.duration));
            println!("🧠 Transcribing...");

            let transcription_result = vosk_engine.transcribe_file(&recording_info.file_path).await?;
            
            println!("📝 Transcription Result:");
            println!("{}", transcription_result.text);
            
            transcription_result
        };

        if auto_copy && !result.text.is_empty() {
            match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                copy_to_clipboard(&result.text)
            ).await {
                Ok(Ok(_)) => println!("\n📋 Copied to clipboard"),
                Ok(Err(e)) => {
                    warn!("Failed to copy to clipboard: {}", e);
                    println!("⚠️  Failed to copy to clipboard: {}", e);
                }
                Err(_) => {
                    warn!("Clipboard operation timed out");
                    println!("⚠️  Clipboard operation timed out");
                }
            }
        }

        // Ask user if they want to continue or quit
        println!("\n➡️ Press 'q' to quit or Enter to record again...");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        
        if input.trim().to_lowercase() == "q" {
            println!("👋 Goodbye!");
            break;
        }
        
        // Clear some space before next recording
        println!();
    }

    Ok(())
}

/// Run batch transcription
async fn run_transcription(
    files: Vec<PathBuf>,
    config: VoskConfig,
    format: String,
    output: Option<PathBuf>,
    clipboard: bool,
) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow::anyhow!("No input files specified"));
    }

    let mut vosk_engine = VoskEngine::new(config)?;
    
    // Load the vosk model
    if let Err(e) = vosk_engine.load_model().await {
        eprintln!("⚠️  Model loading failed: {}. Will download on first use.", e);
    }
    
    let mut results = Vec::new();

    for file in &files {
        println!("🧠 Transcribing: {}", file.display());
        
        match vosk_engine.transcribe_file(file).await {
            Ok(result) => {
                println!("✅ Transcription completed");
                results.push((file.clone(), result));
            }
            Err(e) => {
                error!("Failed to transcribe {}: {}", file.display(), e);
                println!("❌ Failed: {}", e);
            }
        }
    }

    if results.is_empty() {
        return Err(anyhow::anyhow!("No files were successfully transcribed"));
    }

    // Format output
    let output_text = match format.as_str() {
        "json" => format_as_json(&results)?,
        "srt" => format_as_srt(&results)?,
        _ => format_as_text(&results),
    };

    // Write output
    if let Some(output_file) = output {
        std::fs::write(&output_file, &output_text)
            .context("Failed to write output file")?;
        println!("📁 Output written to: {}", output_file.display());
    } else {
        println!("{}", output_text);
    }

    // Copy to clipboard if requested
    if clipboard {
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            copy_to_clipboard(&output_text)
        ).await {
            Ok(Ok(_)) => println!("\n📋 Copied to clipboard"),
            Ok(Err(e)) => {
                warn!("Failed to copy to clipboard: {}", e);
                println!("⚠️  Failed to copy to clipboard: {}", e);
            }
            Err(_) => {
                warn!("Clipboard operation timed out");
                println!("⚠️  Clipboard operation timed out");
            }
        }
    }

    Ok(())
}

/// List available models
async fn list_models(detailed: bool) -> Result<()> {
    let models = VoskEngine::available_models();

    if detailed {
        println!("Available Vosk Models:");
        println!("{:<30} {:<10} {:<15} {}", "Model", "Size", "Language", "Description");
        println!("{}", "-".repeat(80));
        
        for model in models {
            println!("{:<30} {:<10} {:<15} {}", 
                     model.to_string(), 
                     format!("{}MB", model.size_mb()),
                     model.language(),
                     model.description());
        }
    } else {
        println!("Available models:");
        for model in models {
            println!("  • {} ({}MB) - {}", model, model.size_mb(), model.description());
        }
    }

    Ok(())
}

/// Test audio recording setup
async fn test_audio_setup(audio_config: AudioConfig, duration: u64) -> Result<()> {
    println!("🔧 Testing audio setup...");

    // Check audio availability first
    match AudioRecorder::check_audio_availability() {
        Ok(backend_info) => {
            println!("✅ Audio backend available: {}", backend_info);
        }
        Err(e) => {
            println!("❌ Audio not available: {}", e);
            return Ok(());
        }
    }

    let mut recorder = AudioRecorder::new(audio_config)?;

    // List devices
    match recorder.list_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("❌ No audio input devices found");
                return Ok(());
            }
            println!("✅ Found {} audio devices:", devices.len());
            for device in &devices {
                println!("   • {}", device);
            }
        }
        Err(e) => {
            println!("❌ Failed to list devices: {}", e);
            return Ok(());
        }
    }

    // Test recording
    println!("\n🎙️  Testing {}-second recording...", duration);
    
    let recording_id = recorder.start_recording().await?;
    println!("Recording started: {}", recording_id);

    tokio::time::sleep(Duration::from_secs(duration)).await;

    let recording_info = recorder.stop_recording().await?;
    
    println!("✅ Test recording completed:");
    println!("   Duration: {}", audio::format_duration(recording_info.duration));
    println!("   File: {}", recording_info.file_path.display());
    println!("   Size: {}", audio::format_file_size(recording_info.file_size));
    
    // Clean up test file
    if let Err(e) = std::fs::remove_file(&recording_info.file_path) {
        warn!("Failed to remove test file: {}", e);
    }

    println!("\n✅ Audio setup test completed successfully!");
    Ok(())
}

/// Download and cache a Vosk model
async fn download_vosk_model(model_name: &str) -> Result<()> {
    let model = match model_name {
        "small-en-us" => VoskModel::SmallEnUs,
        "large-en-us" => VoskModel::LargeEnUs,
        "small-cn" => VoskModel::SmallCn,
        "small-ru" => VoskModel::SmallRu,
        "small-fr" => VoskModel::SmallFr,
        "small-de" => VoskModel::SmallDe,
        "small-es" => VoskModel::SmallEs,
        _ => {
            println!("Unknown model: {}", model_name);
            println!("Available models: small-en-us, large-en-us, small-cn, small-ru, small-fr, small-de, small-es");
            return Ok(());
        }
    };
    
    println!("📥 Downloading Vosk model: {} ({}MB)", model, model.size_mb());
    
    let config = VoskConfig {
        model,
        ..VoskConfig::default()
    };
    
    let mut engine = VoskEngine::new(config)?;
    engine.load_model().await?;
    
    println!("✅ Vosk model downloaded successfully");
    Ok(())
}

/// Copy text to clipboard (async version)
/// Copy text to clipboard using system tools (matching WhisperNow)
pub async fn copy_to_clipboard(text: &str) -> Result<()> {
    use tokio::process::Command;
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;
    
    // Check if we're on Wayland or X11
    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    let is_x11 = std::env::var("DISPLAY").is_ok();
    
    if is_wayland {
        // Use wl-copy for Wayland (same as WhisperNow)
        let mut child = Command::new("wl-copy")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn wl-copy - is wl-clipboard installed?")?;
        
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes()).await?;
            stdin.shutdown().await?;
        }
        
        let status = child.wait().await?;
        if !status.success() {
            return Err(anyhow::anyhow!("wl-copy failed: {:?}", status));
        }
    } else if is_x11 {
        // Use xclip for X11
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("Failed to spawn xclip - is xclip installed?")?;
        
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes()).await?;
            stdin.shutdown().await?;
        }
        
        let status = child.wait().await?;
        if !status.success() {
            return Err(anyhow::anyhow!("xclip failed: {:?}", status));
        }
    } else {
        return Err(anyhow::anyhow!("No display environment detected"));
    }
    
    Ok(())
}

/// Test clipboard functionality with detailed output
async fn test_clipboard_functionality(text: &str) -> Result<()> {
    println!("🔧 Testing Clipboard Functionality");
    println!("==================================");
    
    // Environment detection
    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    let is_x11 = std::env::var("DISPLAY").is_ok();
    let is_headless = !is_wayland && !is_x11;
    
    println!("Environment Detection:");
    println!("  Wayland: {}", if is_wayland { "✅ Available" } else { "❌ Not detected" });
    println!("  X11: {}", if is_x11 { "✅ Available" } else { "❌ Not detected" });
    println!("  Headless: {}", if is_headless { "⚠️  Yes" } else { "✅ No" });
    
    if let Ok(wayland) = std::env::var("WAYLAND_DISPLAY") {
        println!("  WAYLAND_DISPLAY: {}", wayland);
    }
    if let Ok(display) = std::env::var("DISPLAY") {
        println!("  DISPLAY: {}", display);
    }
    
    println!();
    
    if is_headless {
        println!("❌ Cannot test clipboard in headless environment");
        return Err(anyhow::anyhow!("No display environment detected"));
    }
    
    println!("📝 Attempting to copy text to clipboard:");
    println!("   \"{}\"", text);
    println!();
    
    let start_time = std::time::Instant::now();
    
    // Use the main copy_to_clipboard function with a 10-second timeout
    let clipboard_operation = copy_to_clipboard(text);
    
    match timeout(Duration::from_secs(10), clipboard_operation).await {
        Ok(Ok(())) => {
            let duration = start_time.elapsed();
            println!("✅ Clipboard copy successful! ({:.0}ms)", duration.as_millis());
            
            // Verify the clipboard content
            println!("🔍 Verifying clipboard content...");
            
            let verify_result = if is_wayland {
                tokio::process::Command::new("wl-paste")
                    .output()
                    .await
                    .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
            } else {
                tokio::process::Command::new("xclip")
                    .args(["-o", "-selection", "clipboard"])
                    .output()
                    .await
                    .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
            };
            
            match verify_result {
                Ok(clipboard_content) => {
                    if clipboard_content.trim() == text.trim() {
                        println!("✅ Verification successful - clipboard contains the correct text!");
                        println!("💡 You can paste (Ctrl+V/Cmd+V) to verify manually.");
                    } else {
                        println!("⚠️  Verification failed - clipboard content differs:");
                        println!("   Expected: \"{}\"", text.trim());
                        println!("   Found: \"{}\"", clipboard_content.trim());
                    }
                }
                Err(e) => {
                    println!("⚠️  Could not verify clipboard content: {}", e);
                    println!("💡 Try pasting (Ctrl+V/Cmd+V) to verify manually.");
                }
            }
            
            Ok(())
        }
        Ok(Err(e)) => {
            let duration = start_time.elapsed();
            println!("❌ Clipboard copy failed! ({:.0}ms)", duration.as_millis());
            println!("   Error: {}", e);
            println!();
            println!("🔧 Troubleshooting suggestions:");
            
            if is_wayland {
                println!("   - Install wl-clipboard: sudo apt install wl-clipboard");
                println!("   - Check if wl-copy is available: which wl-copy");
            }
            
            if is_x11 {
                println!("   - Install xclip: sudo apt install xclip");
                println!("   - Check if xclip is available: which xclip");
            }
            
            println!("   - Try running in a different terminal or desktop session");
            println!("   - Check if clipboard manager is running");
            
            Err(e)
        }
        Err(_) => {
            let duration = start_time.elapsed();
            println!("❌ Clipboard test timed out! ({:.0}ms)", duration.as_millis());
            println!("   The clipboard operation took longer than 10 seconds.");
            println!();
            println!("🔧 This usually indicates:");
            println!("   - No clipboard manager is running");
            println!("   - System clipboard is not properly configured");
            println!("   - Desktop environment issues");
            
            Err(anyhow::anyhow!("Clipboard test timed out after 10 seconds"))
        }
    }
}

/// Copy text to clipboard (non-blocking version for GUI)
pub fn copy_to_clipboard_nonblocking(text: &str) -> Result<()> {
    use std::thread;
    use std::sync::mpsc;
    
    let text = text.to_string();
    let (tx, rx) = mpsc::channel();
    
    // Spawn a thread with system clipboard fallback
    thread::spawn(move || {
        let result = copy_to_clipboard_sync(&text);
        let _ = tx.send(result);
    });
    
    // Wait for result with a reasonable timeout
    match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("Clipboard operation timed out after 3 seconds")),
    }
}

/// Synchronous clipboard copy using system tools only
fn copy_to_clipboard_sync(text: &str) -> Result<()> {
    use arboard::Clipboard;
    
    let mut clipboard = Clipboard::new()
        .map_err(|e| anyhow::anyhow!("Failed to access clipboard: {}", e))?;
    
    clipboard
        .set_text(text)
        .map_err(|e| anyhow::anyhow!("Failed to copy to clipboard: {}", e))?;
    
    Ok(())
}

/// Format results as plain text
fn format_as_text(results: &[(PathBuf, vosk_engine::VoskResult)]) -> String {
    let mut output = String::new();
    
    for (file, result) in results {
        if results.len() > 1 {
            output.push_str(&format!("=== {} ===\n", file.display()));
        }
        output.push_str(&result.text);
        if results.len() > 1 {
            output.push_str("\n\n");
        }
    }
    
    output
}

/// Format results as JSON
fn format_as_json(results: &[(PathBuf, vosk_engine::VoskResult)]) -> Result<String> {
    use serde_json::json;
    
    let json_results: Vec<_> = results.iter().map(|(file, result)| {
        json!({
            "file": file.to_string_lossy(),
            "text": result.text,
            "confidence": result.confidence,
            "words": result.words
        })
    }).collect();
    
    let output = json!({
        "transcriptions": json_results,
        "total_files": results.len(),
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    Ok(serde_json::to_string_pretty(&output)?)
}

/// Format results as SRT subtitles
fn format_as_srt(results: &[(PathBuf, vosk_engine::VoskResult)]) -> Result<String> {
    let mut output = String::new();
    let mut segment_id = 1;
    
    for (file, result) in results {
        if results.len() > 1 {
            output.push_str(&format!("// File: {}\n\n", file.display()));
        }
        
        // Vosk doesn't have segments, so create one segment for the whole text
        output.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            segment_id,
            "00:00:00,000", // Start at beginning
            "00:00:10,000", // Assume 10 seconds if no timing info
            result.text
        ));
        segment_id += 1;
    }
    
    Ok(output)
}



/// Truncate string to specified length with ellipsis
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len >= 3 {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.chars().take(max_len).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_name() {
        assert_eq!(parse_model_name("tiny").unwrap(), WhisperModel::Tiny);
        assert_eq!(parse_model_name("distil-small").unwrap(), WhisperModel::DistilSmall);
        assert_eq!(parse_model_name("large-v3").unwrap(), WhisperModel::Large);
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(65.5), "00:01:05,500");
        assert_eq!(format_srt_time(3661.250), "01:01:01,250");
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
        assert_eq!(truncate_string("hi", 1), "h");
    }
}
