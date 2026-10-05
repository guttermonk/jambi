//! Jambi - High-Performance Voice Transcription
//!
//! A blazing-fast voice transcription application built with Rust,
//! providing superior performance over traditional Python implementations.

use anyhow::{Context, Result};
use serde::{Serialize, Deserialize};

use clap::{Parser, Subcommand, ValueEnum};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use std::time::Duration;
use tokio::time::timeout;

use tracing::{debug, info, warn, error};

mod audio;
mod config;
mod daemon;
mod dictate;
mod tray;
mod vosk_engine;

use audio::{AudioRecorder, AudioConfig};
use tray::{TrayColor, TrayIcon};
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

    /// Interaction mode, overriding `mode` in the config file
    #[arg(long, value_enum)]
    mode: Option<Mode>,
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

    /// Hold-to-talk dictation, driven by a compositor keybind
    Dictate {
        #[command(subcommand)]
        action: DictateAction,
    },

    /// Run the background daemon that keeps the model loaded
    ///
    /// Start it once at login and every later command skips the ~800ms model
    /// load. Nothing requires it: each command falls back to loading its own
    /// model when no daemon is listening.
    Daemon {
        #[command(subcommand)]
        action: Option<DaemonAction>,

        /// Tray glyph, overriding `daemon.tray_icon` in the config file
        #[arg(long, value_enum)]
        icon: Option<TrayIcon>,

        /// Tray ink, overriding `daemon.tray_color`. `white` suits a dark
        /// panel, `black` a light one
        #[arg(long, value_enum)]
        color: Option<TrayColor>,

        /// Leave the tray glyph in its usual ink while recording, overriding
        /// `daemon.tray_red_when_recording`. Only turns the tint off -- to
        /// force it on, set the config key
        #[arg(long)]
        no_recording_tint: bool,
    },

    /// Print the active mode and exit
    ///
    /// Exists so a keybind wrapper script can branch on the configured mode
    /// without having to parse config.toml itself.
    Mode,
}

#[derive(Subcommand)]
enum DictateAction {
    /// Start recording; runs until `stop` signals it
    Start,
    /// Stop the running recording and deliver the text
    Stop,
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Load the model and serve requests in the foreground (the default)
    Run,
    /// Report whether a daemon is running, and what it has loaded
    Status,
    /// Ask a running daemon to exit
    Stop,
}

/// How jambi behaves when launched from its keybind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Open the interactive TUI in a terminal window (the historical behaviour)
    #[default]
    Windowed,
    /// Hold-to-talk dictation that types the result at the cursor
    Live,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Windowed => write!(f, "windowed"),
            Mode::Live => write!(f, "live"),
        }
    }
}

/// Settings specific to live dictation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DictateConfig {
    /// Milliseconds to wait after recording stops before typing, giving the
    /// keybind's modifiers time to come back up. See the note in
    /// `dictate::deliver` for why this is not read from evdev.
    #[serde(default = "default_modifier_grace_ms")]
    pub modifier_grace_ms: u64,

    /// Per-keystroke delay passed to wtype/xdotool. Some clients drop
    /// characters from a zero-delay burst.
    #[serde(default = "default_type_delay_ms")]
    pub type_delay_ms: u64,

    /// How long `dictate stop` waits for a recorder to appear before giving up,
    /// covering the case where the key is released while the model is loading.
    #[serde(default = "default_stop_wait_ms")]
    pub stop_wait_ms: u64,

    /// Absolute path to the icon shown on notifications. Falls back to
    /// JAMBI_ICON (the glyph the package ships) when unset, and to no icon at
    /// all when neither names a file that exists.
    ///
    /// Worth setting on a themed desktop: point it at the recolored copy the
    /// theme generates and the notification follows the palette instead of
    /// staying the shipped white.
    #[serde(default)]
    pub icon: Option<String>,
}

fn default_modifier_grace_ms() -> u64 {
    250
}

fn default_type_delay_ms() -> u64 {
    10
}

fn default_stop_wait_ms() -> u64 {
    2000
}

/// Settings for the background daemon that keeps the model warm.
///
/// See `daemon.rs` for what it does. These control how the *front-ends* treat
/// it as much as the daemon itself, since every command tries the socket and
/// falls back to loading its own model.
/// `deny_unknown_fields` so a key that is misspelt, or left over from a
/// rename, is an error naming the offender rather than a setting that silently
/// does nothing. Serde ignores unknown fields otherwise, which for appearance
/// settings is indistinguishable from the feature being broken.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    /// Hand work to a running daemon when one is listening. Turning this off
    /// makes every invocation load its own model again, which is the pre-daemon
    /// behaviour and a useful thing to compare against.
    pub enabled: bool,

    /// Show a tray indicator while the daemon runs, so it is visible among
    /// other background applications. Ignored when built without the `tray`
    /// feature.
    pub tray: bool,

    /// Which glyph the indicator draws.
    pub tray_icon: TrayIcon,

    /// The ink the indicator is drawn in: `white` for a dark panel, `black`
    /// for a light one. There is no reliable way to read the panel's color,
    /// so this is a setting rather than something detected.
    pub tray_color: TrayColor,

    /// Turn the indicator red while a recording is in progress.
    ///
    /// On by default, because it is the only at-a-glance sign that a dictation
    /// whose key release went missing is still holding the microphone. Off
    /// leaves the glyph in its usual ink; the tooltip and menu still report
    /// the state.
    pub tray_red_when_recording: bool,

    /// Hard cap on a single recording, in seconds.
    ///
    /// Only reached when the release half of a keybind never arrives -- a
    /// dropped `bindr`, or a compositor reload mid-press. Without it the
    /// daemon would hold the microphone open for the rest of the session.
    pub max_recording_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tray: true,
            tray_icon: TrayIcon::default(),
            tray_color: TrayColor::default(),
            tray_red_when_recording: true,
            max_recording_secs: 300,
        }
    }
}

impl Default for DictateConfig {
    fn default() -> Self {
        Self {
            modifier_grace_ms: default_modifier_grace_ms(),
            type_delay_ms: default_type_delay_ms(),
            stop_wait_ms: default_stop_wait_ms(),
            icon: None,
        }
    }
}

/// Application state
///
/// Every field defaults, so a config file may set as little as `mode = "live"`
/// and still parse. Before that, `[audio]` and `[vosk]` were mandatory and a
/// one-line config was a hard error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub vosk: VoskConfig,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default)]
    pub dictate: DictateConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
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
            mode: Mode::default(),
            dictate: DictateConfig::default(),
            daemon: DaemonConfig::default(),
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

    // `--mode` wins over the config file so a single invocation can be forced
    // into either mode without editing anything.
    let mode = cli.mode.unwrap_or(config.mode);

    // Override model if specified
    let vosk_config = config.vosk.clone();
    if let Some(_model_name) = &cli.model {
        // For now, model override via CLI is not supported for Vosk
        // You can specify the model in the config file
        eprintln!("Note: Model override via CLI not yet implemented for Vosk. Use config file instead.");
    }

    // Execute command
    match cli.command {
        Some(Commands::Mode) => {
            println!("{}", mode);
            Ok(())
        }
        Some(Commands::Dictate { action }) => {
            let mut vosk_config = vosk_config;
            vosk_config.verbose = cli.verbose;

            match action {
                DictateAction::Start => {
                    dictate::start(vosk_config, config.dictate, config.daemon, config.auto_copy)
                        .await
                }
                DictateAction::Stop => {
                    dictate::stop(mode, &config.dictate, &config.daemon, config.auto_copy).await
                }
            }
        }
        Some(Commands::Daemon {
            action,
            icon,
            color,
            no_recording_tint,
        }) => {
            let mut vosk_config = vosk_config;
            vosk_config.verbose = cli.verbose;

            // Flags win over the config file, so every appearance can be tried
            // without editing anything.
            let mut daemon_config = config.daemon;
            if let Some(icon) = icon {
                daemon_config.tray_icon = icon;
            }
            if let Some(color) = color {
                daemon_config.tray_color = color;
            }
            if no_recording_tint {
                daemon_config.tray_red_when_recording = false;
            }

            match action.unwrap_or(DaemonAction::Run) {
                DaemonAction::Run => daemon::run(vosk_config, daemon_config).await,
                DaemonAction::Status => daemon::print_status().await,
                DaemonAction::Stop => daemon::request_shutdown().await,
            }
        }
        Some(Commands::Record { auto_start, max_duration, output, live }) => {
            let mut audio_config = config.audio.clone();
            if let Some(output_dir) = output {
                audio_config.output_dir = output_dir;
            }
            audio_config.max_duration = Some(max_duration);
            audio_config.verbose = cli.verbose;

            let mut vosk_config = vosk_config;
            vosk_config.verbose = cli.verbose;

            run_recording_session(audio_config, vosk_config, &config.daemon, auto_start, live, config.auto_copy, cli.verbose).await
        }
        Some(Commands::Transcribe { files, format, output, clipboard }) => {
            run_transcription(files, vosk_config, &config.daemon, format, output, clipboard || config.auto_copy).await
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
            // No subcommand: the configured mode decides. Keeping this in step
            // with the keybind means `mode = "live"` is not quietly ignored for
            // anyone who runs a bare `jambi`.
            match mode {
                Mode::Windowed => {
                    let auto_copy = config.auto_copy;
                    run_interactive_mode(config, auto_copy, cli.verbose).await
                }
                Mode::Live => {
                    let mut vosk_config = vosk_config;
                    vosk_config.verbose = cli.verbose;
                    dictate::start(vosk_config, config.dictate, config.daemon, config.auto_copy)
                        .await
                }
            }
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

/// The config file consulted when `--config` is not given.
fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|dir| dir.join("jambi").join("config.toml"))
}

/// Read and parse a config file.
fn read_config(path: &Path) -> Result<AppConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

    info!("Loaded configuration from: {}", path.display());
    Ok(config)
}

/// Load configuration, preferring an explicit `--config` over the default path.
///
/// An explicit path that cannot be read is an error -- the user named it, so
/// silently falling back to defaults would hide the typo. The default path is
/// best-effort: not having one is the normal state for a fresh install.
fn load_config(config_path: Option<&Path>) -> Result<AppConfig> {
    if let Some(path) = config_path {
        return read_config(path);
    }

    if let Some(path) = default_config_path() {
        if path.exists() {
            return read_config(&path);
        }
        debug!("no config file at {}, using defaults", path.display());
    }

    Ok(AppConfig::default())
}



/// Where a front-end does its recognition.
///
/// The two variants exist so that nothing above this line has to care whether
/// a daemon is running. `Daemon` never calls `Model::new` in this process at
/// all -- that is the load time the daemon exists to remove. `Local` is the
/// standalone path, and loads on first use rather than at startup so that both
/// variants behave the same way from the caller's side: construction is cheap,
/// and the first transcription is where the cost lands.
enum Engine {
    Daemon {
        client: daemon::Client,
        model: String,
    },
    /// Boxed because `VoskEngine` is much the larger variant, and this enum is
    /// moved around by value.
    Local(Box<VoskEngine>),
}

impl Engine {
    /// Prefer a running daemon, fall back to an in-process engine.
    async fn connect(vosk_config: VoskConfig, daemon_config: &DaemonConfig) -> Result<Self> {
        if let Some(mut client) = daemon::Client::connect(daemon_config).await {
            // Asked rather than assumed: the daemon's model comes from the
            // config *it* was started with, which may not be this process's.
            match client.request(&daemon::Request::Status).await {
                Ok(daemon::Reply::Status { model, .. }) => {
                    debug!("using the daemon's warm {} model", model);
                    return Ok(Self::Daemon { client, model });
                }
                Ok(other) => warn!("unexpected status from the daemon: {:?}", other),
                Err(e) => warn!("could not query the daemon ({:#}), loading a model here", e),
            }
        }

        Ok(Self::Local(Box::new(VoskEngine::new(vosk_config)?)))
    }

    /// The model in use, for display.
    fn model_label(&self) -> String {
        match self {
            Self::Daemon { model, .. } => format!("{} (warm, from the daemon)", model),
            Self::Local(engine) => engine.config.model.to_string(),
        }
    }

    /// Load the model now, if this is a local engine that has not yet.
    ///
    /// Only worth calling where the caller wants the cost paid at a moment it
    /// has told the user about, rather than inside the first transcription.
    async fn warm_up(&mut self) -> Result<()> {
        if let Self::Local(engine) = self {
            if engine.model.is_none() {
                engine.load_model().await?;
            }
        }
        Ok(())
    }

    /// Recognise an audio file, with word timings and confidence intact so
    /// that `--format json` reports the same thing either way.
    async fn transcribe_file(&mut self, path: &Path) -> Result<vosk_engine::VoskResult> {
        match self {
            Self::Daemon { client, .. } => {
                // The path travels, not the audio: the daemon runs as the same
                // user in the same session, so it can open the file itself.
                //
                // Resolved here first, because the daemon's working directory
                // is wherever it was started from -- a bare `jambi transcribe
                // clip.wav` would otherwise ask it to open a `clip.wav` that
                // means something else entirely, or nothing at all.
                let absolute = std::fs::canonicalize(path)
                    .with_context(|| format!("cannot read {}", path.display()))?;
                let request = daemon::Request::TranscribeFile { path: absolute };
                match client.request(&request).await? {
                    daemon::Reply::Transcribed { result } => Ok(result),
                    daemon::Reply::Error { message } => Err(anyhow::anyhow!(message)),
                    other => Err(anyhow::anyhow!("unexpected reply: {:?}", other)),
                }
            }
            Self::Local(engine) => {
                if engine.model.is_none() {
                    engine.load_model().await?;
                }
                engine.transcribe_file(path).await
            }
        }
    }

    /// Transcribe from the microphone, printing text as it is recognised and
    /// stopping when the user presses Enter.
    async fn transcribe_live(&mut self) -> Result<String> {
        match self {
            Self::Daemon { client, .. } => stream_from_daemon(client).await,
            Self::Local(engine) => Ok(engine.transcribe_live().await?.text),
        }
    }

    /// Switch to a different language model.
    async fn set_model(&mut self, model: VoskModel) -> Result<()> {
        match self {
            Self::Daemon {
                client,
                model: current,
            } => {
                // Swaps the daemon's warm model, so the change outlives this
                // process. That is what someone switching language wants, and
                // worth knowing before they wonder why it stuck.
                match client.request(&daemon::Request::SetModel { model }).await? {
                    daemon::Reply::Ok => {
                        *current = model.to_string();
                        Ok(())
                    }
                    daemon::Reply::Error { message } => Err(anyhow::anyhow!(message)),
                    other => Err(anyhow::anyhow!("unexpected reply: {:?}", other)),
                }
            }
            Self::Local(engine) => {
                let mut config = engine.config.clone();
                config.model = model;
                engine.config = config;
                engine.model = None; // force a reload against the new model
                engine.load_model().await
            }
        }
    }
}

/// Drive a `Stream` request on the daemon: print partial results as they
/// arrive, and stop when the user presses Enter.
///
/// The terminal handling mirrors `VoskEngine::transcribe_live` so the two
/// paths look identical to someone watching, with the only difference being
/// which process owns the model.
async fn stream_from_daemon(client: &mut daemon::Client) -> Result<String> {
    match client.request(&daemon::Request::Stream).await? {
        daemon::Reply::Streaming => {}
        daemon::Reply::AlreadyRecording => {
            anyhow::bail!("the daemon is already recording for something else")
        }
        daemon::Reply::Error { message } => anyhow::bail!(message),
        other => anyhow::bail!("unexpected reply: {:?}", other),
    }

    println!("⚠️ Press Enter to stop");

    // A thread, not a tokio task: a blocking stdin read cannot be cancelled,
    // and this one lives until the process exits either way.
    let (enter_tx, enter_rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        let _ = enter_tx.send(());
    });

    let mut enter = enter_rx;
    let mut asked_to_stop = false;

    let text = loop {
        tokio::select! {
            _ = &mut enter, if !asked_to_stop => {
                asked_to_stop = true;
                client.send(&daemon::Request::Stop).await?;
            }
            reply = client.next_reply() => match reply? {
                daemon::Reply::Partial { text } => {
                    print!("\r\x1b[K📝 {}", text);
                    io::stdout().flush().ok();
                }
                daemon::Reply::Segment { text } => {
                    print!("\r\x1b[K✅ {}\n", text);
                    io::stdout().flush().ok();
                }
                daemon::Reply::Final { text } => break text,
                daemon::Reply::Error { message } => anyhow::bail!(message),
                other => anyhow::bail!("unexpected reply: {:?}", other),
            }
        }
    };

    println!("⏹️ Stopped live transcription");

    Ok(text)
}

/// Run interactive recording mode
async fn run_interactive_mode(config: AppConfig, auto_copy: bool, verbose: bool) -> Result<()> {
    println!("🎙️  Jambi - Interactive Voice Transcription");
    println!("============================================");

    let mut audio_config = config.audio.clone();
    audio_config.verbose = verbose;
    let mut recorder = AudioRecorder::new(audio_config)?;

    let mut vosk_config = config.vosk;
    vosk_config.verbose = verbose;
    let mut engine = Engine::connect(vosk_config, &config.daemon).await?;

    println!("Model: {}", engine.model_label());
    println!("Sample Rate: {}Hz, Channels: {}",
             config.audio.sample_rate, config.audio.channels);
    println!("Output Directory: {}", config.audio.output_dir.display());
    println!();

    // Loading up front rather than on first use, because this menu is a place
    // the user waits anyway -- better here, with a line saying so, than as an
    // unexplained pause after they press R. With a daemon this is a no-op and
    // the message never appears.
    if matches!(engine, Engine::Local(_)) {
        println!("🔄 Loading Vosk model...");
        if let Err(e) = engine.warm_up().await {
            println!("⚠️  Model loading failed: {}. Will download on first use.", e);
        }
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
                if let Err(e) = record_and_transcribe(&mut recorder, &mut engine, auto_copy).await {
                    error!("Recording failed: {}", e);
                    println!("❌ Recording failed: {}", e);
                }
            }
            "T" | "TRANSCRIBE" => {
                if let Err(e) = transcribe_file_interactive(&mut engine, auto_copy).await {
                    error!("Transcription failed: {}", e);
                    println!("❌ Transcription failed: {}", e);
                }
            }
            "M" | "MODEL" => {
                if let Err(e) = switch_model_interactive(&mut engine).await {
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
    engine: &mut Engine,
    auto_copy: bool,
) -> Result<()> {
    let recording_info = recorder.record_audio().await?;

    println!("✅ Recording completed:");
    println!("   Duration: {}", audio::format_duration(recording_info.duration));
    println!("   File: {}", recording_info.file_path.display());
    println!("   Size: {}", audio::format_file_size(recording_info.file_size));

    println!("🧠 Transcribing audio...");
    let text = engine.transcribe_file(&recording_info.file_path).await?.text;

    println!("📝 Transcription completed successfully");
    println!();
    println!("┌{}┐", "─".repeat(60));
    println!("│ {:^58} │", "TRANSCRIPTION RESULT");
    println!("├{}┤", "─".repeat(60));
    for line in text.lines() {
        println!("│ {:<58} │", truncate_string(line, 58));
    }
    println!("└{}┘", "─".repeat(60));
    println!();

    if auto_copy && !text.is_empty() {
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            copy_to_clipboard(&text)
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
async fn transcribe_file_interactive(engine: &mut Engine, auto_copy: bool) -> Result<()> {
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
    let text = engine.transcribe_file(&path).await?.text;

    println!("📝 Transcription completed successfully");
    println!();
    println!("{}", text);
    println!();

    if auto_copy && !text.is_empty() {
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            copy_to_clipboard(&text)
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
async fn switch_model_interactive(engine: &mut Engine) -> Result<()> {
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

            println!("Loading new model: {}", selected_model);
            engine.set_model(selected_model).await?;

            println!("✅ Switched to model: {}", selected_model);
            if matches!(engine, Engine::Daemon { .. }) {
                // Said plainly because it is the surprising part: the daemon
                // holds one warm model, so this outlives the menu and applies
                // to dictation too.
                println!("   (the daemon's warm model changed, so this applies everywhere)");
            }
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
    daemon_config: &DaemonConfig,
    auto_start: bool,
    live: bool,
    auto_copy: bool,
    verbose: bool,
) -> Result<()> {
    let mut recorder = AudioRecorder::new(audio_config)?;
    let mut engine = Engine::connect(vosk_config, daemon_config).await?;

    // Loaded here rather than during the first recording, where it would eat
    // the beginning of what the user says. A daemon has already done it.
    if let Err(e) = engine.warm_up().await {
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

        // Perform recording/transcription based on mode
        let text = if live {
            // Use live transcription mode
            println!("🎤 Live transcription enabled & now listening...");

            let live_text = engine.transcribe_live().await?;

            println!("\n📝 Final Transcription:");
            println!("{}", live_text);

            live_text
        } else {
            // Use the new record_audio function that handles Enter-to-stop
            let recording_info = recorder.record_audio().await?;

            println!("⏱️ Recording completed: {}", audio::format_duration(recording_info.duration));
            println!("🧠 Transcribing...");

            let transcribed = engine.transcribe_file(&recording_info.file_path).await?.text;

            println!("📝 Transcription Result:");
            println!("{}", transcribed);

            transcribed
        };

        if auto_copy && !text.is_empty() {
            match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                copy_to_clipboard(&text)
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
    daemon_config: &DaemonConfig,
    format: String,
    output: Option<PathBuf>,
    clipboard: bool,
) -> Result<()> {
    if files.is_empty() {
        return Err(anyhow::anyhow!("No input files specified"));
    }

    let mut engine = Engine::connect(config, daemon_config).await?;

    if let Err(e) = engine.warm_up().await {
        eprintln!("⚠️  Model loading failed: {}. Will download on first use.", e);
    }

    let mut results = Vec::new();

    for file in &files {
        println!("🧠 Transcribing: {}", file.display());

        match engine.transcribe_file(file).await {
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

    // Whisper-related tests removed - using Vosk instead

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
        assert_eq!(truncate_string("hi", 1), "h");
    }
}
