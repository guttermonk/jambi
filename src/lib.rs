//! Jambi - Voice Transcription Library
//!
//! A native Rust voice transcription library using the Vosk speech recognition engine.
//!
//! # Features
//!
//! - **Native Performance**: Fast startup and low memory usage
//! - **Cross-platform**: Works on Linux, macOS, and Windows
//! - **Audio Recording**: Built-in audio recording with CPAL
//! - **Vosk Integration**: Lightweight speech recognition
//! - **Simple API**: Easy to use async/await interface
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use jambi::{AudioRecorder, AudioConfig, VoskEngine, VoskConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Set up audio recording
//!     let audio_config = AudioConfig::default();
//!     let mut recorder = AudioRecorder::new(audio_config)?;
//!
//!     // Set up Vosk transcription
//!     let vosk_config = VoskConfig::default();
//!     let mut engine = VoskEngine::new(vosk_config)?;
//!     engine.load_model().await?;
//!
//!     // Record and transcribe
//!     let recording = recorder.record_audio().await?;
//!     let result = engine.transcribe_file(&recording.file_path).await?;
//!     println!("Transcription: {}", result.text);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Modules
//!
//! - [`audio`] - Cross-platform audio recording
//! - [`vosk_engine`] - Vosk speech recognition integration
//! - [`config`] - Configuration management

use anyhow::Result;
use std::path::Path;

pub mod audio;
pub mod config;
pub mod vosk_engine;

// Clipboard functionality
use anyhow::Context;

use std::time::Duration;

/// Copy text to clipboard (non-blocking version for GUI)
pub fn copy_to_clipboard_nonblocking(text: &str) -> anyhow::Result<()> {
    use std::process::{Command, Stdio};
    use std::io::Write;
    use std::thread;
    use std::sync::mpsc;
    
    // Check environment
    let is_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    let is_x11 = std::env::var("DISPLAY").is_ok();
    
    if !is_wayland && !is_x11 {
        return Err(anyhow::anyhow!("No display environment detected"));
    }
    
    let text = text.to_string();
    let (tx, rx) = mpsc::channel();
    
    // Spawn a thread with system clipboard tools
    thread::spawn(move || {
        if is_wayland {
            // Try wl-copy
            let child = Command::new("wl-copy")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .context("Failed to spawn wl-copy");
            
            match child {
                Ok(mut child) => {
                    if let Some(mut stdin) = child.stdin.take() {
                        if let Err(e) = stdin.write_all(text.as_bytes()).and_then(|_| stdin.flush()) {
                            let _ = tx.send(Err(anyhow::anyhow!("Failed to write to wl-copy: {}", e)));
                            return;
                        }
                        drop(stdin);
                        
                        match child.wait() {
                            Ok(status) if status.success() => {
                                let _ = tx.send(Ok(()));
                            }
                            Ok(_) => {
                                let _ = tx.send(Err(anyhow::anyhow!("wl-copy failed")));
                            }
                            Err(e) => {
                                let _ = tx.send(Err(anyhow::anyhow!("Failed to wait for wl-copy: {}", e)));
                            }
                        }
                    } else {
                        let _ = tx.send(Err(anyhow::anyhow!("Failed to get wl-copy stdin")));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            }
        } else {
            // Try xclip
            let child = Command::new("xclip")
                .args(["-selection", "clipboard"])
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .context("Failed to spawn xclip");
            
            match child {
                Ok(mut child) => {
                    if let Some(mut stdin) = child.stdin.take() {
                        if let Err(e) = stdin.write_all(text.as_bytes()).and_then(|_| stdin.flush()) {
                            let _ = tx.send(Err(anyhow::anyhow!("Failed to write to xclip: {}", e)));
                            return;
                        }
                        drop(stdin);
                        
                        match child.wait() {
                            Ok(status) if status.success() => {
                                let _ = tx.send(Ok(()));
                            }
                            Ok(_) => {
                                let _ = tx.send(Err(anyhow::anyhow!("xclip failed")));
                            }
                            Err(e) => {
                                let _ = tx.send(Err(anyhow::anyhow!("Failed to wait for xclip: {}", e)));
                            }
                        }
                    } else {
                        let _ = tx.send(Err(anyhow::anyhow!("Failed to get xclip stdin")));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            }
        }
    });
    
    // Wait for result with a reasonable timeout
    match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("Clipboard operation timed out after 3 seconds")),
    }
}

// Re-export main types for convenience
pub use audio::{AudioRecorder, AudioConfig, RecordingInfo, RecordingState};
pub use vosk_engine::{VoskEngine, VoskConfig, VoskModel};
pub use config::*;



/// Application constants
pub const APP_NAME: &str = "Jambi";
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const APP_DESCRIPTION: &str = "High-performance voice transcription";

/// Get the application name
pub fn app_name() -> &'static str {
    APP_NAME
}

/// Get the application version
pub fn app_version() -> &'static str {
    APP_VERSION
}

/// Format duration in seconds as human-readable string
pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor() as u32;
        let secs = seconds % 60.0;
        format!("{}m {:.1}s", minutes, secs)
    } else {
        let hours = (seconds / 3600.0).floor() as u32;
        let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
        let secs = seconds % 60.0;
        format!("{}h {}m {:.1}s", hours, minutes, secs)
    }
}

/// Format file size in bytes as human-readable string
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Validate that a file exists and is readable
pub fn validate_audio_file<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    
    if !path.exists() {
        return Err(anyhow::anyhow!("File does not exist: {}", path.display()));
    }
    
    if !path.is_file() {
        return Err(anyhow::anyhow!("Path is not a file: {}", path.display()));
    }
    
    // Check file extension
    if let Some(extension) = path.extension() {
        let ext = extension.to_string_lossy().to_lowercase();
        if !matches!(ext.as_str(), "wav" | "mp3" | "flac" | "ogg" | "m4a") {
            return Err(anyhow::anyhow!(
                "Unsupported audio format: {} (supported: wav, mp3, flac, ogg, m4a)", 
                ext
            ));
        }
    } else {
        return Err(anyhow::anyhow!("File has no extension: {}", path.display()));
    }
    
    Ok(())
}

/// Generate a unique ID for recordings
pub fn generate_recording_id() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    
    format!("rec_{:08x}", (hasher.finish() as u32))
}

/// Check if CUDA is available for GPU acceleration
pub fn cuda_available() -> bool {
    // In this demo version, GPU acceleration is simulated
    std::env::var("JAMBI_SIMULATE_CUDA").is_ok()
}

/// Check if Metal is available for GPU acceleration (macOS)
pub fn metal_available() -> bool {
    // In this demo version, GPU acceleration is simulated
    cfg!(target_os = "macos") && std::env::var("JAMBI_SIMULATE_METAL").is_ok()
}

/// Get the best available device description
pub fn get_best_device() -> &'static str {
    if cuda_available() {
        "CUDA GPU (simulated)"
    } else if metal_available() {
        "Metal GPU (simulated)"
    } else {
        "CPU"
    }
}

/// Application-wide error type
#[derive(thiserror::Error, Debug)]
pub enum JambiError {
    #[error("Audio error: {0}")]
    Audio(#[from] cpal::BuildStreamError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Transcription error: {0}")]
    Transcription(String),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Runtime error: {0}")]
    Runtime(#[from] anyhow::Error),
}

pub type JambiResult<T> = Result<T, JambiError>;

/// Initialize the library with logging
pub fn init() -> Result<()> {
    init_with_level("info")
}

/// Initialize the library with specific log level
pub fn init_with_level(level: &str) -> Result<()> {
    let env_filter = format!("jambi={},warn", level);
    
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;
    
    tracing::info!("Initialized {} v{}", APP_NAME, APP_VERSION);
    
    // Log system capabilities
    if cuda_available() {
        tracing::info!("CUDA GPU acceleration available");
    }
    if metal_available() {
        tracing::info!("Metal GPU acceleration available");
    }
    
    Ok(())
}

/// Get system information for diagnostics
pub fn system_info() -> SystemInfo {
    SystemInfo {
        app_name: APP_NAME.to_string(),
        app_version: APP_VERSION.to_string(),
        rust_version: "1.89.0".to_string(), // Could be determined at runtime if needed
        target: std::env::consts::ARCH.to_string(),
        cuda_available: cuda_available(),
        metal_available: metal_available(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

/// System information structure
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemInfo {
    pub app_name: String,
    pub app_version: String,
    pub rust_version: String,
    pub target: String,
    pub cuda_available: bool,
    pub metal_available: bool,
    pub os: String,
    pub arch: String,
}

impl std::fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} v{}", self.app_name, self.app_version)?;
        writeln!(f, "Rust: {}", self.rust_version)?;
        writeln!(f, "Target: {}", self.target)?;
        writeln!(f, "OS: {} ({})", self.os, self.arch)?;
        writeln!(f, "CUDA: {}", if self.cuda_available { "Available" } else { "Not available" })?;
        writeln!(f, "Metal: {}", if self.metal_available { "Available" } else { "Not available" })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(1.5), "1.5s");
        assert_eq!(format_duration(65.0), "1m 5.0s");
        assert_eq!(format_duration(3661.0), "1h 1m 1.0s");
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
        assert_eq!(format_file_size(1073741824), "1.0 GB");
    }

    #[test]
    fn test_generate_recording_id() {
        let id1 = generate_recording_id();
        let id2 = generate_recording_id();
        
        assert!(id1.starts_with("rec_"));
        assert!(id2.starts_with("rec_"));
        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 12); // "rec_" + 8 hex chars
    }

    #[test]
    fn test_app_constants() {
        assert_eq!(app_name(), "Jambi");
        assert!(!app_version().is_empty());
    }

    #[test]
    fn test_validate_audio_file() {
        use tempfile::NamedTempFile;
        
        // Test non-existent file
        assert!(validate_audio_file("/nonexistent/file.wav").is_err());
        
        // Test file with wrong extension
        let temp_file = NamedTempFile::new().unwrap();
        assert!(validate_audio_file(temp_file.path()).is_err());
        
        // Test valid extension (can't actually test without creating a proper WAV file)
        let temp_wav = temp_file.path().with_extension("wav");
        std::fs::write(&temp_wav, b"fake wav content").unwrap();
        // This would fail because it's not a real WAV, but extension check should pass
        // We'd need a proper WAV file to test the full validation
    }

    #[test]
    fn test_system_info() {
        let info = system_info();
        assert_eq!(info.app_name, "Jambi");
        assert!(!info.app_version.is_empty());
        assert!(!info.rust_version.is_empty());
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        
        // Test display formatting
        let display = format!("{}", info);
        assert!(display.contains("Jambi"));
        assert!(display.contains("Rust:"));
        assert!(display.contains("CUDA:"));
        assert!(display.contains("Metal:"));
    }

    #[tokio::test]
    async fn test_init_logging() {
        // This test might conflict with other tests if they also initialize logging
        // In a real scenario, you'd want to test this in isolation
        let result = init_with_level("debug");
        // Don't assert success since logging might already be initialized
        // Just ensure it doesn't panic
        println!("Logging initialization result: {:?}", result);
    }

    #[test]
    fn test_device_detection() {
        // These functions should not panic
        let cuda = cuda_available();
        let metal = metal_available();
        let device = get_best_device();
        
        println!("CUDA available: {}", cuda);
        println!("Metal available: {}", metal);
        println!("Best device: {:?}", device);
        
        // Device should return a string description
        assert!(device.len() > 0);
    }
}