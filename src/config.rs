//! Configuration management for Jambi
//!
//! This module handles loading, validation, and management of application
//! configuration from various sources including files, environment variables,
//! and command-line arguments.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Whisper model imports removed - using Vosk instead

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Audio recording configuration
    pub audio: AudioConfig,
    /// Transcription configuration
    pub transcription: TranscriptionConfig,
    /// Application behavior settings
    pub app: AppConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            transcription: TranscriptionConfig::default(),
            app: AppConfig::default(),
        }
    }
}

/// Audio recording configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Sample rate in Hz (typically 16000 for speech)
    pub sample_rate: u32,
    /// Number of audio channels (1 for mono, 2 for stereo)
    pub channels: u16,
    /// Buffer size for audio processing
    pub buffer_size: usize,
    /// Directory to save recordings
    pub output_dir: PathBuf,
    /// Maximum recording duration in seconds (None for unlimited)
    pub max_duration: Option<u64>,
    /// Preferred audio input device (None for default)
    pub input_device: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 1024,
            output_dir: default_recordings_dir(),
            max_duration: Some(300), // 5 minutes
            input_device: None,
        }
    }
}

/// Transcription configuration (for Vosk engine)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Device for inference (cpu, cuda, metal)
    pub device: DeviceType,
    /// Language for transcription (None for auto-detect)
    pub language: Option<String>,
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Enable automatic punctuation
    pub auto_punctuation: bool,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            device: DeviceType::Auto,
            language: Some("en".to_string()),
            sample_rate: 16000,
            auto_punctuation: true,
        }
    }
}

/// Device type for ML inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    /// Use CPU for inference
    Cpu,
    /// Use CUDA GPU for inference
    Cuda,
    /// Use Metal GPU for inference (macOS)
    Metal,
    /// Auto-select best available device
    Auto,
}

impl DeviceType {
}

/// Application behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Automatically copy transcription results to clipboard
    pub auto_copy: bool,
    /// Keep recording files after transcription
    pub keep_recordings: bool,
    /// Show desktop notifications
    pub show_notifications: bool,
    /// Maximum number of recent transcriptions to keep in memory
    pub max_recent_transcriptions: usize,
    /// Auto-start recording on launch
    pub auto_start_recording: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            auto_copy: true,
            keep_recordings: false,
            show_notifications: true,
            max_recent_transcriptions: 50,
            auto_start_recording: false,
        }
    }
}

/// Get the default recordings directory
fn default_recordings_dir() -> PathBuf {
    dirs::data_dir()
        .or_else(|| dirs::home_dir())
        .unwrap_or_else(|| std::env::temp_dir())
        .join("jambi")
        .join("recordings")
}

// Model parsing removed - Vosk uses different model management

// Model serde support removed - Vosk uses different model management

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use tempfile::NamedTempFile;
    
    // Test-only helpers
    impl DeviceType {
        fn description(&self) -> &'static str {
            match self {
                DeviceType::Cpu => "CPU processing",
                DeviceType::Cuda => "CUDA GPU processing",
                DeviceType::Metal => "Metal GPU processing (macOS)",
                DeviceType::Auto => "Auto-select best device",
            }
        }
    }

    fn default_config_path() -> PathBuf {
        dirs::config_dir()
            .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
            .unwrap_or_else(|| std::env::temp_dir())
            .join("jambi")
            .join("config.toml")
    }

    fn parse_bool(s: &str) -> Result<bool> {
        match s.to_lowercase().trim() {
            "true" | "t" | "yes" | "y" | "1" | "on" => Ok(true),
            "false" | "f" | "no" | "n" | "0" | "off" | "" => Ok(false),
            _ => Err(anyhow::anyhow!("Invalid boolean value: {}", s)),
        }
    }

    struct ConfigBuilder {
        config: Config,
    }

    impl ConfigBuilder {
        fn new() -> Self {
            Self {
                config: Config::default(),
            }
        }
        
        fn language(mut self, language: String) -> Self {
            self.config.transcription.language = Some(language);
            self
        }
        
        fn device(mut self, device: DeviceType) -> Self {
            self.config.transcription.device = device;
            self
        }
        
        fn auto_copy(mut self, auto_copy: bool) -> Self {
            self.config.app.auto_copy = auto_copy;
            self
        }
        
        fn build(self) -> Config {
            self.config
        }
    }

    struct ConfigLoader {
        config: Config,
    }

    impl ConfigLoader {
        fn new() -> Self {
            Self {
                config: Config::default(),
            }
        }
        
        fn from_file(mut self, path: &std::path::Path) -> Result<Self> {
            let content = std::fs::read_to_string(path)?;
            self.config = toml::from_str(&content)?;
            Ok(self)
        }
        
        fn validate(self) -> Result<Self> {
            // Validate sample rate
            if self.config.audio.sample_rate < 8000 || self.config.audio.sample_rate > 48000 {
                return Err(anyhow::anyhow!("Sample rate must be between 8000 and 48000 Hz"));
            }
            
            // Validate channels
            if self.config.audio.channels == 0 || self.config.audio.channels > 2 {
                return Err(anyhow::anyhow!("Channels must be 1 (mono) or 2 (stereo)"));
            }
            
            // Validate buffer size
            if self.config.audio.buffer_size < 128 || self.config.audio.buffer_size > 8192 {
                return Err(anyhow::anyhow!("Buffer size must be between 128 and 8192"));
            }
            
            Ok(self)
        }
        
        fn build(self) -> Config {
            self.config
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.audio.channels, 1);
        assert_eq!(config.transcription.sample_rate, 16000);
        assert!(config.app.auto_copy);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .language("en".to_string())
            .device(DeviceType::Cpu)
            .auto_copy(false)
            .build();

        assert_eq!(config.transcription.language, Some("en".to_string()));
        assert_eq!(config.transcription.device, DeviceType::Cpu);
        assert!(!config.app.auto_copy);
    }

    #[test]
    fn test_device_type_description() {
        assert_eq!(DeviceType::Cpu.description(), "CPU processing");
        assert_eq!(DeviceType::Auto.description(), "Auto-select best device");
    }

    // Model parsing tests removed - Vosk uses different model management

    #[test]
    fn test_parse_bool() {
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("1").unwrap());
        assert!(parse_bool("yes").unwrap());
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("0").unwrap());
        assert!(!parse_bool("").unwrap());
        assert!(parse_bool("invalid").is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&toml_str).unwrap();
        
        assert_eq!(config.audio.sample_rate, deserialized.audio.sample_rate);
        assert_eq!(config.transcription.sample_rate, deserialized.transcription.sample_rate);
    }

    #[test]
    fn test_config_loader_validation() {
        let mut config = Config::default();
        
        // Test invalid sample rate
        config.audio.sample_rate = 1000; // Too low
        let loader = ConfigLoader { config: config.clone() };
        assert!(loader.validate().is_err());

        // Test invalid channels
        config.audio.sample_rate = 16000; // Fix sample rate
        config.audio.channels = 0; // Invalid
        let loader = ConfigLoader { config: config.clone() };
        assert!(loader.validate().is_err());

        // Test valid config
        config.audio.channels = 1; // Fix channels
        let loader = ConfigLoader { config };
        assert!(loader.validate().is_ok());
    }

    #[test]
    fn test_config_file_loading() {
        let temp_file = NamedTempFile::new().unwrap();
        let config_content = r#"
[audio]
sample_rate = 22050
channels = 2
buffer_size = 2048
output_dir = "/tmp/test"

[transcription]
device = "cpu"
language = "en"
sample_rate = 16000
auto_punctuation = true

[app]
auto_copy = false
keep_recordings = true
show_notifications = false
max_recent_transcriptions = 50
auto_start_recording = true
"#;
        std::fs::write(temp_file.path(), config_content).unwrap();

        let config = ConfigLoader::new()
            .from_file(temp_file.path()).unwrap()
            .validate().unwrap()
            .build();

        assert_eq!(config.audio.sample_rate, 22050);
        assert_eq!(config.audio.channels, 2);
        assert_eq!(config.audio.buffer_size, 2048);
        assert_eq!(config.transcription.sample_rate, 16000);
        assert!(config.transcription.auto_punctuation);
        assert!(!config.app.auto_copy);
        assert!(config.app.keep_recordings);
    }

    #[test]
    fn test_default_paths() {
        let config_path = default_config_path();
        assert!(config_path.to_string_lossy().contains("jambi"));
        assert!(config_path.to_string_lossy().contains("config.toml"));

        let recordings_path = default_recordings_dir();
        assert!(recordings_path.to_string_lossy().contains("jambi"));
        assert!(recordings_path.to_string_lossy().contains("recordings"));
    }
}