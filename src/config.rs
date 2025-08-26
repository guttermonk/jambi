//! Configuration management for Jambi
//!
//! This module handles loading, validation, and management of application
//! configuration from various sources including files, environment variables,
//! and command-line arguments.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;


use crate::whisper::WhisperModel;

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

/// Transcription configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Whisper model to use
    #[serde(with = "model_serde")]
    pub model: WhisperModel,
    /// Device for inference (cpu, cuda, metal)
    pub device: DeviceType,
    /// Language for transcription (None for auto-detect)
    pub language: Option<String>,
    /// Temperature for sampling (0.0 for deterministic)
    pub temperature: f32,
    /// Number of beams for beam search
    pub num_beams: usize,
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model: WhisperModel::Tiny,
            device: DeviceType::Auto,
            language: Some("en".to_string()),
            temperature: 0.0,
            num_beams: 1,
            max_tokens: 448,
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

/// Parse model name from string
pub fn parse_model_name(name: &str) -> Result<WhisperModel> {
    match name.to_lowercase().trim() {
        "tiny" => Ok(WhisperModel::Tiny),
        "base" => Ok(WhisperModel::Base),
        "small" => Ok(WhisperModel::Small),
        "medium" => Ok(WhisperModel::Medium),
        "large" => Ok(WhisperModel::Large),
        "distil-small" | "distil-small.en" => Ok(WhisperModel::DistilSmall),
        "distil-medium" | "distil-medium.en" => Ok(WhisperModel::DistilMedium),
        _ => Err(anyhow::anyhow!("Unknown model: {}", name)),
    }
}



// Serde support for WhisperModel
mod model_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(model: &WhisperModel, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&model.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<WhisperModel, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_model_name(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.audio.channels, 1);
        assert_eq!(config.transcription.model, WhisperModel::DistilSmall);
        assert!(config.app.auto_copy);
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .model(WhisperModel::Small)
            .device(DeviceType::Cpu)
            .auto_copy(false)
            .build();

        assert_eq!(config.transcription.model, WhisperModel::Small);
        assert_eq!(config.transcription.device, DeviceType::Cpu);
        assert!(!config.app.auto_copy);
    }

    #[test]
    fn test_device_type_description() {
        assert_eq!(DeviceType::Cpu.description(), "CPU processing");
        assert_eq!(DeviceType::Auto.description(), "Auto-select best device");
    }

    #[test]
    fn test_parse_model_name() {
        assert_eq!(parse_model_name("tiny").unwrap(), WhisperModel::Tiny);
        assert_eq!(parse_model_name("distil-small").unwrap(), WhisperModel::DistilSmall);
        assert_eq!(parse_model_name("large-v3").unwrap(), WhisperModel::Large);
        assert!(parse_model_name("invalid").is_err());
    }

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
        assert_eq!(config.transcription.model, deserialized.transcription.model);
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
model = "small"
device = "cpu"
temperature = 0.5
num_beams = 2
max_tokens = 512

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
        assert_eq!(config.transcription.model, WhisperModel::Small);
        assert_eq!(config.transcription.temperature, 0.5);
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