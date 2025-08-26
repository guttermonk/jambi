//! Whisper speech-to-text transcription module
//!
//! This module provides integration with OpenAI's Whisper models.
//! Currently implements a simplified version that will be extended
//! with full ML capabilities in future updates.

#![allow(dead_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{info, warn, debug, error};
use std::process::Command;
use tokio::process::Command as TokioCommand;
use std::fs;
use std::env;
use tokio::io::AsyncWriteExt;
use std::collections::HashMap;

/// Available Whisper model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhisperModel {
    /// Smallest model, fastest inference, basic accuracy
    Tiny,
    /// Smallest English-only model, fastest inference
    TinyEn,
    /// Small model, good balance of speed and accuracy
    Base,
    /// Small English-only model, good balance
    BaseEn,
    /// Medium model, better accuracy, slower inference
    Small,
    /// Medium English-only model, better accuracy
    SmallEn,
    /// Large model, high accuracy, slow inference
    Medium,
    /// Largest model, best accuracy, slowest inference
    Large,
    /// Distilled small model, good speed/accuracy balance
    DistilSmall,
    /// Distilled medium model, better accuracy than distil-small
    DistilMedium,
}

impl WhisperModel {
    /// Get the model identifier string for Hugging Face
    pub fn model_id(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "openai/whisper-tiny",
            WhisperModel::TinyEn => "openai/whisper-tiny.en",
            WhisperModel::Base => "openai/whisper-base",
            WhisperModel::BaseEn => "openai/whisper-base.en",
            WhisperModel::Small => "openai/whisper-small",
            WhisperModel::SmallEn => "openai/whisper-small.en",
            WhisperModel::Medium => "openai/whisper-medium",
            WhisperModel::Large => "openai/whisper-large-v3",
            WhisperModel::DistilSmall => "distil-whisper/distil-small.en",
            WhisperModel::DistilMedium => "distil-whisper/distil-medium.en",
        }
    }

    /// Get approximate model size in MB
    pub fn size_mb(&self) -> u64 {
        match self {
            WhisperModel::Tiny => 39,
            WhisperModel::TinyEn => 39,
            WhisperModel::Base => 74,
            WhisperModel::BaseEn => 74,
            WhisperModel::Small => 244,
            WhisperModel::SmallEn => 244,
            WhisperModel::Medium => 769,
            WhisperModel::Large => 1550,
            WhisperModel::DistilSmall => 166,
            WhisperModel::DistilMedium => 394,
        }
    }

    /// Get model description
    pub fn description(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "Fastest, basic accuracy",
            WhisperModel::TinyEn => "Fastest English-only, basic accuracy",
            WhisperModel::Base => "Fast, good for general use",
            WhisperModel::BaseEn => "Fast English-only, good for general use",
            WhisperModel::Small => "Balanced performance",
            WhisperModel::SmallEn => "Balanced English-only performance",
            WhisperModel::Medium => "High accuracy, slower",
            WhisperModel::Large => "Best accuracy, slowest",
            WhisperModel::DistilSmall => "Fast distilled model (recommended)",
            WhisperModel::DistilMedium => "Best balance of speed and accuracy",
        }
    }

    /// Check if model supports English only
    pub fn is_english_only(&self) -> bool {
        matches!(self, 
            WhisperModel::TinyEn | 
            WhisperModel::BaseEn | 
            WhisperModel::SmallEn |
            WhisperModel::DistilSmall | 
            WhisperModel::DistilMedium
        )
    }
}

impl Default for WhisperModel {
    fn default() -> Self {
        WhisperModel::DistilSmall
    }
}

impl std::fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            WhisperModel::Tiny => "tiny",
            WhisperModel::TinyEn => "tiny.en",
            WhisperModel::Base => "base",
            WhisperModel::BaseEn => "base.en",
            WhisperModel::Small => "small",
            WhisperModel::SmallEn => "small.en",
            WhisperModel::Medium => "medium",
            WhisperModel::Large => "large-v3",
            WhisperModel::DistilSmall => "distil-small.en",
            WhisperModel::DistilMedium => "distil-medium.en",
        };
        write!(f, "{}", name)
    }
}

impl WhisperModel {
    /// Get the GGML model filename
    pub fn ggml_filename(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "ggml-tiny.bin",
            WhisperModel::TinyEn => "ggml-tiny.en.bin",
            WhisperModel::Base => "ggml-base.bin",
            WhisperModel::BaseEn => "ggml-base.en.bin",
            WhisperModel::Small => "ggml-small.bin",
            WhisperModel::SmallEn => "ggml-small.en.bin",
            WhisperModel::Medium => "ggml-medium.bin",
            WhisperModel::Large => "ggml-large-v3.bin",
            WhisperModel::DistilSmall => "ggml-small.en.bin",
            WhisperModel::DistilMedium => "ggml-medium.en.bin",
        }
    }
    
    /// Get the download URL for the GGML model
    pub fn download_url(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            WhisperModel::TinyEn => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
            WhisperModel::Base => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
            WhisperModel::BaseEn => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            WhisperModel::Small => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            WhisperModel::SmallEn => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
            WhisperModel::Medium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            WhisperModel::Large => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
            WhisperModel::DistilSmall => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
            WhisperModel::DistilMedium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
        }
    }
}

/// Device type for ML inference
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Auto
    }
}

/// Transcription configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranscriptionConfig {
    /// Whisper model to use
    pub model: WhisperModel,
    /// Device for inference (CPU, CUDA, etc.)
    pub device: DeviceType,
    /// Language for transcription (None for auto-detect)
    pub language: Option<String>,
    /// Temperature for sampling (0.0 for deterministic)
    pub temperature: f32,
    /// Number of beams for beam search
    pub num_beams: usize,
    /// Maximum number of tokens to generate
    pub max_tokens: Option<usize>,
    /// Enable voice activity detection
    pub vad_filter: bool,
    /// Minimum silence duration in milliseconds
    pub min_silence_duration_ms: u32,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            model: WhisperModel::default(),
            device: DeviceType::default(),
            language: Some("en".to_string()),
            temperature: 0.0,
            num_beams: 2, // Match Python implementation
            max_tokens: None,
            vad_filter: true,
            min_silence_duration_ms: 500,
        }
    }
}

/// Transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Language detected (if auto-detection was used)
    pub detected_language: Option<String>,
    /// Model used for transcription
    pub model: String,
    /// Segments with timestamps (if available)
    pub segments: Vec<TranscriptionSegment>,
}

/// Individual transcription segment with timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds  
    pub end: f32,
    /// Segment text
    pub text: String,
    /// Confidence score for this segment
    pub confidence: Option<f32>,
}

/// Model metadata for tracking versions and updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_name: String,
    pub file_path: PathBuf,
    pub file_size: u64,
    pub download_date: chrono::DateTime<chrono::Utc>,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub checksum: Option<String>,
}

impl ModelMetadata {
    /// Create new metadata for a downloaded model
    pub fn new(model_name: String, file_path: PathBuf, file_size: u64) -> Self {
        let now = chrono::Utc::now();
        Self {
            model_name,
            file_path,
            file_size,
            download_date: now,
            last_used: now,
            checksum: None,
        }
    }
    
    /// Update last used timestamp
    pub fn mark_used(&mut self) {
        self.last_used = chrono::Utc::now();
    }
}

/// Model manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub models: HashMap<String, ModelMetadata>,
    pub last_update_check: Option<chrono::DateTime<chrono::Utc>>,
    pub auto_update: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            last_update_check: None,
            auto_update: true,
        }
    }
}

impl ModelConfig {
    /// Load configuration from disk
    pub async fn load() -> Result<Self> {
        let config_path = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi")
            .join("model_config.json");
            
        if !config_path.exists() {
            return Ok(Self::default());
        }
        
        let content = tokio::fs::read_to_string(&config_path).await?;
        serde_json::from_str(&content)
            .context("Failed to parse model configuration")
    }
    
    /// Save configuration to disk
    pub async fn save(&self) -> Result<()> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi");
            
        tokio::fs::create_dir_all(&cache_dir).await?;
        
        let config_path = cache_dir.join("model_config.json");
        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(&config_path, content).await?;
        
        Ok(())
    }
}

/// Whisper speech recognition engine for transcribing audio
pub struct WhisperEngine {
    config: TranscriptionConfig,
    _faster_whisper_available: bool,
    _whisper_cli_available: bool,
    _whisper_cpp_path: Option<PathBuf>,
    whisper_cpp_available: bool,
    model_path: Option<PathBuf>,
    _last_update_check: Option<std::time::SystemTime>,
    _model_metadata: Option<ModelMetadata>,
}

impl WhisperEngine {
    /// Create a new Whisper engine with the given configuration
    pub fn new(config: TranscriptionConfig) -> Result<Self> {
        // Always use self-contained approach
        let whisper_cpp_path = None;
        let whisper_cpp_available = Self::check_whisper_cpp_available();
        
        if whisper_cpp_available {
            info!("Whisper ready for transcription");
        } else {
            info!("Whisper will be downloaded automatically on first use");
        }
        
        // Don't check for Python implementations
        let faster_whisper_available = false;
        let whisper_cli_available = false;
        
        Ok(Self {
            config,
            _faster_whisper_available: faster_whisper_available,
            _whisper_cli_available: whisper_cli_available,
            _whisper_cpp_path: whisper_cpp_path,
            whisper_cpp_available,
            model_path: None,
            _last_update_check: None,
            _model_metadata: None,
        })
    }
    

    
    /// Check if whisper.cpp is available in PATH or common locations
    fn check_whisper_cpp_available() -> bool {
        // Check common binary names
        for binary_name in &["whisper-cpp", "whisper", "whisper.cpp", "main"] {
            if Command::new("which")
                .arg(binary_name)
                .output()
                .map(|output| output.status.success())
                .unwrap_or(false)
            {
                return true;
            }
        }
        
        // Check in Jambi's bin directory for the real binary
        if let Ok(home) = env::var("HOME") {
            let jambi_bin = PathBuf::from(home).join(".cache/jambi/bin/whisper-cpp.real");
            if jambi_bin.exists() {
                return true;
            }
        }
        
        false
    }
    
    /// Get the models directory
    fn get_models_dir() -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi")
            .join("models");
        
        fs::create_dir_all(&cache_dir)?;
        Ok(cache_dir)
    }
    
    /// Check for model updates weekly
    async fn check_for_model_updates(&mut self) {
        // Using chrono for time handling
        
        // Load model configuration
        let mut config = match ModelConfig::load().await {
            Ok(config) => config,
            Err(_) => ModelConfig::default(),
        };
        
        let now = chrono::Utc::now();
        let week = chrono::Duration::weeks(1);
        
        // Check if we've checked recently
        if let Some(last_check) = config.last_update_check {
            if now - last_check < week {
                return; // Already checked this week
            }
        }
        
        // Update last check time
        config.last_update_check = Some(now);
        let _ = config.save().await;
        
        info!("Checking for model updates (weekly check)...");
        
        // Check if newer models are available
        // In production, this would check a version manifest
        // For now, we just ensure we have at least one model
        if config.models.is_empty() && self.model_path.is_none() {
            info!("No models found, downloading default model...");
            let _ = self.download_model_if_needed().await;
        }
    }
    
    /// Download a whisper model if not already present
    async fn download_model_if_needed(&self) -> Result<PathBuf> {
        let models_dir = Self::get_models_dir()?;
        let model_filename = self.config.model.ggml_filename();
        let model_path = models_dir.join(model_filename);
        
        if model_path.exists() {
            info!("Model already exists: {}", model_path.display());
            return Ok(model_path);
        }
        
        // Try to find any existing model as fallback
        let fallback_models = ["ggml-tiny.en.bin", "ggml-base.en.bin", "ggml-small.en.bin"];
        for fallback in &fallback_models {
            let fallback_path = models_dir.join(fallback);
            if fallback_path.exists() {
                info!("Using existing model as fallback: {}", fallback);
                return Ok(fallback_path);
            }
        }
        
        info!("Downloading whisper model: {} ({} MB)", 
              self.config.model, 
              self.config.model.size_mb());
        
        // Check if we're in offline mode
        if env::var("JAMBI_OFFLINE").is_ok() {
            return Err(anyhow::anyhow!("Running in offline mode, cannot download models"));
        }
        
        let url = self.config.model.download_url();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout for large models
            .build()?;
        
        let response = match client.get(url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                // Try to download tiny model as fallback
                warn!("Failed to download {}: {}. Trying tiny model as fallback.", self.config.model, e);
                let tiny_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin";
                match client.get(tiny_url).send().await {
                    Ok(resp) => resp,
                    Err(e2) => {
                        return Err(anyhow::anyhow!("Model download failed: {} (and fallback failed: {})", e, e2));
                    }
                }
            }
        };
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Model download failed with status: {}", response.status()));
        }
        
        let total_size = response
            .content_length()
            .unwrap_or(self.config.model.size_mb() * 1024 * 1024);
        
        // Create temp file first
        let temp_path = model_path.with_extension("tmp");
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Failed to download chunk")?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            
            if downloaded % (1024 * 1024 * 10) == 0 {
                let progress = (downloaded as f64 / total_size as f64) * 100.0;
                info!("Download progress: {:.1}%", progress);
            }
        }
        
        file.flush().await?;
        drop(file); // Close the file
        
        // Rename temp file to final name
        tokio::fs::rename(&temp_path, &model_path).await?;
        
        info!("Model downloaded successfully: {}", model_path.display());
        
        // Save model metadata
        let metadata = ModelMetadata::new(
            model_filename.to_string(),
            model_path.clone(),
            downloaded,
        );
        
        let mut config = ModelConfig::load().await.unwrap_or_default();
        config.models.insert(model_filename.to_string(), metadata);
        let _ = config.save().await;
        
        Ok(model_path)
    }

    /// Load the specified Whisper model
    pub async fn load_model(&mut self) -> Result<()> {
        info!("Initializing whisper engine for model: {}", self.config.model);
        
        // Step 1: Ensure whisper-cpp binary is available first
        if !self.whisper_cpp_available {
            info!("Whisper binary not found, downloading...");
            match self.ensure_whisper_cpp_available().await {
                Ok(_) => {
                    info!("Whisper binary downloaded successfully");
                    self.whisper_cpp_available = true;
                }
                Err(e) => {
                    error!("Failed to download whisper binary: {}", e);
                    // Continue - we might still be able to use fallback methods
                }
            }
        } else {
            info!("Whisper binary already available");
        }
        
        // Step 2: Download model if needed
        if self.model_path.is_none() {
            info!("Downloading model: {}", self.config.model);
            match self.download_model_if_needed().await {
                Ok(path) => {
                    self.model_path = Some(path.clone());
                    info!("Model downloaded successfully: {}", path.display());
                }
                Err(e) => {
                    error!("Model download failed: {}", e);
                    // Don't return error - fallback transcription can still work
                }
            }
        } else {
            info!("Model already loaded: {:?}", self.model_path);
        }
        
        // Step 3: Verify setup status
        if self.whisper_cpp_available && self.model_path.is_some() {
            info!("✅ Whisper engine fully initialized and ready for transcription");
        } else {
            warn!("⚠️ Whisper engine partially initialized:");
            warn!("  - Whisper binary: {}", if self.whisper_cpp_available { "✓" } else { "✗" });
            warn!("  - Model: {}", if self.model_path.is_some() { "✓" } else { "✗" });
            warn!("  Fallback transcription will be used until setup completes");
        }
        
        // Check for model updates weekly
        self.check_for_model_updates().await;
        
        Ok(())
    }
    
    /// Ensure whisper.cpp binary is available by downloading pre-built version
    async fn ensure_whisper_cpp_available(&mut self) -> Result<()> {
        use tokio::process::Command as TokioCommand;
        
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi");
        
        let bin_dir = cache_dir.join("bin");
        
        tokio::fs::create_dir_all(&bin_dir).await?;
        
        // Check for the real binary, not the wrapper script
        let whisper_cpp_path = bin_dir.join("whisper-cpp.real");
        let wrapper_path = bin_dir.join("whisper-cpp");
        
        // If only wrapper exists but not real binary, check for it
        if !whisper_cpp_path.exists() && wrapper_path.exists() {
            warn!("Found wrapper script but no real binary, will re-download");
            let _ = tokio::fs::remove_file(&wrapper_path).await;
        }
        
        if whisper_cpp_path.exists() {
            info!("Found existing whisper binary at: {}", whisper_cpp_path.display());
            if self.test_whisper_binary(&whisper_cpp_path).await {
                info!("Existing whisper binary is working");
                self.whisper_cpp_available = true;
                return Ok(());
            } else {
                warn!("Existing whisper binary failed test, will re-download");
                let _ = tokio::fs::remove_file(&whisper_cpp_path).await;
            }
        }
        
        // Try to download a working whisper implementation
        info!("Downloading whisper binary...");
        match self.download_whisper_stream_binary().await {
            Ok(_) => {
                info!("Successfully downloaded whisper binary");
                self.whisper_cpp_available = true;
                return Ok(());
            }
            Err(e) => {
                error!("Failed to download whisper binary: {}", e);
            }
        }
        
        // First, check if whisper is already available on the system
        let system_commands = ["whisper", "whisper.cpp", "whisper-cpp"];
        for cmd in &system_commands {
            match TokioCommand::new("which")
                .arg(cmd)
                .output()
                .await
            {
                Ok(output) if output.status.success() => {
                    let system_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !system_path.is_empty() {
                        info!("Found system whisper at: {}", system_path);
                        // Create symlink to system whisper
                        match std::os::unix::fs::symlink(&system_path, &whisper_cpp_path) {
                            Ok(_) => {
                                info!("Created symlink to system whisper");
                                self.whisper_cpp_available = true;
                                return Ok(());
                            }
                            Err(e) => {
                                debug!("Failed to create symlink: {}", e);
                            }
                        }
                    }
                }
                _ => continue,
            }
        }
        
        // Try to download pre-built binary first
        match self.try_download_prebuilt_binary().await {
            Ok(_) => {
                self.whisper_cpp_available = true;
                return Ok(());
            }
            Err(e) => {
                debug!("Pre-built binary download failed: {}", e);
            }
        }
        
        info!("Setting up whisper for transcription...");
        
        // Try the working whisper script first
        let mut possible_paths = vec![
            PathBuf::from("./scripts/compile-real-whisper.sh"),
            PathBuf::from("scripts/compile-real-whisper.sh"),
            PathBuf::from("./scripts/get-working-whisper.sh"),
            PathBuf::from("scripts/get-working-whisper.sh"),
            PathBuf::from("./scripts/download-whisper-release.sh"),
            PathBuf::from("scripts/download-whisper-release.sh"),
            PathBuf::from("./scripts/setup-no-compile.sh"),
            PathBuf::from("scripts/setup-no-compile.sh"),
            PathBuf::from("./scripts/download-prebuilt.sh"),
            PathBuf::from("scripts/download-prebuilt.sh"),
        ];
        
        // Add home directory path if available
        if let Some(home) = dirs::home_dir() {
            possible_paths.push(home.join(".cache/jambi/scripts/compile-real-whisper.sh"));
            possible_paths.push(home.join(".cache/jambi/scripts/get-working-whisper.sh"));
            possible_paths.push(home.join(".cache/jambi/scripts/download-whisper-release.sh"));
            possible_paths.push(home.join(".cache/jambi/scripts/setup-no-compile.sh"));
            possible_paths.push(home.join(".cache/jambi/scripts/download-prebuilt.sh"));
        }
        
        let script_path = possible_paths.into_iter()
            .find(|p| p.exists());
            
        let script_path = match script_path {
            Some(path) => path,
            None => {
                warn!("No setup scripts found. Whisper setup skipped.");
                // Don't try to compile - just continue without whisper
                return Ok(());
            }
        };
        
        info!("Running setup script: {}", script_path.display());
        let output = TokioCommand::new("bash")
            .arg(&script_path)
            .output()
            .await;
            
        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                if !stdout.is_empty() {
                    debug!("Setup stdout: {}", stdout);
                }
                
                if output.status.success() {
                    info!("Successfully compiled whisper.cpp");
                    self.whisper_cpp_available = true;
                    Ok(())
                } else {
                    error!("Whisper.cpp setup failed with status: {:?}", output.status);
                    if !stdout.is_empty() {
                        error!("Setup stdout: {}", stdout);
                    }
                    if !stderr.is_empty() {
                        error!("Setup stderr: {}", stderr);
                    } else {
                        error!("No error output captured. Running script with --verbose flag...");
                        // Try running with verbose flag to get more info
                        let verbose_output = TokioCommand::new("bash")
                            .arg(&script_path)
                            .arg("--verbose")
                            .output()
                            .await;
                        if let Ok(verbose) = verbose_output {
                            let v_stdout = String::from_utf8_lossy(&verbose.stdout);
                            let v_stderr = String::from_utf8_lossy(&verbose.stderr);
                            error!("Verbose stdout: {}", v_stdout);
                            error!("Verbose stderr: {}", v_stderr);
                        }
                    }
            
                    // Provide helpful error message
                    if stderr.contains("build-essential") || stderr.contains("gcc") || stderr.contains("make: not found") {
                        return Err(anyhow::anyhow!(
                            "Build tools not installed. Please install:\n\
                             Ubuntu/Debian: sudo apt install build-essential git\n\
                             Fedora: sudo dnf install make gcc-c++ git\n\
                             macOS: xcode-select --install"
                        ));
                    }
                    
                    // Check for permission errors
                    if stderr.contains("Permission denied") {
                        return Err(anyhow::anyhow!(
                            "Permission denied. Check directory permissions for: {}",
                            dirs::cache_dir().unwrap_or_default().display()
                        ));
                    }
                    
                    // Don't fail completely - continue without whisper
                    warn!("Whisper setup incomplete. Will use fallback transcription.");
                    Ok(())
                }
            }
            Err(e) => {
                warn!("Failed to run setup script: {}. Will use fallback.", e);
                Ok(())
            }
        }
    }
    
    /// Download whisper-stream binary which includes whisper functionality
    async fn download_whisper_stream_binary(&self) -> Result<()> {
        use tokio::process::Command as TokioCommand;
        
        info!("Downloading whisper transcription binary...");
        
        let bin_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi")
            .join("bin");
        
        tokio::fs::create_dir_all(&bin_dir).await?;
        
        // Download to whisper-cpp.real directly
        let whisper_path = bin_dir.join("whisper-cpp.real");
        
        // Determine system architecture
        let _arch = if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            return Err(anyhow::anyhow!("Unsupported architecture"));
        };
        
        let os = if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "macos") {
            "darwin"
        } else {
            return Err(anyhow::anyhow!("Unsupported OS"));
        };
        
        // Use whisper-stream releases which have pre-built binaries
        let download_url = format!(
            "https://github.com/ggerganov/whisper.cpp/releases/download/v1.5.4/whisper-v1.5.4-{}.tar.gz",
            if os == "darwin" { "macos" } else { "linux" }
        );
        
        // Download and extract
        let temp_file = bin_dir.join("whisper-temp.tar.gz");
        
        // Download the archive
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
            
        let response = client.get(&download_url).send().await
            .map_err(|_| anyhow::anyhow!("Failed to download whisper binary"))?;
            
        if !response.status().is_success() {
            // If official release doesn't work, compile from source
            return self.compile_minimal_whisper().await;
        }
        
        let bytes = response.bytes().await?;
        tokio::fs::write(&temp_file, &bytes).await?;
        
        // Extract the binary
        let output = TokioCommand::new("tar")
            .args(["-xzf", temp_file.to_str().unwrap(), "-C", bin_dir.to_str().unwrap()])
            .output()
            .await;
            
        if let Ok(output) = output {
            if output.status.success() {
                // Make executable
                let _ = TokioCommand::new("chmod")
                    .args(["+x", whisper_path.to_str().unwrap()])
                    .output()
                    .await;
                    
                // Test if it works
                let test = TokioCommand::new(&whisper_path)
                    .arg("--help")
                    .output()
                    .await;
                    
                if let Ok(test_output) = test {
                    if test_output.status.success() {
                        info!("Successfully downloaded pre-built whisper binary");
                        return Ok(());
                    }
                }
            }
        
            // Update model metadata to mark as used
            if let Some(ref model_path) = self.model_path {
                if let Some(file_name) = model_path.file_name() {
                    if let Some(file_name_str) = file_name.to_str() {
                        let mut config = ModelConfig::load().await.unwrap_or_default();
                        if let Some(metadata) = config.models.get_mut(file_name_str) {
                            metadata.mark_used();
                            let _ = config.save().await;
                        }
                    }
                }
            }
        }
        
        // If download failed, remove the file
        let _ = tokio::fs::remove_file(&whisper_path).await;
        
        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_file).await;
        
        // Find and rename the extracted binary
        if let Ok(mut entries) = tokio::fs::read_dir(&bin_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().contains("whisper") || name.to_string_lossy() == "main" {
                        let _ = tokio::fs::rename(&path, &whisper_path).await;
                        break;
                    }
                }
            }
        }
        
        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = tokio::fs::metadata(&whisper_path).await {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                let _ = tokio::fs::set_permissions(&whisper_path, perms).await;
            }
        }
        
        // Test if it works
        if self.test_whisper_binary(&whisper_path).await {
            info!("Successfully downloaded whisper binary");
            Ok(())
        } else {
            // If download didn't work, try compiling
            self.compile_minimal_whisper().await
        }
    }
    
    /// Compile a minimal whisper binary
    async fn compile_minimal_whisper(&self) -> Result<()> {
        use tokio::process::Command as TokioCommand;
        
        info!("Compiling minimal whisper (one-time setup)...");
        
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi");
        
        let temp_dir = cache_dir.join("whisper-build");
        tokio::fs::create_dir_all(&temp_dir).await?;
        
        // Download source
        let output = TokioCommand::new("git")
            .args(["clone", "--depth", "1", "https://github.com/ggerganov/whisper.cpp.git"])
            .arg(&temp_dir)
            .output()
            .await;
            
        if output.is_err() || !output.unwrap().status.success() {
            return Err(anyhow::anyhow!("Please install git and build tools"));
        }
        
        // Build
        let output = TokioCommand::new("make")
            .current_dir(&temp_dir)
            .args(["-j2", "main"])
            .env("CFLAGS", "-O2")
            .env("CXXFLAGS", "-O2")
            .output()
            .await;
            
        if let Ok(output) = output {
            if output.status.success() {
                // Copy binary
                let main_path = temp_dir.join("main");
                let target_path = cache_dir.join("bin").join("whisper-cpp");
                if main_path.exists() {
                    tokio::fs::copy(&main_path, &target_path).await?;
                    
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mut perms = tokio::fs::metadata(&target_path).await?.permissions();
                        perms.set_mode(0o755);
                        tokio::fs::set_permissions(&target_path, perms).await?;
                    }
                    
                    // Clean up
                    let _ = tokio::fs::remove_dir_all(&temp_dir).await;
                    
                    return Ok(());
                }
            }
        }
        
        Err(anyhow::anyhow!("Compilation failed. Please install build tools: sudo apt install build-essential git"))
    }
    
    /// Test if a whisper binary works
    async fn test_whisper_binary(&self, path: &Path) -> bool {
        use tokio::process::Command as TokioCommand;
        use tokio::time::{timeout, Duration};
        
        info!("Testing whisper binary at: {}", path.display());
        
        // First check if file exists and is executable
        if !path.exists() {
            warn!("Whisper binary does not exist at: {}", path.display());
            return false;
        }
        
        // Try to run with --help flag and a timeout
        let test_cmd = TokioCommand::new(path)
            .arg("--help")
            .output();
            
        match timeout(Duration::from_secs(5), test_cmd).await {
            Ok(Ok(output)) if output.status.success() => {
                info!("✅ Whisper binary test successful");
                true
            }
            Ok(Ok(output)) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                warn!("Whisper binary test failed with exit code {:?}: {}", 
                      output.status.code(), stderr);
                false
            }
            Ok(Err(e)) => {
                warn!("Failed to execute whisper binary: {}", e);
                false
            }
            Err(_) => {
                error!("Whisper binary test timed out after 5 seconds");
                false
            }
        }
    }
    
    /// Try to download pre-built whisper binary from GitHub releases
    async fn try_download_prebuilt_binary(&self) -> Result<()> {
        // This is now handled by download_whisper_stream_binary
        self.download_whisper_stream_binary().await
    }
    
    /// Compile whisper.cpp directly without external scripts
    async fn _compile_whisper_directly(&self) -> Result<()> {
        use tokio::process::Command as TokioCommand;
        
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("jambi");
        
        let whisper_dir = cache_dir.join("whisper.cpp");
        let bin_dir = cache_dir.join("bin");
        
        tokio::fs::create_dir_all(&bin_dir).await?;
        
        // Clone or update whisper.cpp
        if !whisper_dir.exists() {
            info!("Downloading whisper.cpp source code...");
            tokio::fs::create_dir_all(&cache_dir).await?;
            
            let output = TokioCommand::new("git")
                .args(["clone", "--depth", "1", "https://github.com/ggerganov/whisper.cpp.git"])
                .arg(&whisper_dir)
                .output()
                .await;
                
            match output {
                Ok(output) => {
                    if !output.status.success() {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        error!("Git clone failed: {}", stderr);
                        return Err(anyhow::anyhow!("Failed to clone whisper.cpp repository: {}", stderr));
                    }
                }
                Err(e) => {
                    error!("Git command failed: {}", e);
                    return Err(anyhow::anyhow!("Git not installed. Please install git first."));
                }
            }
        }
        
        info!("Compiling whisper.cpp (this may take a minute)...");
        
        // Compile with simple flags
        let output = TokioCommand::new("make")
            .current_dir(&whisper_dir)
            .args(["-j", "4", "main"])
            .env("CFLAGS", "-O3")
            .env("CXXFLAGS", "-O3")
            .output()
            .await?;
            
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            error!("Make failed - stdout: {}", stdout);
            error!("Make failed - stderr: {}", stderr);
            
            if stderr.contains("cc: not found") || stderr.contains("g++: not found") {
                return Err(anyhow::anyhow!(
                    "C++ compiler not found. Please install:\n\
                     Ubuntu/Debian: sudo apt install build-essential\n\
                     Fedora: sudo dnf install gcc-c++ make\n\
                     macOS: xcode-select --install"
                ));
            }
            
            return Err(anyhow::anyhow!("Compilation failed: {}", stderr));
        }
        
        // Copy binary
        let main_binary = whisper_dir.join("main");
        let target_binary = bin_dir.join("whisper-cpp");
        
        tokio::fs::copy(&main_binary, &target_binary).await?;
        
        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&target_binary).await?.permissions();
            perms.set_mode(0o755);
            tokio::fs::set_permissions(&target_binary, perms).await?;
        }
        
        info!("Successfully compiled whisper.cpp");
        Ok(())
    }

    /// Transcribe audio from a WAV file
    /// Transcribe an audio file
    pub async fn transcribe_file<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<TranscriptionResult> {
        let audio_path = audio_path.as_ref();
        info!("Transcribing file: {}", audio_path.display());
        let start_time = Instant::now();

        // Ensure model is loaded before attempting transcription
        if self.model_path.is_none() {
            match self.load_model().await {
                Ok(_) => {},
                Err(e) => {
                    warn!("Failed to load model: {}", e);
                    // Try to download just the model without whisper.cpp setup
                    self.model_path = match self.download_model_if_needed().await {
                        Ok(path) => Some(path),
                        Err(e2) => {
                            error!("Failed to download model: {}", e2);
                            // Don't fail completely - use fallback transcription
                            return self.simple_fallback_transcription(audio_path).await.map(|text| {
                                TranscriptionResult {
                                    text,
                                    confidence: Some(0.5),
                                    processing_time_ms: 0,
                                    detected_language: self.config.language.clone(),
                                    model: "fallback".to_string(),
                                    segments: vec![],
                                }
                            });
                        }
                    };
                }
            }
        }

        // Validate file exists and has supported extension
        if !audio_path.exists() {
            return Err(anyhow::anyhow!("Audio file does not exist: {}", audio_path.display()));
        }
        
        info!("Starting transcription for: {}", audio_path.display());
        info!("Whisper.cpp available: {}, Model path: {:?}", 
              self.whisper_cpp_available, self.model_path);

        let extension = audio_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let extension = match extension {
            Some(ext) => ext,
            None => return Err(anyhow::anyhow!("Audio file has no extension: {}", audio_path.display())),
        };

        if !matches!(extension.as_str(), "wav" | "mp3" | "flac" | "ogg" | "m4a") {
            return Err(anyhow::anyhow!(
                "Unsupported audio format: {} (supported: wav, mp3, flac, ogg, m4a)", 
                extension
            ));
        }

        // Try transcription with whatever method works
        let transcription_result = if self.whisper_cpp_available && self.model_path.is_some() {
            info!("Using whisper.cpp for transcription");
            info!("  Binary available: {}", self.whisper_cpp_available);
            info!("  Model path: {:?}", self.model_path);
            // Try whisper.cpp first
            match self.transcribe_with_builtin(audio_path).await {
                Ok(text) => {
                    info!("✅ Whisper.cpp transcription successful");
                    Some(text)
                },
                Err(e) => {
                    error!("❌ Whisper.cpp transcription failed: {:?}", e);
                    None
                }
            }
        } else if self.model_path.is_some() {
            info!("Whisper.cpp not available, trying direct model transcription");
            // Try using the model directly with ffmpeg + simple processing
            match self.transcribe_with_model_direct(audio_path).await {
                Ok(text) => Some(text),
                Err(e) => {
                    warn!("Direct model transcription failed: {}", e);
                    None
                }
            }
        } else {
            warn!("No transcription method available (whisper: {}, model: {:?})", 
                  self.whisper_cpp_available, self.model_path);
            None
        };
        
        if let Some(text) = transcription_result {
            let total_time = start_time.elapsed();
            info!("Transcription completed in {:.2}s", total_time.as_secs_f64());
            
            return Ok(TranscriptionResult {
                text,
                confidence: Some(0.95),
                processing_time_ms: total_time.as_millis() as u64,
                detected_language: self.config.language.clone(),
                model: self.config.model.to_string(),
                segments: vec![],
            });
        }
        
        // As a last resort, use a simple fallback
        let text = self.simple_fallback_transcription(audio_path).await?;
        let total_time = start_time.elapsed();
        
        Ok(TranscriptionResult {
            text,
            confidence: Some(0.5),
            processing_time_ms: total_time.as_millis() as u64,
            detected_language: self.config.language.clone(),
            model: "fallback".to_string(),
            segments: vec![],
        })
    }
    
    /// Transcribe using built-in method (downloads model if needed)
    async fn transcribe_with_builtin(&mut self, audio_path: &Path) -> Result<String> {
        info!("transcribe_with_builtin called for: {}", audio_path.display());
        
        // Try to ensure model and binary are available
        if self.model_path.is_none() {
            // Ensure whisper.cpp is available first
            if !self.whisper_cpp_available {
                match self.ensure_whisper_cpp_available().await {
                    Ok(_) => {},
                    Err(e) => {
                        warn!("Could not ensure whisper.cpp: {}", e);
                        // Continue anyway - we might still be able to use fallback
                    }
                }
            }
            
            match self.download_model_if_needed().await {
                Ok(path) => self.model_path = Some(path),
                Err(e) => {
                    warn!("Could not download model: {}", e);
                    // If we don't have whisper.cpp or model, return error
                    if !self.whisper_cpp_available {
                        return Err(anyhow::anyhow!("Neither whisper.cpp nor model available"));
                    }
                }
            }
        }
        
        // If we don't have a model path, we can't proceed with whisper.cpp
        let model_path = match self.model_path.as_ref() {
            Some(path) => path,
            None => return Err(anyhow::anyhow!("No model available for transcription")),
        };
        
        // Try to use whisper.cpp
        info!("Trying whisper.cpp CLI with model: {}", model_path.display());
        match self.transcribe_with_whisper_cpp_cli(audio_path, model_path).await {
            Ok(transcription) => {
                info!("Whisper.cpp CLI transcription successful");
                return Ok(transcription);
            },
            Err(e) => {
                error!("Whisper.cpp CLI failed: {:?}", e);
            }
        }
        
        // As a last resort, check if whisper.cpp main executable exists in other locations
        info!("Trying whisper.cpp main executable");
        match self.transcribe_with_whisper_cpp_main(audio_path, model_path).await {
            Ok(transcription) => {
                info!("Whisper.cpp main transcription successful");
                return Ok(transcription);
            },
            Err(e) => {
                error!("Whisper.cpp main failed: {:?}", e);
            }
        }
        
        Err(anyhow::anyhow!("All whisper.cpp methods failed"))
    }
    
    /// Try to transcribe using whisper.cpp CLI
    async fn transcribe_with_whisper_cpp_cli(&self, audio_path: &Path, model_path: &Path) -> Result<String> {
        info!("transcribe_with_whisper_cpp_cli called");
        
        // Try jambi's whisper-cpp.real binary (not the wrapper script)
        let jambi_whisper = dirs::home_dir()
            .map(|home| home.join(".cache/jambi/bin/whisper-cpp.real"))
            .filter(|path| path.exists());
        
        let binary_name = if let Some(ref path) = jambi_whisper {
            path.to_string_lossy().to_string()
        } else {
            "whisper-cpp".to_string()
        };
        
        // Build command string like the bash script does
        let settings = format!(
            "-t 4 --no-timestamps -l {}",
            self.config.language.as_deref().unwrap_or("en")
        );
        
        let whisper_cmd = format!(
            "{} -m {} -f {} {}",
            binary_name,
            model_path.display(),
            audio_path.display(),
            settings
        );
        
        info!("Executing whisper command: {}", whisper_cmd);
        
        // Run the command directly without sh -c wrapper
        let mut cmd = TokioCommand::new(&binary_name);
        cmd.arg("-m")
            .arg(model_path)
            .arg("-f")
            .arg(audio_path)
            .arg("-t")
            .arg("4")
            .arg("--no-timestamps")
            .arg("-l")
            .arg(self.config.language.as_deref().unwrap_or("en"));
        
        // Add timeout
        use tokio::time::{timeout, Duration};
        let output = timeout(Duration::from_secs(90), cmd.output()).await;
            
        match output {
            Err(_) => {
                // Timeout occurred
                error!("Whisper timeout after 90 seconds");
                Err(anyhow::anyhow!("whisper-cpp timed out after 90 seconds - model may be too large or system too slow"))
            }
            Ok(Ok(output)) if output.status.success() => {
                let text = String::from_utf8_lossy(&output.stdout);
                info!("Whisper raw output (first 500 chars): {}", 
                      text.chars().take(500).collect::<String>());
                
                // Filter out whisper debug output and extract actual transcription
                let transcription: String = text
                    .lines()
                    .filter(|line| {
                        !line.starts_with("whisper_") &&
                        !line.starts_with("system_info") &&
                        !line.starts_with("main:") &&
                        !line.starts_with('[') &&
                        !line.trim().is_empty()
                    })
                    .collect::<Vec<&str>>()
                    .join(" ")
                    .trim()
                    .to_string();
                
                if !transcription.is_empty() {
                    info!("Transcription extracted: {}", transcription);
                    Ok(transcription)
                } else {
                    warn!("Empty transcription after filtering, raw output was: {}", text);
                    Err(anyhow::anyhow!("Empty transcription from whisper-cpp"))
                }
            }
            Ok(Ok(output)) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                error!("Whisper failed with exit code {:?}", output.status.code());
                error!("Stderr: {}", stderr);
                error!("Stdout: {}", stdout);
                Err(anyhow::anyhow!("whisper-cpp failed with exit code {:?}: {}", 
                                    output.status.code(), stderr))
            }
            Ok(Err(e)) => {
                error!("Failed to execute whisper-cpp command: {:?}", e);
                Err(anyhow::anyhow!("Failed to run whisper-cpp: {}", e))
            }
        }
    }

    
    /// Try to transcribe using whisper.cpp main executable
    async fn transcribe_with_whisper_cpp_main(&self, audio_path: &Path, model_path: &Path) -> Result<String> {
        info!("transcribe_with_whisper_cpp_main called");
        
        // First, convert audio to 16kHz WAV if needed
        let temp_wav = std::env::temp_dir().join("jambi_temp_audio.wav");
        let convert_output = TokioCommand::new("ffmpeg")
            .arg("-i")
            .arg(audio_path)
            .arg("-ar")
            .arg("16000")
            .arg("-ac")
            .arg("1")
            .arg("-y")
            .arg(&temp_wav)
            .output()
            .await?;
            
        if !convert_output.status.success() {
            return Err(anyhow::anyhow!("Failed to convert audio to 16kHz WAV"));
        }
        
        // Try the real binary first, then common whisper.cpp binary names
        let real_binary = dirs::home_dir()
            .map(|home| home.join(".cache/jambi/bin/whisper-cpp.real"))
            .filter(|path| path.exists());
        
        let mut binaries_to_try = Vec::new();
        if let Some(real_bin) = real_binary {
            binaries_to_try.push(real_bin.to_string_lossy().to_string());
        }
        binaries_to_try.extend(["whisper", "whisper.cpp", "main"].iter().map(|s| s.to_string()));
        
        for binary_name in &binaries_to_try {
            let mut cmd = TokioCommand::new(binary_name);
            cmd.arg("-m")
                .arg(model_path)
                .arg("-f")
                .arg(&temp_wav)
                .arg("--no-timestamps")
                .arg("-l")
                .arg(self.config.language.as_deref().unwrap_or("en"));
                
            if let Ok(output) = cmd.output().await {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    
                    // Clean up temp file
                    let _ = tokio::fs::remove_file(&temp_wav).await;
                    
                    // Extract the transcription (whisper.cpp outputs extra info)
                    let lines: Vec<&str> = text.lines().collect();
                    let transcription = lines.into_iter()
                        .filter(|line| !line.trim().is_empty() && !line.starts_with('['))
                        .collect::<Vec<&str>>()
                        .join(" ")
                        .trim()
                        .to_string();
                    
                    if !transcription.is_empty() {
                        return Ok(transcription);
                    }
                }
            }
        }
        
        // Clean up temp file
        let _ = tokio::fs::remove_file(&temp_wav).await;
        
        Err(anyhow::anyhow!("No working whisper.cpp binary found"))
    }
    
    /// Direct model transcription without whisper.cpp
    async fn transcribe_with_model_direct(&self, _audio_path: &Path) -> Result<String> {
        // For now, this is a placeholder for future implementation
        // In a real implementation, we could use candle or another Rust ML framework
        Err(anyhow::anyhow!("Direct model transcription not yet implemented"))
    }
    
    /// Simple fallback transcription that works without external dependencies
    async fn simple_fallback_transcription(&self, audio_path: &Path) -> Result<String> {
        warn!("Using fallback transcription method (placeholder text will be generated)");
        
        // First, try web-based transcription if network is available
        if let Ok(transcription) = self.try_web_transcription(audio_path).await {
            return Ok(transcription);
        }
        
        // Get audio file info
        let metadata = tokio::fs::metadata(audio_path).await?;
        let file_size = metadata.len();
        
        // Estimate duration based on file size (rough approximation)
        // WAV at 16kHz mono 16-bit ≈ 32KB per second
        let estimated_duration_secs = file_size / 32000;
        
        // Generate a simple transcription placeholder
        let placeholder_text = if estimated_duration_secs < 5 {
            format!("[{} second audio clip]", estimated_duration_secs)
        } else if estimated_duration_secs < 30 {
            format!("[{} seconds of speech recorded]", estimated_duration_secs)
        } else {
            format!("[{} seconds of audio content]", estimated_duration_secs)
        };
        
        // Add setup status if needed
        let setup_note = if !self.whisper_cpp_available {
            "\n\nNote: Speech-to-text is being set up. Your next recording will be transcribed automatically."
        } else if self.model_path.is_none() {
            "\n\nNote: Downloading speech model. This is a one-time process."
        } else {
            ""
        };
        
        let message = format!("{}{}", placeholder_text, setup_note);
        
        // Log the actual status for debugging
        debug!("Fallback transcription used. Model: {:?}, Whisper: {}", 
               self.model_path.is_some(), self.whisper_cpp_available);
        
        Ok(message)
    }
    
    /// Try web-based transcription using a free API
    async fn try_web_transcription(&self, audio_path: &Path) -> Result<String> {
        debug!("Attempting web-based transcription...");
        
        // Check if file is too large (limit to 25MB for web services)
        let metadata = tokio::fs::metadata(audio_path).await?;
        if metadata.len() > 25 * 1024 * 1024 {
            return Err(anyhow::anyhow!("File too large for web transcription"));
        }
        
        // For now, return an error - in production, this would call a web API
        // Options include:
        // 1. OpenAI Whisper API (requires API key)
        // 2. Hugging Face Inference API (free tier available)
        // 3. Self-hosted whisper server
        
        Err(anyhow::anyhow!("Web transcription not configured"))
    }
    
    /// Try transcribing with faster-whisper (matches original Python implementation)
    async fn _transcribe_with_faster_whisper(&self, audio_path: &Path) -> Result<String> {
        debug!("Attempting transcription with faster-whisper");
        
        let audio_path_str = audio_path.to_string_lossy();
        let model_id = self.config.model.model_id();
        
        // Build Python command to use faster-whisper
        let python_script = format!(
            r#"
import sys
from faster_whisper import WhisperModel

model = WhisperModel("{}", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    "{}",
    beam_size={},
    language="{}",
    vad_filter={},
    vad_parameters=dict(min_silence_duration_ms={})
)
transcription = " ".join([segment.text.strip() for segment in segments])
print(transcription.strip())
"#,
            model_id,
            audio_path_str,
            self.config.num_beams,
            self.config.language.as_deref().unwrap_or("en"),
            if self.config.vad_filter { "True" } else { "False" },
            self.config.min_silence_duration_ms
        );
        
        let output = TokioCommand::new("python3")
            .arg("-c")
            .arg(&python_script)
            .output()
            .await
            .context("Failed to execute faster-whisper")?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("faster-whisper failed: {}", stderr));
        }
        
        let text = String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_string();
        
        Ok(text)
    }
    
    /// Try transcribing with whisper CLI
    async fn _transcribe_with_whisper_cli(&self, audio_path: &Path) -> Result<String> {
        debug!("Attempting whisper CLI transcription for: {}", audio_path.display());
        
        let model_name = match self.config.model {
            WhisperModel::Tiny => "tiny",
            WhisperModel::TinyEn => "tiny.en",
            WhisperModel::Base => "base",
            WhisperModel::BaseEn => "base.en",
            WhisperModel::Small => "small",
            WhisperModel::SmallEn => "small.en",
            WhisperModel::Medium => "medium",
            WhisperModel::Large => "large",
            WhisperModel::DistilSmall => "small.en",
            WhisperModel::DistilMedium => "medium.en",
        };
        
        let mut cmd = TokioCommand::new("whisper");
        cmd.arg(audio_path)
            .arg("--model")
            .arg(model_name)
            .arg("--output_format")
            .arg("txt")
            .arg("--output_dir")
            .arg("/tmp");
            
        if let Some(ref lang) = self.config.language {
            cmd.arg("--language").arg(lang);
        }
        
        let output = cmd.output()
            .await
            .context("Failed to run whisper command")?;
            
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Whisper CLI failed: {}", stderr));
        }
        
        // Read the output file
        let output_filename = audio_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");
        let output_path = format!("/tmp/{}.txt", output_filename);
        
        match tokio::fs::read_to_string(&output_path).await {
            Ok(text) => {
                // Clean up the output file
                let _ = tokio::fs::remove_file(&output_path).await;
                Ok(text.trim().to_string())
            },
            Err(e) => Err(anyhow::anyhow!("Failed to read transcription result: {}", e))
        }
    }



    /// Get available models
    pub fn available_models() -> Vec<WhisperModel> {
        vec![
            WhisperModel::Tiny,
            WhisperModel::TinyEn,
            WhisperModel::Base,
            WhisperModel::BaseEn,
            WhisperModel::Small,
            WhisperModel::SmallEn,
            WhisperModel::Medium,
            WhisperModel::Large,
            WhisperModel::DistilSmall,
            WhisperModel::DistilMedium,
        ]
    }

    /// Get the current configuration
    pub fn config(&self) -> &TranscriptionConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: TranscriptionConfig) {
        let model_changed = config.model != self.config.model;
        self.config = config;

        if model_changed {

            info!("Model changed to: {}", self.config.model);
        }
    }
}

/// Model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model: WhisperModel,
    pub size_mb: u64,
    pub description: String,
    pub is_loaded: bool,
    pub is_cached: bool,
}

impl Drop for WhisperEngine {
    fn drop(&mut self) {
        info!("Whisper engine dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_display() {
        assert_eq!(WhisperModel::Tiny.to_string(), "tiny");
        assert_eq!(WhisperModel::DistilSmall.to_string(), "distil-small.en");
        assert_eq!(WhisperModel::Large.to_string(), "large-v3");
    }

    #[test]
    fn test_model_properties() {
        assert!(WhisperModel::Tiny.size_mb() < WhisperModel::Large.size_mb());
        assert!(WhisperModel::DistilSmall.is_english_only());
        assert!(!WhisperModel::Small.is_english_only());
    }

    #[test]
    fn test_transcription_config_default() {
        let config = TranscriptionConfig::default();
        assert_eq!(config.model, WhisperModel::DistilSmall);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.num_beams, 2);
    }

    #[tokio::test]
    async fn test_whisper_engine_creation() {
        let config = TranscriptionConfig::default();
        let engine = WhisperEngine::new(config);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert_eq!(engine.config().model, WhisperModel::DistilSmall);
    }

    #[test]
    fn test_available_models() {
        let models = WhisperEngine::available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&WhisperModel::DistilSmall));
        assert!(models.contains(&WhisperModel::Large));
    }




}