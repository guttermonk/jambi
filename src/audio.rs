//! Audio recording module using CPAL for cross-platform audio capture
//! 
//! This module provides functionality to record audio from system microphones
//! and save it to WAV files for transcription processing.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleRate, Stream, StreamConfig, BufferSize};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use chrono::Utc;
use hound::{WavSpec, WavWriter};
use std::io::BufWriter;
use std::fs::File;
use std::time::{Duration, Instant};

use std::process::{Command, Stdio};
use tokio::process::{Command as TokioCommand, Child};
use tokio::time::{timeout, sleep};
use tracing::{info, warn, error};


/// Audio recording configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioConfig {
    /// Sample rate in Hz (typically 16000 for speech)
    pub sample_rate: u32,
    /// Number of audio channels (1 for mono, 2 for stereo)
    pub channels: u16,
    /// Buffer size for audio processing
    pub buffer_size: usize,
    /// Directory to save recordings
    pub output_dir: PathBuf,
    /// Maximum recording duration
    pub max_duration: Option<u64>,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            buffer_size: 1024,
            output_dir: std::env::temp_dir().join("jambi"),
            max_duration: Some(300), // 5 minutes default
            verbose: false,
        }
    }
}

/// Information about a completed recording
#[derive(Debug, Clone)]
pub struct RecordingInfo {
    pub id: String,
    pub file_path: PathBuf,
    pub duration: Duration,
    pub file_size: u64,
}

/// Recording backend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecordingBackend {
    /// Use sox command-line tool (like original Python implementation)
    Sox,
    /// Use CPAL library
    Cpal,
}

/// Audio recording state
#[derive(Debug, Clone, PartialEq)]
pub enum RecordingState {
    Idle,
    Recording,
    Stopping,
    Error(String),
}

/// Audio recorder with multiple backend support
pub struct AudioRecorder {
    config: AudioConfig,
    backend: RecordingBackend,
    // CPAL-specific fields
    host: Option<Host>,
    device: Option<Device>,
    stream: Option<Stream>,
    // Sox-specific fields
    sox_process: Option<Child>,
    // Common fields
    state: Arc<Mutex<RecordingState>>,
    writer: Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>,
    start_time: Arc<Mutex<Option<Instant>>>,
    current_recording_path: Option<PathBuf>,
}

impl AudioRecorder {
    /// Check if audio recording is available and return backend info
    pub fn check_audio_availability() -> Result<String> {
        // Check if sox is available
        let sox_available = Command::new("sox")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if sox_available {
            Ok("sox (command-line tool)".to_string())
        } else {
            // Check CPAL
            let host = cpal::default_host();
            match host.default_input_device() {
                Some(device) => {
                    let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                    Ok(format!("CPAL audio library (device: {})", name))
                }
                None => Err(anyhow::anyhow!(
                    "No audio recording method available. Please install 'sox' for audio recording:\n\
                    Ubuntu/Debian: sudo apt install sox\n\
                    Fedora: sudo dnf install sox\n\
                    macOS: brew install sox"
                ))
            }
        }
    }

    /// Create a new audio recorder with the given configuration
    pub fn new(config: AudioConfig) -> Result<Self> {
        // Ensure output directory exists
        std::fs::create_dir_all(&config.output_dir)
            .context("Failed to create output directory")?;

        // Check if sox is available
        let sox_available = Command::new("sox")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        let (backend, host) = if sox_available {
            if config.verbose {
                info!("Using sox for audio recording");
            }
            (RecordingBackend::Sox, None)
        } else {
            if config.verbose {
                info!("sox not found, falling back to CPAL");
            }
            (RecordingBackend::Cpal, Some(cpal::default_host()))
        };
        
        Ok(Self {
            config,
            backend,
            host,
            device: None,
            stream: None,
            sox_process: None,
            state: Arc::new(Mutex::new(RecordingState::Idle)),
            writer: Arc::new(Mutex::new(None)),
            start_time: Arc::new(Mutex::new(None)),
            current_recording_path: None,
        })
    }

    /// List available audio input devices
    pub fn list_devices(&self) -> Result<Vec<String>> {
        match self.backend {
            RecordingBackend::Sox => {
                // For sox, we can't easily list devices, but we can check if it works
                let sox_works = Command::new("sox")
                    .arg("-d")
                    .arg("-n")
                    .arg("trim")
                    .arg("0")
                    .arg("0.001")
                    .output()
                    .map(|output| output.status.success())
                    .unwrap_or(false);
                
                if sox_works {
                    Ok(vec!["Default audio device (via sox)".to_string()])
                } else {
                    warn!("sox is available but cannot access audio devices");
                    Ok(vec![])
                }
            }
            RecordingBackend::Cpal => {
                let devices = self.host.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("CPAL host not initialized"))?
                    .input_devices()
                    .context("Failed to enumerate input devices")?;

                let mut device_names = Vec::new();
                for device in devices {
                    if let Ok(name) = device.name() {
                        device_names.push(name);
                    }
                }

                if device_names.is_empty() {
                    warn!("No audio input devices found");
                } else {
                    info!("Found {} audio input devices", device_names.len());
                }

                Ok(device_names)
            }
        }
    }

    /// Initialize the default audio input device
    fn init_device(&mut self) -> Result<()> {
        if self.backend == RecordingBackend::Sox {
            // Sox doesn't need device initialization
            // Sox doesn't need device initialization
            return Ok(());
        }
        
        if self.device.is_some() {
            return Ok(());
        }

        let host = self.host.as_ref()
            .ok_or_else(|| anyhow::anyhow!("CPAL host not initialized"))?;

        let device = host.default_input_device()
            .context("No default input device available")?;

        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        info!("Using audio device: {}", name);

        self.device = Some(device);
        Ok(())
    }

    /// Start recording using sox (matches original Python implementation)
    async fn start_sox_recording(&mut self, file_path: PathBuf) -> Result<()> {
        // Stop any existing sox process gracefully
        if let Some(mut process) = self.sox_process.take() {
            // Try to terminate gracefully first
            #[cfg(unix)]
            {
                use nix::sys::signal::{self, Signal};
                use nix::unistd::Pid;
                
                if let Some(pid) = process.id() {
                    let _ = signal::kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
                    // Give it a moment to clean up
                    let _ = timeout(Duration::from_millis(500), process.wait()).await;
                }
            }
            
            // Force kill if still running
            let _ = process.kill().await;
        }

        if self.config.verbose {
            info!("Starting sox recording to: {}", file_path.display());
        }
        
        let child = TokioCommand::new("sox")
            .arg("-d")                    // default input device
            .arg("-t").arg("wav")         // explicitly specify WAV format
            .arg("-r").arg("16000")       // sample rate 16kHz
            .arg("-c").arg("1")           // mono
            .arg("-b").arg("16")          // 16-bit
            .arg("-e").arg("signed-integer") // signed PCM format
            .arg(&file_path)              // output file
            .stderr(Stdio::null())        // suppress warnings
            .spawn()
            .context("Failed to start sox process")?;

        self.sox_process = Some(child);
        self.current_recording_path = Some(file_path);
        
        Ok(())
    }

    /// Stop sox recording
    async fn stop_sox_recording(&mut self) -> Result<RecordingInfo> {
        if let Some(mut process) = self.sox_process.take() {
            // Send SIGTERM first for graceful shutdown, then SIGINT if needed
            #[cfg(unix)]
            {
                use nix::sys::signal::{self, Signal};
                use nix::unistd::Pid;
                
                if let Some(pid) = process.id() {
                    // First try SIGTERM for graceful shutdown
                    let _ = signal::kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
                    
                    // Give sox time to flush buffers and write file properly
                    match timeout(Duration::from_secs(3), process.wait()).await {
                        Ok(Ok(status)) => {
                            if self.config.verbose {
                                info!("Sox exited gracefully with status: {:?}", status);
                            }
                        }
                        Ok(Err(e)) => {
                            warn!("Sox process error: {}, trying SIGINT", e);
                            // Try SIGINT (Ctrl+C) as second attempt
                            let _ = signal::kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                            
                            // Wait again after SIGINT
                            match timeout(Duration::from_secs(2), process.wait()).await {
                                Ok(Ok(status)) => {
                                    if self.config.verbose {
                                        info!("Sox exited after SIGINT with status: {:?}", status);
                                    }
                                }
                                Ok(Err(e)) => {
                                    warn!("Sox still erroring after SIGINT: {}", e);
                                }
                                Err(_) => {
                                    warn!("Sox didn't exit after SIGINT, forcing termination");
                                    let _ = process.kill().await;
                                }
                            }
                        }
                        Err(_) => {
                            warn!("Sox didn't exit in 3 seconds with SIGTERM, trying SIGINT");
                            // Try SIGINT
                            let _ = signal::kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                            
                            // Give it one more second
                            if timeout(Duration::from_secs(1), process.wait()).await.is_err() {
                                warn!("Sox still running, force killing");
                                let _ = process.kill().await;
                            }
                        }
                    }
                } else {
                    // Fallback if we can't get PID
                    process.kill().await
                        .context("Failed to stop sox process")?;
                }
            }
            
            #[cfg(not(unix))]
            {
                // On non-Unix systems, just kill the process
                process.kill().await
                    .context("Failed to stop sox process")?;
            }
            
            // Wait longer to ensure file is fully written and finalized
            sleep(Duration::from_secs(1)).await;
            
            // Additional check - wait for file to be readable and have reasonable size
            let file_path_ref = self.current_recording_path.as_ref()
                .ok_or_else(|| anyhow::anyhow!("No recording path available"))?;
            
            let mut attempts = 0;
            let mut last_size = 0u64;
            loop {
                if let Ok(metadata) = std::fs::metadata(file_path_ref) {
                    let current_size = metadata.len();
                    
                    // Check if file is still growing
                    if current_size > last_size {
                        last_size = current_size;
                        attempts = 0; // Reset attempts if file is growing
                    } else if current_size > 1024 { // At least 1KB of audio data
                        // File stopped growing and has reasonable size
                        if self.config.verbose {
                            info!("Recording file ready: {} bytes", current_size);
                        }
                        break;
                    }
                }
                
                attempts += 1;
                if attempts > 20 {
                    // After 2 seconds of no growth, accept whatever we have
                    if last_size > 44 {
                        warn!("Recording file stopped growing at {} bytes after {} attempts", last_size, attempts);
                        break;
                    } else {
                        return Err(anyhow::anyhow!("Recording file too small or missing: {} bytes", last_size));
                    }
                }
                sleep(Duration::from_millis(100)).await;
            }
        }

        let file_path = self.current_recording_path.take()
            .ok_or_else(|| anyhow::anyhow!("No recording path available"))?;

        let start_time = self.start_time.lock().unwrap().take()
            .ok_or_else(|| anyhow::anyhow!("No start time recorded"))?;
        
        let duration = start_time.elapsed();
        
        let metadata = std::fs::metadata(&file_path)
            .context("Failed to get recording file metadata")?;
        
        let id = file_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("recording")
            .to_string();

        Ok(RecordingInfo {
            id,
            file_path,
            duration,
            file_size: metadata.len(),
        })
    }

    /// Start recording audio
    pub async fn start_recording(&mut self) -> Result<String> {
        // Generate recording ID and file path
        let recording_id = generate_recording_id();
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("recording_{}_{}.wav", timestamp, recording_id);
        let file_path = self.config.output_dir.join(&filename);

        *self.start_time.lock().unwrap() = Some(Instant::now());
        *self.state.lock().unwrap() = RecordingState::Recording;

        // Use sox if available
        if self.backend == RecordingBackend::Sox {
            self.start_sox_recording(file_path).await?;
            return Ok(recording_id);
        }

        // Fall back to CPAL
        if self.device.is_none() {
            self.init_device()?;
        }

        let device = self.device.as_ref().unwrap();
        
        if self.config.verbose {
            info!("Recording to: {}", file_path.display());
        }

        // Configure audio stream
        let config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: BufferSize::Fixed(self.config.buffer_size as u32),
        };

        // Create WAV writer for CPAL
        let spec = WavSpec {
            channels: self.config.channels,
            sample_rate: self.config.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let file = File::create(&file_path)
            .context("Failed to create output file")?;
        let writer = WavWriter::new(BufWriter::new(file), spec)
            .context("Failed to create WAV writer")?;

        *self.writer.lock().unwrap() = Some(writer);

        // Build the input stream
        let writer_clone = Arc::clone(&self.writer);
        let state_clone = Arc::clone(&self.state);
        let start_time_clone = Arc::clone(&self.start_time);
        let max_duration = self.config.max_duration;
        let verbose = self.config.verbose;

        let stream = device.build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mut writer_guard = writer_clone.lock().unwrap();
                if let Some(ref mut writer) = *writer_guard {
                    // Write audio data
                    for &sample in data {
                        if let Err(e) = writer.write_sample(sample) {
                            error!("Error writing audio sample: {}", e);
                            *state_clone.lock().unwrap() = RecordingState::Error(e.to_string());
                            return;
                        }
                    }

                    // Check for maximum duration
                    if let Some(max_dur) = max_duration {
                        if let Some(start) = *start_time_clone.lock().unwrap() {
                            if start.elapsed().as_secs() >= max_dur {
                                if verbose {
                                    info!("Maximum recording duration reached");
                                }
                                *state_clone.lock().unwrap() = RecordingState::Stopping;
                                return;
                            }
                        }
                    }
                }
            },
            move |err| {
                error!("Audio stream error: {}", err);
            },
            None,
        ).context("Failed to build input stream")?;

        // Start the stream
        stream.play().context("Failed to start audio stream")?;
        self.stream = Some(stream);

        if self.config.verbose {
            info!("Started recording: {} -> {}", recording_id, file_path.display());
        }
        Ok(recording_id)
    }

    /// Stop the current recording and return recording information
    pub async fn stop_recording(&mut self) -> Result<RecordingInfo> {
        // Handle sox backend first (before taking start_time)
        if self.backend == RecordingBackend::Sox {
            let recording_info = self.stop_sox_recording().await?;
            *self.state.lock().unwrap() = RecordingState::Idle;
            return Ok(recording_info);
        }

        // For CPAL backend, take start_time
        let start_time = self.start_time.lock().unwrap().take()
            .context("No recording in progress")?;
        let duration = start_time.elapsed();

        // Handle CPAL backend
        // Stop the stream
        if let Some(stream) = self.stream.take() {
            drop(stream); // This stops the stream
        }

        // Finalize the WAV file
        let mut writer_guard = self.writer.lock().unwrap();
        let writer = writer_guard.take()
            .context("No WAV writer available")?;
        
        writer.finalize()
            .context("Failed to finalize WAV file")?;

        // Get file path and size
        let file_path = self.get_latest_recording_path()?;
        let metadata = std::fs::metadata(&file_path)
            .context("Failed to get file metadata")?;

        let recording_info = RecordingInfo {
            id: extract_id_from_path(&file_path)?,
            file_path,
            duration,
            file_size: metadata.len(),
        };

        *self.state.lock().unwrap() = RecordingState::Idle;
        
        if self.config.verbose {
            info!("Stopped recording: {} ({:.2}s, {} bytes)", 
                  recording_info.id, 
                  recording_info.duration.as_secs_f64(), 
                  recording_info.file_size);
        }

        Ok(recording_info)
    }



    /// Get path to the most recent recording file
    fn get_latest_recording_path(&self) -> Result<PathBuf> {
        let mut entries: Vec<_> = std::fs::read_dir(&self.config.output_dir)
            .context("Failed to read output directory")?
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("wav"))
                    .unwrap_or(false)
            })
            .collect();

        entries.sort_by_key(|entry| {
            entry.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH)
        });

        entries.last()
            .map(|entry| entry.path())
            .context("No recording files found")
    }


    /// Record audio until Enter key is pressed or timeout
    pub async fn record_audio(&mut self) -> Result<RecordingInfo> {
        println!("🔴 Recording...");
        println!("⚠️ Press Enter to stop");

        // Generate recording path
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let recording_id = generate_recording_id();
        let filename = format!("recording_{}_{}.wav", timestamp, recording_id);
        let file_path = self.config.output_dir.join(filename);
        
        // Ensure output directory exists
        std::fs::create_dir_all(&self.config.output_dir)
            .context("Failed to create output directory")?;
        
        // Set start time and state before starting recording
        *self.start_time.lock().unwrap() = Some(Instant::now());
        *self.state.lock().unwrap() = RecordingState::Recording;
        
        // Start sox recording
        self.start_sox_recording(file_path).await?;
        
        // Give sox a moment to initialize and start writing
        sleep(Duration::from_secs(1)).await;
        
        // Verify sox is actually running
        if let Some(ref process) = self.sox_process {
            if self.config.verbose {
                info!("Sox process started with PID: {:?}", process.id());
            }
        }
        
        // Set up channel to wait for Enter key
        let (tx, rx) = tokio::sync::oneshot::channel();
        
        // Spawn task to wait for Enter
        tokio::spawn(async move {
            let mut buffer = String::new();
            std::io::stdin().read_line(&mut buffer).unwrap();
            let _ = tx.send(());
        });
        
        // Wait for Enter or timeout
        tokio::select! {
            _ = rx => {
                // User pressed Enter - stop recording
                if self.config.verbose {
                    info!("User pressed Enter, stopping recording");
                }
                let recording_info = self.stop_sox_recording().await?;
                *self.state.lock().unwrap() = RecordingState::Idle;
                Ok(recording_info)
            }
            _ = sleep(Duration::from_secs(self.config.max_duration.unwrap_or(300))) => {
                // Timeout reached - stop recording
                if self.config.verbose {
                    info!("Max duration reached, stopping recording");
                }
                let recording_info = self.stop_sox_recording().await?;
                *self.state.lock().unwrap() = RecordingState::Idle;
                Ok(recording_info)
            }
        }
    }



}

/// Generate a unique recording ID
fn generate_recording_id() -> String {
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

/// Extract recording ID from file path
fn extract_id_from_path(path: &PathBuf) -> Result<String> {
    let filename = path.file_name()
        .and_then(|name| name.to_str())
        .context("Invalid file path")?;

    // Try to parse format: recording_TIMESTAMP_ID.wav
    let parts: Vec<&str> = filename.split('_').collect();
    if parts.len() >= 3 {
        let id = parts[2].replace(".wav", "");
        if !id.is_empty() {
            return Ok(id);
        }
    }

    // Fallback to filename without extension
    Ok(path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string())
}

/// Format duration as human-readable string
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    let milliseconds = duration.subsec_millis();

    if minutes > 0 {
        format!("{}:{:02}.{:03}", minutes, seconds, milliseconds)
    } else {
        format!("{}.{:03}s", seconds, milliseconds)
    }
}

/// Format file size as human-readable string
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{:.0} {}", size, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.buffer_size, 1024);
        assert!(config.max_duration.is_some());
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
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.500s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1:05.000");
        assert_eq!(format_duration(Duration::from_secs(3661)), "61:01.000");
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
    }

    #[test]
    fn test_extract_id_from_path() {
        let path = PathBuf::from("recording_20231201_120000_rec_abcd1234.wav");
        let id = extract_id_from_path(&path).unwrap();
        assert_eq!(id, "120000");
    }

    #[tokio::test]
    async fn test_audio_recorder_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = AudioConfig {
            output_dir: temp_dir.path().to_path_buf(),
            ..AudioConfig::default()
        };

        let recorder = AudioRecorder::new(config);
        assert!(recorder.is_ok());

        let _recorder = recorder.unwrap();
        // Recorder created successfully
    }
}
