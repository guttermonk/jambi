//! Vosk-based speech recognition engine for fast CPU transcription
//! 
//! This module provides an alternative to Whisper using Vosk, which is
//! specifically optimized for real-time CPU-based speech recognition.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn, debug};
use vosk::{Model, Recognizer, DecodingState};


/// Vosk model variants with their download URLs and sizes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoskModel {
    /// Small English model (40MB) - fastest, good accuracy
    SmallEnUs,
    /// Large English model (1.8GB) - best accuracy
    LargeEnUs,
    /// Small Indian English model (40MB)
    SmallEnIn,
    /// Small Chinese model (40MB)
    SmallCn,
    /// Small Russian model (40MB)
    SmallRu,
    /// Small French model (40MB)
    SmallFr,
    /// Small German model (40MB)
    SmallDe,
    /// Small Spanish model (40MB)
    SmallEs,
    /// Small Portuguese model (40MB)
    SmallPt,
    /// Small Italian model (40MB)
    SmallIt,
    /// Small Dutch model (40MB)
    SmallNl,
    /// Small Japanese model (40MB)
    SmallJa,
}

impl VoskModel {
    /// Get the download URL for the model
    pub fn download_url(&self) -> &'static str {
        match self {
            VoskModel::SmallEnUs => "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            VoskModel::LargeEnUs => "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            VoskModel::SmallEnIn => "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
            VoskModel::SmallCn => "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
            VoskModel::SmallRu => "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
            VoskModel::SmallFr => "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
            VoskModel::SmallDe => "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
            VoskModel::SmallEs => "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
            VoskModel::SmallPt => "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
            VoskModel::SmallIt => "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
            VoskModel::SmallNl => "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip",
            VoskModel::SmallJa => "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip",
        }
    }
    
    /// Get the model directory name
    pub fn model_name(&self) -> &'static str {
        match self {
            VoskModel::SmallEnUs => "vosk-model-small-en-us-0.15",
            VoskModel::LargeEnUs => "vosk-model-en-us-0.22",
            VoskModel::SmallEnIn => "vosk-model-small-en-in-0.4",
            VoskModel::SmallCn => "vosk-model-small-cn-0.22",
            VoskModel::SmallRu => "vosk-model-small-ru-0.22",
            VoskModel::SmallFr => "vosk-model-small-fr-0.22",
            VoskModel::SmallDe => "vosk-model-small-de-0.15",
            VoskModel::SmallEs => "vosk-model-small-es-0.42",
            VoskModel::SmallPt => "vosk-model-small-pt-0.3",
            VoskModel::SmallIt => "vosk-model-small-it-0.22",
            VoskModel::SmallNl => "vosk-model-small-nl-0.22",
            VoskModel::SmallJa => "vosk-model-small-ja-0.22",
        }
    }
    
    /// Get approximate model size in MB
    pub fn size_mb(&self) -> u64 {
        match self {
            VoskModel::SmallEnUs => 40,
            VoskModel::LargeEnUs => 1800,
            VoskModel::SmallEnIn => 40,
            VoskModel::SmallCn => 40,
            VoskModel::SmallRu => 40,
            VoskModel::SmallFr => 40,
            VoskModel::SmallDe => 40,
            VoskModel::SmallEs => 40,
            VoskModel::SmallPt => 40,
            VoskModel::SmallIt => 40,
            VoskModel::SmallNl => 40,
            VoskModel::SmallJa => 40,
        }
    }
    
    /// Get model description
    pub fn description(&self) -> &'static str {
        match self {
            VoskModel::SmallEnUs => "Small US English model - fast and accurate",
            VoskModel::LargeEnUs => "Large US English model - best accuracy",
            VoskModel::SmallEnIn => "Small Indian English model",
            VoskModel::SmallCn => "Small Chinese model",
            VoskModel::SmallRu => "Small Russian model",
            VoskModel::SmallFr => "Small French model",
            VoskModel::SmallDe => "Small German model",
            VoskModel::SmallEs => "Small Spanish model",
            VoskModel::SmallPt => "Small Portuguese model",
            VoskModel::SmallIt => "Small Italian model",
            VoskModel::SmallNl => "Small Dutch model",
            VoskModel::SmallJa => "Small Japanese model",
        }
    }
    
    /// Get the language code
    pub fn language(&self) -> &'static str {
        match self {
            VoskModel::SmallEnUs | VoskModel::LargeEnUs => "en-US",
            VoskModel::SmallEnIn => "en-IN",
            VoskModel::SmallCn => "zh",
            VoskModel::SmallRu => "ru",
            VoskModel::SmallFr => "fr",
            VoskModel::SmallDe => "de",
            VoskModel::SmallEs => "es",
            VoskModel::SmallPt => "pt",
            VoskModel::SmallIt => "it",
            VoskModel::SmallNl => "nl",
            VoskModel::SmallJa => "ja",
        }
    }
}

impl Default for VoskModel {
    fn default() -> Self {
        VoskModel::SmallEnUs
    }
}

impl std::fmt::Display for VoskModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model_name())
    }
}

/// Vosk transcription configuration
// See the note on AudioConfig: `serde(default)` keeps every key optional so a
// config file can set just `model` without restating the rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoskConfig {
    /// Model to use for transcription
    pub model: VoskModel,
    /// Sample rate (typically 16000)
    pub sample_rate: f32,
    /// Maximum alternatives to return
    pub max_alternatives: usize,
    /// Whether to return word-level timestamps
    pub show_words: bool,
    /// Whether to return partial results
    pub partial_results: bool,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for VoskConfig {
    fn default() -> Self {
        Self {
            model: VoskModel::default(),
            sample_rate: 16000.0,
            max_alternatives: 0,
            show_words: true,
            partial_results: true,
            verbose: false,
        }
    }
}

/// Vosk transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoskResult {
    /// The transcribed text
    pub text: String,
    /// Word-level information if available
    pub words: Option<Vec<WordInfo>>,
    /// Confidence score if available
    pub confidence: Option<f32>,
}

/// Word-level information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordInfo {
    pub word: String,
    pub start: f32,
    pub end: f32,
    pub confidence: f32,
}

/// Decoded audio, ready for the recogniser: a single channel, carrying the
/// rate the file declared rather than the one the config assumed.
///
/// The rate travels with the samples deliberately. It used to be dropped on
/// the floor -- the header was never read, and every file was handed to Vosk
/// as though it were `vosk.sample_rate` -- so an ordinary 44.1kHz recording was
/// decoded at 16kHz and transcribed into nonsense, with nothing to indicate
/// anything had gone wrong.
#[derive(Debug)]
struct Pcm {
    samples: Vec<i16>,
    sample_rate: f32,
}

/// Read a WAV file into mono samples at its own rate.
fn read_wav(path: &Path) -> Result<Pcm> {
    use hound::{SampleFormat, WavReader};
    use std::fs::File;
    use std::io::{BufReader, Read, Seek};

    let mut file = File::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;

    // Sniffed before handing the file to hound, whose complaint for anything
    // else is "Ill-formed WAVE file: no RIFF tag found" -- true, but it never
    // occurs to someone holding an .m4a that WAV was ever the requirement.
    let mut header = [0u8; 12];
    let read = file.read(&mut header)?;
    let header = &header[..read];
    if !(starts_with(header, 0, b"RIFF") && starts_with(header, 8, b"WAVE")) {
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        anyhow::bail!(
            "{} is {}, and jambi reads WAV.\n\
             Convert it first, to mono at the rate the model wants:\n    \
             ffmpeg -i {} -ar 16000 -ac 1 {}.wav",
            path.display(),
            describe_format(header),
            path.display(),
            stem,
        );
    }
    file.rewind()?;

    let mut wav = WavReader::new(BufReader::new(file))
        .with_context(|| format!("cannot read {} as WAV", path.display()))?;
    let spec = wav.spec();

    // Widened or scaled to i16, which is what Vosk takes. A 24-bit or float
    // WAV is ordinary output from any recorder or DAW, and reading one as i16
    // outright is an error rather than a conversion.
    let interleaved: Vec<i16> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 0..=16) => wav
            .samples::<i16>()
            .collect::<std::result::Result<_, _>>()?,
        (SampleFormat::Int, bits) => {
            let shift = bits - 16;
            wav.samples::<i32>()
                .map(|s| s.map(|s| (s >> shift) as i16))
                .collect::<std::result::Result<_, _>>()?
        }
        (SampleFormat::Float, _) => wav
            .samples::<f32>()
            .map(|s| s.map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16))
            .collect::<std::result::Result<_, _>>()?,
    };

    let samples = downmix(interleaved, spec.channels);

    debug!(
        "{}: {}Hz, {} channel(s), {} bit {:?} -> {} mono samples",
        path.display(),
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample,
        spec.sample_format,
        samples.len()
    );

    Ok(Pcm {
        samples,
        sample_rate: spec.sample_rate as f32,
    })
}

/// Average the channels of interleaved audio down to one.
///
/// Vosk takes mono. Feeding it interleaved stereo hands it two channels'
/// samples as though they were one channel at twice the rate, which recognises
/// as nothing at all.
fn downmix(interleaved: Vec<i16>, channels: u16) -> Vec<i16> {
    if channels <= 1 {
        return interleaved;
    }

    interleaved
        .chunks_exact(channels as usize)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|&s| i32::from(s)).sum();
            (sum / frame.len() as i32) as i16
        })
        .collect()
}

fn starts_with(header: &[u8], offset: usize, tag: &[u8]) -> bool {
    header.len() >= offset + tag.len() && &header[offset..offset + tag.len()] == tag
}

/// Name what a file's leading bytes say it actually is.
///
/// Only ever used to make the "this is not a WAV" error say something the
/// reader can act on; the extension is not consulted, because a mislabelled
/// file is exactly the case where a confusing error is most likely.
fn describe_format(header: &[u8]) -> &'static str {
    let mp3_frame = header.first() == Some(&0xFF)
        && header.get(1).is_some_and(|b| b & 0xE0 == 0xE0);

    if starts_with(header, 4, b"ftyp") {
        "MPEG-4 audio (m4a/aac)"
    } else if starts_with(header, 0, b"fLaC") {
        "FLAC"
    } else if starts_with(header, 0, b"OggS") {
        "Ogg (vorbis or opus)"
    } else if starts_with(header, 0, b"ID3") || mp3_frame {
        "MP3"
    } else if starts_with(header, 0, b"FORM")
        && (starts_with(header, 8, b"AIFF") || starts_with(header, 8, b"AIFC"))
    {
        "AIFF"
    } else if starts_with(header, 0, b"caff") {
        "CAF"
    } else if starts_with(header, 0, b"wvpk") {
        "WavPack"
    } else if starts_with(header, 0, b"\x1a\x45\xdf\xa3") {
        "Matroska or WebM"
    } else if starts_with(header, 0, b"RIFF") {
        "a RIFF file but not WAVE"
    } else {
        "not audio jambi recognises"
    }
}

/// Vosk speech recognition engine
pub struct VoskEngine {
    pub config: VoskConfig,
    pub model_path: Option<PathBuf>,
    pub model: Option<Arc<Model>>,
}

impl VoskEngine {
    /// Create a new Vosk engine with the given configuration
    pub fn new(config: VoskConfig) -> Result<Self> {
        if config.verbose {
            info!("Initializing Vosk engine with model: {}", config.model);
        }
        
        Ok(Self {
            config,
            model_path: None,
            model: None,
        })
    }

    /// Build an engine around a model that is already in memory.
    ///
    /// This is the daemon's whole reason for existing. `Model::new` costs
    /// ~800ms and is otherwise paid by every invocation; the daemon pays it
    /// once at startup and hands the same `Arc<Model>` to each recording, so
    /// recognition starts as soon as the microphone is open.
    pub fn with_model(config: VoskConfig, model: Arc<Model>) -> Self {
        Self {
            config,
            model_path: None,
            model: Some(model),
        }
    }

    /// The loaded model, for handing to another engine instance.
    pub fn loaded_model(&self) -> Option<Arc<Model>> {
        self.model.clone()
    }

    /// Get the models directory
    fn get_models_dir() -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?;
        Ok(cache_dir.join("jambi").join("vosk-models"))
    }
    
    /// Check if a model is already downloaded
    #[allow(dead_code)]
    pub async fn is_model_downloaded(&self) -> bool {
        if let Ok(models_dir) = Self::get_models_dir() {
            let model_dir = models_dir.join(self.config.model.model_name());
            model_dir.exists() && model_dir.is_dir()
        } else {
            false
        }
    }
    
    /// Download a Vosk model if not already present
    pub async fn download_model(&mut self) -> Result<PathBuf> {
        let models_dir = Self::get_models_dir()?;
        tokio::fs::create_dir_all(&models_dir).await?;
        
        let model_dir = models_dir.join(self.config.model.model_name());
        
        // Check if model already exists
        if model_dir.exists() && model_dir.is_dir() {
            if self.config.verbose {
                info!("Model already exists: {}", model_dir.display());
            }
            self.model_path = Some(model_dir.clone());
            return Ok(model_dir);
        }
        
        if self.config.verbose {
            info!("Downloading Vosk model: {}", self.config.model);
        }
        
        // Download the zip file
        let url = self.config.model.download_url();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        
        let response = client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model: HTTP {}",
                response.status()
            ));
        }
        
        // Get the content length for progress tracking
        let total_size = response
            .content_length()
            .unwrap_or(self.config.model.size_mb() * 1024 * 1024);
        
        info!("Downloading {} MB...", total_size / 1024 / 1024);
        
        // Download to a temporary file
        let temp_file = models_dir.join(format!("{}.zip.tmp", self.config.model.model_name()));
        let mut file = tokio::fs::File::create(&temp_file).await?;
        
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            
            if downloaded % (10 * 1024 * 1024) == 0 {
                let progress = (downloaded as f64 / total_size as f64) * 100.0;
                info!("Download progress: {:.1}%", progress);
            }
        }
        
        file.sync_all().await?;
        drop(file);
        
        info!("Download complete, extracting...");
        
        // Extract the zip file
        use tokio::process::Command;
        
        let output = Command::new("unzip")
            .arg("-q")
            .arg(&temp_file)
            .arg("-d")
            .arg(&models_dir)
            .output()
            .await;
        
        match output {
            Ok(output) if output.status.success() => {
                if self.config.verbose {
                    info!("Model extracted successfully");
                }
                // Clean up temp file
                let _ = tokio::fs::remove_file(&temp_file).await;
            }
            _ => {
                // Try using Rust's zip crate as fallback
                warn!("unzip command failed, trying built-in extraction");
                self.extract_zip_fallback(&temp_file, &models_dir).await?;
                let _ = tokio::fs::remove_file(&temp_file).await;
            }
        }
        
        // Verify the model directory exists
        if !model_dir.exists() {
            return Err(anyhow::anyhow!(
                "Model directory not found after extraction: {}",
                model_dir.display()
            ));
        }
        
        self.model_path = Some(model_dir.clone());
        info!("Model ready at: {}", model_dir.display());
        
        Ok(model_dir)
    }
    
    /// Fallback zip extraction using zip crate
    async fn extract_zip_fallback(&self, zip_path: &Path, target_dir: &Path) -> Result<()> {
        use zip::ZipArchive;
        
        let file = std::fs::File::open(zip_path)?;
        let mut archive = ZipArchive::new(file)?;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = target_dir.join(file.mangled_name());
            
            if file.name().ends_with('/') {
                std::fs::create_dir_all(&outpath)?;
            } else {
                if let Some(p) = outpath.parent() {
                    if !p.exists() {
                        std::fs::create_dir_all(p)?;
                    }
                }
                let mut outfile = std::fs::File::create(&outpath)?;
                std::io::copy(&mut file, &mut outfile)?;
            }
        }
        
        Ok(())
    }
    
    /// Load the Vosk model into memory
    pub async fn load_model(&mut self) -> Result<()> {
        // Suppress Vosk's internal debug messages
        // Use Error level to only show errors, suppressing debug/info/warning messages
        vosk::set_log_level(vosk::LogLevel::Error);
        
        // Download model if needed
        if self.model_path.is_none() {
            self.download_model().await?;
        }
        
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;
        
        if self.config.verbose {
            info!("Loading Vosk model from: {}", model_path.display());
        }
        
        // Load the model synchronously (Vosk doesn't have async loading)
        let model_path_str = model_path.to_string_lossy().to_string();
        let model = tokio::task::spawn_blocking(move || -> Result<Model> {
            Model::new(&model_path_str)
                .ok_or_else(|| anyhow::anyhow!("Failed to load Vosk model from path: {}", model_path_str))
        }).await??;
        
        self.model = Some(Arc::new(model));
        if self.config.verbose {
            info!("Vosk model loaded successfully");
        }
        
        Ok(())
    }
    
    /// Transcribe audio data
    pub async fn transcribe(&self, audio_data: &[i16], sample_rate: f32) -> Result<VoskResult> {
        // Suppress Vosk's internal debug messages
        vosk::set_log_level(vosk::LogLevel::Error);
        
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?
            .clone();
        
        let config = self.config.clone();
        
        // Run transcription in blocking thread pool
        let audio_data = audio_data.to_vec();
        
        let result = tokio::task::spawn_blocking(move || {
            Self::transcribe_sync(model, &audio_data, sample_rate, &config)
        }).await??;
        
        Ok(result)
    }
    
    /// Synchronous transcription function
    fn transcribe_sync(
        model: Arc<Model>,
        audio_data: &[i16],
        sample_rate: f32,
        config: &VoskConfig,
    ) -> Result<VoskResult> {
        let mut recognizer = Recognizer::new(&model, sample_rate)
            .ok_or_else(|| anyhow::anyhow!("Failed to create recognizer with sample rate: {}", sample_rate))?;
        
        // Configure recognizer
        recognizer.set_max_alternatives(config.max_alternatives as u16);
        recognizer.set_words(config.show_words);
        recognizer.set_partial_words(config.partial_results);
        
        // Process audio in chunks
        const CHUNK_SIZE: usize = 8000; // 0.5 seconds at 16kHz
        
        for chunk in audio_data.chunks(CHUNK_SIZE) {
            match recognizer.accept_waveform(chunk) {
                Ok(DecodingState::Finalized) => {
                    // We have a final result for this chunk
                    debug!("Chunk finalized");
                }
                Ok(DecodingState::Running) => {
                    // Still processing
                    if config.partial_results {
                        let partial = recognizer.partial_result();
                        debug!("Partial: {:?}", partial);
                    }
                }
                Ok(DecodingState::Failed) => {
                    warn!("Decoding failed for audio chunk");
                }
                Err(e) => {
                    warn!("Error processing audio chunk: {:?}", e);
                }
            }
        }
        
        // Get final result
        let final_result = recognizer.final_result();
        
        // Parse the result
        let result = if config.max_alternatives > 0 {
            // Multiple alternatives requested
            match final_result.multiple() {
                Some(alts) => {
                    if let Some(first) = alts.alternatives.first() {
                        VoskResult {
                            text: first.text.to_string(),
                            words: None, // Would need to parse word info
                            confidence: Some(first.confidence),
                        }
                    } else {
                        VoskResult {
                            text: String::new(),
                            words: None,
                            confidence: None,
                        }
                    }
                }
                None => VoskResult {
                    text: String::new(),
                    words: None,
                    confidence: None,
                }
            }
        } else {
            // Single result
            match final_result.single() {
                Some(single) => {
                    VoskResult {
                        text: single.text.to_string(),
                        words: if config.show_words {
                            Some(single.result.iter().map(|w| WordInfo {
                                word: w.word.to_string(),
                                start: w.start,
                                end: w.end,
                                confidence: w.conf,
                            }).collect())
                        } else {
                            None
                        },
                        confidence: None, // Single result doesn't have overall confidence
                    }
                }
                None => VoskResult {
                    text: String::new(),
                    words: None,
                    confidence: None,
                }
            }
        };
        
        Ok(result)
    }
    
    /// Transcribe an audio file
    pub async fn transcribe_file(&self, audio_path: &Path) -> Result<VoskResult> {
        if self.config.verbose {
            info!("Transcribing file: {}", audio_path.display());
        }
        
        // Load model if not already loaded
        if self.model.is_none() {
            return Err(anyhow::anyhow!("Model not loaded. Call load_model() first"));
        }
        
        // The rate comes from the file, never from the config. Vosk's
        // `Recognizer::new` documents its parameter as "the sample rate of the
        // audio you going to feed into the recognizer" and resamples to the
        // model's rate itself, so declaring the truth is both correct and
        // enough.
        let audio = self.read_audio_file(audio_path).await?;

        let result = self.transcribe(&audio.samples, audio.sample_rate).await?;

        if self.config.verbose {
            info!("Transcription complete: {} words",
                  result.text.split_whitespace().count());
        }

        Ok(result)
    }

    /// Read an audio file into mono samples at the file's own rate.
    async fn read_audio_file(&self, audio_path: &Path) -> Result<Pcm> {
        let file_path = audio_path.to_path_buf();
        tokio::task::spawn_blocking(move || read_wav(&file_path)).await?
    }
    
    /// Get available models
    pub fn available_models() -> Vec<VoskModel> {
        vec![
            VoskModel::SmallEnUs,
            VoskModel::LargeEnUs,
            VoskModel::SmallEnIn,
            VoskModel::SmallCn,
            VoskModel::SmallRu,
            VoskModel::SmallFr,
            VoskModel::SmallDe,
            VoskModel::SmallEs,
            VoskModel::SmallPt,
            VoskModel::SmallIt,
            VoskModel::SmallNl,
            VoskModel::SmallJa,
        ]
    }
    
    /// Stream from the default input device, feeding Vosk as the audio arrives,
    /// and return the accumulated text once `stop` resolves.
    ///
    /// This is the shared engine behind both live-transcription front-ends: the
    /// windowed TUI stops on Enter and prints as it goes, while live dictation
    /// stops on a signal and stays silent. What differs between them is only the
    /// stop future and the two callbacks, so neither owns the recognition loop.
    ///
    /// `on_partial` fires with the in-progress guess for the current utterance
    /// (it is revised repeatedly, so treat it as redraw-in-place, not append).
    /// `on_segment` fires once per finalised utterance and is append-only.
    pub async fn transcribe_stream<S>(
        &mut self,
        stop: S,
        mut on_partial: impl FnMut(&str),
        mut on_segment: impl FnMut(&str),
    ) -> Result<VoskResult>
    where
        S: std::future::Future<Output = ()>,
    {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use cpal::{SampleRate, StreamConfig};
        use std::sync::mpsc;

        // Suppress Vosk's internal debug messages
        vosk::set_log_level(vosk::LogLevel::Error);

        // Audio capture is opened BEFORE the model is loaded, and the order
        // matters more than it looks. Loading the small English model costs
        // ~800ms (the on-disk check beside it is ~1ms -- the time is all in
        // Vosk's Model::new), and on a push-to-talk key the user is already
        // speaking during it. Opening the device first costs tens of
        // milliseconds and lets the capture callback pile samples into the
        // channel meanwhile, so the first word survives instead of being
        // clipped. The channel is unbounded and 800ms of 16kHz mono i16 is
        // ~26KB, so the backlog is free; the drain in the loop below picks it
        // up on the first poll.
        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        let config = StreamConfig {
            channels: 1,
            sample_rate: SampleRate(16000),
            buffer_size: cpal::BufferSize::Fixed(1024),
        };

        // Channel for sending audio data from callback to main thread
        let (tx, rx) = mpsc::channel::<Vec<i16>>();

        // Build input stream
        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert f32 samples to i16
                let samples: Vec<i16> = data.iter()
                    .map(|&s| (s * 32767.0) as i16)
                    .collect();
                let _ = tx.send(samples);
            },
            |err| eprintln!("Audio stream error: {}", err),
            None
        )?;

        // Start the audio stream
        stream.play()?;

        // Now pay the model cost, with the microphone already live
        if self.model.is_none() {
            self.load_model().await?;
        }

        let model = self.model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;

        // Create recognizer with streaming configuration
        let mut recognizer = Recognizer::new(model, 16000.0)
            .ok_or_else(|| anyhow::anyhow!("Failed to create recognizer"))?;
        let _ = recognizer.set_max_alternatives(0);
        let _ = recognizer.set_words(true);

        let mut full_text = String::new();
        let mut last_partial = String::new();
        let mut stop = std::pin::pin!(stop);

        // Process audio in real-time
        loop {
            tokio::select! {
                _ = &mut stop => break,
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(10)) => {
                    // Check for audio data
                    while let Ok(samples) = rx.try_recv() {
                        match recognizer.accept_waveform(&samples) {
                            Ok(DecodingState::Running) => {
                                // Revised guess for the utterance still being spoken
                                let partial = recognizer.partial_result().partial;
                                if !partial.is_empty() && partial != last_partial {
                                    on_partial(partial);
                                    last_partial = partial.to_string();
                                }
                            }
                            Ok(DecodingState::Finalized) => {
                                // final_result() both reads and resets the
                                // recognizer, so each utterance is reported once
                                // and the post-loop call below can only return
                                // audio this branch has not already consumed.
                                if let Some(single) = recognizer.final_result().single() {
                                    if !single.text.is_empty() {
                                        on_segment(single.text);
                                        if !full_text.is_empty() {
                                            full_text.push(' ');
                                        }
                                        full_text.push_str(single.text);
                                        last_partial.clear();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Drain whatever the capture callback queued between the last poll and
        // the stop. Without this the tail of the final word is dropped, which on
        // a push-to-talk key is exactly where the user stops speaking.
        while let Ok(samples) = rx.try_recv() {
            let _ = recognizer.accept_waveform(&samples);
        }

        // Flush the utterance in progress when the stop arrived
        if let Some(single) = recognizer.final_result().single() {
            if !single.text.is_empty() {
                on_segment(single.text);
                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(single.text);
            }
        }

        // Stop the stream
        drop(stream);

        Ok(VoskResult {
            text: full_text.trim().to_string(),
            words: None,
            confidence: None,
        })
    }

    /// Live transcription for the windowed TUI - prints text as you speak and
    /// stops when the user presses Enter.
    pub async fn transcribe_live(&mut self) -> Result<VoskResult> {
        use std::io::{self, Write};

        println!("⚠️ Press Enter to stop");

        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let mut buffer = String::new();
            let _ = std::io::stdin().read_line(&mut buffer);
            let _ = stop_tx.send(());
        });

        let result = self
            .transcribe_stream(
                async {
                    let _ = stop_rx.await;
                },
                |partial| {
                    print!("\r\x1b[K📝 {}", partial);
                    io::stdout().flush().ok();
                },
                |segment| {
                    print!("\r\x1b[K✅ {}\n", segment);
                    io::stdout().flush().ok();
                },
            )
            .await?;

        println!("⏹️ Stopped live transcription");

        Ok(result)
    }
    
    /// Clean up old models to save disk space
    #[allow(dead_code)]
    pub async fn cleanup_old_models(&self) -> Result<()> {
        let models_dir = Self::get_models_dir()?;
        
        if !models_dir.exists() {
            return Ok(());
        }
        
        let current_model = self.config.model.model_name();
        
        let mut entries = tokio::fs::read_dir(&models_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                let dir_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                
                if dir_name != current_model && dir_name.starts_with("vosk-model-") {
                    info!("Removing old model: {}", dir_name);
                    tokio::fs::remove_dir_all(&path).await?;
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vosk_model_properties() {
        let model = VoskModel::SmallEnUs;
        assert_eq!(model.language(), "en-US");
        assert_eq!(model.size_mb(), 40);
        assert!(model.download_url().contains("vosk-model-small-en-us"));
    }
    
    #[tokio::test]
    async fn test_vosk_engine_creation() {
        let config = VoskConfig::default();
        let engine = VoskEngine::new(config);
        assert!(engine.is_ok());
    }

    /// Write a WAV of `channels` channels at `rate`, for the reader tests.
    fn write_wav(path: &std::path::Path, rate: u32, channels: u16, frames: &[i16]) {
        let spec = hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for &frame in frames {
            for _ in 0..channels {
                writer.write_sample(frame).unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    /// The bug this replaced: the header was never read, so every file was
    /// transcribed as though it were the rate in the config. A 44.1kHz file
    /// went to Vosk labelled 16kHz and came back as nonsense, silently.
    #[test]
    fn the_rate_comes_from_the_file_not_the_config() {
        let dir = tempfile::tempdir().unwrap();

        for rate in [8_000, 16_000, 44_100, 48_000] {
            let path = dir.path().join(format!("{rate}.wav"));
            write_wav(&path, rate, 1, &[0, 1000, -1000, 0]);

            let pcm = read_wav(&path).unwrap();
            assert_eq!(
                pcm.sample_rate, rate as f32,
                "{rate}Hz file reported {}Hz",
                pcm.sample_rate
            );
        }
    }

    /// Stereo has to be averaged down, not handed over interleaved -- which
    /// would be two channels' samples read as one channel at twice the rate.
    #[test]
    fn channels_are_mixed_down_to_one() {
        let dir = tempfile::tempdir().unwrap();
        let frames = [0i16, 1000, -1000, 500];

        for channels in [1u16, 2] {
            let path = dir.path().join(format!("{channels}ch.wav"));
            write_wav(&path, 16_000, channels, &frames);

            let pcm = read_wav(&path).unwrap();
            assert_eq!(
                pcm.samples.len(),
                frames.len(),
                "{channels} channels produced {} samples for {} frames",
                pcm.samples.len(),
                frames.len()
            );
            assert_eq!(pcm.samples, frames, "{channels} channels changed the audio");
        }
    }

    #[test]
    fn downmix_averages_each_frame() {
        assert_eq!(downmix(vec![100, 200, -100, 100], 2), vec![150, 0]);
        assert_eq!(downmix(vec![1, 2, 3], 1), vec![1, 2, 3]);
    }

    /// The reported bug: an .m4a produced "Ill-formed WAVE file: no RIFF tag
    /// found", which is true and useless. The error has to name the format and
    /// say what to do.
    #[test]
    fn a_non_wav_file_is_reported_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.m4a");
        // The ftyp box an MPEG-4 file opens with.
        std::fs::write(&path, b"\0\0\0\x20ftypM4A \0\0\0\0M4A mp42isom").unwrap();

        let error = read_wav(&path).unwrap_err().to_string();
        assert!(error.contains("MPEG-4"), "did not name the format: {error}");
        assert!(error.contains("ffmpeg"), "did not say how to convert: {error}");
        assert!(
            !error.contains("RIFF tag"),
            "still leaks hound's wording: {error}"
        );
    }

    #[test]
    fn formats_are_told_apart_by_their_leading_bytes() {
        assert_eq!(describe_format(b"fLaC\0\0\0\0"), "FLAC");
        assert_eq!(describe_format(b"OggS\0\0\0\0"), "Ogg (vorbis or opus)");
        assert_eq!(describe_format(b"ID3\x04\0\0\0\0"), "MP3");
        assert_eq!(describe_format(b"\0\0\0 ftypM4A "), "MPEG-4 audio (m4a/aac)");
        assert_eq!(describe_format(b"FORM\0\0\0\0AIFF"), "AIFF");
        // Short reads must not panic.
        assert_eq!(describe_format(b""), "not audio jambi recognises");
        assert_eq!(describe_format(b"ID"), "not audio jambi recognises");
    }
}
