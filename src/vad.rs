//! Voice Activity Detection (VAD) module for audio preprocessing
//! 
//! This module provides various Voice Activity Detection algorithms to identify
//! speech segments in audio recordings and remove silence periods before
//! transcription processing.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::VadConfig;

/// Voice Activity Detection processor with multiple algorithm support
pub struct VadProcessor {
    config: VadConfig,
    algorithm: VadAlgorithm,
}

/// Available VAD algorithms
#[derive(Debug, Clone)]
pub enum VadAlgorithm {
    /// Simple energy-based detection
    Energy(EnergyVad),
    /// Spectral-based detection using frequency analysis
    Spectral(SpectralVad),
    /// Hybrid approach combining energy and spectral features
    Hybrid(HybridVad),
}

/// Energy-based VAD using RMS energy calculation
#[derive(Debug, Clone)]
pub struct EnergyVad {
    energy_threshold: f32,
    window_size: usize,
    smoothing_factor: f32,
}

/// Spectral-based VAD using frequency domain analysis
#[derive(Debug, Clone)]
pub struct SpectralVad {
    energy_threshold: f32,
    spectral_threshold: f32,
    window_size: usize,
    hop_size: usize,
    low_freq_cutoff: f32,
    high_freq_cutoff: f32,
}

/// Hybrid VAD combining multiple features
#[derive(Debug, Clone)]
pub struct HybridVad {
    energy_vad: EnergyVad,
    spectral_vad: SpectralVad,
    energy_weight: f32,
    spectral_weight: f32,
}

/// VAD processing result
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Processed audio samples with silence removed
    pub audio_samples: Vec<f32>,
    /// Speech segments with timing information
    pub speech_segments: Vec<SpeechSegment>,
    /// Original duration in seconds
    pub original_duration: f64,
    /// Processed duration in seconds
    pub processed_duration: f64,
    /// Percentage of audio that was speech
    pub speech_percentage: f64,
}

/// Speech segment with timing information
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start_time: f64,
    /// End time in seconds
    pub end_time: f64,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Average energy level
    pub energy: f32,
}

/// Audio frame analysis result
#[derive(Debug, Clone)]
struct FrameAnalysis {
    is_speech: bool,
    confidence: f32,
    energy: f32,
    spectral_features: Option<SpectralFeatures>,
}

/// Spectral features for a frame
#[derive(Debug, Clone)]
struct SpectralFeatures {
    spectral_centroid: f32,
    spectral_rolloff: f32,
    zero_crossing_rate: f32,
    spectral_flux: f32,
}

impl VadProcessor {
    /// Create a new VAD processor with default energy-based algorithm
    pub fn new(config: VadConfig) -> Self {
        let algorithm = VadAlgorithm::Energy(EnergyVad::new(
            config.energy_threshold,
            config.window_size,
        ));

        Self { config, algorithm }
    }

    /// Create VAD processor with specific algorithm
    pub fn with_algorithm(config: VadConfig, algorithm: VadAlgorithm) -> Self {
        Self { config, algorithm }
    }

    /// Process audio samples and remove silence
    pub fn process_audio(&self, samples: Vec<f32>, sample_rate: u32) -> Result<VadResult> {
        if samples.is_empty() {
            return Ok(VadResult {
                audio_samples: samples,
                speech_segments: Vec::new(),
                original_duration: 0.0,
                processed_duration: 0.0,
                speech_percentage: 0.0,
            });
        }

        let original_duration = samples.len() as f64 / sample_rate as f64;
        
        // Analyze frames to detect speech
        let frame_analyses = self.analyze_frames(&samples, sample_rate)?;
        
        // Extract speech segments
        let speech_segments = self.extract_speech_segments(&frame_analyses, sample_rate);
        
        // Remove silence based on speech segments
        let processed_samples = if self.config.enabled {
            self.remove_silence(&samples, &speech_segments, sample_rate)?
        } else {
            samples
        };

        let processed_duration = processed_samples.len() as f64 / sample_rate as f64;
        let speech_percentage = if original_duration > 0.0 {
            processed_duration / original_duration * 100.0
        } else {
            0.0
        };

        Ok(VadResult {
            audio_samples: processed_samples,
            speech_segments,
            original_duration,
            processed_duration,
            speech_percentage,
        })
    }

    /// Analyze audio frames to detect speech activity
    fn analyze_frames(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<FrameAnalysis>> {
        let window_size = self.config.window_size;
        let hop_size = window_size / 2; // 50% overlap
        let mut analyses = Vec::new();

        for (frame_idx, window) in samples.windows(window_size).step_by(hop_size).enumerate() {
            let analysis = match &self.algorithm {
                VadAlgorithm::Energy(energy_vad) => {
                    energy_vad.analyze_frame(window, sample_rate)?
                },
                VadAlgorithm::Spectral(spectral_vad) => {
                    spectral_vad.analyze_frame(window, sample_rate)?
                },
                VadAlgorithm::Hybrid(hybrid_vad) => {
                    hybrid_vad.analyze_frame(window, sample_rate)?
                },
            };

            analyses.push(analysis);
        }

        // Apply smoothing to reduce spurious detections
        self.smooth_detections(&mut analyses);

        Ok(analyses)
    }

    /// Smooth detection results to reduce false positives
    fn smooth_detections(&self, analyses: &mut [FrameAnalysis]) {
        let smoothing_window = 5; // Smooth over 5 frames
        let mut smoothed_speech = Vec::with_capacity(analyses.len());

        for i in 0..analyses.len() {
            let start = i.saturating_sub(smoothing_window / 2);
            let end = (i + smoothing_window / 2 + 1).min(analyses.len());
            
            let speech_count = analyses[start..end]
                .iter()
                .filter(|a| a.is_speech)
                .count();
            
            let total_count = end - start;
            let speech_ratio = speech_count as f32 / total_count as f32;
            
            // Require majority vote for speech detection
            smoothed_speech.push(speech_ratio > 0.6);
        }

        // Update analyses with smoothed results
        for (analysis, &is_speech) in analyses.iter_mut().zip(smoothed_speech.iter()) {
            analysis.is_speech = is_speech;
        }
    }

    /// Extract speech segments from frame analyses
    fn extract_speech_segments(&self, analyses: &[FrameAnalysis], sample_rate: u32) -> Vec<SpeechSegment> {
        let mut segments = Vec::new();
        let hop_size = self.config.window_size / 2;
        let frame_duration = hop_size as f64 / sample_rate as f64;
        let min_segment_duration = 0.1; // Minimum 100ms segment
        let min_silence_duration = self.config.min_silence_duration_ms as f64 / 1000.0;

        let mut current_segment_start: Option<f64> = None;
        let mut last_speech_time = 0.0;

        for (frame_idx, analysis) in analyses.iter().enumerate() {
            let frame_time = frame_idx as f64 * frame_duration;

            if analysis.is_speech {
                if current_segment_start.is_none() {
                    current_segment_start = Some(frame_time);
                }
                last_speech_time = frame_time;
            } else if let Some(segment_start) = current_segment_start {
                // Check if silence duration exceeds threshold
                if frame_time - last_speech_time >= min_silence_duration {
                    let segment_duration = last_speech_time - segment_start;
                    
                    if segment_duration >= min_segment_duration {
                        // Calculate average confidence and energy for the segment
                        let segment_start_frame = (segment_start / frame_duration) as usize;
                        let segment_end_frame = (last_speech_time / frame_duration) as usize;
                        
                        let segment_analyses = &analyses[segment_start_frame..=segment_end_frame.min(analyses.len() - 1)];
                        let avg_confidence = segment_analyses.iter()
                            .map(|a| a.confidence)
                            .sum::<f32>() / segment_analyses.len() as f32;
                        let avg_energy = segment_analyses.iter()
                            .map(|a| a.energy)
                            .sum::<f32>() / segment_analyses.len() as f32;

                        segments.push(SpeechSegment {
                            start_time: segment_start,
                            end_time: last_speech_time,
                            confidence: avg_confidence,
                            energy: avg_energy,
                        });
                    }
                    
                    current_segment_start = None;
                }
            }
        }

        // Handle final segment if it exists
        if let Some(segment_start) = current_segment_start {
            let segment_duration = last_speech_time - segment_start;
            if segment_duration >= min_segment_duration {
                let segment_start_frame = (segment_start / frame_duration) as usize;
                let segment_end_frame = (last_speech_time / frame_duration) as usize;
                
                let segment_analyses = &analyses[segment_start_frame..=segment_end_frame.min(analyses.len() - 1)];
                let avg_confidence = segment_analyses.iter()
                    .map(|a| a.confidence)
                    .sum::<f32>() / segment_analyses.len() as f32;
                let avg_energy = segment_analyses.iter()
                    .map(|a| a.energy)
                    .sum::<f32>() / segment_analyses.len() as f32;

                segments.push(SpeechSegment {
                    start_time: segment_start,
                    end_time: last_speech_time,
                    confidence: avg_confidence,
                    energy: avg_energy,
                });
            }
        }

        segments
    }

    /// Remove silence from audio based on detected speech segments
    fn remove_silence(&self, samples: &[f32], segments: &[SpeechSegment], sample_rate: u32) -> Result<Vec<f32>> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }

        let mut output_samples = Vec::new();
        let padding_samples = (0.05 * sample_rate as f32) as usize; // 50ms padding

        for segment in segments {
            let start_sample = (segment.start_time * sample_rate as f64) as usize;
            let end_sample = (segment.end_time * sample_rate as f64) as usize;
            
            // Add padding around speech segments
            let padded_start = start_sample.saturating_sub(padding_samples);
            let padded_end = (end_sample + padding_samples).min(samples.len());
            
            if padded_start < samples.len() && padded_end > padded_start {
                output_samples.extend_from_slice(&samples[padded_start..padded_end]);
                
                // Add small gap between segments
                let gap_samples = (0.02 * sample_rate as f32) as usize; // 20ms gap
                output_samples.extend(std::iter::repeat(0.0).take(gap_samples));
            }
        }

        Ok(output_samples)
    }
}

impl EnergyVad {
    /// Create new energy-based VAD
    pub fn new(energy_threshold: f32, window_size: usize) -> Self {
        Self {
            energy_threshold,
            window_size,
            smoothing_factor: 0.1,
        }
    }

    /// Analyze a single frame using energy-based detection
    pub fn analyze_frame(&self, window: &[f32], _sample_rate: u32) -> Result<FrameAnalysis> {
        let energy = self.calculate_rms_energy(window);
        let is_speech = energy > self.energy_threshold;
        let confidence = if is_speech {
            (energy / self.energy_threshold).min(1.0)
        } else {
            1.0 - (energy / self.energy_threshold).min(1.0)
        };

        Ok(FrameAnalysis {
            is_speech,
            confidence,
            energy,
            spectral_features: None,
        })
    }

    /// Calculate RMS energy of a window
    fn calculate_rms_energy(&self, window: &[f32]) -> f32 {
        if window.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = window.iter().map(|&x| x * x).sum();
        (sum_squares / window.len() as f32).sqrt()
    }
}

impl SpectralVad {
    /// Create new spectral-based VAD
    pub fn new(energy_threshold: f32, spectral_threshold: f32, window_size: usize) -> Self {
        Self {
            energy_threshold,
            spectral_threshold,
            window_size,
            hop_size: window_size / 2,
            low_freq_cutoff: 80.0,  // Hz
            high_freq_cutoff: 8000.0, // Hz
        }
    }

    /// Analyze frame using spectral features
    pub fn analyze_frame(&self, window: &[f32], sample_rate: u32) -> Result<FrameAnalysis> {
        let energy = self.calculate_rms_energy(window);
        let spectral_features = self.extract_spectral_features(window, sample_rate)?;
        
        // Combine energy and spectral features for decision
        let energy_vote = energy > self.energy_threshold;
        let spectral_vote = spectral_features.spectral_centroid > self.spectral_threshold &&
                           spectral_features.zero_crossing_rate < 0.5;
        
        let is_speech = energy_vote && spectral_vote;
        let confidence = if is_speech {
            0.8 // High confidence for spectral detection
        } else {
            0.6
        };

        Ok(FrameAnalysis {
            is_speech,
            confidence,
            energy,
            spectral_features: Some(spectral_features),
        })
    }

    /// Calculate RMS energy
    fn calculate_rms_energy(&self, window: &[f32]) -> f32 {
        if window.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = window.iter().map(|&x| x * x).sum();
        (sum_squares / window.len() as f32).sqrt()
    }

    /// Extract spectral features from audio window
    fn extract_spectral_features(&self, window: &[f32], sample_rate: u32) -> Result<SpectralFeatures> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window.len());
        
        // Convert to complex numbers for FFT
        let mut buffer: Vec<Complex<f32>> = window.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        fft.process(&mut buffer);
        
        // Calculate magnitude spectrum
        let magnitude_spectrum: Vec<f32> = buffer.iter()
            .take(buffer.len() / 2) // Only first half due to symmetry
            .map(|c| c.norm())
            .collect();

        let spectral_centroid = self.calculate_spectral_centroid(&magnitude_spectrum, sample_rate);
        let spectral_rolloff = self.calculate_spectral_rolloff(&magnitude_spectrum, sample_rate);
        let zero_crossing_rate = self.calculate_zero_crossing_rate(window);
        let spectral_flux = self.calculate_spectral_flux(&magnitude_spectrum);

        Ok(SpectralFeatures {
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
            spectral_flux,
        })
    }

    /// Calculate spectral centroid
    fn calculate_spectral_centroid(&self, spectrum: &[f32], sample_rate: u32) -> f32 {
        let freq_resolution = sample_rate as f32 / (2.0 * spectrum.len() as f32);
        
        let weighted_sum: f32 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| (i as f32 * freq_resolution) * mag)
            .sum();
        
        let total_magnitude: f32 = spectrum.iter().sum();
        
        if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            0.0
        }
    }

    /// Calculate spectral rolloff (frequency below which 85% of energy lies)
    fn calculate_spectral_rolloff(&self, spectrum: &[f32], sample_rate: u32) -> f32 {
        let total_energy: f32 = spectrum.iter().map(|&x| x * x).sum();
        let threshold = total_energy * 0.85;
        
        let mut cumulative_energy = 0.0;
        let freq_resolution = sample_rate as f32 / (2.0 * spectrum.len() as f32);
        
        for (i, &mag) in spectrum.iter().enumerate() {
            cumulative_energy += mag * mag;
            if cumulative_energy >= threshold {
                return i as f32 * freq_resolution;
            }
        }
        
        (spectrum.len() - 1) as f32 * freq_resolution
    }

    /// Calculate zero crossing rate
    fn calculate_zero_crossing_rate(&self, window: &[f32]) -> f32 {
        if window.len() < 2 {
            return 0.0;
        }

        let zero_crossings = window.windows(2)
            .filter(|pair| (pair[0] >= 0.0 && pair[1] < 0.0) || (pair[0] < 0.0 && pair[1] >= 0.0))
            .count();

        zero_crossings as f32 / (window.len() - 1) as f32
    }

    /// Calculate spectral flux (measure of spectral change)
    fn calculate_spectral_flux(&self, spectrum: &[f32]) -> f32 {
        // Simple implementation - would need previous frame for proper calculation
        spectrum.iter().map(|&x| x.abs()).sum::<f32>() / spectrum.len() as f32
    }
}

impl HybridVad {
    /// Create new hybrid VAD combining energy and spectral approaches
    pub fn new(energy_threshold: f32, spectral_threshold: f32, window_size: usize) -> Self {
        Self {
            energy_vad: EnergyVad::new(energy_threshold, window_size),
            spectral_vad: SpectralVad::new(energy_threshold, spectral_threshold, window_size),
            energy_weight: 0.4,
            spectral_weight: 0.6,
        }
    }

    /// Analyze frame using hybrid approach
    pub fn analyze_frame(&self, window: &[f32], sample_rate: u32) -> Result<FrameAnalysis> {
        let energy_analysis = self.energy_vad.analyze_frame(window, sample_rate)?;
        let spectral_analysis = self.spectral_vad.analyze_frame(window, sample_rate)?;

        // Weighted combination of decisions
        let combined_confidence = energy_analysis.confidence * self.energy_weight +
                                 spectral_analysis.confidence * self.spectral_weight;

        let is_speech = energy_analysis.is_speech || spectral_analysis.is_speech;

        Ok(FrameAnalysis {
            is_speech,
            confidence: combined_confidence,
            energy: energy_analysis.energy,
            spectral_features: spectral_analysis.spectral_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_vad() {
        let config = VadConfig::default();
        let energy_vad = EnergyVad::new(config.energy_threshold, config.window_size);
        
        // Test with silent window
        let silent_window = vec![0.0; 512];
        let result = energy_vad.analyze_frame(&silent_window, 16000).unwrap();
        assert!(!result.is_speech);
        assert!(result.energy < 0.01);

        // Test with speech-like window
        let speech_window: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let result = energy_vad.analyze_frame(&speech_window, 16000).unwrap();
        assert!(result.energy > 0.0);
    }

    #[test]
    fn test_vad_processor() {
        let config = VadConfig::default();
        let processor = VadProcessor::new(config);
        
        // Create test audio with speech and silence
        let sample_rate = 16000;
        let mut samples = Vec::new();
        
        // Add silence (1 second)
        samples.extend(std::iter::repeat(0.0).take(sample_rate));
        
        // Add speech-like signal (1 second)
        for i in 0..sample_rate {
            samples.push((i as f32 * 0.01).sin() * 0.3);
        }
        
        // Add more silence
        samples.extend(std::iter::repeat(0.0).take(sample_rate));

        let result = processor.process_audio(samples, sample_rate as u32).unwrap();
        
        assert!(result.speech_segments.len() > 0);
        assert!(result.processed_duration < result.original_duration);
    }

    #[test]
    fn test_spectral_features() {
        let spectral_vad = SpectralVad::new(0.01, 1000.0, 512);
        
        // Create a test signal with known frequency content
        let sample_rate = 16000;
        let window: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let features = spectral_vad.extract_spectral_features(&window, sample_rate).unwrap();
        
        // Spectral centroid should be around 440 Hz for pure tone
        assert!(features.spectral_centroid > 400.0 && features.spectral_centroid < 500.0);
        assert!(features.zero_crossing_rate > 0.0);
    }
}