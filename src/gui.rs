//! GUI module using egui for cross-platform native interface
//! 
//! This module provides a modern, responsive GUI interface using egui for
//! the WhisperNow transcription application with real-time updates and
//! comprehensive controls.

use anyhow::{Context, Result};
use eframe::egui::{self, *};
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::Utc;

use crate::{
    WhisperApp, AppState, AppEvent, TranscriptionResult, RecordingState, 
    TranscriptionState, AppConfig, ModelConfig, AudioConfig, VadConfig,
    AudioDeviceInfo, ModelInfo, copy_to_clipboard_nonblocking, show_notification,
    format_duration, format_file_size
};

/// Main GUI application structure
pub struct WhisperGui {
    app: WhisperApp,
    state: Arc<AppState>,
    gui_state: GuiState,
    event_receiver: tokio::sync::broadcast::Receiver<AppEvent>,
}

/// GUI-specific state management
#[derive(Debug)]
struct GuiState {
    // UI state
    selected_transcriptions: Vec<bool>,
    show_settings: bool,
    show_about: bool,
    show_model_downloader: bool,
    
    // Recording state
    recording_duration: f64,
    last_recording_update: Instant,
    
    // Transcription queue animation
    spinner_rotation: f32,
    last_spinner_update: Instant,
    
    // Settings panels
    settings_tab: SettingsTab,
    
    // Model management
    available_models: Vec<ModelInfo>,
    model_download_progress: Option<ModelDownloadProgress>,
    
    // Audio devices
    audio_devices: Vec<AudioDeviceInfo>,
    
    // Error handling
    error_message: Option<String>,
    show_error_dialog: bool,
    
    // Status messages
    status_message: Option<StatusMessage>,
    
    // Configuration
    temp_config: AppConfig,
    config_dirty: bool,
    
    // Window state
    window_focused: bool,
    
    // Search and filtering
    transcription_search: String,
    show_search: bool,
}

/// Settings panel tabs
#[derive(Debug, Clone, Copy, PartialEq)]
enum SettingsTab {
    General,
    Audio,
    Model,
    Advanced,
}

/// Model download progress tracking
#[derive(Debug, Clone)]
struct ModelDownloadProgress {
    model_name: String,
    downloaded_bytes: u64,
    total_bytes: u64,
    status: String,
}

/// Status message with automatic dismissal
#[derive(Debug, Clone)]
struct StatusMessage {
    text: String,
    message_type: StatusMessageType,
    created_at: Instant,
    duration: Duration,
}

#[derive(Debug, Clone, PartialEq)]
enum StatusMessageType {
    Info,
    Success,
    Warning,
    Error,
}

impl WhisperGui {
    /// Create new GUI application
    pub fn new(app: WhisperApp) -> Self {
        let state = app.state.clone();
        let event_receiver = state.subscribe_events();
        let config = state.get_config();

        let gui_state = GuiState {
            selected_transcriptions: Vec::new(),
            show_settings: false,
            show_about: false,
            show_model_downloader: false,
            recording_duration: 0.0,
            last_recording_update: Instant::now(),
            spinner_rotation: 0.0,
            last_spinner_update: Instant::now(),
            settings_tab: SettingsTab::General,
            available_models: crate::whisper::WhisperEngine::list_available_models(),
            model_download_progress: None,
            audio_devices: Vec::new(),
            error_message: None,
            show_error_dialog: false,
            status_message: None,
            temp_config: config,
            config_dirty: false,
            window_focused: true,
            transcription_search: String::new(),
            show_search: false,
        };

        Self {
            app,
            state,
            gui_state,
            event_receiver,
        }
    }

    /// Handle application events
    fn handle_events(&mut self) {
        while let Ok(event) = self.event_receiver.try_recv() {
            match event {
                AppEvent::RecordingStarted(_) => {
                    self.gui_state.last_recording_update = Instant::now();
                },
                AppEvent::RecordingStopped(_) => {
                    self.gui_state.recording_duration = 0.0;
                },
                AppEvent::TranscriptionCompleted(result) => {
                    self.update_transcription_list();
                    self.show_status_message(
                        format!("Transcription completed: {}", 
                               result.text.chars().take(30).collect::<String>()),
                        StatusMessageType::Success
                    );
                },
                AppEvent::TranscriptionError(_, error) => {
                    self.show_error(&format!("Transcription failed: {}", error));
                },
                AppEvent::ModelLoaded(model_name) => {
                    self.show_status_message(
                        format!("Model '{}' loaded successfully", model_name),
                        StatusMessageType::Success
                    );
                },
                AppEvent::ModelLoadError(error) => {
                    self.show_error(&format!("Failed to load model: {}", error));
                },
                AppEvent::ConfigurationChanged(config) => {
                    self.gui_state.temp_config = config;
                    self.gui_state.config_dirty = false;
                },
                _ => {}
            }
        }
    }

    /// Show status message
    fn show_status_message(&mut self, text: String, message_type: StatusMessageType) {
        let duration = match message_type {
            StatusMessageType::Error => Duration::from_secs(10),
            StatusMessageType::Warning => Duration::from_secs(5),
            _ => Duration::from_secs(3),
        };

        self.gui_state.status_message = Some(StatusMessage {
            text,
            message_type,
            created_at: Instant::now(),
            duration,
        });
    }

    /// Show error dialog
    fn show_error(&mut self, message: &str) {
        self.gui_state.error_message = Some(message.to_string());
        self.gui_state.show_error_dialog = true;
    }

    /// Update transcription list selection state
    fn update_transcription_list(&mut self) {
        let transcriptions = self.state.get_transcriptions();
        self.gui_state.selected_transcriptions.resize(transcriptions.len(), false);
    }

    /// Update recording duration
    fn update_recording_duration(&mut self) {
        if matches!(self.state.get_recording_state(), RecordingState::Recording) {
            let elapsed = self.gui_state.last_recording_update.elapsed();
            self.gui_state.recording_duration += elapsed.as_secs_f64();
            self.gui_state.last_recording_update = Instant::now();
        }
    }

    /// Update spinner animation
    fn update_spinner(&mut self) {
        if matches!(self.state.get_transcription_state(), TranscriptionState::Processing | TranscriptionState::QueueProcessing(_)) {
            let elapsed = self.gui_state.last_spinner_update.elapsed();
            self.gui_state.spinner_rotation += elapsed.as_secs_f32() * 360.0; // One rotation per second
            self.gui_state.spinner_rotation %= 360.0;
            self.gui_state.last_spinner_update = Instant::now();
        }
    }

    /// Dismiss status message if expired
    fn update_status_message(&mut self) {
        if let Some(ref status) = self.gui_state.status_message.clone() {
            if status.created_at.elapsed() > status.duration {
                self.gui_state.status_message = None;
            }
        }
    }
}

impl eframe::App for WhisperGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle events and updates
        self.handle_events();
        self.update_recording_duration();
        self.update_spinner();
        self.update_status_message();

        // Request repaint for animations
        ctx.request_repaint_after(Duration::from_millis(50));

        // Handle keyboard shortcuts
        if ctx.input(|i| i.key_pressed(Key::Space)) && !self.gui_state.show_settings {
            self.toggle_recording();
        }

        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.gui_state.show_settings = false;
            self.gui_state.show_about = false;
            self.gui_state.show_error_dialog = false;
        }

        // Main UI panels
        self.render_top_panel(ctx);
        self.render_central_panel(ctx);
        self.render_bottom_panel(ctx);

        // Modal dialogs
        if self.gui_state.show_settings {
            self.render_settings_window(ctx);
        }

        if self.gui_state.show_about {
            self.render_about_window(ctx);
        }

        if self.gui_state.show_error_dialog {
            self.render_error_dialog(ctx);
        }

        if self.gui_state.show_model_downloader {
            self.render_model_downloader(ctx);
        }
    }
}

impl WhisperGui {
    /// Render top control panel
    fn render_top_panel(&mut self, ctx: &egui::Context) {
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Recording button
                self.render_recording_button(ui);

                ui.separator();

                // Queue status
                self.render_queue_status(ui);

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // Settings button
                    if ui.button("⚙ Settings").clicked() {
                        self.gui_state.show_settings = true;
                    }

                    // Search button
                    if ui.button("🔍").clicked() {
                        self.gui_state.show_search = !self.gui_state.show_search;
                    }
                });
            });

            // Search bar
            if self.gui_state.show_search {
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    ui.text_edit_singleline(&mut self.gui_state.transcription_search);
                    if ui.button("Clear").clicked() {
                        self.gui_state.transcription_search.clear();
                    }
                });
            }

            // Status message bar
            if let Some(ref status) = self.gui_state.status_message {
                self.render_status_bar(ui, status);
            }
        });
    }

    /// Render recording button with state
    fn render_recording_button(&mut self, ui: &mut Ui) {
        let recording_state = self.state.get_recording_state();
        let (text, color, enabled) = match recording_state {
            RecordingState::Idle => ("🎙 Record (Space)", Color32::GREEN, true),
            RecordingState::Recording => ("⏹ Stop (Space)", Color32::RED, true),
            RecordingState::Stopping => ("Stopping...", Color32::YELLOW, false),
            RecordingState::Error(_) => ("❌ Error", Color32::RED, false),
        };

        let button = Button::new(text)
            .fill(color.gamma_multiply(0.2))
            .min_size(Vec2::new(120.0, 30.0));

        if ui.add_enabled(enabled, button).clicked() {
            self.toggle_recording();
        }

        // Show recording duration
        if matches!(recording_state, RecordingState::Recording) {
            ui.label(format!("Duration: {}", format_duration(self.gui_state.recording_duration)));
        }
    }

    /// Render transcription queue status
    fn render_queue_status(&mut self, ui: &mut Ui) {
        let transcription_state = self.state.get_transcription_state();
        
        match transcription_state {
            TranscriptionState::Processing => {
                ui.horizontal(|ui| {
                    self.render_spinner(ui);
                    ui.label("Processing transcription...");
                });
            },
            TranscriptionState::QueueProcessing(count) => {
                ui.horizontal(|ui| {
                    self.render_spinner(ui);
                    ui.label(format!("Queue: {} items", count));
                });
            },
            TranscriptionState::LoadingModel => {
                ui.horizontal(|ui| {
                    self.render_spinner(ui);
                    ui.label("Loading model...");
                });
            },
            TranscriptionState::Error(ref error) => {
                ui.horizontal(|ui| {
                    ui.label("❌");
                    ui.label(format!("Error: {}", error));
                });
            },
            TranscriptionState::Idle => {
                ui.label("Ready");
            },
        }
    }

    /// Render spinning animation
    fn render_spinner(&mut self, ui: &mut Ui) {
        let size = 16.0;
        let center = ui.min_rect().center() + Vec2::new(size / 2.0, 0.0);
        
        ui.allocate_space(Vec2::new(size, size));
        
        let painter = ui.painter();
        let angle = self.gui_state.spinner_rotation * std::f32::consts::PI / 180.0;
        
        // Draw spinning circle segments
        for i in 0..8 {
            let segment_angle = angle + (i as f32 * std::f32::consts::PI / 4.0);
            let opacity = (255.0 * (i as f32 / 8.0)) as u8;
            let color = Color32::from_rgba_unmultiplied(100, 150, 255, opacity);
            
            let start = center + Vec2::angled(segment_angle) * (size / 3.0);
            let end = center + Vec2::angled(segment_angle) * (size / 2.0);
            
            painter.line_segment([start.to_pos2(), end.to_pos2()], Stroke::new(2.0, color));
        }
    }

    /// Render status message bar
    fn render_status_bar(&self, ui: &mut Ui, status: &StatusMessage) {
        let color = match status.message_type {
            StatusMessageType::Success => Color32::GREEN,
            StatusMessageType::Warning => Color32::YELLOW,
            StatusMessageType::Error => Color32::RED,
            StatusMessageType::Info => Color32::BLUE,
        };

        ui.horizontal(|ui| {
            ui.colored_label(color, &status.text);
        });
    }

    /// Render main transcription panel
    fn render_central_panel(&mut self, ctx: &egui::Context) {
        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    self.render_transcription_list(ui);
                });
        });
    }

    /// Render transcription list
    fn render_transcription_list(&mut self, ui: &mut Ui) {
        let transcriptions = self.state.get_transcriptions();
        
        if transcriptions.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(50.0);
                ui.heading("No transcriptions yet");
                ui.label("Press the Record button or Space to start recording");
            });
            return;
        }

        // Filter transcriptions based on search
        let filtered_transcriptions: Vec<_> = transcriptions.iter()
            .enumerate()
            .filter(|(_, t)| {
                if self.gui_state.transcription_search.is_empty() {
                    true
                } else {
                    t.text.to_lowercase().contains(&self.gui_state.transcription_search.to_lowercase())
                }
            })
            .collect();

        // Ensure selection state matches filtered list
        if self.gui_state.selected_transcriptions.len() != transcriptions.len() {
            self.gui_state.selected_transcriptions.resize(transcriptions.len(), false);
        }

        for (original_idx, transcription) in filtered_transcriptions {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    // Selection checkbox
                    let mut selected = self.gui_state.selected_transcriptions.get(original_idx).copied().unwrap_or(false);
                    if ui.checkbox(&mut selected, "").changed() {
                        if let Some(sel) = self.gui_state.selected_transcriptions.get_mut(original_idx) {
                            *sel = selected;
                        }
                    }

                    // Transcription content
                    ui.vertical(|ui| {
                        // Metadata
                        ui.horizontal(|ui| {
                            ui.small_text(format!("⏱ {}", transcription.timestamp.format("%H:%M:%S")));
                            ui.small_text(format!("🕐 {}", format_duration(transcription.processing_time_ms as f64 / 1000.0)));
                            ui.small_text(format!("🎯 {:.0}%", transcription.confidence * 100.0));
                            ui.small_text(format!("🤖 {}", transcription.model_name));
                        });

                        // Transcription text
                        ui.label(&transcription.text);
                    });

                    // Copy button
                    if ui.button("📋 Copy").clicked() {
                        if let Err(e) = copy_to_clipboard_nonblocking(&transcription.text) {
                            self.show_error(&format!("Failed to copy to clipboard: {}", e));
                        } else {
                            self.show_status_message("Copied to clipboard".to_string(), StatusMessageType::Success);
                        }
                    }
                });
            });
        }
    }

    /// Render bottom action panel
    fn render_bottom_panel(&mut self, ctx: &egui::Context) {
        TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Bulk actions
                if ui.button("Select All").clicked() {
                    self.gui_state.selected_transcriptions.fill(true);
                }

                if ui.button("Select None").clicked() {
                    self.gui_state.selected_transcriptions.fill(false);
                }

                ui.separator();

                // Copy actions
                let selected_count = self.gui_state.selected_transcriptions.iter().filter(|&&x| x).count();
                
                if ui.button(format!("Copy Selected ({})", selected_count))
                    .on_hover_text("Copy selected transcriptions to clipboard")
                    .clicked() 
                {
                    self.copy_selected_transcriptions();
                }

                if ui.button("Copy All").clicked() {
                    self.copy_all_transcriptions();
                }

                ui.separator();

                // Clear actions
                if ui.button("Clear All")
                    .on_hover_text("Clear all transcriptions")
                    .clicked() 
                {
                    self.state.clear_transcriptions();
                    self.gui_state.selected_transcriptions.clear();
                }

                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    // About button
                    if ui.button("About").clicked() {
                        self.gui_state.show_about = true;
                    }

                    // Status info
                    let transcription_count = self.state.get_transcriptions().len();
                    ui.label(format!("Transcriptions: {}", transcription_count));
                });
            });
        });
    }

    /// Render settings window
    fn render_settings_window(&mut self, ctx: &egui::Context) {
        Window::new("⚙ Settings")
            .resizable(true)
            .default_size([600.0, 500.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Settings tabs
                    ui.selectable_value(&mut self.gui_state.settings_tab, SettingsTab::General, "General");
                    ui.selectable_value(&mut self.gui_state.settings_tab, SettingsTab::Audio, "Audio");
                    ui.selectable_value(&mut self.gui_state.settings_tab, SettingsTab::Model, "Model");
                    ui.selectable_value(&mut self.gui_state.settings_tab, SettingsTab::Advanced, "Advanced");
                });

                ui.separator();

                ScrollArea::vertical().show(ui, |ui| {
                    match self.gui_state.settings_tab {
                        SettingsTab::General => self.render_general_settings(ui),
                        SettingsTab::Audio => self.render_audio_settings(ui),
                        SettingsTab::Model => self.render_model_settings(ui),
                        SettingsTab::Advanced => self.render_advanced_settings(ui),
                    }
                });

                ui.separator();

                // Settings action buttons
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.save_config();
                    }

                    if ui.button("Reset to Defaults").clicked() {
                        self.gui_state.temp_config = AppConfig::default();
                        self.gui_state.config_dirty = true;
                    }

                    if ui.button("Cancel").clicked() {
                        self.gui_state.temp_config = self.state.get_config();
                        self.gui_state.config_dirty = false;
                        self.gui_state.show_settings = false;
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            if self.gui_state.config_dirty {
                                self.save_config();
                            }
                            self.gui_state.show_settings = false;
                        }
                    });
                });
            });
    }

    /// Render general settings tab
    fn render_general_settings(&mut self, ui: &mut Ui) {
        ui.heading("Application Settings");

        ui.horizontal(|ui| {
            ui.label("Auto-copy to clipboard:");
            if ui.checkbox(&mut self.gui_state.temp_config.app.auto_copy_to_clipboard, "Automatically copy transcriptions to clipboard").changed() {
                self.gui_state.config_dirty = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Show notifications:");
            if ui.checkbox(&mut self.gui_state.temp_config.app.show_notifications, "Show desktop notifications for completed transcriptions").changed() {
                self.gui_state.config_dirty = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Keep recording files:");
            if ui.checkbox(&mut self.gui_state.temp_config.app.keep_recording_files, "Keep audio files after transcription").changed() {
                self.gui_state.config_dirty = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Max recent transcriptions:");
            if ui.add(Slider::new(&mut self.gui_state.temp_config.app.max_recent_transcriptions, 10..=1000).text("items")).changed() {
                self.gui_state.config_dirty = true;
            }
        });
    }

    /// Render audio settings tab
    fn render_audio_settings(&mut self, ui: &mut Ui) {
        ui.heading("Audio Settings");

        ui.horizontal(|ui| {
            ui.label("Sample rate:");
            let mut sample_rate = self.gui_state.temp_config.audio.sample_rate as i32;
            if ui.add(Slider::new(&mut sample_rate, 8000..=48000).text("Hz").logarithmic(true)).changed() {
                self.gui_state.temp_config.audio.sample_rate = sample_rate as u32;
                self.gui_state.config_dirty = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Channels:");
            let mut channels = self.gui_state.temp_config.audio.channels as i32;
            if ui.add(Slider::new(&mut channels, 1..=2).text("channels")).changed() {
                self.gui_state.temp_config.audio.channels = channels as u16;
                self.gui_state.config_dirty = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Recording directory:");
            ui.text_edit_singleline(&mut self.gui_state.temp_config.audio.recording_path.to_string_lossy().to_string());
            if ui.button("Browse").clicked() {
                // TODO: Add file dialog
            }
        });
    }

    /// Render model settings tab
    fn render_model_settings(&mut self, ui: &mut Ui) {
        ui.heading("Whisper Model Settings");

        ui.horizontal(|ui| {
            ui.label("Model:");
            egui::ComboBox::from_label("Select Model")
                .selected_text(&self.gui_state.temp_config.model.name)
                .show_ui(ui, |ui| {
                    for model in &self.gui_state.available_models {
                        let selected = ui.selectable_value(
                            &mut self.gui_state.temp_config.model.name, 
                            model.name.clone(), 
                            format!("{} - {}", model.name, model.description)
                        );
                        if selected.changed() {
                            self.gui_state.config_dirty = true;
                        }
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Device:");
            egui::ComboBox::from_label("Compute Device")
                .selected_text(&self.gui_state.temp_config.model.device)
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.gui_state.temp_config.model.device, "cpu".to_string(), "CPU").changed() {
                        self.gui_state.config_dirty = true;
                    }
                    if candle_core::utils::cuda_is_available() {
                        if ui.selectable_value(&mut self.gui_state.temp_config.model.device, "cuda".to_string(), "CUDA GPU").changed() {
                            self.gui_state.config_dirty = true;
                        }
                    }
                    if candle_core::utils::metal_is_available() {
                        if ui.selectable_value(&mut self.gui_state.temp_config.model.device, "metal".to_string(), "Metal GPU").changed() {
                            self.gui_state.config_dirty = true;
                        }
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Beam size:");
            if ui.add(Slider::new(&mut self.gui_state.temp_config.model.beam_size, 1..=10).text("beams")).changed() {
                self.gui_state.config_dirty = true;
            }
        });

        if ui.button("Download Models").clicked() {
            self.gui_state.show_model_downloader = true;
        }
    }

    /// Render advanced settings tab
    fn render_advanced_settings(&mut self, ui: &mut Ui) {
        ui.heading("Voice Activity Detection");

        ui.horizontal(|ui| {
            ui.label("Enable VAD:");
            if ui.checkbox(&mut self.gui_state.temp_config.vad.enabled, "Remove silence before transcription").changed() {
                self.gui_state.config_dirty = true;
            }
        });

        if self.gui_state.temp_config.vad.enabled {
            ui.horizontal(|ui| {
                ui.label("Energy threshold:");
                if ui.add(Slider::new(&mut self.gui_state.temp_config.vad.energy_threshold, 0.001..=0.1).logarithmic(true)).changed() {
                    self.gui_state.config_dirty = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Min silence duration (ms):");
                let mut silence_ms = self.gui_state.temp_config.vad.min_silence_duration_ms as i32;
                if ui.add(Slider::new(&mut silence_ms, 100..=2000).text("ms")).changed() {
                    self.gui_state.temp_config.vad.min_silence_duration_ms = silence_ms as u64;
                    self.gui_state.config_dirty = true;
                }
            });
        }
    }

    /// Render about dialog
    fn render_about_window(&mut self, ctx: &egui::Context) {
        Window::new("About WhisperNow")
            .resizable(false)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("WhisperNow");
                    ui.label(format!("Version {}", crate::app_version()));
                    ui.add_space(10.0);
                    ui.label("High-performance voice transcription tool");
                    ui.label("Built with Rust and Candle ML framework");
                    ui.add_space(10.0);
                    ui.hyperlink_to("GitHub Repository", "https://github.com/username/whisper-now");
                    ui.add_space(10.0);
                    if ui.button("Close").clicked() {
                        self.gui_state.show_about = false;
                    }
                });
            });
    }

    /// Render error dialog
    fn render_error_dialog(&mut self, ctx: &egui::Context) {
        if let Some(ref error_msg) = self.gui_state.error_message {
            Window::new("❌ Error")
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(error_msg);
                    ui.add_space(10.0);
                    if ui.button("OK").clicked() {
                        self.gui_state.show_error_dialog = false;
                        self.gui_state.error_message = None;
                    }
                });
        }
    }

    /// Render model downloader dialog
    fn render_model_downloader(&mut self, ctx: &egui::Context) {
        Window::new("📥 Download Models")
            .resizable(true)
            .default_size([400.0, 300.0])
            .show(ctx, |ui| {
                ui.heading("Available Models");
                
                for model in &self.gui_state.available_models.clone() {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.strong(&model.name);
                                ui.label(&model.description);
                                ui.small_text(format!("Size: {}", model.size));
                            });
                            
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button("Download").clicked() {
                                    // TODO: Implement model download
                                    self.show_status_message(
                                        format!("Downloading {} not yet implemented", model.name),
                                        StatusMessageType::Info
                                    );
                                }
                            });
                        });
                    });
                }
                
                ui.separator();
                
                if ui.button("Close").clicked() {
                    self.gui_state.show_model_downloader = false;
                }
            });
    }

    /// Toggle recording state
    fn toggle_recording(&mut self) {
        let current_state = self.state.get_recording_state();
        match current_state {
            RecordingState::Idle => {
                if let Some(ref recorder) = self.app.audio_recorder {
                    let recorder_clone = recorder.clone();
                    let state_clone = self.state.clone();
                    tokio::spawn(async move {
                        if let Err(e) = recorder_clone.start_recording().await {
                            let _ = state_clone.send_event(AppEvent::RecordingError(format!("Failed to start recording: {}", e)));
                        }
                    });
                }
            },
            RecordingState::Recording => {
                if let Some(ref recorder) = self.app.audio_recorder {
                    let recorder_clone = recorder.clone();
                    let state_clone = self.state.clone();
                    tokio::spawn(async move {
                        if let Ok(Some(recording)) = recorder_clone.stop_recording().await {
                            let _ = state_clone.send_event(AppEvent::RecordingStopped(recording.clone()));
                            
                            // Queue for transcription
                            if let Some(ref engine) = state_clone.clone().whisper_engine {
                                if let Err(e) = engine.queue_transcription(recording).await {
                                    let _ = state_clone.send_event(AppEvent::TranscriptionError(
                                        "unknown".to_string(),
                                        format!("Failed to queue transcription: {}", e)
                                    ));
                                }
                            }
                        }
                    });
                }
            },
            _ => {
                // Do nothing for stopping or error states
            }
        }
    }

    /// Copy selected transcriptions to clipboard
    fn copy_selected_transcriptions(&mut self) {
        let transcriptions = self.state.get_transcriptions();
        let selected_texts: Vec<String> = transcriptions.iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if self.gui_state.selected_transcriptions.get(i).copied().unwrap_or(false) {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect();
        
        if !selected_texts.is_empty() {
            let combined_text = selected_texts.join("\n");
            if let Err(e) = copy_to_clipboard_nonblocking(&combined_text) {
                self.show_error(&format!("Failed to copy to clipboard: {}", e));
            } else {
                self.show_status_message(
                    format!("Copied {} transcriptions to clipboard", selected_texts.len()),
                    StatusMessageType::Success
                );
            }
        }
    }

    /// Copy all transcriptions to clipboard
    fn copy_all_transcriptions(&mut self) {
        let transcriptions = self.state.get_transcriptions();
        if !transcriptions.is_empty() {
            let all_texts: Vec<String> = transcriptions.iter()
                .map(|t| t.text.clone())
                .collect();
            let combined_text = all_texts.join("\n");
            
            if let Err(e) = copy_to_clipboard_nonblocking(&combined_text) {
                self.show_error(&format!("Failed to copy to clipboard: {}", e));
            } else {
                self.show_status_message(
                    format!("Copied all {} transcriptions to clipboard", all_texts.len()),
                    StatusMessageType::Success
                );
            }
        }
    }

    /// Save configuration changes
    fn save_config(&mut self) {
        if self.gui_state.config_dirty {
            let config = self.gui_state.temp_config.clone();
            if let Err(e) = self.state.update_config(config) {
                self.show_error(&format!("Failed to save configuration: {}", e));
            } else {
                self.gui_state.config_dirty = false;
                self.show_status_message("Configuration saved".to_string(), StatusMessageType::Success);
                
                // TODO: Trigger model reload if model settings changed
            }
        }
    }
}

/// Run the GUI application
pub async fn run_gui_app(mut app: WhisperApp) -> Result<()> {
    // Initialize components
    app.init_audio().await
        .context("Failed to initialize audio recorder")?;
    
    app.init_whisper().await
        .context("Failed to initialize Whisper engine")?;

    // Load initial model
    let config = app.state.get_config();
    if let Some(ref engine) = app.whisper_engine {
        engine.load_model(&config.model).await
            .context("Failed to load initial Whisper model")?;
        
        // Start transcription processing loop
        let engine_clone = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = engine_clone.start_processing_loop().await {
                tracing::error!("Transcription processing loop error: {}", e);
            }
        });
    }

    // Configure and run egui application
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([600.0, 400.0])
            .with_icon(
                // Add a simple icon
                eframe::icon_data::from_png_bytes(&[]).unwrap_or_default()
            ),
        ..Default::default()
    };

    let gui_app = WhisperGui::new(app);
    
    eframe::run_native(
        "WhisperNow - Voice Transcription",
        options,
        Box::new(|_cc| Box::new(gui_app)),
    ).map_err(|e| anyhow::anyhow!("GUI application error: {}", e))