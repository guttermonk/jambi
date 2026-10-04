      888888        d8888 888b     d888 888888b.  8888888
        "88b       d88888 8888b   d8888 888  "88b   888  
         888      d88P888 88888b.d88888 888  .88P   888  
         888     d88P 888 888Y88888P888 8888888K.   888  
         888    d88P  888 888 Y888P 888 888  "Y88b  888  
         888   d88P   888 888  Y8P  888 888    888  888  
         88P  d8888888888 888   "   888 888   d88P  888  
         888 d88P     888 888       888 8888888P" 8888888
       .d88P             a blazing-fast                  
     .d88P"      voice transcription application         
    888P"                built with Rust                 

# Jambi
Jambi's mission is to transcribe audio to your clipboard, as quickly and accurately as possible, while staying privacy-focused and open-source — on everyday machines with integrated graphics, not just those with a discrete GPU. That's why it runs on Vosk: recognition happens as you speak, on a couple of CPU cores, where GPU-oriented Whisper models would leave you waiting.

Jambi aims to help computer users with disabilities, such as vision or physical impairments, by providing real-time transcription of their speech. It's also a great tool for anyone who wants to transcribe audio quickly and easily.

This is the alpha release and the project is still in early development. Currently looking for feedback and contributors. If you are a developer, you can contribute to the project by submitting pull requests or reporting issues.

If you like the project, please show your support by leaving a star. Thanks!


## Features

- **Real-time transcription** - Processes audio faster than playback speed on CPU
- **Small footprint** - Only 40MB model size (vs 75MB+ for Whisper)
- **Low latency** - Instant results without GPU requirements
- **Multiple languages** - Supports 12+ languages including English, Spanish, French, German, Chinese, etc.
- **Privacy-focused** - Everything runs locally after initial setup, no cloud services required

## Quick Start

### Prerequisites

**For running a pre-compiled binary:**
- Standard audio libraries (typically already installed)

**On Ubuntu/Debian:**
```bash
sudo apt-get install libasound2-dev libssl-dev build-essential
```

**On Fedora:**
```bash
sudo dnf install alsa-lib-devel openssl-devel
```

**On NixOS:** See the [NixOS Installation](#nixos-installation) section below.

### Installation

#### Standard Build Installation

**Prereqs for building from source:**
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))
- ALSA development libraries (Linux)
- OpenSSL development libraries

1. Clone the repository:
```bash
git clone https://github.com/guttermonk/jambi.git
cd jambi
```

2. Build the project:
```bash
cargo build --release
```

3. Download the Vosk library and model (automatic on first run):
```bash
./jambi --help
```

#### NixOS Installation

For NixOS users, you can install jambi directly using the flake:

```bash
# Install permanently to your system
nix profile install github:guttermonk/jambi

# Or run once without installing
nix run github:guttermonk/jambi

# For development
nix develop github:guttermonk/jambi
```

After installation, `jambi` will be available in your PATH. The flake automatically handles all dependencies including ALSA, audio libraries, and runtime requirements.

**Note for NixOS systems:** For system-wide installation, add to your `configuration.nix`:
```nix
{
  inputs.jambi.url = "github:guttermonk/jambi";
  
  environment.systemPackages = [ inputs.jambi.packages.${system}.default ];
}
```

### Usage

#### Interactive Mode (Default)
Start Jambi in interactive mode for recording and transcription:
```bash
./jambi
```

#### Record Audio
Record audio and transcribe it:
```bash
./jambi record
# Press Enter to start, Enter again to stop
```

#### Live transcription
Speak freely while your words are converted to text:
```bash
./jambi record --live
# Press Enter to start, Enter again to stop
```

#### Live Dictation (hold-to-talk)
Dictate into whatever window has focus, with no terminal involved: hold a key,
speak, release, and the text is typed at the cursor.

Set the mode once in `~/.config/jambi/config.toml`:
```toml
mode = "live"
```
then bind a key to the two halves of the hold. For Hyprland:
```
bind  = SUPER SHIFT, D, exec, jambi dictate start
bindr = SUPER SHIFT, D, exec, jambi dictate stop
```
`start` records until signalled; `stop` signals it and exits immediately. The
text goes to the cursor via `wtype` (Wayland) or `xdotool` (X11), and to the
clipboard as well when `auto_copy` is on, so nothing is lost if typing fails.

Because Vosk recognises speech *as you speak*, releasing the key is near
instant -- there is no post-hoc pass over the clip to wait through.

#### Transcribe File
Transcribe an existing audio file:
```bash
./jambi transcribe audio.wav
```

#### List Available Models
See all available language models:
```bash
./jambi models
```

## Configuration

### Command-Line Options

The `--verbose` flag controls the display of informational messages:
```bash
# Default (quiet mode) - only shows warnings and errors
./jambi record

# Verbose mode - shows detailed progress and debug information
./jambi --verbose record
```

When verbose mode is disabled (default), the following messages are suppressed:
- Model loading notifications
- Recording progress updates
- File path information
- Transcription statistics

### Configuration File

Copy the example configuration to the path Jambi reads by default:
```bash
mkdir -p ~/.config/jambi
cp config.example.toml ~/.config/jambi/config.toml
```

Any other location works too, passed explicitly:
```bash
./jambi --config ./config.toml
```

Edit it to change:
- Interaction mode (windowed TUI or live hold-to-talk dictation)
- Model selection (language)
- Sample rate
- Auto-copy to clipboard
- Output directory

Every key is optional -- a file setting nothing but `mode` is valid, and the
rest falls back to defaults.

Example configuration:
```toml
# Top-level keys must come BEFORE the first [table] header. A bare key after a
# header belongs to that table, so `auto_copy` placed below [audio] would
# silently become `audio.auto_copy` and do nothing.
mode = "windowed"   # or "live" for hold-to-talk dictation
auto_copy = true
keep_recordings = false

[vosk]
model = "SmallEnUs"  # Options: SmallEnUs, SmallEs, SmallFr, etc.
sample_rate = 16000.0
show_words = true

[audio]
sample_rate = 16000
channels = 1
output_dir = "~/jambi_recordings"

[dictate]            # only used when mode = "live"
modifier_grace_ms = 250  # wait for hotkey modifiers to lift before typing
type_delay_ms = 10       # per-keystroke delay for wtype/xdotool
# icon = "/home/you/.icons/microphone.svg"  # see Notifications below
```

### Notifications

A dictation cycle reports itself through `notify-send`, replacing its own
popup rather than stacking: listening, then done, or the specific failure
(microphone unavailable, no speech detected, typing failed with the text
left on the clipboard). These matter more than usual in live mode, since it
runs detached from a keybind and nothing else would surface an error.

Notifications carry a microphone glyph, shipped in `assets/` with white
strokes so it reads on a dark panel. If your desktop recolours its icons,
point `dictate.icon` at its copy instead and the notification follows your
palette:
```toml
[dictate]
icon = "/home/you/.icons/microphone.svg"
```
The path is resolved each time a notification fires, so a theme switch needs
no restart. A path that does not exist falls back to no icon rather than a
broken image.

### Command-Line Mode Override

`--mode` overrides whatever the config file says, for a single run:
```bash
./jambi --mode live       # force hold-to-talk dictation
./jambi --mode windowed   # force the interactive TUI
./jambi mode              # print the mode currently in effect
```

## Project Structure

```
jambi/
├── src/                 # Rust source code
│   ├── main.rs         # CLI entry point
│   ├── lib.rs          # Library interface
│   ├── vosk_engine.rs  # Vosk speech recognition
│   ├── audio.rs        # Audio recording
│   └── config.rs       # Configuration handling
├── target/release/     # Compiled binary (after build)
├── Cargo.toml         # Rust dependencies
├── config.example.toml # Example configuration
└── jambi              # Main launcher script
```


## Performance

Vosk provides real-time transcription on CPU:
- **Speed**: 0.1-0.5x real-time factor (faster than audio playback)
- **Memory**: ~300MB RAM usage
- **Model size**: 40MB (small model)

Compare with Whisper on CPU:
- Whisper: 30-60+ seconds for 3 seconds of audio
- Vosk: 0.3-1.5 seconds for 3 seconds of audio

## Supported Languages

| Model | Language | Size |
|-------|----------|------|
| SmallEnUs | English (US) | 40MB |
| LargeEnUs | English (US) | 1.8GB |
| SmallEs | Spanish | 40MB |
| SmallFr | French | 40MB |
| SmallDe | German | 40MB |
| SmallRu | Russian | 40MB |
| SmallIt | Italian | 40MB |
| SmallPt | Portuguese | 40MB |
| SmallNl | Dutch | 40MB |
| SmallCn | Chinese | 40MB |
| SmallJa | Japanese | 40MB |

## Development

### Building from Source

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt

# Run linter
cargo clippy
```

### Troubleshooting

If you encounter library loading issues:
```bash
# Check that Vosk library is downloaded
ls ~/.cache/jambi/vosk-lib/libvosk.so

# Verify Vosk model is downloaded
ls ~/.cache/jambi/vosk-models/

# Check library dependencies (Linux)
ldd ~/.cache/jambi/vosk-lib/libvosk.so

# Test basic functionality
./jambi --help
```

**Note**: Utility scripts for advanced diagnostics and testing are not included in the repository but can be created locally in a `scripts/` directory if needed.

## Architecture

Jambi is built with a modular architecture:

- **Audio Module**: Handles cross-platform audio recording using CPAL
- **Vosk Engine**: Manages speech recognition with Vosk models
- **Config Module**: Handles configuration and settings
- **CLI Interface**: Provides user-friendly command-line interface

The application uses async Rust (Tokio) for efficient I/O handling and can process multiple audio streams concurrently.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition API
- [CPAL](https://github.com/RustAudio/cpal) - Cross-platform audio library for Rust
- Original inspiration from [WhisperNow](https://github.com/shinglyu/WhisperNow)

## FAQ

**Q: Why Vosk instead of Whisper?**
A: Vosk is optimized for real-time CPU-based transcription, making it 10-50x faster than Whisper on CPU. It's ideal for live transcription applications.

**Q: Can I use my own models?**
A: Yes! Download any Vosk model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and place it in `~/.cache/jambi/vosk-models/`.

**Q: Does it work offline?**
A: Yes, everything runs locally on your machine. No internet connection required after initial setup.

**Q: How accurate is it?**
A: Vosk provides good accuracy for real-time transcription. For highest accuracy with more processing time, consider using Whisper models instead.

**Q: In Hyprland, how do I make Jambi open in the same workspace every time?**
A: Yes, you can use Hyprland's window rules feature to achieve this. For example, if you always want Jambi to open in the Special Workspace, add the following line to your `~/.config/hypr/hyprland.conf` file:
```
windowrulev2 = workspace special silent, class:jambi
```

**Q: In Hyprland, is there a way to only allow one instance of Jambi to run at a time?**
A: Yes, you can launch Jambi with the following script, which will check to see if Jambi is already running before starting a new instance:
```bash
#!usr/bin/env bash

    if hyprctl clients | grep -q "class: jambi"; then
      workspace=$(hyprctl clients | grep "class: jambi" -B4 | grep "workspace:" | head -n1 | awk '{print $2}')
      if [[ "$workspace" == "-99" ]]; then
        hyprctl dispatch togglespecialworkspace
      else
        hyprctl dispatch workspace $workspace
      fi
    else
      kitty --class jambi -e jambi record --live &
      sleep 0.5
      workspace=$(hyprctl clients | grep "class: jambi" -B4 | grep "workspace:" | head -n1 | awk '{print $2}')
      if [[ "$workspace" == "-99" ]]; then
        hyprctl dispatch togglespecialworkspace
      else
        hyprctl dispatch workspace $workspace
      fi
    fi
```
