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
- **Optional daemon** - Keeps the model warm in the background so commands start instantly, with a tray indicator to show it is there

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

#### Background Daemon (optional, for faster startup)

Loading the Vosk model takes most of a second, and a one-shot process pays it
every single time. Run the daemon once and that cost is paid once, at login:

```bash
jambi daemon          # loads the model, then serves requests
jambi daemon status   # is one running, and what has it loaded?
jambi daemon stop     # ask it to exit
```

##### Start-up and run times

| | without daemon | with daemon | saved |
|---|---|---|---|
| **Start up** -- `jambi daemon`, paid once at login | n/a | 645 ms | -- |
| **Run** -- `jambi transcribe`, 1s clip | 1352 ms | 616 ms | 736 ms |
| **Run** -- `jambi transcribe`, 10s clip | 1405 ms | 747 ms | 658 ms |
| Reference -- `jambi mode`, loads no model | ~25 ms | ~25 ms | -- |

The saving is near-constant rather than proportional to the audio, and it
matches the start-up row almost exactly: what the daemon removes is the model
load, not any part of the recognition. The last row is the yardstick -- a
command that touches no model runs in about 25ms either way, so essentially the
whole difference is the model, paid once instead of every time.

<sub>Release build, `vosk-model-small-en-us-0.15`, median of 15 runs on an idle
machine with a warm page cache; the start-up figure is the median of 4 cold
starts (577--724 ms). Measured with `WAYLAND_DISPLAY`/`DISPLAY` unset so
`auto_copy` fails immediately rather than spawning `wl-copy`, whose 2s timeout
otherwise swamps everything here. This is a faster machine than the one used
for the cross-tool comparison under [Performance](#performance), so compare the
two columns here, not the two tables -- and expect your own absolute numbers to
differ. The gap is the part that travels.</sub>

Nothing requires it. Every command tries the daemon's socket
(`$XDG_RUNTIME_DIR/jambi/daemon.sock`) and falls back to loading its own model
when none is listening, so starting or stopping the daemon changes speed and
nothing else. `JAMBI_NO_DAEMON=1` bypasses it for a single run, and
`enabled = false` under `[daemon]` turns it off for good.

For hold-to-talk dictation the difference is more than the clock: without a
daemon the model is being read off disk while you are already talking, and with
one, recognition starts the moment the microphone opens.

**Tray indicator.** While the daemon runs it publishes a StatusNotifierItem --
the same mechanism Vorta, KeePassXC and OpenSnitch use -- so it appears in your
tray with the other background applications. It turns red while a recording is
in progress, which is how a dictation whose key release went missing becomes
visible instead of silently holding the microphone. Its menu shows the warm
model and offers **Quit Jambi daemon**. Set `tray = false` under `[daemon]` to
run without one; on a desktop with no tray at all the daemon just carries on
and logs that it has no indicator.

Appearance is chosen under `[daemon]`:

```toml
[daemon]
tray_icon = "lamp"              # or "microphone"
tray_color = "white"            # or "black" for a light panel
tray_red_when_recording = true  # false to keep one color throughout
```

`tray_icon` picks between a genie lamp -- Jambi being a genie, and the default
-- and the microphone, matching the notification glyph. The lamp is drawn solid
and unadorned, so every pixel goes to the shape itself; being wider than it is
tall, it fills the width of the icon slot and rather less of the height.

`tray_color` is the ink: `white` for a dark panel, `black` for a light one.
Nothing can reliably read your panel's color, so it is a setting rather than
something detected; `"dark"` and `"light"` are accepted as aliases naming the
panel instead of the ink, so `tray_color = "dark"` means the same as
`"white"`.

`tray_red_when_recording` is the color change itself. Leaving it on is worth
it -- the red is the only at-a-glance sign that a dictation whose key release
never arrived is still holding the microphone -- but if you would rather the
tray stayed one color, turning it off does not hide the state: the tooltip and
the tray menu still both say "Recording".

Each has a flag, so you can see the effect before committing to a config edit:

```bash
jambi daemon --icon microphone --color black --no-recording-tint
```

`--no-recording-tint` only turns the tint off; to force it back on, set the
config key.

The glyphs are drawn in code rather than shipped as bitmaps, so they stay crisp
at whatever size your panel asks for instead of being scaled from one image.

**Starting it at login.** For Hyprland, the simplest route is your config:

```
exec-once = jambi daemon
```

Or as a systemd user service, which also restarts it if it ever dies:

```bash
mkdir -p ~/.config/systemd/user
cp packaging/systemd/jambi-daemon.service ~/.config/systemd/user/
# set ExecStart to the output of `command -v jambi`
systemctl --user daemon-reload
systemctl --user enable --now jambi-daemon
```

On Nix the unit ships ready to use, with its path already filled in:

```bash
mkdir -p ~/.config/systemd/user
ln -sf "$(nix build --no-link --print-out-paths github:guttermonk/jambi)/share/systemd/user/jambi-daemon.service" \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jambi-daemon
```

The daemon never types the transcription itself. `jambi dictate stop` does
that, because typing needs the compositor's environment (`WAYLAND_DISPLAY`,
`wtype` on PATH) which the keybind has and a systemd unit may not -- so the
daemon hands over the text and delivery takes exactly the same path it does
without a daemon.

It also does not hold the microphone open between recordings. That would shave
a few more milliseconds off, at the cost of showing jambi as permanently
recording in PipeWire and holding the device against everything else.

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
- Whether to use the background daemon, and its tray indicator's glyph and colors

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

[daemon]                   # see Background Daemon above
enabled = true             # use a running daemon when one is listening
tray = true                # show a tray indicator while it runs
tray_icon = "lamp"         # or "microphone"
tray_color = "white"       # or "black" for a light panel
tray_red_when_recording = true  # false to keep one color throughout
max_recording_secs = 300   # safety cap if a key release is ever missed
```

### Notifications

A dictation cycle reports itself through `notify-send`, replacing its own
popup rather than stacking: listening, then done, or the specific failure
(microphone unavailable, no speech detected, typing failed with the text
left on the clipboard). These matter more than usual in live mode, since it
runs detached from a keybind and nothing else would surface an error.

"Listening..." stays up for as long as you hold the key, rather than timing
out part-way through: a recording has no fixed length, and a popup that
expired early would be saying the dictation had stopped when it had not.
Whatever ends the recording replaces it. A lost key release is the one case
nothing replaces it, so it also expires after `daemon.max_recording_secs` --
the same cap that stops a missed release holding the microphone open.

Notifications carry a microphone glyph, shipped in `assets/` with white
strokes so it reads on a dark panel. If your desktop recolors its icons,
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
│   ├── daemon.rs       # Background daemon that keeps the model warm
│   ├── tray.rs         # Tray indicator for the daemon
│   ├── vosk_engine.rs  # Vosk speech recognition
│   ├── audio.rs        # Audio recording
│   └── config.rs       # Configuration handling
├── target/release/     # Compiled binary (after build)
├── Cargo.toml         # Rust dependencies
├── config.example.toml # Example configuration
└── jambi              # Main launcher script
```


## Performance

Measured against two Whisper-based dictation tools on the same 11.0s clip
(`jfk.wav` from whisper.cpp's test suite), same machine, median of 3 runs.
The machine is the kind Jambi targets: an Intel i5-4250U, 2 cores at 1.3GHz,
integrated graphics, no discrete GPU.

| Tool | Engine | Model | Time | vs realtime | Peak RAM |
|---|---|---|---|---|---|
| **Jambi** | Vosk (Kaldi) | small-en-us, 68MB | **3.4s** | **0.31x** | **175MB** |
| whisp-away | faster-whisper (CTranslate2) | base.en, 141MB | 6.7s | 0.61x | 430MB |
| voxtype | whisper.cpp (ggml) | tiny.en, 75MB | 22.5s | 2.05x | 220MB |
| voxtype | whisper.cpp (ggml) | base.en, 142MB | 53.1s | 4.83x | 332MB |

All four transcribed all 21 words correctly. Jambi was the fastest and the
lightest: 2x faster than the next tool and 6.6x faster than whisper.cpp at a
comparable model size, in a third of whisp-away's memory. Jambi and
whisp-away both finished before the clip would have finished playing;
whisper.cpp did not, at either size. The headroom is what makes hold-to-talk
feel immediate rather than merely quick.

Two honest caveats. **Vosk returns no capitalisation or punctuation** --
Jambi gave `and so my fellow americans ask not what your country...` where
both Whisper tools produced fully punctuated text. If you need prose rather
than words, that is a real cost, and the right reason to pick a Whisper tool
over this one. And the clip is clean studio speech, which flatters every
engine relative to a laptop microphone in a noisy room; treat the ratios as
sound and the absolute numbers as a best case.

Worth noting what the middle two rows isolate: same `base.en` weights, two
runtimes, 6.7s against 53.1s. Roughly 8x of the gap between these tools is
the inference runtime, not the model.

### With and without the daemon

Every row above is a one-shot run, which loads the model before it can
recognise anything. The optional [background daemon](#background-daemon-optional-for-faster-startup)
takes that load out of each invocation and pays it once at login instead,
worth roughly 0.7s per command. The start-up and run-time table is in
that section -- kept there rather than repeated here because it was measured
on a different, faster machine than the comparison above, and the two sets of
absolute numbers should not be read against each other.

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
