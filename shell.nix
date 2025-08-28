{ pkgs ? import <nixpkgs> {
    overlays = [
      (import (builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz"))
    ];
  }
}:

let
  # Rust toolchain with required components
  rustToolchain = pkgs.rust-bin.stable.latest.default.override {
    extensions = [ "rust-src" "clippy" "rustfmt" ];
    targets = [ "x86_64-unknown-linux-gnu" ];
  };

  # System libraries needed for building
  systemLibs = with pkgs; [
    # System libraries
    bzip2
    stdenv.cc.cc

    # Audio support
    alsa-lib
    alsa-lib.dev
    pulseaudio
    jack2
    
    # GUI libraries (X11 and Wayland)
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    xorg.libXext
    xorg.libXfixes
    wayland
    wayland-protocols
    libxkbcommon
    
    # Graphics and rendering
    mesa
    vulkan-loader
    vulkan-headers
    
    # System integration
    dbus
    dbus.dev
    fontconfig
    fontconfig.dev
    
    # Networking and TLS
    openssl
    openssl.dev
    
    # Audio/video processing
    ffmpeg
    sox
    
    # Clipboard support
    wl-clipboard
    xclip
    
    # Notifications
    libnotify
  ];

  # Development tools
  devTools = with pkgs; [
    # Rust toolchain
    rustToolchain
    
    # Build tools
    pkg-config
    cmake
    gcc
    lld
    
    # Development utilities
    rust-analyzer
    bacon
    cargo-watch
    cargo-edit
    cargo-audit
    cargo-deny
    cargo-nextest
    
    # Debugging and profiling
    gdb
    valgrind
    heaptrack
    
    # Documentation
    mdbook
    
    # Version control
    git
    
    # Text editors (optional)
    # vscode-with-extensions
    # vim
    # emacs
  ];

  # Runtime dependencies
  runtimeDeps = with pkgs; [
    sox
    wl-clipboard
    xclip
    libnotify
    pulseaudio
  ];

in pkgs.mkShell {
  buildInputs = systemLibs ++ devTools;

  # Environment variables for development
  shellHook = ''
    export PKG_CONFIG_PATH="${pkgs.lib.makeSearchPath "lib/pkgconfig" systemLibs}:$PKG_CONFIG_PATH"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath systemLibs}:$LD_LIBRARY_PATH"
    export PATH="${pkgs.lib.makeBinPath runtimeDeps}:$PATH"
    
    # Rust environment
    export RUST_SRC_PATH="${rustToolchain}/lib/rustlib/src/rust/library"
    export RUST_BACKTRACE=1
    export CARGO_TARGET_DIR="target"
    
    # Audio configuration
    export ALSA_PCM_CARD=default
    export ALSA_PCM_DEVICE=0
    
    # GUI configuration
    export XDG_DATA_DIRS="${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}:${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:$XDG_DATA_DIRS"
    
    # Development environment info
    echo "🎙️  Jambi Development Environment"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Rust version: $(rustc --version 2>/dev/null || echo 'Not found')"
    echo "Cargo version: $(cargo --version 2>/dev/null || echo 'Not found')"
    echo ""
    echo "📦 Available commands:"
    echo "  cargo run                    # Run main binary (auto-detects GUI/CLI)"
    echo "  cargo run --bin jambi-gui    # Run GUI version"
    echo "  cargo run --bin jambi-cli    # Run CLI version"
    echo "  cargo build                  - Build the project"
    echo "  cargo build --release        - Build optimized release"
    echo "  cargo test                   - Run tests"
    echo "  cargo clippy                 - Run linter"
    echo "  cargo fmt                    - Format code"
    echo "  bacon                        - Continuous testing"
    echo "  cargo nextest run            - Fast test runner"
    echo ""
    echo "🔧 Development tools:"
    echo "  rust-analyzer               - LSP server (for editors)"
    echo "  cargo-watch                 - Watch for changes"
    echo "  cargo-edit                  - Edit Cargo.toml"
    echo "  cargo-audit                 - Security audit"
    echo "  gdb                         - Debugger"
    echo ""
    
    # Check system capabilities
    echo "🔍 System check:"
    
    # Audio recording check
    if command -v sox >/dev/null 2>&1; then
      echo "  Audio recording: Available (sox)"
      sox --version 2>/dev/null | head -1 | sed 's/^/    /'
    else
      echo "  ⚠️  Audio recording: Not available (sox not found)"
      echo "     Install sox for audio recording: sudo apt install sox"
    fi
    
    # Graphics check
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
      echo "  ✅ Display: Available"
    else
      echo "  ⚠️  Display: No X11/Wayland display detected"
    fi
    
    # CUDA check
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "  🚀 CUDA GPU detected:"
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^/    /' || echo "    Failed to query GPU"
      echo "    To enable CUDA support: cargo build --features cuda"
    else
      echo "  💻 CUDA: Not available"
    fi
    
    echo ""
    echo "🚀 Ready to develop! Run 'cargo run' to start Jambi"
    echo ""
  '';

  # Additional environment variables
  RUSTFLAGS = "-C link-arg=-fuse-ld=lld"; # Use LLD linker for faster linking
  RUST_LOG = "info"; # Default log level
  
  # Cargo configuration
  CARGO_BUILD_INCREMENTAL = "true";
  CARGO_BUILD_PIPELINING = "true";
  
  # OpenSSL configuration (for HTTPS downloads)
  OPENSSL_DIR = "${pkgs.openssl.dev}";
  OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
  
  # pkg-config paths
  PKG_CONFIG_PATH = pkgs.lib.makeSearchPath "lib/pkgconfig" systemLibs;
}