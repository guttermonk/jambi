{
  description = "Jambi - a blazing-fast voice transcription application built with Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Rust toolchain
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "clippy" "rustfmt" ];
        };

        # Build inputs
        commonBuildInputs = with pkgs; [
          # Audio libraries
          alsa-lib
          pulseaudio
          jack2
          
          # System libraries
          openssl
          pkg-config
          
          # Runtime dependencies
          sox
          wl-clipboard
          xclip
          libnotify
        ];

        # Build jambi
        jambi = pkgs.rustPlatform.buildRustPackage {
          pname = "jambi";
          version = "0.1.0";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            rustToolchain
            pkg-config
            makeWrapper
            wget
            unzip
          ];

          buildInputs = commonBuildInputs;

          # Download and set up vosk library before build
          preBuild = ''
            export VOSK_VERSION="0.3.45"
            export VOSK_DIR="$TMPDIR/vosk"
            mkdir -p "$VOSK_DIR"
            
            echo "Downloading vosk library..."
            ${pkgs.wget}/bin/wget -q "https://github.com/alphacep/vosk-api/releases/download/v$VOSK_VERSION/vosk-linux-x86_64-$VOSK_VERSION.zip" -O "$VOSK_DIR/vosk.zip"
            ${pkgs.unzip}/bin/unzip -q "$VOSK_DIR/vosk.zip" -d "$VOSK_DIR"
            
            export VOSK_LIB_DIR="$VOSK_DIR/vosk-linux-x86_64-$VOSK_VERSION"
            export RUSTFLAGS="-L $VOSK_LIB_DIR"
            export LD_LIBRARY_PATH="$VOSK_LIB_DIR:$LD_LIBRARY_PATH"
          '';

          # Environment variables for build
          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs}";

          # Wrap binary with runtime dependencies and vosk library
          postInstall = ''
            # Copy vosk library to output
            mkdir -p $out/lib
            cp $VOSK_LIB_DIR/libvosk.so $out/lib/ || true
            
            wrapProgram $out/bin/jambi \
              --prefix PATH : ${pkgs.lib.makeBinPath (with pkgs; [ sox wl-clipboard xclip ])} \
              --prefix LD_LIBRARY_PATH : $out/lib \
              --set ALSA_PCM_CARD default \
              --set ALSA_PCM_DEVICE 0
          '';

          meta = with pkgs.lib; {
            description = "Fast Voice Transcription with Vosk";
            homepage = "https://github.com/guttermonk/jambi";
            license = with licenses; [ mit asl20 ];
            maintainers = [ ];
            platforms = platforms.linux;
          };
        };

      in
      {
        # Default package
        packages.default = jambi;
        packages.jambi = jambi;

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = commonBuildInputs ++ (with pkgs; [
            rustToolchain
            rust-analyzer
            bacon
            cargo-watch
            gdb
            wget
            unzip
          ]);

          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs}";
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = "1";
          ALSA_PCM_CARD = "default";
          ALSA_PCM_DEVICE = "0";
          
          # Setup vosk for development
          shellHook = ''
            echo "🎙️  Jambi Development Environment"
            echo "Rust version: $(rustc --version)"
            
            # Download vosk for development if needed
            VOSK_VERSION="0.3.45"
            VOSK_DEV_DIR="$HOME/.cache/vosk"
            if [ ! -f "$VOSK_DEV_DIR/libvosk.so" ]; then
              echo "Setting up Vosk library for development..."
              mkdir -p "$VOSK_DEV_DIR"
              wget -q "https://github.com/alphacep/vosk-api/releases/download/v$VOSK_VERSION/vosk-linux-x86_64-$VOSK_VERSION.zip" -O "/tmp/vosk.zip"
              unzip -q "/tmp/vosk.zip" -d "/tmp"
              cp "/tmp/vosk-linux-x86_64-$VOSK_VERSION/libvosk.so" "$VOSK_DEV_DIR/"
              rm -rf "/tmp/vosk.zip" "/tmp/vosk-linux-x86_64-$VOSK_VERSION"
              echo "Vosk library installed to $VOSK_DEV_DIR"
            fi
            
            export RUSTFLAGS="-L $VOSK_DEV_DIR"
            export LD_LIBRARY_PATH="$VOSK_DEV_DIR:$LD_LIBRARY_PATH"
            
            echo "Run 'cargo run' to start jambi"
          '';
        };

        # App for `nix run`
        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/jambi";
        };

        # Formatter
        formatter = pkgs.nixpkgs-fmt;
      }
    );
}
