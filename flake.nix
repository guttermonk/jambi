{
  description = "Jambi - a blazing-fast voice transcription application built with Rust";

  inputs = {
    # No rust-overlay: the toolchain comes from nixpkgs so that a consumer can
    # set `inputs.jambi.inputs.nixpkgs.follows` and have it actually work.
    #
    # That matters for more than tidiness. jambi captures audio through ALSA,
    # and on a PipeWire system ALSA reaches the server by dlopen'ing
    # libasound_module_pcm_pipewire.so, whose absolute path is baked into
    # /etc/alsa/conf.d by the host. That plugin is built against the host's
    # alsa-lib, so if jambi's alsa-lib comes from a different nixpkgs the
    # dlopen fails and the default capture device cannot be opened at all --
    # ALSA reports only "cannot be opened or _snd_pcm_pipewire_open was not
    # defined inside". Following the host nixpkgs keeps one alsa-lib in play
    # and avoids the whole class of problem.
    #
    # rust-toolchain.toml asks for "stable" with no version pin, so nixpkgs'
    # rustc satisfies it; that file is now only consulted by rustup-based
    # (non-Nix) setups.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # Vosk library - fetch as a fixed-output derivation
        voskVersion = "0.3.45";
        voskLibrary = pkgs.fetchzip {
          url = "https://github.com/alphacep/vosk-api/releases/download/v${voskVersion}/vosk-linux-x86_64-${voskVersion}.zip";
          sha256 = "sha256-ToMDbD5ooFMHU0nNlfpLynF29kkfMknBluKO5PipLFY=";  # You'll need to update this
          stripRoot = true;
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
          stdenv.cc.cc
          bzip2
          
          # Runtime dependencies
          sox
          wl-clipboard
          xclip
          libnotify
          # Keystroke synthesis for live dictation: wtype on Wayland, xdotool on
          # X11. Only one is used per session; which one is decided at runtime
          # from WAYLAND_DISPLAY/DISPLAY.
          wtype
          xdotool
        ];

        # Build jambi
        jambi = pkgs.rustPlatform.buildRustPackage {
          pname = "jambi";
          version = "0.1.0";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          # No toolchain listed: buildRustPackage brings nixpkgs' cargo/rustc
          # in itself, and adding a second one here would shadow it.
          nativeBuildInputs = with pkgs; [
            pkg-config
            makeWrapper
          ];

          buildInputs = commonBuildInputs;

          # Set up vosk library before build
          #
          # libstdc++ sits alongside libvosk on LD_LIBRARY_PATH for the same
          # reason the dev shell lists it: libvosk.so links against it, and
          # nothing puts it on the binary's RPATH -- it is a transitive
          # dependency of libvosk rather than something the crate links
          # directly, so the cc wrapper has no -L to turn into an rpath entry.
          # The installed binary gets it from wrapProgram below; the test
          # binary checkPhase runs is unwrapped and would die on startup with
          # "libstdc++.so.6: cannot open shared object file" (exit 127).
          # These exports carry into checkPhase because the phases share one
          # shell.
          preBuild = ''
            export VOSK_LIB_DIR="${voskLibrary}"
            export RUSTFLAGS="-L ${voskLibrary}"
            export LD_LIBRARY_PATH="${voskLibrary}:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
          '';

          # Environment variables for build
          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs}";

          # Wrap binary with runtime dependencies and vosk library
          postInstall = ''
            # Copy vosk library to output
            mkdir -p $out/lib
            cp $VOSK_LIB_DIR/libvosk.so $out/lib/

            # Notification glyph. Shipped untinted (white strokes) so it reads
            # on a dark panel anywhere; JAMBI_ICON is only the default, and
            # `dictate.icon` in config.toml overrides it -- which is how a
            # themed desktop points at its own recoloured copy instead.
            mkdir -p $out/share/jambi
            cp ${./assets/microphone.svg} $out/share/jambi/microphone.svg

            # systemd user unit for the daemon, with ExecStart pointed at this
            # store path so it needs no editing. Shipped but not activated --
            # see the Daemon section of the README for enabling it.
            mkdir -p $out/share/systemd/user
            substitute ${./packaging/systemd/jambi-daemon.service} \
              $out/share/systemd/user/jambi-daemon.service \
              --replace-fail "/usr/local/bin/jambi" "$out/bin/jambi"

            wrapProgram $out/bin/jambi \
              --prefix PATH : ${pkgs.lib.makeBinPath (with pkgs; [ sox wl-clipboard xclip wtype xdotool libnotify ])} \
              --prefix LD_LIBRARY_PATH : "$out/lib:${pkgs.stdenv.cc.cc.lib}/lib" \
              --set-default JAMBI_ICON "$out/share/jambi/microphone.svg" \
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
            rustc
            cargo
            clippy
            rustfmt
            rust-analyzer
            bacon
            cargo-watch
            gdb
            wget
            unzip
          ]);

          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs}";
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
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
            # libstdc++ alongside libvosk: libvosk.so links against it, and
            # without this `cargo run`/`cargo test` in the shell die with
            # "libstdc++.so.6: cannot open shared object file". The packaged
            # binary gets this from wrapProgram; the dev shell has to say it
            # itself. Taken from this nixpkgs so it matches the glibc the
            # toolchain here links against.
            export LD_LIBRARY_PATH="$VOSK_DEV_DIR:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

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
