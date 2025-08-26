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
          ];

          buildInputs = commonBuildInputs;

          # Environment variables for build
          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs}";

          # Wrap binary with runtime dependencies
          postInstall = ''
            wrapProgram $out/bin/jambi \
              --prefix PATH : ${pkgs.lib.makeBinPath (with pkgs; [ sox wl-clipboard xclip ])} \
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
          ]);

          PKG_CONFIG_PATH = "${pkgs.lib.makeSearchPath "lib/pkgconfig" commonBuildInputs}";
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = "1";
          ALSA_PCM_CARD = "default";
          ALSA_PCM_DEVICE = "0";
          
          shellHook = ''
            echo "🎙️  Jambi Development Environment"
            echo "Rust version: $(rustc --version)"
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
