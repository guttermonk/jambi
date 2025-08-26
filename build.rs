//! Build script for Jambi
//! 
//! This script helps locate and link the Vosk library for speech recognition.

use std::env;
use std::path::PathBuf;

fn main() {
    // Always link Vosk since we're using it as the primary engine
    link_vosk_library();
}

fn link_vosk_library() {
    // Check environment variable first
    if let Ok(vosk_lib_dir) = env::var("VOSK_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", vosk_lib_dir);
        println!("cargo:rustc-link-lib=dylib=vosk");
        
        // Also set rpath for runtime linking
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", vosk_lib_dir);
        return;
    }
    
    // Get home directory properly
    let home = env::var("HOME").unwrap_or_else(|_| "/home/user".to_string());
    
    // Common paths where Vosk library might be installed
    let search_paths = vec![
        // User's cache directory (where our test script downloads it)
        PathBuf::from(&home).join(".cache/jambi/vosk-lib"),
        // System paths
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib/aarch64-linux-gnu"),
        // Homebrew on macOS
        PathBuf::from("/opt/homebrew/lib"),
        PathBuf::from("/usr/local/opt/vosk/lib"),
        // Custom installation paths
        PathBuf::from("/opt/vosk/lib"),
    ];
    
    // Try to find libvosk.so or libvosk.dylib
    for path in &search_paths {
        if path.exists() {
            let vosk_so = path.join("libvosk.so");
            let vosk_dylib = path.join("libvosk.dylib");
            let vosk_dll = path.join("vosk.dll");
            
            if vosk_so.exists() || vosk_dylib.exists() || vosk_dll.exists() {
                println!("cargo:rustc-link-search=native={}", path.display());
                println!("cargo:rustc-link-lib=dylib=vosk");
                println!("cargo:warning=Found Vosk library at: {}", path.display());
                
                // Also set rpath for runtime linking
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
                return;
            }
        }
    }
    
    // If not found, provide instructions
    println!("cargo:warning=Vosk library not found!");
    println!("cargo:warning=Please install Vosk library:");
    println!("cargo:warning=  1. Download from: https://github.com/alphacep/vosk-api/releases");
    println!("cargo:warning=  2. Extract libvosk.so to one of these locations:");
    for path in &search_paths {
        println!("cargo:warning=     - {}", path.display());
    }
    println!("cargo:warning=  Or set VOSK_LIB_DIR environment variable to the directory containing libvosk.so");
    
    // Still try to link, it might be in a standard location
    println!("cargo:rustc-link-lib=dylib=vosk");
}