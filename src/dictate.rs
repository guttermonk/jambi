//! Live dictation: hold-to-talk transcription driven by a compositor keybind.
//!
//! The workflow mirrors whisp-away -- press the key to start, release it to
//! stop, and the text lands at the cursor -- but the mechanics differ in a way
//! that matters. Vosk recognises *while* you speak, so by the time the key
//! comes up the text already exists and the release is near-instant. A
//! Whisper-based tool has to run its encoder over the whole clip after the
//! fact, which is the multi-second pause this mode exists to avoid.
//!
//! Two processes are involved, because a keybind cannot hold state:
//!
//!   press   -> `jambi dictate start`, which records until signalled
//!   release -> `jambi dictate stop`, which signals it and exits immediately
//!
//! They find each other through a pidfile under $XDG_RUNTIME_DIR.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::vosk_engine::{VoskConfig, VoskEngine};
use crate::{DictateConfig, Mode};

/// Runtime state lives under $XDG_RUNTIME_DIR so it is cleared on logout and is
/// never world-readable; a stale pidfile in /tmp would otherwise outlive the
/// session and make `stop` signal an unrelated process.
fn state_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("jambi")
}

fn pid_path() -> PathBuf {
    state_dir().join("dictate.pid")
}

/// Removes the pidfile however `start` returns -- including on the error paths,
/// which would otherwise leave a pidfile pointing at a dead process and make
/// every later press report "already recording".
struct PidGuard(PathBuf);

impl Drop for PidGuard {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_file(&self.0) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("failed to remove pidfile {}: {}", self.0.display(), e);
            }
        }
    }
}

/// Read the pidfile and return the pid only if that process is still alive.
///
/// The liveness check is what makes a crashed or killed recorder self-healing:
/// signal 0 performs the permission and existence checks without delivering
/// anything, so a stale pidfile reads as "nothing running" rather than wedging
/// the mode until the file is deleted by hand.
fn running_pid() -> Option<i32> {
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    let raw = std::fs::read_to_string(pid_path()).ok()?;
    let pid: i32 = raw.trim().parse().ok()?;

    match kill(Pid::from_raw(pid), None) {
        Ok(()) => Some(pid),
        Err(_) => {
            debug!("ignoring stale pidfile for pid {}", pid);
            None
        }
    }
}

/// Begin recording and block until signalled, then deliver the text.
///
/// Runs in the foreground; the compositor's `exec` is what detaches it.
pub async fn start(
    vosk_config: VoskConfig,
    dictate_config: DictateConfig,
    auto_copy: bool,
) -> Result<()> {
    use tokio::signal::unix::{signal, SignalKind};

    if let Some(existing) = running_pid() {
        // Not an error: holding the key sends one press, but key repeat or a
        // second tap while the first recording is still open would otherwise
        // stack recorders onto the same microphone.
        warn!("dictation already running as pid {}", existing);
        notify("⚠️ Already recording");
        return Ok(());
    }

    let dir = state_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create state directory {}", dir.display()))?;

    let pid_file = pid_path();
    std::fs::write(&pid_file, std::process::id().to_string())
        .with_context(|| format!("failed to write pidfile {}", pid_file.display()))?;
    let _guard = PidGuard(pid_file);

    // Signals are registered before the model loads. A key held only briefly
    // means `stop` can fire while the model is still loading, and a handler
    // installed after that point would miss the signal entirely -- leaving a
    // recorder running with no key left to stop it.
    let mut sigusr1 = signal(SignalKind::user_defined1())?;
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    // Deliberately not calling load_model() here. transcribe_stream opens the
    // microphone before loading the model so the ~800ms load overlaps with the
    // user already talking; pre-loading it here would serialise the two again
    // and clip the first word.
    let mut engine = VoskEngine::new(vosk_config)?;

    notify("🎤 Listening...");

    let stop = async move {
        tokio::select! {
            _ = sigusr1.recv() => debug!("stopping on SIGUSR1"),
            _ = sigterm.recv() => debug!("stopping on SIGTERM"),
            _ = sigint.recv() => debug!("stopping on SIGINT"),
        }
    };

    // Callbacks are no-ops here: nothing is watching stdout, and the partial
    // text would be half-recognised anyway. Only the final string is wanted.
    let result = engine.transcribe_stream(stop, |_| {}, |_| {}).await?;

    if result.text.is_empty() {
        info!("no speech recognised");
        notify("❌ No speech detected");
        return Ok(());
    }

    info!("dictated: {}", result.text);
    deliver(&result.text, &dictate_config, auto_copy).await;

    Ok(())
}

/// Signal the running recorder to finish. Returns as soon as the signal is
/// delivered -- the recorder does the typing.
///
/// `mode` is taken so that one keybind can serve both modes: the release half
/// fires regardless of which mode the press acted on, and in windowed mode
/// there is never a recording to stop. Without this check every press in
/// windowed mode would poll for `stop_wait_ms` and then pop up a "no recording"
/// notification about a recording the user never started.
pub async fn stop(mode: Mode, dictate_config: &DictateConfig) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    if mode != Mode::Live {
        debug!("mode is {}, nothing to stop", mode);
        return Ok(());
    }

    // The press half has to create the pidfile before this can find it, and on
    // a quick tap the release arrives first. Poll briefly rather than reporting
    // "nothing recording" for what is really a race the user cannot see.
    let deadline = Duration::from_millis(dictate_config.stop_wait_ms);
    let started = std::time::Instant::now();
    let pid = loop {
        if let Some(pid) = running_pid() {
            break pid;
        }
        if started.elapsed() >= deadline {
            warn!("no dictation process found after {:?}", deadline);
            notify("❌ No recording in progress");
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };

    kill(Pid::from_raw(pid), Signal::SIGUSR1)
        .with_context(|| format!("failed to signal dictation process {}", pid))?;

    debug!("signalled pid {} to stop", pid);
    Ok(())
}

/// Put the text where the user is typing, falling back to the clipboard.
fn notify(body: &str) {
    // Replace the previous popup instead of stacking: a dictation cycle emits
    // several of these in a row and they describe one operation, not many.
    let _ = std::process::Command::new("notify-send")
        .args([
            "-h",
            "string:x-canonical-private-synchronous:jambi",
            "-t",
            "2000",
            "Jambi",
            body,
        ])
        .status();
}

/// Type the text at the cursor, falling back to the clipboard if no typing tool
/// is available or the typing attempt fails.
async fn deliver(text: &str, config: &DictateConfig, auto_copy: bool) {
    // Copy first and unconditionally when auto_copy is on, so the text survives
    // even if typing fails below -- a transcription that reached neither the
    // cursor nor the clipboard is lost for good.
    let copied = if auto_copy {
        match crate::copy_to_clipboard(text).await {
            Ok(()) => true,
            Err(e) => {
                warn!("failed to copy to clipboard: {}", e);
                false
            }
        }
    } else {
        false
    };

    // The keybind's modifiers are very likely still held at this instant: the
    // release of D is what triggered `stop`, but Super and Shift come up
    // whenever the user's hand gets round to it. Typing into a held Super turns
    // every character into a compositor shortcut, so wait for the modifiers to
    // clear. A fixed grace period is deliberately chosen over reading evdev --
    // it needs no input-group membership and no device access.
    if config.modifier_grace_ms > 0 {
        tokio::time::sleep(Duration::from_millis(config.modifier_grace_ms)).await;
    }

    match type_text(text, config).await {
        Ok(()) => notify("✅ Done"),
        Err(e) => {
            warn!("failed to type text: {}", e);
            if copied {
                notify("📋 Copied to clipboard (typing failed)");
            } else {
                notify("❌ Could not deliver text");
            }
        }
    }
}

/// Synthesise the keystrokes, picking the tool that matches the session type.
async fn type_text(text: &str, config: &DictateConfig) -> Result<()> {
    use tokio::process::Command;

    let delay = config.type_delay_ms.to_string();

    // Wayland first: on a Wayland session an inherited DISPLAY usually points at
    // Xwayland, where xdotool would type into the wrong place or nowhere.
    let (program, args) = if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        ("wtype", vec!["-d".to_string(), delay, text.to_string()])
    } else if std::env::var_os("DISPLAY").is_some() {
        (
            "xdotool",
            vec![
                "type".to_string(),
                "--delay".to_string(),
                delay,
                text.to_string(),
            ],
        )
    } else {
        anyhow::bail!("no Wayland or X11 session detected");
    };

    let status = Command::new(program)
        .args(&args)
        .status()
        .await
        .with_context(|| format!("failed to run {} - is it installed?", program))?;

    if !status.success() {
        anyhow::bail!("{} exited with {}", program, status);
    }

    Ok(())
}
