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
//!
//! When a daemon is running (see `daemon.rs`) the recording happens there
//! instead, on a model that is already in memory, and `stop` receives the text
//! over the socket rather than signalling a process. Delivery stays here either
//! way: typing needs the compositor's environment, which this process has and
//! a daemon started from a systemd unit may not.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::daemon::{Client, Reply, Request};
use crate::vosk_engine::{VoskConfig, VoskEngine};
use crate::{DaemonConfig, DictateConfig, Mode};

/// Runtime state lives under $XDG_RUNTIME_DIR so it is cleared on logout and is
/// never world-readable; a stale pidfile in /tmp would otherwise outlive the
/// session and make `stop` signal an unrelated process.
pub fn state_dir() -> PathBuf {
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
///
/// Hands the recording to a running daemon when there is one, which is the
/// fast path: the model is already in memory, so recognition begins as soon as
/// the microphone opens instead of racing a disk read against the first word.
/// This process then exits immediately and `stop` collects the text.
pub async fn start(
    vosk_config: VoskConfig,
    dictate_config: DictateConfig,
    daemon_config: DaemonConfig,
    auto_copy: bool,
) -> Result<()> {
    use tokio::signal::unix::{signal, SignalKind};

    if let Some(mut client) = Client::connect(&daemon_config).await {
        match client.request(&Request::Start).await {
            Ok(Reply::Started) => {
                debug!("daemon is recording");
                notify(
                    &dictate_config,
                    "🎤 Listening...",
                    Expiry::while_recording(Some(daemon_config.max_recording_secs)),
                );
                return Ok(());
            }
            Ok(Reply::AlreadyRecording) => {
                warn!("the daemon is already recording");
                notify(&dictate_config, "⚠️ Already recording", Expiry::BRIEF);
                return Ok(());
            }
            Ok(Reply::Error { message }) => {
                error!("daemon refused to start recording: {}", message);
                notify(&dictate_config, "❌ Recording failed - check the microphone", Expiry::BRIEF);
                return Err(anyhow::anyhow!(message));
            }
            Ok(other) => {
                // Falling through to the in-process path would be worse than
                // failing loudly: a daemon answering nonsense is a bug, and
                // recording twice over it would hide that.
                anyhow::bail!("unexpected reply from the daemon: {:?}", other);
            }
            Err(e) => {
                // The daemon was reachable a moment ago and is not now -- it
                // was shut down mid-request, most likely. Recording in-process
                // is exactly the right answer.
                warn!("daemon request failed ({:#}), recording in-process", e);
            }
        }
    }

    if let Some(existing) = running_pid() {
        // Not an error: holding the key sends one press, but key repeat or a
        // second tap while the first recording is still open would otherwise
        // stack recorders onto the same microphone.
        warn!("dictation already running as pid {}", existing);
        notify(&dictate_config, "⚠️ Already recording", Expiry::BRIEF);
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

    notify(
        &dictate_config,
        "🎤 Listening...",
        Expiry::while_recording(None),
    );

    let stop = async move {
        tokio::select! {
            _ = sigusr1.recv() => debug!("stopping on SIGUSR1"),
            _ = sigterm.recv() => debug!("stopping on SIGTERM"),
            _ = sigint.recv() => debug!("stopping on SIGINT"),
        }
    };

    // Callbacks are no-ops here: nothing is watching stdout, and the partial
    // text would be half-recognised anyway. Only the final string is wanted.
    //
    // Failures are notified rather than just returned. This runs detached from
    // a keybind, so stderr goes nowhere the user will ever look: without this
    // a microphone that cannot be opened looks identical to one that heard
    // nothing -- the "Listening..." popup appears and then silence, with no
    // hint that anything broke.
    let result = match engine.transcribe_stream(stop, |_| {}, |_| {}).await {
        Ok(result) => result,
        Err(e) => {
            error!("dictation failed: {:#}", e);
            notify(&dictate_config, "❌ Recording failed - check the microphone", Expiry::BRIEF);
            return Err(e);
        }
    };

    if result.text.is_empty() {
        info!("no speech recognised");
        notify(&dictate_config, "❌ No speech detected", Expiry::BRIEF);
        return Ok(());
    }

    info!("dictated: {}", result.text);
    deliver(&result.text, &dictate_config, auto_copy).await;

    Ok(())
}

/// End the running recording and deliver its text.
///
/// Two shapes of recorder may be out there, and both are checked each time
/// round the poll loop rather than being chosen up front: a daemon can be
/// started or stopped between a press and its release, and committing to one
/// answer at the top would make that window silently drop a dictation.
///
///   in-process -- signal the pidfile's process; it does its own delivery
///   daemon     -- collect the text over the socket and deliver it here
///
/// Delivery stays in this process for the daemon case because typing needs the
/// compositor's environment (WAYLAND_DISPLAY, and `wtype` on PATH). This half
/// is spawned by the keybind so it always has that; a daemon under a systemd
/// unit may not.
///
/// `mode` is taken so that one keybind can serve both modes: the release half
/// fires regardless of which mode the press acted on, and in windowed mode
/// there is never a recording to stop. Without this check every press in
/// windowed mode would poll for `stop_wait_ms` and then pop up a "no recording"
/// notification about a recording the user never started.
pub async fn stop(
    mode: Mode,
    dictate_config: &DictateConfig,
    daemon_config: &DaemonConfig,
    auto_copy: bool,
) -> Result<()> {
    if mode != Mode::Live {
        debug!("mode is {}, nothing to stop", mode);
        return Ok(());
    }

    // The key is up, so stop claiming to be listening -- immediately, before
    // any of the work below. This process is the only one that knows the
    // release happened: the in-process recorder learns it from a signal, and
    // the daemon not until its reply is already being written, by which point
    // the recognising is done. Waiting for either would leave "Listening..."
    // up through the flush, the modifier grace and the typing, which is most
    // of the gap the user actually sees.
    //
    // Emitted before knowing whether anything is recording at all. A `stop`
    // with no press behind it therefore flashes this before correcting itself
    // to "Recording already stopped" -- the rarer case, and the one where
    // being briefly wrong costs least.
    notify(dictate_config, "🧠 Transcribing...", Expiry::WHILE_DELIVERING);

    // The press half has to register itself before this can find it, and on a
    // quick tap the release arrives first. Poll briefly rather than reporting
    // "nothing recording" for what is really a race the user cannot see.
    let deadline = Duration::from_millis(dictate_config.stop_wait_ms);
    let started = std::time::Instant::now();

    // One connection, reused across the polls: a `stop` that finds nothing is
    // retried, and reconnecting every 25ms to ask again would be wasteful.
    let mut client = Client::connect(daemon_config).await;

    loop {
        if let Some(pid) = running_pid() {
            return signal_recorder(pid);
        }

        // Taken rather than borrowed so that a failed request simply drops the
        // connection: the remaining polls then watch the pidfile alone.
        if let Some(mut connected) = client.take() {
            match collect_from_daemon(&mut connected).await {
                Ok(Collected::Text(text)) => {
                    return finish(text, dictate_config, auto_copy).await;
                }
                Ok(Collected::Elsewhere) => return Ok(()),
                Ok(Collected::Failed(message)) => {
                    error!("dictation failed: {}", message);
                    notify(dictate_config, "❌ Recording failed - check the microphone", Expiry::BRIEF);
                    return Err(anyhow::anyhow!(message));
                }
                Ok(Collected::Pending) => client = Some(connected),
                Err(e) => warn!(
                    "daemon request failed ({:#}), watching the pidfile instead",
                    e
                ),
            }
        }

        if started.elapsed() >= deadline {
            // Reached either because no recording was running, or because the
            // one that was died on its own -- a microphone that cannot be
            // opened being the usual reason. "No recording in progress" read
            // as a denial of something the user had just plainly started, so
            // this states the outcome instead of arguing about the premise;
            // the recorder itself notifies the specific failure.
            warn!("no dictation in progress after {:?}", deadline);
            notify(dictate_config, "⏹️ Recording already stopped", Expiry::BRIEF);
            return Ok(());
        }

        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Signal an in-process recorder to finish. Returns as soon as the signal is
/// delivered -- that process does its own typing.
fn signal_recorder(pid: i32) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    kill(Pid::from_raw(pid), Signal::SIGUSR1)
        .with_context(|| format!("failed to signal dictation process {}", pid))?;

    debug!("signalled pid {} to stop", pid);
    Ok(())
}

/// What the daemon had to say about the recording.
enum Collected {
    /// Nothing is recording *yet* -- the race the poll loop exists to absorb.
    Pending,
    /// The recording ended. Empty means it heard no speech.
    Text(String),
    /// A streaming front-end owns the recording and shows the text itself, so
    /// there is nothing here to deliver. Distinct from an empty transcript,
    /// which would otherwise be reported as "no speech detected".
    Elsewhere,
    /// The recording itself failed -- an unavailable microphone, usually.
    /// Reported separately from a transport error because there is no point
    /// retrying it, and the user needs to be told why nothing was typed.
    Failed(String),
}

/// Ask the daemon to end the recording and hand over its text.
async fn collect_from_daemon(client: &mut Client) -> Result<Collected> {
    match client.request(&Request::Stop).await? {
        Reply::Stopped { text: Some(text) } => Ok(Collected::Text(text)),
        Reply::Stopped { text: None } => {
            debug!("the recording belongs to a streaming client");
            Ok(Collected::Elsewhere)
        }
        Reply::NotRecording => Ok(Collected::Pending),
        Reply::Error { message } => Ok(Collected::Failed(message)),
        other => Err(anyhow::anyhow!("unexpected reply: {:?}", other)),
    }
}

/// Report and deliver a transcript collected from the daemon.
async fn finish(text: String, config: &DictateConfig, auto_copy: bool) -> Result<()> {
    if text.is_empty() {
        info!("no speech recognised");
        notify(config, "❌ No speech detected", Expiry::BRIEF);
        return Ok(());
    }

    info!("dictated: {}", text);
    deliver(&text, config, auto_copy).await;
    Ok(())
}

/// Icon shown on notifications: `dictate.icon` if set, otherwise the glyph the
/// package ships via JAMBI_ICON. Returns None rather than a missing path so a
/// stale setting degrades to a plain notification instead of a broken image.
fn icon(config: &DictateConfig) -> Option<String> {
    config
        .icon
        .clone()
        .or_else(|| std::env::var("JAMBI_ICON").ok())
        .filter(|path| {
            let present = std::path::Path::new(path).exists();
            if !present {
                debug!("notification icon {} does not exist, omitting", path);
            }
            present
        })
}

/// How long a popup stays on screen, in milliseconds.
///
/// Zero is "until something replaces it" to a notification daemon that follows
/// the specification.
#[derive(Debug, Clone, Copy)]
struct Expiry(u64);

impl Expiry {
    /// For anything reporting an operation that has already finished.
    const BRIEF: Self = Self(2000);

    /// For "Listening...", which has to last as long as the key is held.
    ///
    /// A recording has no fixed length, so any timeout is a guess -- and the
    /// guess that expires early claims the dictation stopped when it had not,
    /// which is the worst thing this particular popup can say, since the
    /// reason to look at it is to know whether you are still being heard.
    /// Every path out of a recording notifies, and all of them carry the
    /// replacement hint below, so this is superseded rather than left behind.
    ///
    /// `cap` bounds it, so that a lost key release cannot leave the popup
    /// insisting it is listening long after the tray has gone back to idle.
    /// It is `daemon.max_recording_secs` as this process reads it, which is
    /// the daemon's own cap whenever both are reading one config file -- and
    /// only a bound on a popup either way, so a daemon started against some
    /// other config costs nothing worse than a stale notification. The
    /// in-process recorder has no cap at all, running until signalled, and
    /// passes `None`.
    fn while_recording(cap: Option<u64>) -> Self {
        Self(cap.map_or(0, |secs| secs.saturating_mul(1000)))
    }

    /// For "Transcribing...", which covers everything between the key coming
    /// up and the text landing: the recogniser's final flush, the modifier
    /// grace period, and the typing itself -- which at `type_delay_ms` per
    /// character is the long pole for a long dictation.
    ///
    /// Generous rather than fitted. It only has to outlast delivery, and
    /// delivery replaces it the moment it ends; the bound exists so that a
    /// `stop` killed mid-typing cannot leave the popup up for the session.
    const WHILE_DELIVERING: Self = Self(60_000);
}

fn notify(config: &DictateConfig, body: &str, expiry: Expiry) {
    let mut args: Vec<String> = Vec::new();

    if let Some(path) = icon(config) {
        args.push("-i".into());
        args.push(path);
    }

    // Replace the previous popup instead of stacking: a dictation cycle emits
    // several of these in a row and they describe one operation, not many.
    // This is also what retires a `UntilReplaced` popup.
    args.push("-h".into());
    args.push("string:x-canonical-private-synchronous:jambi".into());
    args.push("-t".into());
    args.push(expiry.0.to_string());
    args.push("Jambi".into());
    args.push(body.into());

    let _ = std::process::Command::new("notify-send").args(&args).status();
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
        Ok(()) => notify(config, "✅ Done", Expiry::BRIEF),
        Err(e) => {
            warn!("failed to type text: {}", e);
            if copied {
                notify(config, "📋 Copied to clipboard (typing failed)", Expiry::BRIEF);
            } else {
                notify(config, "❌ Could not deliver text", Expiry::BRIEF);
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
