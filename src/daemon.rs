//! A background daemon that keeps the Vosk model warm.
//!
//! Every standalone jambi invocation pays `Model::new` -- ~800ms for the small
//! English model, and far more for the large one. That cost is unavoidable in a
//! one-shot process, and it is exactly the cost the push-to-talk path cannot
//! afford: `transcribe_stream` opens the microphone before loading so the load
//! overlaps with the user already talking, which rescues the first word but
//! still means the model is being read off disk while they speak.
//!
//! Running one long-lived process sidesteps it. The daemon loads the model at
//! startup and holds it, and each front-end asks the daemon to do the
//! recognition instead of loading a model of its own:
//!
//!   jambi daemon            -- loads the model, then serves requests
//!   jambi dictate start     -- asks the daemon to begin recording
//!   jambi dictate stop      -- asks for the text, then types it
//!   jambi / jambi record    -- stream or file recognition over the socket
//!
//! Nothing requires it. Every caller tries the socket, and falls back to its
//! own in-process engine when no daemon answers, so the daemon is an
//! optimisation a user opts into at login and never a dependency.
//!
//! # Off, without being gone
//!
//! A left click on the tray icon turns the daemon off: it releases the model
//! and answers `NotServing` -- "do it yourself" -- so every front-end goes
//! back to the in-process path above, while the process and its icon stay for
//! the click that turns it on again. See `toggle_daemon`.
//!
//! # Why the text comes back to the client
//!
//! The daemon records, but it deliberately does *not* type the result. Typing
//! needs WAYLAND_DISPLAY/DISPLAY (and `wtype`/`xdotool` on PATH), and a daemon
//! started from a systemd unit may have neither. `jambi dictate stop` is
//! spawned by the compositor keybind and therefore always has the right
//! environment, so the daemon hands it the string and it does the delivery --
//! the same code path, with the same environment, as when no daemon is running.
//!
//! # Why the microphone is not held open
//!
//! Opening the capture device costs tens of milliseconds, so the daemon could
//! shave that too by keeping a stream open permanently. It does not: that would
//! show jambi as continuously recording in PipeWire's indicators and hold the
//! device against other applications, which is too much to pay for a few
//! milliseconds.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot, Mutex, Notify};
use tracing::{debug, error, info, warn};
use vosk::Model;

use crate::tray::{Activity, Tray, TrayState, TrayStyle};
use crate::vosk_engine::{VoskConfig, VoskEngine, VoskModel, VoskResult};
use crate::DaemonConfig;

/// How long `stop` waits for the recording thread to hand over its text. The
/// thread only has to drain the capture backlog and flush the last utterance,
/// so reaching this means something is genuinely stuck.
const RESULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Socket the daemon listens on.
///
/// Shares the per-session state directory with the dictation pidfile: under
/// $XDG_RUNTIME_DIR it is cleared on logout and is not reachable by other
/// users, so a socket that accepts "record from the microphone" is not exposed
/// beyond the session that created it.
pub fn socket_path() -> PathBuf {
    if let Some(explicit) = std::env::var_os("JAMBI_DAEMON_SOCKET") {
        return PathBuf::from(explicit);
    }
    crate::dictate::state_dir().join("daemon.sock")
}

/// One request, one line of JSON.
///
/// A line-delimited protocol rather than a framed one because every message is
/// small and the only variable-length field is transcribed text, which JSON
/// escapes for us -- a transcript containing a newline would otherwise split
/// into two messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
pub enum Request {
    /// Liveness check, used to tell a running daemon from a stale socket.
    Ping,
    /// Report the warm model and whether a recording is in progress.
    Status,
    /// Begin a recording that outlives this connection; `Stop` collects it.
    Start,
    /// End the current recording and return its text.
    Stop,
    /// Begin a recording bound to this connection, with partial results
    /// streamed back as they are recognised.
    Stream,
    /// Recognise an existing audio file using the warm model.
    TranscribeFile { path: PathBuf },
    /// Swap the warm model, loading the new one before replying.
    SetModel { model: VoskModel },
    /// Ask the daemon to exit.
    Shutdown,
}

/// One reply, or -- during a `Stream` -- one event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum Reply {
    Ok,
    Error {
        message: String,
    },
    Status {
        model: String,
        recording: bool,
        uptime_secs: u64,
        pid: u32,
        /// False when the daemon has been clicked off in the tray: it is still
        /// here, but holding no model and taking no work. `model` then names
        /// what the next click will load rather than what is in memory.
        serving: bool,
    },
    Started,
    /// The daemon has been turned off from its tray icon: it holds no model
    /// and is taking no work, so the front-end should do the job in-process
    /// exactly as it would with no daemon running at all.
    ///
    /// Deliberately not an `Error`. Nothing failed, and a front-end that
    /// treated this as a failure would refuse to dictate while the daemon was
    /// off -- turning a daemon that is merely parked into a jambi that is
    /// broken.
    NotServing,
    /// A recording was already in progress, so this request did nothing.
    AlreadyRecording,
    /// Nothing was recording, so there was nothing to stop.
    NotRecording,
    /// `text` is `None` when a streaming client owns the recording and will
    /// receive the transcript on its own connection. Distinct from an empty
    /// string, which means "recorded, but no speech was recognised".
    Stopped {
        text: Option<String>,
    },
    Streaming,
    /// Revised guess for the utterance being spoken; redraw, do not append.
    Partial {
        text: String,
    },
    /// A finalised utterance; append-only.
    Segment {
        text: String,
    },
    /// Everything recognised during a `Stream`, sent once as it ends.
    Final {
        text: String,
    },
    /// Carries the whole `VoskResult`, not just its text: `jambi transcribe
    /// --format json` reports word timings and confidence, and sending only
    /// the string would quietly empty those fields whenever a daemon was
    /// running.
    Transcribed {
        result: VoskResult,
    },
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// The warm model and the config it was loaded from, swapped together by
/// `SetModel` so the two can never disagree.
///
/// `model` is `None` while the daemon is turned off from the tray. The config
/// stays: it is what names the model in the status and the tooltip, and what
/// the next click loads.
struct Loaded {
    config: VoskConfig,
    model: Option<Arc<Model>>,
}

/// Handles on a recording in flight.
///
/// Both halves are shared rather than owned because two different clients may
/// need them: a streaming client awaits its own transcript while any later
/// `Stop` must still be able to end the recording.
#[derive(Clone)]
struct Recording {
    /// Signalled to end the recording. `Notify` keeps a permit when nobody is
    /// waiting yet, so a stop that arrives while the microphone is still
    /// opening is not lost -- which is the common case on a quick key tap.
    stop: Arc<Notify>,
    /// Set by the recording thread as it returns. This is what tells "still
    /// recording" apart from "died on its own", so a microphone that cannot be
    /// opened does not leave a phantom session wedging every later press.
    finished: Arc<AtomicBool>,
    /// Signalled once, after `finished` is set. The flag answers "is it still
    /// going?" for a request that happens to ask; this wakes a watcher for the
    /// case nobody asks -- a recording that hit the duration cap, say, whose
    /// tray icon would otherwise stay red until the next key press.
    done: Arc<Notify>,
}

impl Recording {
    fn is_running(&self) -> bool {
        !self.finished.load(Ordering::Acquire)
    }
}

struct Session {
    recording: Recording,
    /// `None` when a streaming client is awaiting the transcript itself.
    result: Option<oneshot::Receiver<Result<String>>>,
}

struct State {
    loaded: Mutex<Loaded>,
    session: Mutex<Option<Session>>,
    max_recording: Duration,
    started: Instant,
    /// Shared with the tray's "Quit" item so both routes to exit are the one
    /// that runs the socket cleanup.
    shutdown: Arc<Notify>,
    tray: Tray,
}

impl State {
    /// Is a recording in progress?
    ///
    /// A session that is present but finished reads as `false`: that is a
    /// recording which died on its own and whose text nobody collected.
    async fn is_recording(&self) -> bool {
        self.session
            .lock()
            .await
            .as_ref()
            .is_some_and(|session| session.recording.is_running())
    }

    /// Is the daemon doing a daemon's job -- holding a model and taking work?
    ///
    /// False only between a click that turned it off and the one that turns it
    /// back on.
    async fn is_serving(&self) -> bool {
        self.loaded.lock().await.model.is_some()
    }

    /// The warm model and the config it came from, or `None` while the daemon
    /// is turned off.
    ///
    /// Clones both so the caller does not hold the lock across a recognition
    /// that may run for minutes.
    async fn warm(&self) -> Option<(VoskConfig, Arc<Model>)> {
        let loaded = self.loaded.lock().await;
        loaded
            .model
            .clone()
            .map(|model| (loaded.config.clone(), model))
    }

    /// Push the current state to the tray indicator.
    ///
    /// Called after anything that changes it, so the icon answers "is this
    /// thing on, and is it recording right now?" honestly rather than going
    /// stale.
    ///
    /// The two locks are taken one at a time, never nested. The recording
    /// paths below hold `session` while they read `loaded`, so anything that
    /// took them the other way round would be a deadlock waiting for a model
    /// switch to coincide with a key press.
    async fn refresh_tray(&self) {
        let recording = self.is_recording().await;
        let loaded = self.loaded.lock().await;
        let activity = match (&loaded.model, recording) {
            (None, _) => Activity::Stopped,
            (Some(_), true) => Activity::Recording,
            (Some(_), false) => Activity::Idle,
        };
        let model = loaded.config.model.to_string();
        drop(loaded);

        self.tray.set(TrayState { model, activity }).await;
    }

    /// Show one state outright, for the one that cannot be derived: a model
    /// being read off disk is "on its way up", which looks exactly like "off"
    /// in the fields above until the load lands.
    async fn show_tray(&self, activity: Activity) {
        let model = self.loaded.lock().await.config.model.to_string();
        self.tray.set(TrayState { model, activity }).await;
    }
}

/// Removes the socket on the way out, including on the error paths. A leftover
/// socket file makes the next `bind` fail with EADDRINUSE even though nothing
/// is listening, so the daemon would refuse to start until it was deleted by
/// hand.
struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_file(&self.0) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("failed to remove socket {}: {}", self.0.display(), e);
            }
        }
    }
}

/// Load the model, then serve requests until told to exit.
///
/// Runs in the foreground: whatever started it -- a systemd user unit, the
/// compositor's autostart -- owns the backgrounding.
pub async fn run(vosk_config: VoskConfig, daemon_config: DaemonConfig) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    use tokio::signal::unix::{signal, SignalKind};

    let path = socket_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create state directory {}", dir.display()))?;
        // Belt and braces for the case where XDG_RUNTIME_DIR is unset and the
        // state directory lands in a world-writable /tmp.
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }

    if path.exists() {
        if ping(&path).await {
            anyhow::bail!(
                "a jambi daemon is already listening on {}",
                path.display()
            );
        }
        warn!("removing stale socket {}", path.display());
        std::fs::remove_file(&path)
            .with_context(|| format!("failed to remove stale socket {}", path.display()))?;
    }

    // Bound before the model is loaded, and the order is the whole point.
    //
    // The socket is how a front-end decides whether a daemon exists at all. If
    // it appeared only once the model was in memory, then every press during
    // that second-or-so would find nothing listening, fall back to its own
    // in-process engine, and load a *second* copy of the same model alongside
    // the one the daemon was already loading -- two 40MB loads competing for
    // the same couple of cores, each making the other slower. That is a
    // pathology, not a fallback, and it is exactly what someone sees when they
    // start the daemon and immediately try it.
    //
    // Binding first closes the window. A press that lands mid-load sits in the
    // listen backlog and is served the moment the accept loop starts, which
    // costs it the remainder of the load but never a redundant one. If loading
    // fails the guard below removes the socket and those clients are
    // disconnected, which is the signal to fall back in-process -- so the
    // escape hatch survives for the case that actually warrants it.
    let listener = UnixListener::bind(&path)
        .with_context(|| format!("failed to bind {}", path.display()))?;
    let _guard = SocketGuard(path.clone());
    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));

    info!("loading model {}...", vosk_config.model);
    let load_started = Instant::now();
    let model = {
        let mut engine = VoskEngine::new(vosk_config.clone())?;
        engine
            .load_model()
            .await
            .context("failed to load the Vosk model")?;
        // Scoped so the engine is dropped here, which is load-bearing rather
        // than tidiness: `loaded_model` hands back a *clone* of the `Arc`, so
        // an engine kept alive for the daemon's lifetime would hold a second
        // reference to the model -- and the tray's off-click, dropping only
        // the daemon's own, would free nothing at all. That is exactly what it
        // did: RSS unmoved across a release, and up by another model's worth
        // on the click back on.
        engine
            .loaded_model()
            .ok_or_else(|| anyhow::anyhow!("model reported loaded but is not present"))?
    };
    info!(
        "model {} ready in {}ms",
        vosk_config.model,
        load_started.elapsed().as_millis()
    );

    let shutdown = Arc::new(Notify::new());
    let toggle = Arc::new(Notify::new());

    // The indicator is what makes the daemon visible among the other
    // background applications, and the one place a recording still running
    // past its release shows up.
    let tray = if daemon_config.tray {
        Tray::spawn(
            TrayStyle {
                icon: daemon_config.tray_icon,
                color: daemon_config.tray_color,
                red_when_recording: daemon_config.tray_red_when_recording,
            },
            TrayState {
                model: vosk_config.model.to_string(),
                activity: Activity::Idle,
            },
            Arc::clone(&shutdown),
            Arc::clone(&toggle),
        )
        .await
    } else {
        debug!("tray disabled in config");
        Tray::disabled()
    };

    let state = Arc::new(State {
        loaded: Mutex::new(Loaded {
            config: vosk_config,
            model: Some(model),
        }),
        session: Mutex::new(None),
        max_recording: Duration::from_secs(daemon_config.max_recording_secs),
        started: Instant::now(),
        shutdown,
        tray,
    });

    watch_for_tray_clicks(Arc::clone(&state), toggle);

    // Caught so that `systemctl --user stop` and Ctrl-C both run the socket
    // guard above instead of leaving the socket behind.
    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;

    info!("listening on {}", path.display());

    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    let state = Arc::clone(&state);
                    tokio::spawn(async move {
                        if let Err(e) = serve(stream, state).await {
                            debug!("connection ended: {:#}", e);
                        }
                    });
                }
                Err(e) => warn!("failed to accept a connection: {}", e),
            },
            _ = state.shutdown.notified() => {
                info!("shutting down on request");
                break;
            }
            _ = sigterm.recv() => {
                info!("shutting down on SIGTERM");
                break;
            }
            _ = sigint.recv() => {
                info!("shutting down on SIGINT");
                break;
            }
        }
    }

    // Any recording still in flight is ended so its thread releases the
    // microphone before the process goes away.
    if let Some(session) = state.session.lock().await.take() {
        session.recording.stop.notify_one();
    }

    state.tray.shutdown().await;

    Ok(())
}

/// Serve one connection. Requests are read until the client disconnects, so a
/// front-end can hold the connection open across several requests.
async fn serve(stream: UnixStream, state: Arc<State>) -> Result<()> {
    let (read, mut write) = stream.into_split();
    let mut lines = BufReader::new(read).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(e) => {
                send(
                    &mut write,
                    &Reply::Error {
                        message: format!("could not parse request: {}", e),
                    },
                )
                .await?;
                continue;
            }
        };

        debug!("request: {:?}", request);

        // Streaming takes over the connection: it writes events until the
        // recording ends, and reads the stop request from the same stream.
        if matches!(request, Request::Stream) {
            stream_session(&state, &mut lines, &mut write).await?;
            return Ok(());
        }

        let reply = dispatch(&state, request).await;
        send(&mut write, &reply).await?;
    }

    Ok(())
}

async fn send(write: &mut OwnedWriteHalf, reply: &Reply) -> Result<()> {
    let mut line = serde_json::to_vec(reply)?;
    line.push(b'\n');
    write.write_all(&line).await?;
    write.flush().await?;
    Ok(())
}

async fn dispatch(state: &Arc<State>, request: Request) -> Reply {
    match request {
        Request::Ping => Reply::Ok,

        Request::Status => {
            // Session before model, and never both at once: see the note on
            // `State::refresh_tray`.
            let recording = state.is_recording().await;
            let loaded = state.loaded.lock().await;
            Reply::Status {
                model: loaded.config.model.to_string(),
                serving: loaded.model.is_some(),
                recording,
                uptime_secs: state.started.elapsed().as_secs(),
                pid: std::process::id(),
            }
        }

        Request::Start => {
            let reply = start_detached(state).await;
            state.refresh_tray().await;
            reply
        }

        Request::Stop => {
            let reply = stop_session(state).await;
            state.refresh_tray().await;
            reply
        }

        Request::TranscribeFile { path } => {
            let Some((config, model)) = state.warm().await else {
                return Reply::NotServing;
            };
            let engine = VoskEngine::with_model(config, model);
            match engine.transcribe_file(&path).await {
                Ok(result) => Reply::Transcribed { result },
                Err(e) => Reply::Error {
                    message: format!("{:#}", e),
                },
            }
        }

        Request::SetModel { model } => {
            let reply = set_model(state, model).await;
            state.refresh_tray().await;
            reply
        }

        Request::Shutdown => {
            state.shutdown.notify_one();
            Reply::Ok
        }

        // Handled in `serve`, which owns the connection for a stream.
        Request::Stream => Reply::Error {
            message: "stream must be the request that takes over the connection".into(),
        },
    }
}

/// Begin a recording whose transcript a later `Stop` collects.
async fn start_detached(state: &Arc<State>) -> Reply {
    let mut slot = state.session.lock().await;

    if let Some(existing) = slot.as_ref() {
        if existing.recording.is_running() {
            // Not an error: key repeat, or a second tap before the first
            // release landed, would otherwise stack recorders onto one
            // microphone.
            return Reply::AlreadyRecording;
        }
        warn!("discarding a finished recording whose text was never collected");
    }

    // Turned off from the tray: the front-end records with its own engine,
    // exactly as it would with no daemon running at all.
    let Some((config, model)) = state.warm().await else {
        return Reply::NotServing;
    };

    let (recording, result) = spawn_recording(config, model, state.max_recording, None);
    watch_for_completion(Arc::clone(state), recording.clone());
    *slot = Some(Session {
        recording,
        result: Some(result),
    });

    Reply::Started
}

/// End the current recording and, unless a streaming client owns it, return
/// its transcript.
async fn stop_session(state: &State) -> Reply {
    let Some(session) = state.session.lock().await.take() else {
        return Reply::NotRecording;
    };

    session.recording.stop.notify_one();

    let Some(result) = session.result else {
        return Reply::Stopped { text: None };
    };

    // A recording that already died on its own still has its error waiting
    // here, which is how an unavailable microphone reaches the user instead of
    // looking like silence.
    match tokio::time::timeout(RESULT_TIMEOUT, result).await {
        Ok(Ok(Ok(text))) => Reply::Stopped { text: Some(text) },
        Ok(Ok(Err(e))) => Reply::Error {
            message: format!("{:#}", e),
        },
        Ok(Err(_)) => Reply::Error {
            message: "the recording thread exited without a result".into(),
        },
        Err(_) => Reply::Error {
            message: format!(
                "the recording did not finish within {}s",
                RESULT_TIMEOUT.as_secs()
            ),
        },
    }
}

/// Act on left clicks of the tray indicator for as long as the daemon runs.
///
/// Its own task rather than part of the accept loop, which has to stay free to
/// answer the socket while a model is being read off disk.
fn watch_for_tray_clicks(state: Arc<State>, toggle: Arc<Notify>) {
    tokio::spawn(async move {
        loop {
            toggle.notified().await;
            toggle_daemon(&state).await;
        }
    });
}

/// Turn the daemon off, or back on, on a left click of the tray icon.
///
/// Off is not a pause that merely declines work: the warm model is released
/// and its memory goes back to the system -- 120MB of resident memory for the
/// small English model, far more for the large ones (see
/// `release_warm_model`). That is the point of the toggle; a daemon that keeps
/// its model is the thing one would otherwise just leave running.
///
/// What off deliberately keeps is the process, and with it two things:
///
///   the icon, because a toggle that removed the icon could not be clicked
///   back on -- that is "Quit", and it is still in the menu for anyone who
///   means it;
///
///   the socket, so that a second `jambi daemon` still refuses to start,
///   `jambi daemon status` can report "turned off" rather than "no daemon
///   running", and front-ends get a straight `NotServing` -- "do it
///   yourself" -- instead of a connection error.
///
/// On loads the model the daemon was configured with, which costs the same
/// ~800ms (far more for the large models) that the front-ends were paying for
/// themselves while it was off.
async fn toggle_daemon(state: &Arc<State>) {
    if state.is_serving().await {
        // Any recording in flight goes with it: "off" that leaves the
        // microphone open is not off. Its text is dropped, and that is the
        // honest outcome -- a click is not a request for a transcript, and no
        // front-end is waiting on this one.
        if let Some(session) = state.session.lock().await.take() {
            warn!("ending the recording in progress: the daemon was turned off");
            session.recording.stop.notify_one();
        }

        release_warm_model(state).await;
        state.refresh_tray().await;
        return;
    }

    // Shown before the load rather than after, because the load is the part
    // worth reporting: for a large model it is seconds, and an icon that sat
    // dimmed throughout would read as a click that did nothing.
    state.show_tray(Activity::Loading).await;

    match load_warm_model(state).await {
        Ok(model) => info!("turned on: the {} model is warm again", model),
        // Left off rather than half on: `loaded.model` is untouched on this
        // path, so the icon goes back to dimmed and the next click tries
        // again. Only reachable if the model has become unreadable since
        // startup -- deleted from the cache, or no memory for it.
        Err(e) => error!("could not load the model, staying off: {:#}", e),
    }

    state.refresh_tray().await;
}

/// Drop the warm model and hand its memory back.
///
/// The drop is the easy half. Giving the memory back is not: freeing returns
/// the model's blocks to glibc's free lists, which is not the same as
/// returning them to the kernel. Measured on the small English model, from an
/// RSS of 146MB warm:
///
///   drop alone                      78MB -- 52MB left on the free lists
///   drop + malloc_trim              26MB -- the model's memory, returned
///
/// Hence the trim. The numbers only hold while the daemon's `Arc` is the last
/// one; see the scoped engine in `run`, where a second reference once made the
/// release free nothing whatsoever.
///
/// glibc only: `malloc_trim` is a GNU extension, and musl has no equivalent to
/// call. There the model is still dropped and the daemon still stops serving;
/// only the moment the pages return to the system is left to the allocator.
async fn release_warm_model(state: &State) {
    // Inside the lock, so nothing can see a daemon that still claims to be
    // serving with the model already gone. A recording that was just told to
    // stop holds an `Arc` of its own, so the last of it comes back when that
    // thread winds up rather than here.
    let mut loaded = state.loaded.lock().await;
    loaded.model = None;

    #[cfg(target_env = "gnu")]
    // SAFETY: `malloc_trim` takes no pointers and promises nothing beyond
    // releasing free memory, so it is safe to call at any point. It returns 1
    // when it managed to hand something back.
    let trimmed = unsafe { libc::malloc_trim(0) } == 1;
    #[cfg(not(target_env = "gnu"))]
    let trimmed = false;

    info!(
        "turned off: the {} model is released{}",
        loaded.config.model,
        if trimmed {
            ""
        } else {
            " (its memory returns to the system when the allocator decides)"
        }
    );
}

/// Load the configured model and make it the warm one.
///
/// Holds the lock across the load for the reason `set_model` does: a request
/// that arrives mid-load waits, rather than seeing a daemon that is half on.
async fn load_warm_model(state: &State) -> Result<VoskModel> {
    let mut loaded = state.loaded.lock().await;

    let mut engine = VoskEngine::new(loaded.config.clone())?;
    engine
        .load_model()
        .await
        .context("failed to load the Vosk model")?;
    let model = engine
        .loaded_model()
        .ok_or_else(|| anyhow::anyhow!("model reported loaded but is not present"))?;

    loaded.model = Some(model);
    Ok(loaded.config.model)
}

/// Load a different model and make it the warm one.
///
/// The load happens inline, holding the lock: a front-end switching language
/// waits exactly as long as it would have in-process, and no recording can
/// start against a half-swapped model.
async fn set_model(state: &State, model: VoskModel) -> Reply {
    let mut loaded = state.loaded.lock().await;

    // Turned off from the tray, so there is no warm model to swap and the
    // front-end is using its own anyway. Loading one here would turn the
    // daemon back on behind the user's click.
    if loaded.model.is_none() {
        return Reply::NotServing;
    }

    if loaded.config.model == model {
        return Reply::Ok;
    }

    let mut config = loaded.config.clone();
    config.model = model;

    let mut engine = match VoskEngine::new(config.clone()) {
        Ok(engine) => engine,
        Err(e) => {
            return Reply::Error {
                message: format!("{:#}", e),
            }
        }
    };

    match engine.load_model().await {
        Ok(()) => match engine.loaded_model() {
            Some(new_model) => {
                info!("warm model is now {}", model);
                // A recording in flight holds its own Arc, so it finishes on
                // the model it started with.
                *loaded = Loaded {
                    config,
                    model: Some(new_model),
                };
                Reply::Ok
            }
            None => Reply::Error {
                message: "model reported loaded but is not present".into(),
            },
        },
        Err(e) => Reply::Error {
            message: format!("{:#}", e),
        },
    }
}

/// Record for as long as this connection wants, streaming partial results back
/// as they are recognised.
async fn stream_session(
    state: &Arc<State>,
    lines: &mut Lines<BufReader<OwnedReadHalf>>,
    write: &mut OwnedWriteHalf,
) -> Result<()> {
    let (events_tx, mut events_rx) = mpsc::unbounded_channel();

    let (recording, result) = {
        let mut slot = state.session.lock().await;
        if slot
            .as_ref()
            .is_some_and(|session| session.recording.is_running())
        {
            send(write, &Reply::AlreadyRecording).await?;
            return Ok(());
        }

        let Some((config, model)) = state.warm().await else {
            send(write, &Reply::NotServing).await?;
            return Ok(());
        };
        let (recording, result) =
            spawn_recording(config, model, state.max_recording, Some(events_tx));
        *slot = Some(Session {
            recording: recording.clone(),
            result: None,
        });
        (recording, result)
    };

    state.refresh_tray().await;

    // Nothing from here to the cleanup below may return early. A write that
    // fails -- the client was killed mid-stream, the usual cause -- must not
    // skip clearing the session, or the slot keeps a recording that has
    // already stopped and the tray stays red for the rest of the session.
    let outcome = pump_stream(lines, write, &recording, result, &mut events_rx).await;

    // Clear the slot only if it still holds *this* recording; a later client
    // may already have started another one.
    {
        let mut slot = state.session.lock().await;
        let ours = slot
            .as_ref()
            .is_some_and(|session| Arc::ptr_eq(&session.recording.finished, &recording.finished));
        if ours {
            *slot = None;
        }
    }

    state.refresh_tray().await;

    let reply = match outcome {
        Ok(Ok(text)) => Reply::Final { text },
        Ok(Err(e)) => Reply::Error {
            message: format!("{:#}", e),
        },
        Err(_) => Reply::Error {
            message: "the recording thread exited without a result".into(),
        },
    };

    // Best-effort: the client may already be gone, which is the normal end of
    // a disconnect-driven stop.
    let _ = send(write, &reply).await;

    Ok(())
}

/// Forward recognition events to the client until the recording produces its
/// transcript, watching the connection for a stop request meanwhile.
///
/// Returns the recording thread's outcome, and never fails: every way this can
/// go wrong is a reason to end the recording, not to abandon it.
async fn pump_stream(
    lines: &mut Lines<BufReader<OwnedReadHalf>>,
    write: &mut OwnedWriteHalf,
    recording: &Recording,
    result: oneshot::Receiver<Result<String>>,
    events: &mut mpsc::UnboundedReceiver<Reply>,
) -> std::result::Result<Result<String>, oneshot::error::RecvError> {
    if send(write, &Reply::Streaming).await.is_err() {
        recording.stop.notify_one();
    }

    let mut result = result;
    let mut asked_to_stop = false;

    let outcome = loop {
        tokio::select! {
            // Biased so the transcript wins over a pending event read once the
            // recording is done, rather than leaving it to whichever the
            // runtime happens to poll first.
            biased;

            outcome = &mut result => break outcome,

            Some(event) = events.recv() => {
                if send(write, &event).await.is_err() {
                    debug!("streaming client went away mid-stream");
                    recording.stop.notify_one();
                    asked_to_stop = true;
                }
            }

            line = lines.next_line(), if !asked_to_stop => {
                asked_to_stop = true;
                // A stop request and a disconnect both end the recording: a
                // front-end that is killed mid-stream must not leave the
                // daemon holding the microphone.
                match line {
                    Ok(Some(_)) => debug!("streaming client asked to stop"),
                    Ok(None) => debug!("streaming client disconnected"),
                    Err(e) => debug!("streaming client read failed: {}", e),
                }
                recording.stop.notify_one();
            }
        }
    };

    // Flush whatever the recognition thread queued just before it finished, so
    // the last utterance appears in the stream and not only in the final text.
    while let Ok(event) = events.try_recv() {
        if send(write, &event).await.is_err() {
            break;
        }
    }

    outcome
}

/// Start recognising on a dedicated thread, returning the handles to stop it
/// and to collect its transcript.
///
/// A thread rather than `tokio::spawn` because the cpal stream and the Vosk
/// recognizer are both `!Send`, so the recognition loop cannot be moved
/// between worker threads. It also keeps the loop off the runtime serving the
/// socket, so `stop` is answered without waiting on a 10ms poll tick.
fn spawn_recording(
    config: VoskConfig,
    model: Arc<Model>,
    max_recording: Duration,
    events: Option<mpsc::UnboundedSender<Reply>>,
) -> (Recording, oneshot::Receiver<Result<String>>) {
    let recording = Recording {
        stop: Arc::new(Notify::new()),
        finished: Arc::new(AtomicBool::new(false)),
        done: Arc::new(Notify::new()),
    };
    let (result_tx, result_rx) = oneshot::channel();

    let stop = Arc::clone(&recording.stop);
    let finished = Arc::clone(&recording.finished);
    let done = Arc::clone(&recording.done);

    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(e) => {
                finished.store(true, Ordering::Release);
                done.notify_one();
                let _ = result_tx.send(Err(anyhow::Error::from(e)));
                return;
            }
        };

        let outcome = runtime.block_on(async move {
            // The model it started with, held by this `Arc` for as long as the
            // recording runs -- so a click that releases the daemon's copy
            // mid-recording cannot pull the model out from under it.
            let mut engine = VoskEngine::with_model(config, model);

            let stop = async move {
                tokio::select! {
                    _ = stop.notified() => {}
                    _ = tokio::time::sleep(max_recording) => {
                        // Reached when the release half of a keybind never
                        // arrives. Finishing the recording keeps the
                        // microphone from staying open for the rest of the
                        // session; the text is still collectable.
                        warn!(
                            "recording reached the {}s cap, finishing it",
                            max_recording.as_secs()
                        );
                    }
                }
            };

            let partial_tx = events.clone();
            let segment_tx = events;

            engine
                .transcribe_stream(
                    stop,
                    |partial| {
                        if let Some(tx) = &partial_tx {
                            let _ = tx.send(Reply::Partial {
                                text: partial.to_string(),
                            });
                        }
                    },
                    |segment| {
                        if let Some(tx) = &segment_tx {
                            let _ = tx.send(Reply::Segment {
                                text: segment.to_string(),
                            });
                        }
                    },
                )
                .await
                .map(|result| result.text)
        });

        if let Err(e) = &outcome {
            error!("recording failed: {:#}", e);
        }

        finished.store(true, Ordering::Release);
        done.notify_one();
        let _ = result_tx.send(outcome);
    });

    (recording, result_rx)
}

/// Refresh the tray once the recording ends, however it ends.
///
/// `Stop` refreshes the tray itself, so this only matters for the endings
/// nobody asked for: a microphone that failed, or a recording that reached the
/// duration cap because the release half of a keybind never arrived. Without
/// it the icon would sit on "recording" until the next request came in.
fn watch_for_completion(state: Arc<State>, recording: Recording) {
    tokio::spawn(async move {
        recording.done.notified().await;
        state.refresh_tray().await;
    });
}

/// Is something listening on `path`?
///
/// Used to tell a running daemon from the socket a killed one left behind.
async fn ping(path: &Path) -> bool {
    match UnixStream::connect(path).await {
        Ok(stream) => {
            let (read, mut write) = stream.into_split();
            let mut lines = BufReader::new(read).lines();
            // The handshake is deliberately a real request: a socket can accept
            // a connection and then never answer, which a connect alone would
            // read as healthy.
            let request = match serde_json::to_vec(&Request::Ping) {
                Ok(mut request) => {
                    request.push(b'\n');
                    request
                }
                Err(_) => return false,
            };
            if write.write_all(&request).await.is_err() {
                return false;
            }
            matches!(
                tokio::time::timeout(Duration::from_secs(2), lines.next_line()).await,
                Ok(Ok(Some(_)))
            )
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A connection to a running daemon.
pub struct Client {
    lines: Lines<BufReader<OwnedReadHalf>>,
    write: OwnedWriteHalf,
}

impl Client {
    /// Connect to a running daemon, or return `None` when there is none.
    ///
    /// "No daemon" is a normal outcome rather than an error: every caller has
    /// a standalone path to fall back to. Unexpected failures degrade the same
    /// way, logged but not propagated, so a broken daemon cannot stop
    /// dictation from working at all.
    pub async fn connect(config: &DaemonConfig) -> Option<Self> {
        if !config.enabled {
            debug!("daemon disabled in config, using an in-process engine");
            return None;
        }
        if std::env::var_os("JAMBI_NO_DAEMON").is_some() {
            debug!("JAMBI_NO_DAEMON is set, using an in-process engine");
            return None;
        }

        let path = socket_path();
        match UnixStream::connect(&path).await {
            Ok(stream) => {
                let (read, write) = stream.into_split();
                Some(Self {
                    lines: BufReader::new(read).lines(),
                    write,
                })
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::NotFound) => {
                debug!("no daemon at {}, using an in-process engine", path.display());
                None
            }
            Err(e) => {
                warn!(
                    "could not reach the daemon at {} ({}), using an in-process engine",
                    path.display(),
                    e
                );
                None
            }
        }
    }

    /// Send a request without waiting for its reply. Used to stop a stream,
    /// whose reply arrives interleaved with the remaining events.
    pub async fn send(&mut self, request: &Request) -> Result<()> {
        let mut line = serde_json::to_vec(request)?;
        line.push(b'\n');
        self.write.write_all(&line).await?;
        self.write.flush().await?;
        Ok(())
    }

    /// Read the next reply or stream event.
    pub async fn next_reply(&mut self) -> Result<Reply> {
        let line = self
            .lines
            .next_line()
            .await
            .context("failed to read from the daemon")?
            .ok_or_else(|| anyhow::anyhow!("the daemon closed the connection"))?;
        serde_json::from_str(&line).context("could not parse the daemon's reply")
    }

    /// Send a request and read its reply.
    pub async fn request(&mut self, request: &Request) -> Result<Reply> {
        self.send(request).await?;
        self.next_reply().await
    }
}

fn format_uptime(secs: u64) -> String {
    let (hours, minutes, seconds) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Print what the daemon is doing, for `jambi daemon status`.
pub async fn print_status() -> Result<()> {
    let config = DaemonConfig::default();
    let Some(mut client) = Client::connect(&config).await else {
        println!("⚪ No daemon running (socket: {})", socket_path().display());
        println!("   Start one with: jambi daemon");
        return Ok(());
    };

    match client.request(&Request::Status).await? {
        Reply::Status {
            model,
            recording,
            uptime_secs,
            pid,
            serving,
        } => {
            // A daemon turned off from the tray is still a running process,
            // and saying "running" of one that is holding nothing and
            // answering nothing would be the least useful reading of the
            // word.
            if serving {
                println!("🟢 Daemon running (pid {})", pid);
            } else {
                println!("⏸️  Daemon turned off (pid {})", pid);
            }
            println!("   Socket:  {}", socket_path().display());
            if serving {
                println!("   Model:   {} (warm)", model);
            } else {
                println!("   Model:   {} (released)", model);
            }
            println!("   Uptime:  {}", format_uptime(uptime_secs));
            println!(
                "   State:   {}",
                match (serving, recording) {
                    (false, _) => "off - click the tray icon to start it",
                    (true, true) => "recording",
                    (true, false) => "idle",
                }
            );
            Ok(())
        }
        Reply::Error { message } => Err(anyhow::anyhow!(message)),
        other => Err(anyhow::anyhow!("unexpected reply: {:?}", other)),
    }
}

/// Ask a running daemon to exit, for `jambi daemon stop`.
pub async fn request_shutdown() -> Result<()> {
    let config = DaemonConfig::default();
    let Some(mut client) = Client::connect(&config).await else {
        println!("⚪ No daemon running");
        return Ok(());
    };

    match client.request(&Request::Shutdown).await? {
        Reply::Ok => {
            println!("⏹️  Daemon shutting down");
            Ok(())
        }
        Reply::Error { message } => Err(anyhow::anyhow!(message)),
        other => Err(anyhow::anyhow!("unexpected reply: {:?}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_round_trip_as_tagged_json() {
        let encoded = serde_json::to_string(&Request::Start).unwrap();
        assert_eq!(encoded, r#"{"cmd":"start"}"#);

        let decoded: Request = serde_json::from_str(r#"{"cmd":"ping"}"#).unwrap();
        assert!(matches!(decoded, Request::Ping));
    }

    /// `NotServing` has to stay a reply of its own rather than collapsing into
    /// an `Error`: every front-end branches on it to fall back to its own
    /// engine, and one that read it as a failure would refuse to work while
    /// the daemon was merely turned off.
    #[test]
    fn being_turned_off_is_its_own_reply() {
        assert_eq!(
            serde_json::to_string(&Reply::NotServing).unwrap(),
            r#"{"reply":"not_serving"}"#
        );

        let decoded: Reply = serde_json::from_str(r#"{"reply":"not_serving"}"#).unwrap();
        assert!(matches!(decoded, Reply::NotServing));
    }

    #[test]
    fn transcripts_with_newlines_stay_on_one_line() {
        let reply = Reply::Stopped {
            text: Some("one\ntwo".to_string()),
        };
        let encoded = serde_json::to_string(&reply).unwrap();
        assert!(!encoded.contains('\n'));

        let decoded: Reply = serde_json::from_str(&encoded).unwrap();
        match decoded {
            Reply::Stopped { text } => assert_eq!(text.as_deref(), Some("one\ntwo")),
            other => panic!("unexpected reply: {:?}", other),
        }
    }

    #[test]
    fn an_empty_transcript_is_distinct_from_no_transcript() {
        let empty = serde_json::to_string(&Reply::Stopped {
            text: Some(String::new()),
        })
        .unwrap();
        let none = serde_json::to_string(&Reply::Stopped { text: None }).unwrap();
        assert_ne!(empty, none);
    }

    #[test]
    fn socket_path_follows_the_environment_override() {
        std::env::set_var("JAMBI_DAEMON_SOCKET", "/run/user/1000/custom.sock");
        assert_eq!(
            socket_path(),
            PathBuf::from("/run/user/1000/custom.sock")
        );
        std::env::remove_var("JAMBI_DAEMON_SOCKET");
    }
}
