//! System tray indicator for the daemon.
//!
//! The daemon is otherwise invisible: it holds the model, answers the socket,
//! and never draws anything. That is fine until you want to know whether it is
//! actually running -- so it publishes a StatusNotifierItem, the same protocol
//! Vorta, KeePassXC and OpenSnitch use, and shows up in the tray beside them.
//!
//! The icon also carries state the socket would otherwise keep to itself: it
//! turns red while a recording is in progress, so a dictation that never
//! received its release (and is quietly recording until the cap) is visible
//! rather than something you find out about from the microphone light.
//!
//! The indicator is strictly decoration. `spawn` reports failures and returns
//! a handle whose methods become no-ops, because a desktop with no tray -- or
//! a daemon started before the bar comes up -- must not stop the daemon from
//! serving requests.

use std::sync::Arc;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tracing::debug;

/// Which drawing the indicator uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TrayIcon {
    /// A genie lamp, Jambi being a genie. Drawn solid and unadorned: it is
    /// wider than it is tall, so it fills the width of the panel's icon slot
    /// and rather less of the height.
    #[default]
    Lamp,
    /// The microphone, matching the glyph on notifications.
    Microphone,
}

/// The color the glyph is drawn in.
///
/// Named for the ink rather than the desktop, because "dark mode" is ambiguous
/// about which one it asks for: a dark panel needs a *light* icon. The `dark`
/// and `light` aliases accept the other vocabulary and map to the color that
/// suits a panel of that shade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TrayColor {
    /// Near-white, for a dark panel.
    #[default]
    #[serde(alias = "dark")]
    White,
    /// Near-black, for a light panel.
    #[serde(alias = "light")]
    Black,
}

/// How the indicator looks. Comes from config and does not change while the
/// daemon runs, which is why it is kept apart from the `TrayState` below.
#[cfg_attr(not(feature = "tray"), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub struct TrayStyle {
    pub icon: TrayIcon,
    pub color: TrayColor,
    /// Turn the glyph red while a recording is in progress.
    ///
    /// On by default: it is the only at-a-glance sign that a dictation whose
    /// key release went missing is still holding the microphone. Turning it off
    /// does not hide that state, it just stops announcing it in color -- the
    /// tooltip and the menu still say "Recording".
    pub red_when_recording: bool,
}

/// Hand-written rather than derived: `red_when_recording` has to default to
/// true, and `bool`'s own default is false.
impl Default for TrayStyle {
    fn default() -> Self {
        Self {
            icon: TrayIcon::default(),
            color: TrayColor::default(),
            red_when_recording: true,
        }
    }
}

/// What the indicator displays. Mirrored from the daemon rather than read back
/// out of it, because the tray host polls these properties at its leisure and
/// cannot be given a future to await.
///
/// The daemon builds one of these either way, so without the `tray` feature
/// the fields are written and never read.
#[cfg_attr(not(feature = "tray"), allow(dead_code))]
#[derive(Debug, Clone)]
pub struct TrayState {
    pub model: String,
    pub recording: bool,
}

#[cfg(feature = "tray")]
mod imp {
    use super::*;
    use tracing::{info, warn};

    use ksni::menu::StandardItem;
    use ksni::{Category, Icon, MenuItem, Status, ToolTip, TrayMethods};

    /// Pixmap heights offered to the host. A tray host picks the closest to its
    /// panel height, and offering several avoids it scaling one up into mush.
    ///
    /// Heights rather than squares: the pixmaps are cropped to their glyph, so
    /// width follows from the shape. Hosts scale an icon to the panel's height
    /// and take the width from the aspect ratio, which is the dimension worth
    /// supplying at native resolution.
    const SIZES: [u32; 5] = [16, 22, 24, 32, 48];

    struct JambiTray {
        style: TrayStyle,
        state: TrayState,
        /// Shared with the daemon's accept loop, so "Quit" takes the same exit
        /// path as SIGTERM and the socket is cleaned up either way.
        shutdown: Arc<Notify>,
    }

    impl ksni::Tray for JambiTray {
        fn id(&self) -> String {
            "jambi".into()
        }

        fn title(&self) -> String {
            "Jambi".into()
        }

        fn category(&self) -> Category {
            Category::ApplicationStatus
        }

        fn status(&self) -> Status {
            // Always Active, never Passive: some hosts hide passive items, and
            // the entire point here is to be visible among the other
            // background applications.
            Status::Active
        }

        fn icon_pixmap(&self) -> Vec<Icon> {
            let color = ink(self.style, self.state.recording);
            SIZES
                .iter()
                .map(|&size| render(self.style.icon, size, color))
                .collect()
        }

        fn tool_tip(&self) -> ToolTip {
            ToolTip {
                title: "Jambi".into(),
                description: if self.state.recording {
                    format!("Recording · {}", self.state.model)
                } else {
                    format!("Idle · {} (warm)", self.state.model)
                },
                ..Default::default()
            }
        }

        fn menu(&self) -> Vec<MenuItem<Self>> {
            let shutdown = Arc::clone(&self.shutdown);

            vec![
                // Disabled items: this is a status readout, not a control
                // panel. Dictation is driven from the keybind, which is also
                // the only context that has the environment to type the result
                // -- offering "start recording" here would record into nowhere.
                StandardItem {
                    label: if self.state.recording {
                        "● Recording".into()
                    } else {
                        "Idle".into()
                    },
                    enabled: false,
                    ..Default::default()
                }
                .into(),
                StandardItem {
                    label: format!("Model: {}", self.state.model),
                    enabled: false,
                    ..Default::default()
                }
                .into(),
                MenuItem::Separator,
                StandardItem {
                    label: "Quit Jambi daemon".into(),
                    icon_name: "application-exit".into(),
                    activate: Box::new(move |_: &mut Self| shutdown.notify_one()),
                    ..Default::default()
                }
                .into(),
            ]
        }
    }

    pub struct Tray(Option<ksni::Handle<JambiTray>>);

    impl Tray {
        /// An indicator the user turned off, or that there is no desktop for.
        /// Every method below is a no-op on it.
        pub fn disabled() -> Self {
            Self(None)
        }

        pub async fn spawn(style: TrayStyle, state: TrayState, shutdown: Arc<Notify>) -> Self {
            let tray = JambiTray {
                style,
                state,
                shutdown,
            };

            // `assume_sni_available` matters for the case this daemon is built
            // for: started at login, very possibly before the bar that hosts
            // the tray. Without it a missing host is a hard error here and the
            // icon never appears for the rest of the session; with it, ksni
            // keeps waiting and registers when the host shows up.
            match tray.assume_sni_available(true).spawn().await {
                Ok(handle) => {
                    info!("tray indicator registered");
                    Self(Some(handle))
                }
                Err(e) => {
                    warn!("no tray indicator ({}); the daemon runs without one", e);
                    Self(None)
                }
            }
        }

        /// Replace what the indicator shows. Silently does nothing when there
        /// is no indicator, which is the normal state on a headless session.
        pub async fn set(&self, state: TrayState) {
            let Some(handle) = &self.0 else {
                return;
            };
            if handle.update(move |tray| tray.state = state).await.is_none() {
                debug!("tray service is gone, leaving the indicator alone");
            }
        }

        pub async fn shutdown(&self) {
            if let Some(handle) = &self.0 {
                handle.shutdown().await;
            }
        }
    }

    /// Non-premultiplied ARGB, as the StatusNotifierItem specification's
    /// `IconPixmap` wants. Alpha comes from coverage, so these are the opaque
    /// colors of the strokes.
    type Rgb = [u8; 3];

    /// The idle and recording colors for a chosen ink.
    ///
    /// Recording stays red in both, because it reports state rather than
    /// following the theme -- but the shade differs, since the red that reads
    /// best on a dark panel is washed out on a light one.
    fn palette(color: TrayColor) -> (Rgb, Rgb) {
        match color {
            // Near-white rather than pure, matching the notification glyph in
            // `assets/`, which reads on the dark panel a tray usually sits on.
            TrayColor::White => ([0xEC, 0xEF, 0xF4], [0xE5, 0x4B, 0x4B]),
            // Near-black rather than pure, which sits heavily beside themed
            // panel icons, with a deeper red to match.
            TrayColor::Black => ([0x2E, 0x34, 0x40], [0xC0, 0x39, 0x2B]),
        }
    }

    /// The color to draw in, for a given style and recording state.
    ///
    /// Separate from `icon_pixmap` so it can be tested without standing up a
    /// tray service: that the recording tint can be switched off is the kind
    /// of thing that silently stops working.
    fn ink(style: TrayStyle, recording: bool) -> Rgb {
        let (idle, active) = palette(style.color);
        if recording && style.red_when_recording {
            active
        } else {
            idle
        }
    }

    /// A shape to draw, with the box it occupies in its own space.
    ///
    /// Carrying the bounds alongside the membership test is what lets one
    /// renderer fit any glyph to the pixmap. Both are needed, not just the
    /// height: the microphone is taller than it is wide and the lamp is nearly
    /// twice as wide as it is tall, so which dimension binds differs between
    /// them. `glyph_bounds_match_the_geometry` checks these against the shapes.
    struct Glyph {
        inside: fn(f32, f32) -> bool,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    }

    impl Glyph {
        fn width(&self) -> f32 {
            self.right - self.left
        }

        fn height(&self) -> f32 {
            self.bottom - self.top
        }
    }

    fn glyph(icon: TrayIcon) -> Glyph {
        match icon {
            TrayIcon::Microphone => Glyph {
                inside: microphone_inside,
                left: 0.5 - CRADLE_OUTER,
                top: MIC_TOP,
                right: 0.5 + CRADLE_OUTER,
                bottom: MIC_BOTTOM,
            },
            TrayIcon::Lamp => Glyph {
                inside: lamp_inside,
                left: LAMP_LEFT,
                top: LAMP_TOP,
                right: LAMP_RIGHT,
                bottom: LAMP_BOTTOM,
            },
        }
    }

    // The microphone's geometry, in its own space: a unit square with v
    // downwards. Hoisted out of the membership test so that the extent below is
    // derived from these same numbers rather than a second copy that could
    // drift.
    const CAP_R: f32 = 0.115;
    const CAP_TOP: f32 = 0.19;
    const CAP_BOTTOM: f32 = 0.49;
    const CRADLE_CY: f32 = 0.44;
    const CRADLE_OUTER: f32 = 0.275;
    const CRADLE_INNER: f32 = 0.225;
    const STEM_DX: f32 = 0.035;
    const STEM_TOP: f32 = 0.70;
    const STEM_BOTTOM: f32 = 0.86;
    const FOOT_DX: f32 = 0.18;
    const FOOT_TOP: f32 = 0.845;
    const FOOT_BOTTOM: f32 = 0.885;

    /// Vertical extent of the microphone: the top of the capsule's cap down to
    /// the underside of the foot. `glyph_extents_match_the_geometry` keeps
    /// these honest against the constants above.
    pub(super) const MIC_TOP: f32 = CAP_TOP - CAP_R;
    pub(super) const MIC_BOTTOM: f32 = FOOT_BOTTOM;

    // The genie lamp -- Jambi is a genie -- drawn solid, with nothing around
    // it. Two shapes were tried and rejected on legibility before this one,
    // and both are worth recording so they are not re-attempted:
    //
    //   a ring with the lamp inside it. The ring is crisp, but it takes the
    //   outer third of the icon and leaves the lamp drawn in strokes 1-2px
    //   wide at panel sizes, which is exactly the "hard to perceive" this
    //   replaced.
    //
    //   a filled disc with the lamp knocked out of it. Bolder, but worse: the
    //   knockout has to stay clear of the rim or it breaks the disc, which
    //   caps the lamp at about half the diameter, and a 1-2px *gap* is flooded
    //   by the fill around it far more readily than a 1-2px stroke is lost.
    //   Below ~24px it reads as a wedge-shaped void, not a lamp.
    //
    // Solid and unadorned spends every pixel on the lamp itself. The trade is
    // that a lamp is nearly twice as wide as it is tall, so it fills the width
    // of a square pixmap and only ~60% of the height -- which is why `Glyph`
    // carries both bounds and `render` fits whichever binds.
    const BODY_CX: f32 = 0.520;
    const BODY_CY: f32 = 0.560;
    const BODY_RX: f32 = 0.230;
    const BODY_RY: f32 = 0.165;
    const KNOB_CX: f32 = 0.520;
    const KNOB_CY: f32 = 0.388;
    const KNOB_R: f32 = 0.062;
    const LAMP_FOOT_X0: f32 = 0.395;
    const LAMP_FOOT_X1: f32 = 0.645;
    const LAMP_FOOT_TOP: f32 = 0.690;
    const LAMP_FOOT_BOTTOM: f32 = 0.745;
    // Spout: a tapering stroke rising from the body to the left. Angled rather
    // than flat, which reads as a spout instead of a stub.
    const SPOUT_AX: f32 = 0.400;
    const SPOUT_AY: f32 = 0.545;
    const SPOUT_BX: f32 = 0.215;
    const SPOUT_BY: f32 = 0.455;
    const SPOUT_W0: f32 = 0.100;
    const SPOUT_W1: f32 = 0.034;
    const HANDLE_CX: f32 = 0.745;
    const HANDLE_CY: f32 = 0.580;
    const HANDLE_OUTER: f32 = 0.132;
    const HANDLE_INNER: f32 = 0.072;

    /// The lamp's bounding box. Hand-derived from the geometry above and
    /// checked against it by `glyph_bounds_match_the_geometry`, because the
    /// spout's rounded tip and the handle's arc do not reduce to a constant
    /// cleanly enough to compute here.
    const LAMP_LEFT: f32 = 0.182;
    const LAMP_TOP: f32 = 0.326;
    const LAMP_RIGHT: f32 = 0.877;
    const LAMP_BOTTOM: f32 = 0.745;

    /// Blank margin above and below the glyph, as a fraction of its longer
    /// side.
    ///
    /// This is the knob for apparent size, and only this one. A host scales
    /// the pixmap so its height matches the panel's icon height, so the glyph
    /// ends up drawn at `height / (height + 2 * margin)` of its neighbours --
    /// a bigger margin means a smaller glyph. Flush to the glyph drew the lamp
    /// noticeably larger than everything else on the bar, and padding out to a
    /// square drew it noticeably smaller; this sits between the two.
    ///
    /// Measured against the longer side so the border reads as the same
    /// thickness as the horizontal one below, rather than being stretched
    /// along whichever axis is shorter.
    pub(super) const GLYPH_MARGIN_Y: f32 = 0.09;

    /// Blank margin to left and right of the glyph, in pixels.
    ///
    /// Deliberately not tied to the vertical margin, because the two do
    /// entirely different jobs: height drives the scale, so vertical padding
    /// resizes the glyph, while horizontal padding cannot -- a host lays the
    /// item out at the pixmap's own width, so every blank column is just a gap
    /// on the bar. One pixel is enough for the antialiased edge to land in and
    /// no more, because anything wider shows up as this icon sitting further
    /// from its neighbour than they sit from each other.
    pub(super) const GLYPH_MARGIN_X_PX: f32 = 1.0;

    /// Draw `icon` at `height` pixels tall, in `color`.
    ///
    /// The pixmap is cropped to the glyph rather than padded out to a square.
    /// That is what keeps the icon spaced like its neighbours in the tray: a
    /// host lays an item out at the pixmap's own width, so transparent columns
    /// inside the pixmap become visible gaps either side of the glyph. The
    /// microphone is barely two thirds as wide as it is tall, which read as
    /// noticeably more air around it than around everything else on the bar.
    ///
    /// Drawn rather than decoded from a bitmap, which buys three things worth
    /// more than the arithmetic below: no image-decoding dependency, no second
    /// copy of the glyph to keep in step with `assets/microphone.svg`, and a
    /// crisp result at whatever size the host asks for instead of one blurred
    /// from a single bitmap. The shipped SVG cannot be used directly -- the
    /// specification takes pixels, not vectors.
    fn render(icon: TrayIcon, height: u32, color: Rgb) -> Icon {
        let shape = glyph(icon);

        /// Samples per axis. 3x3 is enough to take the stair-steps off strokes
        /// this thick, and keeps a 16px icon at 2,304 distance evaluations.
        const SAMPLES: u32 = 3;

        // One scale for both axes, set by the height. The width then follows
        // the glyph's own aspect; deriving it from a second scale would let
        // rounding stretch the drawing.
        let margin_y = GLYPH_MARGIN_Y * shape.width().max(shape.height());
        let units_per_px = (shape.height() + 2.0 * margin_y) / height as f32;
        let width = (shape.width() / units_per_px + 2.0 * GLYPH_MARGIN_X_PX)
            .round()
            .max(1.0) as u32;

        let centre_u = (shape.left + shape.right) / 2.0;
        let centre_v = (shape.top + shape.bottom) / 2.0;

        let mut data = vec![0u8; (width * height * 4) as usize];

        for y in 0..height {
            for x in 0..width {
                let mut hits = 0u32;
                for sy in 0..SAMPLES {
                    for sx in 0..SAMPLES {
                        // Sample at subpixel centres, so coverage is symmetric
                        // about the pixel rather than biased to one corner.
                        let px = x as f32 + (sx as f32 + 0.5) / SAMPLES as f32;
                        let py = y as f32 + (sy as f32 + 0.5) / SAMPLES as f32;
                        let u = centre_u + (px - width as f32 / 2.0) * units_per_px;
                        let v = centre_v + (py - height as f32 / 2.0) * units_per_px;
                        if (shape.inside)(u, v) {
                            hits += 1;
                        }
                    }
                }

                let alpha = (hits * 255 / (SAMPLES * SAMPLES)) as u8;
                let offset = ((y * width + x) * 4) as usize;
                data[offset] = alpha;
                data[offset + 1] = color[0];
                data[offset + 2] = color[1];
                data[offset + 3] = color[2];
            }
        }

        Icon {
            width: width as i32,
            height: height as i32,
            data,
        }
    }

    /// Is the point `(u, v)` -- both in 0..1, v downwards -- part of the
    /// microphone?
    ///
    /// The shape is the conventional microphone: a capsule, the U-shaped
    /// cradle around its lower half, and a stem down to a foot.
    fn microphone_inside(u: f32, v: f32) -> bool {
        let dx = (u - 0.5).abs();

        // Capsule: a rectangle with semicircular caps, which is a distance
        // test against the vertical segment at its core.
        let core = v.clamp(CAP_TOP, CAP_BOTTOM);
        if (dx * dx + (v - core) * (v - core)).sqrt() <= CAP_R {
            return true;
        }

        // Cradle: the outer half of an annulus, kept to the lower half so it
        // reads as a U rather than a ring around the capsule.
        let r = (dx * dx + (v - CRADLE_CY) * (v - CRADLE_CY)).sqrt();
        if v >= CRADLE_CY && (CRADLE_INNER..=CRADLE_OUTER).contains(&r) {
            return true;
        }

        // Stem, from the cradle's base to the foot.
        if dx <= STEM_DX && (STEM_TOP..=STEM_BOTTOM).contains(&v) {
            return true;
        }

        // Foot.
        dx <= FOOT_DX && (FOOT_TOP..=FOOT_BOTTOM).contains(&v)
    }

    /// Distance from `(px, py)` to the segment `a`-`b`, with how far along the
    /// segment the nearest point fell. The fraction is what lets a stroke
    /// taper along its length.
    fn segment_distance(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> (f32, f32) {
        let (vx, vy) = (bx - ax, by - ay);
        let (wx, wy) = (px - ax, py - ay);
        let t = ((wx * vx + wy * vy) / (vx * vx + vy * vy)).clamp(0.0, 1.0);
        let (nx, ny) = (ax + t * vx, ay + t * vy);
        (((px - nx).powi(2) + (py - ny).powi(2)).sqrt(), t)
    }

    /// Is the point part of the genie lamp?
    ///
    /// A squat body, a tapering spout rising to the left, a looped handle to
    /// the right, a knob on the lid and a foot beneath.
    fn lamp_inside(u: f32, v: f32) -> bool {
        let bx = (u - BODY_CX) / BODY_RX;
        let by = (v - BODY_CY) / BODY_RY;
        if bx * bx + by * by <= 1.0 {
            return true;
        }

        let kx = u - KNOB_CX;
        let ky = v - KNOB_CY;
        if kx * kx + ky * ky <= KNOB_R * KNOB_R {
            return true;
        }

        if (LAMP_FOOT_X0..=LAMP_FOOT_X1).contains(&u)
            && (LAMP_FOOT_TOP..=LAMP_FOOT_BOTTOM).contains(&v)
        {
            return true;
        }

        // Spout: widest where it meets the body, so the two join without a
        // seam, tapering to the tip.
        let (distance, along) = segment_distance(u, v, SPOUT_AX, SPOUT_AY, SPOUT_BX, SPOUT_BY);
        if distance <= SPOUT_W0 + along * (SPOUT_W1 - SPOUT_W0) {
            return true;
        }

        // Handle: an annulus whose left side is swallowed by the body, leaving
        // a loop. Hollow rather than solid -- the hole is what makes it read as
        // a handle rather than a lump, and it survives down to 22px.
        let hx = u - HANDLE_CX;
        let hy = v - HANDLE_CY;
        let hr = (hx * hx + hy * hy).sqrt();
        (HANDLE_INNER..=HANDLE_OUTER).contains(&hr)
    }

    #[cfg(test)]
    pub(super) fn test_icon(icon: TrayIcon, size: u32) -> Icon {
        render(icon, size, palette(TrayColor::White).0)
    }

    #[cfg(test)]
    pub(super) fn test_palette(color: TrayColor) -> (Rgb, Rgb) {
        palette(color)
    }

    #[cfg(test)]
    pub(super) fn test_ink(style: TrayStyle, recording: bool) -> Rgb {
        ink(style, recording)
    }

    /// A glyph's own bounding box, found by sampling rather than read off the
    /// constants. Lets the tests check that the bounds in `glyph` really do
    /// bound the shapes they claim to.
    #[cfg(test)]
    pub(super) fn sampled_bounds(icon: TrayIcon) -> (f32, f32, f32, f32) {
        const STEPS: u32 = 2000;
        let shape = glyph(icon).inside;
        let (mut left, mut top) = (f32::MAX, f32::MAX);
        let (mut right, mut bottom) = (f32::MIN, f32::MIN);
        for i in 0..=STEPS {
            let v = i as f32 / STEPS as f32;
            for j in 0..=STEPS {
                let u = j as f32 / STEPS as f32;
                if shape(u, v) {
                    left = left.min(u);
                    right = right.max(u);
                    top = top.min(v);
                    bottom = bottom.max(v);
                }
            }
        }
        (left, top, right, bottom)
    }

    /// The bounds `glyph` declares, for the tests to compare against.
    #[cfg(test)]
    pub(super) fn declared_bounds(icon: TrayIcon) -> (f32, f32, f32, f32) {
        let g = glyph(icon);
        (g.left, g.top, g.right, g.bottom)
    }
}

#[cfg(not(feature = "tray"))]
mod imp {
    use super::*;

    /// Stand-in for builds without the `tray` feature, so the daemon needs no
    /// conditional compilation of its own.
    pub struct Tray;

    impl Tray {
        pub fn disabled() -> Self {
            Self
        }

        pub async fn spawn(
            _style: TrayStyle,
            _state: TrayState,
            _shutdown: Arc<Notify>,
        ) -> Self {
            debug!("built without the tray feature; no indicator");
            Self
        }

        pub async fn set(&self, _state: TrayState) {}

        pub async fn shutdown(&self) {}
    }
}

pub use imp::Tray;

#[cfg(all(test, feature = "tray"))]
mod tests {
    use super::imp;
    use super::{TrayColor, TrayIcon, TrayStyle};

    const ICONS: [TrayIcon; 2] = [TrayIcon::Microphone, TrayIcon::Lamp];
    const SIZES: [u32; 5] = [16, 22, 24, 32, 48];

    /// Every size the host may ask for has to come back as a buffer matching
    /// its declared dimensions; a short one is read past the end by the host.
    /// The height is what was asked for, and the width follows the glyph.
    #[test]
    fn every_icon_size_is_a_complete_argb_buffer() {
        for icon in ICONS {
            for size in SIZES {
                let rendered = imp::test_icon(icon, size);
                assert_eq!(rendered.height, size as i32, "{icon:?} at {size}");
                assert!(rendered.width > 0, "{icon:?} at {size}: zero width");
                assert_eq!(
                    rendered.data.len(),
                    (rendered.width * rendered.height * 4) as usize,
                    "{icon:?} at {size}: buffer does not match {}x{}",
                    rendered.width,
                    rendered.height
                );
            }
        }
    }

    /// A glyph that came out blank or solid would still pass the size check,
    /// and both are silent failures in a tray.
    #[test]
    fn no_glyph_is_blank_or_solid() {
        for icon in ICONS {
            let rendered = imp::test_icon(icon, 32);
            let opaque = rendered.data.chunks_exact(4).filter(|p| p[0] > 128).count();
            let total = (rendered.width * rendered.height) as usize;
            assert!(opaque > total / 50, "{icon:?} is nearly blank: {opaque} px");
            assert!(
                opaque < total * 7 / 8,
                "{icon:?} covers the whole pixmap: {opaque} px"
            );
        }
    }

    /// The pixmap must be cropped to the glyph and its margin, with nothing
    /// spare. Transparent columns beyond that become visible gaps either side
    /// of the icon, which is what made this one sit further from its
    /// neighbours than they sat from each other -- a square pixmap left the
    /// microphone filling only 65% of its own width.
    ///
    /// Checked as a fraction rather than a row count because the margin is
    /// deliberate; what must not come back is *slack*.
    #[test]
    fn pixmaps_are_cropped_to_glyph_and_margin() {
        for icon in ICONS {
            let rendered = imp::test_icon(icon, 48);
            let (w, h) = (rendered.width as usize, rendered.height as usize);
            let opaque = |x: usize, y: usize| rendered.data[(y * w + x) * 4] > 0;

            let rows: Vec<usize> = (0..h).filter(|&y| (0..w).any(|x| opaque(x, y))).collect();
            let cols: Vec<usize> = (0..w).filter(|&x| (0..h).any(|y| opaque(x, y))).collect();

            let tall = (rows.last().unwrap() - rows.first().unwrap() + 1) as f32 / h as f32;
            let wide = (cols.last().unwrap() - cols.first().unwrap() + 1) as f32 / w as f32;

            assert!(
                tall > 0.70,
                "{icon:?}: glyph is only {:.0}% of the pixmap's height",
                tall * 100.0
            );
            assert!(
                wide > 0.70,
                "{icon:?}: glyph is only {:.0}% of the pixmap's width",
                wide * 100.0
            );

            // Centred: an off-centre glyph would read as the icon sitting
            // closer to one neighbour than the other.
            let left = *cols.first().unwrap();
            let right = w - 1 - cols.last().unwrap();
            assert!(
                left.abs_diff(right) <= 1,
                "{icon:?}: not centred, {left} blank columns left and {right} right"
            );
        }
    }

    /// The width must be the glyph's own, at the scale the height sets, plus
    /// the horizontal margin and nothing else. Getting this wrong either
    /// stretches the drawing or pads it back out into the gap that started
    /// all this.
    #[test]
    fn pixmap_width_follows_the_glyph() {
        for icon in ICONS {
            let (left, top, right, bottom) = imp::declared_bounds(icon);
            let (gw, gh) = (right - left, bottom - top);
            let margin_y = imp::GLYPH_MARGIN_Y * gw.max(gh);

            for size in SIZES {
                let rendered = imp::test_icon(icon, size);
                let units_per_px = (gh + 2.0 * margin_y) / size as f32;
                let expected = (gw / units_per_px + 2.0 * imp::GLYPH_MARGIN_X_PX).round() as i32;
                assert!(
                    (rendered.width - expected).abs() <= 1,
                    "{icon:?} at {size}: pixmap is {}px wide, expected {expected}px",
                    rendered.width
                );
            }
        }
    }

    /// Horizontal padding is pure spacing on the bar -- a host lays the item
    /// out at the pixmap's width -- so it has to stay at the one pixel the
    /// antialiased edge needs. This is the regression guard for the icon
    /// sitting further from its neighbour than they sit from each other.
    #[test]
    fn horizontal_padding_stays_hairline() {
        for icon in ICONS {
            for size in SIZES {
                let rendered = imp::test_icon(icon, size);
                let (w, h) = (rendered.width as usize, rendered.height as usize);
                let opaque = |x: usize, y: usize| rendered.data[(y * w + x) * 4] > 0;

                let cols: Vec<usize> =
                    (0..w).filter(|&x| (0..h).any(|y| opaque(x, y))).collect();
                let left = *cols.first().unwrap();
                let right = w - 1 - cols.last().unwrap();

                assert!(
                    left <= 1 && right <= 1,
                    "{icon:?} at {size}: {left} blank columns left, {right} right"
                );
            }
        }
    }

    /// The bounds in `glyph` are hand-derived from the geometry constants, so
    /// they can fall out of step with the shapes -- a taller foot, or a spout
    /// reaching further left, would leave the drawing fitted to a box it no
    /// longer fits inside and silently clip an edge.
    #[test]
    fn glyph_bounds_match_the_geometry() {
        for icon in ICONS {
            let (left, top, right, bottom) = imp::sampled_bounds(icon);
            let declared = imp::declared_bounds(icon);
            for (name, found, said) in [
                ("left", left, declared.0),
                ("top", top, declared.1),
                ("right", right, declared.2),
                ("bottom", bottom, declared.3),
            ] {
                assert!(
                    (found - said).abs() < 0.01,
                    "{icon:?} {name}: shape is at {found}, bounds say {said}"
                );
            }
        }
    }

    /// The handle's loop is the lamp's one interior hole, and the feature that
    /// makes it read as a handle rather than a lump. If it filled in, the
    /// drawing would still pass every check above.
    #[test]
    fn the_lamp_handle_is_a_loop() {
        let rendered = imp::test_icon(TrayIcon::Lamp, 48);
        let (w, h) = (rendered.width as usize, rendered.height as usize);
        let opaque = |x: usize, y: usize| rendered.data[(y * w + x) * 4] > 128;

        // A row through the handle crosses: body, gap, handle. Scanning every
        // row and taking the best avoids pinning the handle's exact height.
        let most_crossings = (0..h)
            .map(|y| {
                let row: Vec<bool> = (0..w).map(|x| opaque(x, y)).collect();
                row.windows(2).filter(|w| w[0] != w[1]).count()
            })
            .max()
            .unwrap();

        assert!(
            most_crossings >= 4,
            "no row crosses the lamp more than {most_crossings} times, so the handle has no hole"
        );
    }

    /// Print both glyphs as ASCII, at the sizes a panel actually asks for.
    ///
    /// Not an assertion -- a drawing is judged by eye, and the checks above
    /// can only catch a glyph that is blank, solid, or badly placed, not one
    /// that has stopped looking like a microphone. Ignored by default; run it
    /// after touching any of the geometry constants:
    ///
    ///   cargo test tray::tests::show_the_glyphs -- --ignored --nocapture
    #[test]
    #[ignore = "visual aid, not a check"]
    fn show_the_glyphs() {
        for icon in ICONS {
            for size in [16u32, 22, 32] {
                let rendered = imp::test_icon(icon, size);
                let (w, h) = (rendered.width as usize, rendered.height as usize);
                println!("\n{icon:?} at {size}px tall -> {w}x{h} pixmap");
                for y in 0..h {
                    let row: String = (0..w)
                        .map(|x| match rendered.data[(y * w + x) * 4] {
                            0..=63 => "  ",
                            64..=191 => "++",
                            _ => "##",
                        })
                        .collect();
                    println!("{row}");
                }
            }
        }
    }

    /// `tray_red_when_recording = false` has to actually stop the color
    /// changing, and must leave the idle color alone while doing it. A setting
    /// that reads fine but does nothing is the failure to guard against here.
    #[test]
    fn the_recording_tint_can_be_switched_off() {
        for color in [TrayColor::White, TrayColor::Black] {
            let on = TrayStyle {
                color,
                red_when_recording: true,
                ..Default::default()
            };
            let off = TrayStyle {
                red_when_recording: false,
                ..on
            };

            assert_ne!(
                imp::test_ink(on, true),
                imp::test_ink(on, false),
                "{color:?}: tint on, but recording draws the idle color"
            );
            assert_eq!(
                imp::test_ink(off, true),
                imp::test_ink(off, false),
                "{color:?}: tint off, but recording still changes color"
            );
            assert_eq!(
                imp::test_ink(on, false),
                imp::test_ink(off, false),
                "{color:?}: the switch moved the idle color too"
            );
        }
    }

    /// Both inks have to differ, and recording has to differ from idle within
    /// each -- otherwise one of the two settings, or the recording state,
    /// silently does nothing.
    #[test]
    fn the_palettes_are_distinguishable() {
        let (white_idle, white_rec) = imp::test_palette(TrayColor::White);
        let (black_idle, black_rec) = imp::test_palette(TrayColor::Black);

        assert_ne!(white_idle, black_idle, "both inks draw the same color");
        assert_ne!(white_idle, white_rec, "white: recording looks idle");
        assert_ne!(black_idle, black_rec, "black: recording looks idle");

        let luma = |c: [u8; 3]| c[0] as u32 + c[1] as u32 + c[2] as u32;
        assert!(
            luma(white_idle) > luma(black_idle),
            "the white ink is not lighter than the black one"
        );
    }
}
