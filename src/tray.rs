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
    /// The microphone, matching the glyph on notifications.
    #[default]
    Microphone,
    /// A genie lamp, Jambi being a genie. Drawn solid and unadorned: it is
    /// wider than it is tall, so it fills the width of the panel's icon slot
    /// and rather less of the height.
    Lamp,
}

/// The colour the glyph is drawn in.
///
/// Named for the ink rather than the desktop, because "dark mode" is ambiguous
/// about which one it asks for: a dark panel needs a *light* icon. The `dark`
/// and `light` aliases accept the other vocabulary and map to the colour that
/// suits a panel of that shade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum TrayColour {
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
#[derive(Debug, Clone, Copy, Default)]
pub struct TrayStyle {
    pub icon: TrayIcon,
    pub colour: TrayColour,
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

    /// Sizes offered to the host. A tray host picks the closest to its panel
    /// height, and offering several avoids it scaling one up into mush.
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
            let (idle, recording) = palette(self.style.colour);
            let colour = if self.state.recording { recording } else { idle };
            SIZES
                .iter()
                .map(|&size| render(self.style.icon, size, colour))
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
    /// colours of the strokes.
    type Rgb = [u8; 3];

    /// The idle and recording colours for a chosen ink.
    ///
    /// Recording stays red in both, because it reports state rather than
    /// following the theme -- but the shade differs, since the red that reads
    /// best on a dark panel is washed out on a light one.
    fn palette(colour: TrayColour) -> (Rgb, Rgb) {
        match colour {
            // Near-white rather than pure, matching the notification glyph in
            // `assets/`, which reads on the dark panel a tray usually sits on.
            TrayColour::White => ([0xEC, 0xEF, 0xF4], [0xE5, 0x4B, 0x4B]),
            // Near-black rather than pure, which sits heavily beside themed
            // panel icons, with a deeper red to match.
            TrayColour::Black => ([0x2E, 0x34, 0x40], [0xC0, 0x39, 0x2B]),
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

    /// Fraction of the icon a glyph's binding dimension fills.
    ///
    /// The microphone's geometry is laid out with a generous margin, spanning
    /// only ~0.81 of its square vertically. Rendered straight, that margin is
    /// dead space inside the pixmap and the icon reads a size smaller than
    /// everything beside it in the tray. Fitting the glyph's own bounding box
    /// uses the room the host actually gave us. Short of 1.0 so the antialiased
    /// edge has somewhere to land rather than being clipped flat against the
    /// border.
    const GLYPH_FILL: f32 = 0.96;

    /// Draw `icon` at `size` square, in `colour`.
    ///
    /// Drawn rather than decoded from a bitmap, which buys three things worth
    /// more than the arithmetic below: no image-decoding dependency, no second
    /// copy of the glyph to keep in step with `assets/microphone.svg`, and a
    /// crisp result at whatever size the host asks for instead of one blurred
    /// from a single bitmap. The shipped SVG cannot be used directly -- the
    /// specification takes pixels, not vectors.
    fn render(icon: TrayIcon, size: u32, colour: Rgb) -> Icon {
        let shape = glyph(icon);

        /// Samples per axis. 3x3 is enough to take the stair-steps off strokes
        /// this thick, and keeps a 16px icon at 2,304 distance evaluations.
        const SAMPLES: u32 = 3;

        let mut data = vec![0u8; (size * size * 4) as usize];
        let pixels = size as f32;

        // Fits the glyph's bounding box into the square pixmap, scaled by
        // whichever dimension binds and centred on the box in both axes. One
        // span for both axes, not two: scaling each to fill independently
        // would stretch the microphone narrow and the lamp tall.
        let span = shape.width().max(shape.height()) / GLYPH_FILL;
        let centre_u = (shape.left + shape.right) / 2.0;
        let centre_v = (shape.top + shape.bottom) / 2.0;

        for y in 0..size {
            for x in 0..size {
                let mut hits = 0u32;
                for sy in 0..SAMPLES {
                    for sx in 0..SAMPLES {
                        // Sample at subpixel centres, so coverage is symmetric
                        // about the pixel rather than biased to one corner.
                        let px = (x as f32 + (sx as f32 + 0.5) / SAMPLES as f32) / pixels;
                        let py = (y as f32 + (sy as f32 + 0.5) / SAMPLES as f32) / pixels;
                        let u = centre_u + (px - 0.5) * span;
                        let v = centre_v + (py - 0.5) * span;
                        if (shape.inside)(u, v) {
                            hits += 1;
                        }
                    }
                }

                let alpha = (hits * 255 / (SAMPLES * SAMPLES)) as u8;
                let offset = ((y * size + x) * 4) as usize;
                data[offset] = alpha;
                data[offset + 1] = colour[0];
                data[offset + 2] = colour[1];
                data[offset + 3] = colour[2];
            }
        }

        Icon {
            width: size as i32,
            height: size as i32,
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
        render(icon, size, palette(TrayColour::White).0)
    }

    #[cfg(test)]
    pub(super) fn test_palette(colour: TrayColour) -> (Rgb, Rgb) {
        palette(colour)
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
    use super::{TrayColour, TrayIcon};

    const ICONS: [TrayIcon; 2] = [TrayIcon::Microphone, TrayIcon::Lamp];
    const SIZES: [u32; 5] = [16, 22, 24, 32, 48];

    /// Every size the host may ask for has to come back as a correctly sized
    /// ARGB32 buffer; a short one is read past the end by the host.
    #[test]
    fn every_icon_size_is_a_complete_argb_buffer() {
        for icon in ICONS {
            for size in SIZES {
                let rendered = imp::test_icon(icon, size);
                assert_eq!(rendered.width, size as i32);
                assert_eq!(rendered.height, size as i32);
                assert_eq!(rendered.data.len(), (size * size * 4) as usize);
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
            let total = (32 * 32) as usize;
            assert!(
                opaque > total / 50,
                "{icon:?} is nearly blank: {opaque} px"
            );
            assert!(
                opaque < total * 3 / 4,
                "{icon:?} covers the whole icon: {opaque} px"
            );
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

    /// The microphone was originally drawn inside its own margin, which made it
    /// read a size smaller than the other tray items. This pins the fix: each
    /// glyph has to reach both edges of the pixmap along whichever dimension
    /// binds -- the microphone is taller than wide and fills the height, the
    /// lamp is wider than tall and fills the width.
    #[test]
    fn every_glyph_fills_its_binding_dimension() {
        let size = 32usize;
        for icon in ICONS {
            let rendered = imp::test_icon(icon, size as u32);
            let opaque = |x: usize, y: usize| rendered.data[(y * size + x) * 4] > 0;

            let rows: Vec<usize> = (0..size).filter(|&y| (0..size).any(|x| opaque(x, y))).collect();
            let cols: Vec<usize> = (0..size).filter(|&x| (0..size).any(|y| opaque(x, y))).collect();

            let spans_rows = *rows.first().unwrap() <= 1 && *rows.last().unwrap() >= size - 2;
            let spans_cols = *cols.first().unwrap() <= 1 && *cols.last().unwrap() >= size - 2;

            assert!(
                spans_rows || spans_cols,
                "{icon:?} fills neither dimension: rows {}..{}, cols {}..{} of {size}",
                rows.first().unwrap(),
                rows.last().unwrap(),
                cols.first().unwrap(),
                cols.last().unwrap()
            );
        }
    }

    /// The handle's loop is the lamp's one interior hole, and the feature that
    /// makes it read as a handle rather than a lump. If it filled in, the
    /// drawing would still pass every check above.
    #[test]
    fn the_lamp_handle_is_a_loop() {
        let size = 48usize;
        let rendered = imp::test_icon(TrayIcon::Lamp, size as u32);
        let opaque = |x: usize, y: usize| rendered.data[(y * size + x) * 4] > 128;

        // A row through the handle crosses: body, gap, handle. Scanning every
        // row and taking the best avoids pinning the handle's exact height.
        let most_crossings = (0..size)
            .map(|y| {
                let row: Vec<bool> = (0..size).map(|x| opaque(x, y)).collect();
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
            for size in [16usize, 22, 32] {
                println!("\n{icon:?} at {size}px");
                let rendered = imp::test_icon(icon, size as u32);
                for y in 0..size {
                    let row: String = (0..size)
                        .map(|x| {
                            match rendered.data[(y * size + x) * 4] {
                                0..=63 => "  ",
                                64..=191 => "++",
                                _ => "##",
                            }
                        })
                        .collect();
                    println!("{row}");
                }
            }
        }
    }

    /// Both inks have to differ, and recording has to differ from idle within
    /// each -- otherwise one of the two settings, or the recording state,
    /// silently does nothing.
    #[test]
    fn the_palettes_are_distinguishable() {
        let (white_idle, white_rec) = imp::test_palette(TrayColour::White);
        let (black_idle, black_rec) = imp::test_palette(TrayColour::Black);

        assert_ne!(white_idle, black_idle, "both inks draw the same colour");
        assert_ne!(white_idle, white_rec, "white: recording looks idle");
        assert_ne!(black_idle, black_rec, "black: recording looks idle");

        let luma = |c: [u8; 3]| c[0] as u32 + c[1] as u32 + c[2] as u32;
        assert!(
            luma(white_idle) > luma(black_idle),
            "the white ink is not lighter than the black one"
        );
    }
}
