# neck_relief_viewer_single_string.py
# Run: python neck_relief_viewer_single_string.py
# Dependencies: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection

# -----------------------------
# Constants / defaults
# -----------------------------
SCALE_MM_DEFAULT = 648.0        # 25.5" scale
N_FRETS = 24
BEND_END_FRET = 17
X_SAMPLES = 900

# Pick geometry (fixed)
PICK_FROM_BRIDGE_MM = 105.0

# Envelope quality/perf
ENVELOPE_MODES = 24
ENVELOPE_T_SAMPLES = 96

# View padding
TOP_HEADROOM_MM = 10.0
BOTTOM_HEADROOM_MM = 5.0

# Visual tuning
FRET_LINEWIDTH = 3.3
FRET_FONT_SIZE = 13
FRET_LABEL_OFFSET = 0.45        # below fret base
NECK_FILL_COLOR = "0.85"
NECK_FILL_ALPHA = 0.25

# Table highlighting
CLR_BAD = (1.00, 0.86, 0.86)    # light red
CLR_OK  = (1.00, 1.00, 1.00)

DEFAULTS = dict(
    lift_mm=0.5,
    pressed_fret=0,
    base_dev_open_mm=1.5,
    nut_clearance=0.10,
    bridge_clearance=2.0,
    neck_thickness=3.0,
    fret_height=1.0,
)

# -----------------------------
# Helpers
# -----------------------------
def fret_positions_mm(scale_mm: float, n_frets: int) -> np.ndarray:
    n = np.arange(0, n_frets + 1)
    return scale_mm - scale_mm / (2 ** (n / 12))

def neck_curve(x: np.ndarray, x_end: float, lift_mm: float) -> np.ndarray:
    """Bow only from nut to x_end: y = lift*(1 - x/x_end)^2 for x<=x_end."""
    y = np.zeros_like(x, dtype=float)
    if x_end <= 1e-9:
        return y
    mask = x <= x_end
    t = x[mask] / x_end
    y[mask] = lift_mm * (1 - t) ** 2
    return y

def speaking_endpoints(pressed: int, x_frets: np.ndarray, scale_mm: float):
    if pressed <= 0:
        return 0.0, scale_mm
    pf = int(np.clip(pressed, 1, len(x_frets) - 1))
    return float(x_frets[pf]), float(scale_mm)

def pick_point_x_from_bridge_mm(x_left: float, x_right: float, pick_mm: float) -> float:
    """Fixed pick point measured from the bridge, clamped into the speaking segment."""
    L = x_right - x_left
    if L <= 1e-9:
        return x_left
    eps = 1e-6
    xp = x_right - float(pick_mm)
    return float(np.clip(xp, x_left + eps, x_right - eps))

def effective_dev_constant_energy(dev_open: float, x_left: float, x_right: float, scale_mm: float) -> float:
    """Auto-scale: dev_eff = dev_open * sqrt(L / L_open)."""
    L = max(x_right - x_left, 1e-9)
    return float(dev_open * np.sqrt(L / max(scale_mm, 1e-9)))

def string_equilibrium(
    x_arr: np.ndarray,
    scale_mm: float,
    x_frets: np.ndarray,
    x_end: float,
    lift_val: float,
    fret_h: float,
    nut_clearance: float,
    bridge_clearance: float,
    pressed_fret: int,
) -> np.ndarray:
    # Heights are measured relative to a "fret-top plane":
    # y = neck_curve(x) + fret_h + clearance_at_that_anchor
    y_nut = neck_curve(np.array([0.0]), x_end, lift_val)[0] + fret_h + nut_clearance
    y_bridge = 0.0 + fret_h + bridge_clearance

    if pressed_fret <= 0:
        return y_nut + (y_bridge - y_nut) * (x_arr / scale_mm)

    pf = int(np.clip(pressed_fret, 1, len(x_frets) - 1))
    xk = x_frets[pf]
    yk = neck_curve(np.array([xk]), x_end, lift_val)[0] + fret_h  # touches fret top

    y = np.empty_like(x_arr, dtype=float)
    left = x_arr <= xk

    if np.any(left):
        t = x_arr[left] / max(xk, 1e-9)
        y[left] = y_nut + (yk - y_nut) * t

    if np.any(~left):
        t = (x_arr[~left] - xk) / max(scale_mm - xk, 1e-9)
        y[~left] = yk + (y_bridge - yk) * t

    return y

def poly_xy_between(x: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> np.ndarray:
    xs = np.concatenate([x, x[::-1]])
    ys = np.concatenate([y_upper, y_lower[::-1]])
    return np.column_stack([xs, ys])

# -----------------------------
# Plucked-string envelope (modal peak)
# -----------------------------
def pluck_envelope_modal_peak(
    x_arr: np.ndarray,
    x_left: float,
    x_right: float,
    x_pick: float,
    dev_at_pick: float,
    n_modes: int = ENVELOPE_MODES,
    n_time: int = ENVELOPE_T_SAMPLES,
) -> np.ndarray:
    """
    Peak (over time) transverse displacement magnitude for an ideal plucked string:
    - initial displacement is triangular with peak dev_at_pick at x_pick
    - fixed ends at x_left, x_right
    - sine modes + cosine time evolution (zero initial velocity)
    """
    amp = np.zeros_like(x_arr, dtype=float)
    L = x_right - x_left
    if dev_at_pick <= 0 or L <= 1e-9:
        return amp

    mask = (x_arr >= x_left) & (x_arr <= x_right)
    if not np.any(mask):
        return amp

    u = (x_arr[mask] - x_left) / L
    alpha = (x_pick - x_left) / L
    alpha = float(np.clip(alpha, 1e-6, 1 - 1e-6))

    n = np.arange(1, n_modes + 1, dtype=float)
    k = n * np.pi

    # Fourier sine series coefficients for triangular initial displacement
    b = (2.0 * dev_at_pick) * np.sin(k * alpha) / ((k ** 2) * (alpha * (1.0 - alpha)))

    S = np.sin(np.outer(n * np.pi, u))                     # (N, M)

    theta = np.linspace(0.0, 2.0 * np.pi, n_time, endpoint=False)
    C = np.cos(np.outer(n, theta))                         # (N, T)

    Y = (b[:, None] * S).T @ C                             # (M, T)
    a = np.max(np.abs(Y), axis=1)

    # Normalize so peak-at-pick corresponds to dev_at_pick (keeps slider intuitive)
    S_pick = np.sin(n * np.pi * alpha)
    y_pick = (b * S_pick) @ C                               # (T,)
    a_pick = float(np.max(np.abs(y_pick)))
    if a_pick > 1e-9:
        a *= (dev_at_pick / a_pick)

    amp[mask] = a
    return amp

# -----------------------------
# Scene setup
# -----------------------------
scale_mm = SCALE_MM_DEFAULT
x_frets = fret_positions_mm(scale_mm, N_FRETS)
x_end = x_frets[min(BEND_END_FRET, N_FRETS)]
x = np.linspace(0, scale_mm, X_SAMPLES)

# Parameters
lift_mm = DEFAULTS["lift_mm"]
neck_thickness = DEFAULTS["neck_thickness"]
fret_height = DEFAULTS["fret_height"]
nut_clearance = DEFAULTS["nut_clearance"]
bridge_clearance = DEFAULTS["bridge_clearance"]
base_dev_open_mm = DEFAULTS["base_dev_open_mm"]
pressed_fret = DEFAULTS["pressed_fret"]

# -----------------------------
# Figure / axes
# -----------------------------
plt.close("all")
fig, ax = plt.subplots(figsize=(12.6, 5.9))
plt.subplots_adjust(left=0.07, right=0.99, bottom=0.47, top=0.92)

ax.set_title("Guitar Neck Relief (Side View) — Bow from Nut to 17th Fret")
ax.set_xlabel("Distance from nut (mm)")
ax.set_ylabel("Height (mm)")
ax.set_xlim(-10, scale_mm + 10)

# -----------------------------
# Artists
# -----------------------------
y_neck = neck_curve(x, x_end, lift_mm)

neck_xy = poly_xy_between(x, y_neck - neck_thickness, y_neck)
neck_poly = ax.fill(neck_xy[:, 0], neck_xy[:, 1], color=NECK_FILL_COLOR, alpha=NECK_FILL_ALPHA, linewidth=0)[0]
(neck_line,) = ax.plot(x, y_neck, linewidth=2.0, color="black")
(rigid_line,) = ax.plot([x_end, scale_mm], [0, 0], linestyle="--", linewidth=1.0, color="black")

def make_fret_segments(lift_val: float, fh: float) -> list:
    y_neck_f = neck_curve(x_frets, x_end, lift_val)
    return [[(x_frets[i], y_neck_f[i]), (x_frets[i], y_neck_f[i] + fh)] for i in range(1, N_FRETS + 1)]

fret_collection = LineCollection(
    make_fret_segments(lift_mm, fret_height),
    linewidths=FRET_LINEWIDTH,
    colors="black"
)
ax.add_collection(fret_collection)

# Fret numbers (horizontal, below fret poles)
fret_texts = []
for i in range(1, N_FRETS + 1):
    xf = x_frets[i]
    yb = neck_curve(np.array([xf]), x_end, lift_mm)[0]
    fret_texts.append(
        ax.text(
            xf, yb - FRET_LABEL_OFFSET, str(i),
            fontsize=FRET_FONT_SIZE, fontweight="bold",
            ha="center", va="top",
            rotation=0, clip_on=True, color="black"
        )
    )

# String + envelope
y_str = string_equilibrium(
    x, scale_mm, x_frets, x_end,
    lift_mm, fret_height, nut_clearance, bridge_clearance, pressed_fret
)
(string_line,) = ax.plot(x, y_str, linewidth=1.8, color="black")

xL, xR = speaking_endpoints(pressed_fret, x_frets, scale_mm)
x_pick = pick_point_x_from_bridge_mm(xL, xR, PICK_FROM_BRIDGE_MM)
dev_eff = effective_dev_constant_energy(base_dev_open_mm, xL, xR, scale_mm)

amp = pluck_envelope_modal_peak(x, xL, xR, x_pick, dev_eff)
y_upper = y_str + amp
y_lower = y_str - amp

env_xy = poly_xy_between(x, y_lower, y_upper)
env_poly = ax.fill(env_xy[:, 0], env_xy[:, 1], color="black", alpha=0.08, linewidth=0)[0]
(env_upper_line,) = ax.plot(x, y_upper, linestyle=":", linewidth=1.0, color="black")
(env_lower_line,) = ax.plot(x, y_lower, linestyle=":", linewidth=1.0, color="black")

measure_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", ha="left", color="black")

# -----------------------------
# Clearance table panel (bottom-right)
# -----------------------------
TABLE_LEFT = 0.69
TABLE_BOTTOM = 0.05
TABLE_WIDTH = 0.30
TABLE_HEIGHT = 0.40

ax_tbl = fig.add_axes([TABLE_LEFT, TABLE_BOTTOM, TABLE_WIDTH, TABLE_HEIGHT])
ax_tbl.set_facecolor("0.97")
ax_tbl.set_xticks([])
ax_tbl.set_yticks([])
for sp in ax_tbl.spines.values():
    sp.set_visible(True)

tbl_header = ax_tbl.text(
    0.02, 0.98, "",
    transform=ax_tbl.transAxes, va="top", ha="left",
    fontsize=9, fontweight="bold"
)

def compute_min_clearances_all_frets(lift_val, fh, nut_c, bridge_c, pressed, dev_open):
    y_neck_f = neck_curve(x_frets, x_end, lift_val)
    y_fret_top = y_neck_f + fh

    y_str_f = string_equilibrium(
        x_frets, scale_mm, x_frets, x_end,
        lift_val, fh, nut_c, bridge_c, pressed
    )

    x_left, x_right = speaking_endpoints(pressed, x_frets, scale_mm)
    xp = pick_point_x_from_bridge_mm(x_left, x_right, PICK_FROM_BRIDGE_MM)
    dev_eff_local = effective_dev_constant_energy(dev_open, x_left, x_right, scale_mm)

    amp_f = pluck_envelope_modal_peak(x_frets, x_left, x_right, xp, dev_eff_local)
    return (y_str_f - amp_f) - y_fret_top

def build_table_cells(clearances):
    half = (N_FRETS + 1) // 2
    rows = half
    cell_text = []
    for r in range(rows):
        f1 = r + 1
        f2 = r + 1 + half
        c1 = clearances[f1] if f1 <= N_FRETS else None
        c2 = clearances[f2] if f2 <= N_FRETS else None
        cell_text.append([
            str(f1) if f1 <= N_FRETS else "",
            f"{c1:+.2f}" if c1 is not None else "",
            str(f2) if f2 <= N_FRETS else "",
            f"{c2:+.2f}" if c2 is not None else "",
        ])
    return cell_text

_initial_clear = compute_min_clearances_all_frets(
    lift_mm, fret_height, nut_clearance, bridge_clearance, pressed_fret, base_dev_open_mm
)

col_labels = ["Fret", "Min clr (mm)", "Fret", "Min clr (mm)"]
table = ax_tbl.table(
    cellText=build_table_cells(_initial_clear),
    colLabels=col_labels,
    cellLoc="center",
    colLoc="center",
    bbox=[0.0, 0.0, 1.0, 0.86],
)
table.auto_set_font_size(False)
table.set_fontsize(8)

def update_table(clearances):
    half = (N_FRETS + 1) // 2
    rows = half
    cells = table.get_celld()

    for r in range(rows):
        f1 = r + 1
        f2 = r + 1 + half
        c1 = float(clearances[f1]) if f1 <= N_FRETS else None
        c2 = float(clearances[f2]) if f2 <= N_FRETS else None

        cells[(r + 1, 0)].get_text().set_text(str(f1))
        cells[(r + 1, 1)].get_text().set_text(f"{c1:+.2f}" if c1 is not None else "")
        cells[(r + 1, 1)].set_facecolor(CLR_BAD if (c1 is not None and c1 < 0.0) else CLR_OK)

        if f2 <= N_FRETS:
            cells[(r + 1, 2)].get_text().set_text(str(f2))
            cells[(r + 1, 3)].get_text().set_text(f"{c2:+.2f}" if c2 is not None else "")
            cells[(r + 1, 3)].set_facecolor(CLR_BAD if (c2 is not None and c2 < 0.0) else CLR_OK)
        else:
            cells[(r + 1, 2)].get_text().set_text("")
            cells[(r + 1, 3)].get_text().set_text("")
            cells[(r + 1, 3)].set_facecolor(CLR_OK)

update_table(_initial_clear)

# -----------------------------
# Sliders + scales (ticks)
# -----------------------------
SL_LEFT = 0.10
SL_WIDTH = 0.52
SL_H = 0.030
SL_GAP = 0.018
y0 = 0.40

def _fmt_tick(v: float) -> str:
    # nicer tick labels
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"

def style_slider_axis(ax_s, vmin, vmax, ticks=None):
    ax_s.set_yticks([])
    if ticks is None:
        mid = (vmin + vmax) / 2.0
        ticks = [vmin, mid, vmax]
    ax_s.set_xticks(ticks)
    ax_s.set_xticklabels([_fmt_tick(t) for t in ticks], fontsize=8)
    ax_s.tick_params(axis="x", pad=2, length=2)
    ax_s.set_xlim(vmin, vmax)

def add_slider(y, label, vmin, vmax, vinit, vstep, ticks=None):
    ax_s = plt.axes([SL_LEFT, y, SL_WIDTH, SL_H])
    s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=vstep)
    s.valtext.set_fontsize(9)
    style_slider_axis(ax_s, vmin, vmax, ticks=ticks)
    return s

s_lift   = add_slider(y0,                    "Headstock lift (mm)",          -10.0,  15.0, lift_mm,            0.05,
                      ticks=[-10, 0, 15])
s_press  = add_slider(y0 - 1*(SL_H+SL_GAP),   "Pressed fret (0=open)",          0,      N_FRETS, pressed_fret,   1,
                      ticks=[0, N_FRETS//2, N_FRETS])
s_dev    = add_slider(y0 - 2*(SL_H+SL_GAP),   "Base dev@pick (mm, open)",       0.0,    5.0,  base_dev_open_mm, 0.05,
                      ticks=[0, 2.5, 5.0])
s_nut    = add_slider(y0 - 3*(SL_H+SL_GAP),   "Nut clearance (mm)",             -5.0,   10.0, nut_clearance,     0.10,
                      ticks=[-5, 0, 10])
s_bridge = add_slider(y0 - 4*(SL_H+SL_GAP),   "Bridge clearance (mm)",          -5.0,   15.0, bridge_clearance,  0.10,
                      ticks=[-5, 5, 15])
s_thick  = add_slider(y0 - 5*(SL_H+SL_GAP),   "Neck thickness (visual, mm)",     3.0,    5.0, neck_thickness,    0.10,
                      ticks=[3.0, 4.0, 5.0])
s_fh     = add_slider(y0 - 6*(SL_H+SL_GAP),   "Fret height (visual, mm)",        0.6,    2.0, fret_height,       0.10,
                      ticks=[0.6, 1.3, 2.0])

# Reset button
btn_ax = plt.axes([SL_LEFT, 0.07, 0.12, 0.045])
btn_reset = Button(btn_ax, "Reset", hovercolor="0.90")

# -----------------------------
# Update logic
# -----------------------------
_bulk_setting = False

def update(_):
    if _bulk_setting:
        return

    lift_val = float(s_lift.val)
    pressed = int(s_press.val)
    dev_open = float(s_dev.val)
    nut_c = float(s_nut.val)
    bridge_c = float(s_bridge.val)
    thick = float(s_thick.val)
    fh = float(s_fh.val)

    # neck curve
    y_n = neck_curve(x, x_end, lift_val)
    neck_line.set_ydata(y_n)
    neck_poly.set_xy(poly_xy_between(x, y_n - thick, y_n))

    # frets + labels
    fret_collection.set_segments(make_fret_segments(lift_val, fh))
    fret_collection.set_linewidths(FRET_LINEWIDTH)
    for i, txt in enumerate(fret_texts, start=1):
        xf = x_frets[i]
        yb = neck_curve(np.array([xf]), x_end, lift_val)[0]
        txt.set_position((xf, yb - FRET_LABEL_OFFSET))

    # string equilibrium
    y_s = string_equilibrium(
        x, scale_mm, x_frets, x_end,
        lift_val, fh, nut_c, bridge_c, pressed
    )
    string_line.set_ydata(y_s)

    # pluck envelope (modal peak) + auto dev scaling
    x_left, x_right = speaking_endpoints(pressed, x_frets, scale_mm)
    xp = pick_point_x_from_bridge_mm(x_left, x_right, PICK_FROM_BRIDGE_MM)
    dev_eff_local = effective_dev_constant_energy(dev_open, x_left, x_right, scale_mm)

    amp_local = pluck_envelope_modal_peak(x, x_left, x_right, xp, dev_eff_local)
    y_u = y_s + amp_local
    y_l = y_s - amp_local

    env_upper_line.set_ydata(y_u)
    env_lower_line.set_ydata(y_l)
    env_poly.set_xy(poly_xy_between(x, y_l, y_u))

    # table
    clearances = compute_min_clearances_all_frets(lift_val, fh, nut_c, bridge_c, pressed, dev_open)
    update_table(clearances)

    mode = "open" if pressed <= 0 else f"fretted @ {pressed}"
    tbl_header.set_text(
        "Clearance table (min during pluck)\n"
        f"Mode: {mode}\n"
        f"Pick: {PICK_FROM_BRIDGE_MM:.0f} mm from bridge (fixed)\n"
        f"Nut: {nut_c:+.2f} | Bridge: {bridge_c:+.2f} | Lift: {lift_val:+.2f}\n"
        f"Dev@pick: {dev_eff_local:.2f} mm (auto-scaled)"
    )

    # y limits
    y_top = max(float(np.max(y_n) + fh), float(np.max(y_u)))
    y_bot = min(float(np.min(y_n - thick)), float(np.min(y_l)))
    ax.set_ylim(y_bot - 1.0, y_top + 3.0)

    fig.canvas.draw_idle()

def on_reset(_event):
    global _bulk_setting
    _bulk_setting = True
    # prevent multiple redraws while resetting
    for s in (s_lift, s_press, s_dev, s_nut, s_bridge, s_thick, s_fh):
        s.eventson = False

    s_lift.set_val(DEFAULTS["lift_mm"])
    s_press.set_val(DEFAULTS["pressed_fret"])
    s_dev.set_val(DEFAULTS["base_dev_open_mm"])
    s_nut.set_val(DEFAULTS["nut_clearance"])
    s_bridge.set_val(DEFAULTS["bridge_clearance"])
    s_thick.set_val(DEFAULTS["neck_thickness"])
    s_fh.set_val(DEFAULTS["fret_height"])

    for s in (s_lift, s_press, s_dev, s_nut, s_bridge, s_thick, s_fh):
        s.eventson = True

    _bulk_setting = False
    update(None)

btn_reset.on_clicked(on_reset)

for sld in (s_lift, s_press, s_dev, s_nut, s_bridge, s_thick, s_fh):
    sld.on_changed(update)

update(None)
plt.show()
