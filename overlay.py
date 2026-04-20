"""
overlay.py — Professional monitoring HUD overlay for the MJPEG camera feed.

Renders onto each video frame:
  • Radar-style DOA semi-circle compass  (Sound Direction — Microphone View)
  • Bark probability bar with threshold marker
  • Peak / average audio-level readout
  • BARKING / MONITORING status pill
  • Timestamp + branding bar

All drawing uses Pillow (PIL).  If Pillow is not installed the module
exposes a no-op ``draw_overlay`` that returns the original JPEG bytes.
"""

import io
import math
import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ── Colour palette ────────────────────────────────────────────────────────────

_C = {
    "bg":           (12, 14, 18, 175),
    "bg_dark":      (6,  8, 12, 210),
    "text":         (210, 215, 225, 255),
    "text_dim":     (120, 128, 145, 220),
    "accent":       (70, 195, 255, 255),
    "green":        (45, 200, 110, 255),
    "red":          (240, 60, 60, 255),
    "orange":       (245, 175, 35, 255),
    "bartlett":     (70, 195, 255, 240),
    "capon":        (255, 145, 45, 240),
    "mem":          (175, 115, 255, 240),
    "bar_bg":       (35, 38, 48, 200),
    "grid":         (55, 60, 72, 140),
    "grid_bright":  (80, 88, 105, 180),
}


# ── Font helpers ──────────────────────────────────────────────────────────────

_font_cache: dict = {}

_FONT_SEARCH = [
    # Ubuntu / Debian (Docker image)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    # Arch / Manjaro
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
]


def _font(size: int = 14, bold: bool = False):
    """Load a TrueType font with caching; falls back to PIL default."""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]

    # Try bold-matching candidates first, then anything available
    prefer = [p for p in _FONT_SEARCH if ("Bold" in p) == bold]
    rest   = [p for p in _FONT_SEARCH if ("Bold" in p) != bold]
    for path in prefer + rest:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[key] = f
            return f
        except (IOError, OSError):
            continue

    # Pillow ≥ 10.1 accepts a size arg; older versions do not
    try:
        f = ImageFont.load_default(size=size)
    except TypeError:
        f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _rrect(draw, bbox, radius, fill):
    """Rounded rectangle with fallback for Pillow < 8.2."""
    try:
        draw.rounded_rectangle(bbox, radius=radius, fill=fill)
    except AttributeError:
        draw.rectangle(bbox, fill=fill)


def _doa_xy(cx, cy, radius, doa_deg):
    """Map a DOA angle (0°=left … 90°=centre … 180°=right) → screen (x, y).

    The semi-circle opens *upward*: 0° at the left, 90° at the top, 180° right.
    """
    rad = math.radians(180 - max(0, min(180, doa_deg)))
    return cx + radius * math.cos(rad), cy - radius * math.sin(rad)


# ── DOA compass ───────────────────────────────────────────────────────────────

def _draw_doa_compass(draw, W, y_top, doa1, doa2, doa3):
    """Radar-style semi-circular DOA compass at top-centre of the frame."""
    radius = 58
    cx = W // 2
    cy = y_top + 24 + radius               # centre of semi-circle (baseline)

    # ── background panel ──────────────────────────────────────────────────
    pw, ph = 300, radius + 50
    px = cx - pw // 2
    _rrect(draw, (px, y_top, px + pw, y_top + ph), 8, _C["bg"])

    # ── title ─────────────────────────────────────────────────────────────
    draw.text((cx, y_top + 9), "SOUND DIRECTION (DOA) — MIC VIEW",
              fill=_C["accent"], font=_font(10), anchor="mt")

    # ── concentric range rings ────────────────────────────────────────────
    for frac in (0.33, 0.66, 1.0):
        r = int(radius * frac)
        draw.arc((cx - r, cy - r, cx + r, cy + r), 180, 360,
                 fill=_C["grid"], width=1)

    # ── radial grid lines every 30° ───────────────────────────────────────
    for deg in range(0, 181, 30):
        xi, yi = _doa_xy(cx, cy, 10, deg)
        xo, yo = _doa_xy(cx, cy, radius, deg)
        draw.line([(xi, yi), (xo, yo)], fill=_C["grid"], width=1)

    # ── outer arc (brighter) ─────────────────────────────────────────────
    draw.arc((cx - radius, cy - radius, cx + radius, cy + radius),
             180, 360, fill=_C["grid_bright"], width=2)

    # ── baseline ──────────────────────────────────────────────────────────
    draw.line([(cx - radius, cy), (cx + radius, cy)],
              fill=_C["grid"], width=1)

    # ── algorithm markers on the arc ──────────────────────────────────────
    for doa_val, col_key in [(doa1, "bartlett"),
                              (doa2, "capon"),
                              (doa3, "mem")]:
        mx, my = _doa_xy(cx, cy, radius, doa_val)
        draw.ellipse((mx - 5, my - 5, mx + 5, my + 5),
                     fill=_C[col_key], outline=(255, 255, 255, 90))

    # ── average DOA pointer (bright line from centre) ─────────────────────
    doa_avg = (doa1 + doa2 + doa3) / 3.0
    ex, ey = _doa_xy(cx, cy, radius - 12, doa_avg)
    draw.line([(cx, cy), (ex, ey)], fill=_C["accent"], width=2)
    draw.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=_C["accent"])

    # ── centre dot ────────────────────────────────────────────────────────
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3),
                 fill=_C["text"], outline=_C["accent"])

    # ── angle labels along the rim ────────────────────────────────────────
    lf = _font(9)
    for deg, label in [(0, "0°"), (45, "45°"), (90, "90°"),
                       (135, "135°"), (180, "180°")]:
        lx, ly = _doa_xy(cx, cy, radius + 14, deg)
        draw.text((lx, ly), label, fill=_C["text_dim"], font=lf, anchor="mm")

    # ── compact legend ────────────────────────────────────────────────────
    ly = cy + 6
    sf = _font(8)
    items = [("Bartlett", "bartlett"), ("Capon", "capon"), ("MEM", "mem")]
    # Fixed spacing keeps it simple
    start_x = cx - 55
    for i, (lbl, ck) in enumerate(items):
        lx = start_x + i * 40
        draw.ellipse((lx, ly, lx + 7, ly + 7), fill=_C[ck])
        draw.text((lx + 10, ly + 3), lbl,
                  fill=_C["text_dim"], font=sf, anchor="lm")

    return doa_avg


# ── Status pill ───────────────────────────────────────────────────────────────

def _draw_status(draw, W, is_barking, prob):
    """Red BARKING or green MONITORING pill at top-right."""
    f = _font(12, bold=True)

    if is_barking:
        txt = f"\u25cf BARKING  {prob:.0%}"
        bg  = (195, 35, 35, 215)
        fg  = (255, 255, 255, 255)
    else:
        txt = "\u25cf MONITORING"
        bg  = (25, 130, 55, 200)
        fg  = (195, 230, 210, 255)

    bb = draw.textbbox((0, 0), txt, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pw, ph = tw + 20, th + 14
    px = W - pw - 8
    py = 8

    _rrect(draw, (px, py, px + pw, py + ph), 5, bg)
    draw.text((px + 10, py + 7), txt, fill=fg, font=f)


# ── Info panel ────────────────────────────────────────────────────────────────

def _draw_info_panel(draw, W, H, prob, threshold,
                     peak_dbfs, avg_dbfs, doa_avg):
    """Technical stats panel at bottom-left."""
    pw, ph = 205, 115
    px = 8
    py = H - 24 - 8 - ph           # above timestamp bar + gap

    _rrect(draw, (px, py, px + pw, py + ph), 6, _C["bg"])

    x = px + 10
    y = py + 8
    fs = _font(10)
    fm = _font(12)

    # ── bark probability header ───────────────────────────────────────────
    prob_col = (_C["red"] if prob >= threshold
                else (_C["orange"] if prob >= 0.5 else _C["green"]))
    draw.text((x, y), "BARK", fill=_C["text_dim"], font=fs)
    draw.text((x + 38, y), f"{prob:.0%}",
              fill=prob_col, font=_font(12, bold=True))
    y += 16

    # ── probability bar ───────────────────────────────────────────────────
    bw = pw - 20
    bh = 8
    _rrect(draw, (x, y, x + bw, y + bh), 3, _C["bar_bg"])

    fw = max(0, min(bw, int(bw * prob)))
    if fw > 1:
        _rrect(draw, (x, y, x + fw, y + bh), 3, prob_col)

    # threshold tick
    tx = x + int(bw * min(threshold, 1.0))
    draw.line([(tx, y - 2), (tx, y + bh + 2)], fill=_C["text"], width=1)
    y += 18

    # ── audio levels ──────────────────────────────────────────────────────
    draw.text((x, y), f"PEAK  {peak_dbfs:+.1f} dBFS",
              fill=_C["text"], font=fm)
    y += 16
    draw.text((x, y), f"AVG   {avg_dbfs:+.1f} dBFS",
              fill=_C["text"], font=fm)
    y += 18

    # ── DOA angle ─────────────────────────────────────────────────────────
    draw.text((x, y), f"DOA   {doa_avg:.0f}\u00b0",
              fill=_C["accent"], font=_font(13, bold=True))


# ── Timestamp bar ─────────────────────────────────────────────────────────────

def _draw_timestamp(draw, W, H):
    """Thin branding + clock bar at the very bottom of the frame."""
    bh = 24
    by = H - bh
    draw.rectangle((0, by, W, H), fill=(0, 0, 0, 165))

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    draw.text((10, by + 6), ts, fill=_C["text_dim"], font=_font(11))
    draw.text((W - 10, by + 6), "WOOFALYTICS",
              fill=_C["accent"], font=_font(10, bold=True), anchor="ra")


# ── Public API ────────────────────────────────────────────────────────────────

def draw_overlay(jpeg_bytes: bytes, overlay_data: dict) -> bytes:
    """Render full HUD overlay onto a JPEG camera frame.

    Parameters
    ----------
    jpeg_bytes : bytes
        Raw JPEG image data from the camera.
    overlay_data : dict
        Expected keys:
          doa         – {"doa1": int, "doa2": int, "doa3": int}
          bark_prob   – {"bark_probability": [float, …], "datetime": str}
          audio_level – {"peak_dbfs": float, "avg_dbfs": float}
          threshold   – float (bark probability threshold, default 0.88)

    Returns
    -------
    bytes
        Decorated JPEG, or the original bytes if anything goes wrong.
    """
    if not _PIL_OK or not jpeg_bytes or not overlay_data:
        return jpeg_bytes or b""

    try:
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGBA")
        ov  = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)
        W, H = img.size

        doa   = overlay_data.get("doa", {})
        pred  = overlay_data.get("bark_prob", {})
        audio = overlay_data.get("audio_level", {})
        thr   = overlay_data.get("threshold", 0.88)

        probs = pred.get("bark_probability", [])
        prob  = max(probs) if probs else 0.0
        is_barking = prob >= thr

        doa1 = doa.get("doa1", 90)
        doa2 = doa.get("doa2", 90)
        doa3 = doa.get("doa3", 90)

        peak = audio.get("peak_dbfs", -60.0)
        avg  = audio.get("avg_dbfs",  -60.0)

        # ── render each element ───────────────────────────────────────────
        doa_avg = _draw_doa_compass(draw, W, 5, doa1, doa2, doa3)
        _draw_status(draw, W, is_barking, prob)
        _draw_info_panel(draw, W, H, prob, thr, peak, avg, doa_avg)
        _draw_timestamp(draw, W, H)

        # ── composite and re-encode ───────────────────────────────────────
        img = Image.alpha_composite(img, ov).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    except Exception:
        return jpeg_bytes
