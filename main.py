import logging
import json
import os
import signal
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import glob
import settings as cfg_store
from record import Woofalytics
import trainer as _trainer_mod

try:
    from overlay import draw_overlay as _draw_overlay
except ImportError:
    _draw_overlay = None


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in its own thread."""
    daemon_threads = True


# ── MJPEG camera streamer ─────────────────────────────────────────────────────

class MJPEGStreamer:
    """Captures frames from a V4L2 device via ffmpeg and serves them as MJPEG.
    Device is resolved from settings at stream-start time so Config page changes
    take effect without a container restart."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._frame  = b""
        self._proc   = None
        self._thread = None
        self._data_provider = None   # set via set_data_provider()

    def _current_device(self) -> str:
        # settings.py takes precedence; fall back to env var
        s = cfg_store.get_public()
        return s.get("video_device") or os.environ.get("VIDEO_DEVICE", "/dev/video2")

    def _capture_loop(self, device: str):
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-f", "v4l2", "-framerate", "15",
            "-i", device,
            "-vf", "scale=640:480",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            buf = b""
            while True:
                chunk = self._proc.stdout.read(16384)
                if not chunk:
                    break
                buf += chunk
                # Carve JPEG frames by SOI/EOI markers
                while True:
                    s = buf.find(b"\xFF\xD8")
                    e = buf.find(b"\xFF\xD9", s + 2) if s >= 0 else -1
                    if s >= 0 and e >= 0:
                        with self._lock:
                            self._frame = buf[s:e + 2]
                        buf = buf[e + 2:]
                    else:
                        break
        except Exception as exc:
            logger.warning(f"MJPEG capture error: {exc}")

    def ensure_running(self):
        if self._thread and self._thread.is_alive():
            return
        device = self._current_device()
        logger.info(f"Starting MJPEG capture from {device}")
        self._thread = threading.Thread(target=self._capture_loop, args=(device,), daemon=True)
        self._thread.start()

    def set_device(self, device: str):
        """Hot-swap to a new device: stop current ffmpeg, next ensure_running picks it up."""
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        with self._lock:
            self._frame = b""

    def set_data_provider(self, provider):
        """Attach a Woofalytics instance so the overlay can read live data."""
        self._data_provider = provider

    def _overlay_data(self):
        """Build the overlay-data dict from the attached provider."""
        dp = self._data_provider
        if dp is None:
            return None
        return {
            "doa":         dp._last_doa,
            "bark_prob":   dp._model_last_pred,
            "audio_level": dp._current_audio_level,
            "threshold":   dp._bark_prob_threshold,
        }

    def get_frame(self) -> bytes:
        with self._lock:
            return self._frame

    def stream_to(self, wfile):
        """Write a continuous MJPEG response to wfile until the client disconnects."""
        self.ensure_running()
        boundary = b"--woof_frame"
        try:
            while True:
                frame = self.get_frame()
                if frame:
                    # Apply HUD overlay (DOA compass, bark prob, audio levels)
                    if _draw_overlay:
                        data = self._overlay_data()
                        if data:
                            frame = _draw_overlay(frame, data)
                    wfile.write(boundary + b"\r\n")
                    wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    wfile.write(frame + b"\r\n")
                    wfile.flush()
                time.sleep(1 / 15)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected


_mjpeg = MJPEGStreamer()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Main")

LOG_PATH = "./log.txt"

import mqtt_manager as _mqtt_mod


# ── Clip slicing helper (with smart silence removal) ─────────────────────────

def _detect_sound_regions(clip_path, silence_thresh=-35, min_silence_dur=0.5):
    """Use ffmpeg silencedetect to find non-silent regions in an audio file.

    Returns list of (start, end) tuples in seconds.
    """
    cmd = [
        "ffmpeg", "-i", clip_path, "-af",
        f"silencedetect=noise={silence_thresh}dB:d={min_silence_dur}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr
    except Exception as exc:
        logger.warning(f"silencedetect failed: {exc}")
        return []

    # Parse silence_start / silence_end from stderr
    import re
    silence_starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", stderr)]
    silence_ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", stderr)]

    # Get total duration
    dur_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", stderr)
    if dur_match:
        total_dur = int(dur_match.group(1)) * 3600 + int(dur_match.group(2)) * 60 + float(dur_match.group(3))
    else:
        # Fallback: ffprobe
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", clip_path],
                capture_output=True, text=True, timeout=10,
            )
            total_dur = float(probe.stdout.strip())
        except Exception:
            total_dur = 60.0

    # Build sound regions (inverse of silence regions)
    sound_regions = []
    pos = 0.0

    for i, s_start in enumerate(silence_starts):
        if s_start > pos + 0.3:  # sound region must be at least 0.3s
            sound_regions.append((pos, s_start))
        if i < len(silence_ends):
            pos = silence_ends[i]
        else:
            pos = total_dur

    # Trailing sound after last silence
    if pos < total_dur - 0.3:
        sound_regions.append((pos, total_dur))

    # If no silence was detected, the whole file is sound
    if not silence_starts and total_dur > 0:
        sound_regions = [(0, total_dur)]

    return sound_regions


def _slice_clip(bark_logger, event_id, seg_seconds=5, smart=False,
                silence_thresh=-35):
    """Split a long clip into shorter segments using ffmpeg.

    If smart=True, uses silencedetect to skip silent sections.
    Each segment gets its own MP3 file and event record in the DB.
    The original event is kept untouched.
    """
    ev = bark_logger.get_event_by_id(event_id)
    if not ev:
        return {"error": f"Event {event_id} not found"}
    clip_path = ev.get("clip_path")
    if not clip_path or not os.path.isfile(clip_path):
        return {"error": "No clip file found for this event"}

    # Probe actual duration
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip_path],
            capture_output=True, text=True, timeout=10,
        )
        actual_dur = float(probe.stdout.strip())
    except Exception:
        actual_dur = ev.get("duration") or 60

    base_ts = ev.get("timestamp", "")
    base_name = os.path.splitext(os.path.basename(clip_path))[0]
    segment_ids = []

    if smart:
        # Smart mode: detect sound regions, then extract only those
        regions = _detect_sound_regions(clip_path, silence_thresh)
        if not regions:
            return {"error": "No sound detected in clip (all silence)"}

        for i, (start, end) in enumerate(regions):
            seg_dur = end - start
            if seg_dur < 0.5:
                continue  # too short

            # If a sound region is longer than seg_seconds, sub-split it
            sub_regions = []
            if seg_dur > seg_seconds * 1.5:
                pos = start
                while pos < end - 0.5:
                    sub_end = min(pos + seg_seconds, end)
                    sub_regions.append((pos, sub_end))
                    pos += seg_seconds
            else:
                sub_regions = [(start, end)]

            for j, (ss, se) in enumerate(sub_regions):
                out_path = f"./clips/{base_name}_v{i:02d}_{j:02d}.mp3"
                dur = se - ss
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}",
                    "-i", clip_path,
                    "-ac", "1", "-q:a", "4", out_path,
                ]
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=15)
                    if result.returncode != 0:
                        continue
                except Exception:
                    continue

                new_id = bark_logger.log_event(
                    timestamp=base_ts, clip_path=out_path,
                    bark_prob=ev.get("bark_prob", 0),
                    peak_dbfs=ev.get("peak_dbfs", -60),
                    avg_dbfs=ev.get("avg_dbfs", -60),
                    duration=round(dur, 1),
                    doa=ev.get("doa", 90), dog_id=ev.get("dog_id"),
                )
                if new_id:
                    segment_ids.append(new_id)
    else:
        # Dumb mode: fixed-length segments
        duration = ev.get("duration") or 0
        if actual_dur <= seg_seconds:
            return {"error": f"Clip is only {actual_dur:.0f}s, shorter than segment size {seg_seconds}s"}

        n_segments = int(actual_dur // seg_seconds)
        if actual_dur % seg_seconds >= 1.0:
            n_segments += 1

        for i in range(n_segments):
            start = i * seg_seconds
            out_path = f"./clips/{base_name}_s{i:03d}.mp3"
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start)]
            if i < n_segments - 1:
                cmd += ["-t", str(seg_seconds)]
            cmd += ["-i", clip_path, "-ac", "1", "-q:a", "4", out_path]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=15)
                if result.returncode != 0:
                    continue
            except Exception:
                continue

            seg_dur = min(seg_seconds, actual_dur - start)
            new_id = bark_logger.log_event(
                timestamp=base_ts, clip_path=out_path,
                bark_prob=ev.get("bark_prob", 0),
                peak_dbfs=ev.get("peak_dbfs", -60),
                avg_dbfs=ev.get("avg_dbfs", -60),
                duration=round(seg_dur, 1),
                doa=ev.get("doa", 90), dog_id=ev.get("dog_id"),
            )
            if new_id:
                segment_ids.append(new_id)

    return {"segments": segment_ids, "count": len(segment_ids),
            "segment_seconds": seg_seconds, "smart": smart}


# ── AI labeling helper ───────────────────────────────────────────────────────

import collections, datetime as _dt
_ai_label_log = collections.deque(maxlen=50)  # last 50 calls

def _ai_label_clip(bark_logger, event_id, settings):
    """Use Google Gemini to classify an audio clip as bark or not-bark."""
    api_key = settings.get("gemini_api_key", "")
    ts = _dt.datetime.now().isoformat()
    if not api_key:
        entry = {"ts": ts, "event_id": event_id, "status": "error", "error": "No API key"}
        _ai_label_log.append(entry)
        return {"error": "Gemini API key not configured. Set it in Settings."}

    ev = bark_logger.get_event_by_id(event_id)
    if not ev:
        return {"error": f"Event {event_id} not found"}
    clip_path = ev.get("clip_path")
    if not clip_path or not os.path.isfile(clip_path):
        return {"error": "No clip file found"}

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Upload the audio file
        with open(clip_path, "rb") as f:
            audio_data = f.read()

        gemini_model = settings.get("gemini_model", "gemini-2.5-flash")
        prompt = settings.get("gemini_prompt", "Is there a dog barking? Respond as JSON: {\"bark\": true/false, \"description\": \"...\"}")

        response = client.models.generate_content(
            model=gemini_model,
            contents=[
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/mpeg",
                                "data": __import__("base64").b64encode(audio_data).decode(),
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
        )

        # Parse response
        text = response.text.strip()
        # Handle markdown code block wrapping
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)

        # Support both old format {"label": "bark"} and new {"bark": true}
        if "bark" in result and isinstance(result["bark"], bool):
            is_bark = result["bark"]
        else:
            is_bark = result.get("label") == "bark"
        label_val = 1 if is_bark else 0
        description = result.get("description", "")

        out = {
            "event_id": event_id,
            "ai_label": "bark" if is_bark else "not_bark",
            "label": label_val,
            "confidence": result.get("confidence", 1.0),
            "description": description,
        }
        _ai_label_log.append({"ts": ts, "event_id": event_id, "status": "ok",
                              "clip": clip_path, "model": gemini_model, **out})
        return out
    except json.JSONDecodeError as exc:
        err = f"Failed to parse AI response: {text[:200]}"
        _ai_label_log.append({"ts": ts, "event_id": event_id, "status": "error",
                              "clip": clip_path, "error": err})
        return {"error": err}
    except Exception as exc:
        err = f"Gemini API error: {str(exc)}"
        _ai_label_log.append({"ts": ts, "event_id": event_id, "status": "error",
                              "clip": clip_path, "error": err})
        return {"error": err}


# ── Augmentation helper ──────────────────────────────────────────────────────

def _augment_clips(bark_logger, event_ids, augmentations):
    """Generate augmented versions of labelled clips using ffmpeg.

    augmentations: dict with keys:
      - gain: list of dB values (e.g. [-12, -6, 6, 12])
      - speed: list of speed factors (e.g. [0.9, 1.1])
      - reverb: bool
    """
    results = []
    gains = augmentations.get("gain", [])
    speeds = augmentations.get("speed", [])
    add_reverb = augmentations.get("reverb", False)

    for eid in event_ids:
        ev = bark_logger.get_event_by_id(eid)
        if not ev or not ev.get("clip_path") or ev.get("label") is None:
            continue
        clip_path = ev["clip_path"]
        if not os.path.isfile(clip_path):
            continue

        base_name = os.path.splitext(os.path.basename(clip_path))[0]
        label = ev["label"]
        base_ts = ev.get("timestamp", "")

        # Gain augmentations
        for g in gains:
            out = f"./clips/{base_name}_g{'+' if g >= 0 else ''}{g}dB.mp3"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", clip_path,
                "-af", f"volume={g}dB",
                "-ac", "1", "-q:a", "4", out,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
                if os.path.isfile(out):
                    new_id = bark_logger.log_event(
                        timestamp=base_ts, clip_path=out,
                        bark_prob=ev.get("bark_prob", 0),
                        peak_dbfs=ev.get("peak_dbfs", -60) + g,
                        avg_dbfs=ev.get("avg_dbfs", -60) + g,
                        duration=ev.get("duration", 0),
                        doa=ev.get("doa", 90), dog_id=ev.get("dog_id"),
                    )
                    if new_id:
                        bark_logger.set_label(new_id, label)
                        results.append(new_id)
            except Exception:
                pass

        # Speed augmentations
        for s in speeds:
            suffix = f"_sp{s:.1f}x"
            out = f"./clips/{base_name}{suffix}.mp3"
            # atempo only accepts 0.5-2.0
            tempo = max(0.5, min(2.0, s))
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", clip_path,
                "-af", f"atempo={tempo}",
                "-ac", "1", "-q:a", "4", out,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
                if os.path.isfile(out):
                    new_dur = (ev.get("duration", 0) or 0) / tempo
                    new_id = bark_logger.log_event(
                        timestamp=base_ts, clip_path=out,
                        bark_prob=ev.get("bark_prob", 0),
                        peak_dbfs=ev.get("peak_dbfs", -60),
                        avg_dbfs=ev.get("avg_dbfs", -60),
                        duration=round(new_dur, 1),
                        doa=ev.get("doa", 90), dog_id=ev.get("dog_id"),
                    )
                    if new_id:
                        bark_logger.set_label(new_id, label)
                        results.append(new_id)
            except Exception:
                pass

        # Reverb augmentation (simple aecho)
        if add_reverb:
            out = f"./clips/{base_name}_reverb.mp3"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", clip_path,
                "-af", "aecho=0.8:0.88:60:0.4",
                "-ac", "1", "-q:a", "4", out,
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
                if os.path.isfile(out):
                    new_id = bark_logger.log_event(
                        timestamp=base_ts, clip_path=out,
                        bark_prob=ev.get("bark_prob", 0),
                        peak_dbfs=ev.get("peak_dbfs", -60),
                        avg_dbfs=ev.get("avg_dbfs", -60),
                        duration=ev.get("duration", 0),
                        doa=ev.get("doa", 90), dog_id=ev.get("dog_id"),
                    )
                    if new_id:
                        bark_logger.set_label(new_id, label)
                        results.append(new_id)
            except Exception:
                pass

    return {"augmented_ids": results, "count": len(results)}


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request access logs

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        with open(path, "r") as f:
            self.wfile.write(f.read().encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))

        # ── Multipart file upload: /api/clips/import ──────────────────────
        if parsed.path == "/api/clips/import":
            import cgi
            import datetime as dt
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self._send_json({"error": "multipart/form-data required"}, 400)
                return
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST",
                         "CONTENT_TYPE": ctype,
                         "CONTENT_LENGTH": str(length)},
            )
            files = form["files"] if "files" in form else []
            if not isinstance(files, list):
                files = [files]
            os.makedirs("./clips", exist_ok=True)
            imported = 0
            for f in files:
                if not f.filename:
                    continue
                ext = os.path.splitext(f.filename)[1].lower()
                if ext not in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
                    continue
                dest = f"./clips/{time.time_ns()}{ext}"
                with open(dest, "wb") as out:
                    out.write(f.file.read())
                # Convert non-MP3 to MP3 for consistency
                if ext != ".mp3":
                    mp3_dest = dest.rsplit(".", 1)[0] + ".mp3"
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-loglevel", "error",
                             "-i", dest, "-ac", "1", "-q:a", "4", mp3_dest],
                            capture_output=True, timeout=30,
                        )
                        os.remove(dest)
                        dest = mp3_dest
                    except Exception:
                        pass  # keep original format
                # Probe duration
                try:
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "csv=p=0", dest],
                        capture_output=True, text=True, timeout=10,
                    )
                    dur = float(probe.stdout.strip())
                except Exception:
                    dur = 0
                wa._bark_logger.log_event(
                    timestamp=dt.datetime.now().isoformat(),
                    clip_path=dest, bark_prob=0, peak_dbfs=-30,
                    avg_dbfs=-30, duration=round(dur, 1), doa=90,
                )
                imported += 1
            self._send_json({"imported": imported})
            return

        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except Exception:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        if parsed.path == "/api/config":
            result = wa.set_config(data)
            self._send_json(result)
        elif parsed.path == "/api/dogs":
            name = data.get("name")
            dog_id = wa._bark_logger.create_dog(name=name)
            self._send_json({"dog_id": dog_id})
        elif parsed.path == "/api/stream":
            action = data.get("action")
            if action == "start":
                wa._streamer.start()
                self._send_json({"ok": True, "running": wa._streamer.is_running()})
            elif action == "stop":
                wa._streamer.stop()
                self._send_json({"ok": True, "running": False})
            else:
                self._send_json({"error": "action must be start or stop"}, 400)
        elif parsed.path == "/api/settings":
            # Save to persistent settings file; apply runtime-safe ones immediately
            updated = cfg_store.update(data)
            if "video_device" in data:
                _mjpeg.set_device(data["video_device"])   # hot-swap camera
            self._send_json(cfg_store.get_public())
        elif parsed.path == "/api/train":
            clips  = data.get("clips", [])   # [{path, label}, ...]
            mode   = data.get("mode", "fine_tune")
            epochs = int(data.get("epochs", 20))
            lr     = float(data.get("lr", 1e-3))
            result = _trainer_mod.get_job().start(clips, mode=mode, epochs=epochs, lr=lr)
            self._send_json(result)
        elif parsed.path.startswith("/api/clips/") and parsed.path.endswith("/slice"):
            # POST /api/clips/<id>/slice  { "segment_seconds": 5, "smart": true }
            parts = parsed.path.strip("/").split("/")
            try:
                event_id = int(parts[2])
            except (IndexError, ValueError):
                self._send_json({"error": "invalid event id"}, 400)
                return
            seg_sec = int(data.get("segment_seconds", 5))
            seg_sec = max(2, min(seg_sec, 30))
            smart = data.get("smart", False)
            silence_thresh = int(data.get("silence_thresh", -35))
            result = _slice_clip(wa._bark_logger, event_id, seg_sec,
                                 smart=smart, silence_thresh=silence_thresh)
            self._send_json(result)
        elif parsed.path == "/api/clips/slice-all":
            # POST /api/clips/slice-all — batch smart-slice all long clips
            settings = cfg_store.get()
            silence_thresh = int(data.get("silence_thresh",
                                          settings.get("silence_thresh_db", -35)))
            max_seg = float(data.get("max_segment_seconds",
                                     settings.get("max_segment_seconds", 8)))
            min_duration = float(data.get("min_duration", max_seg + 1))

            all_events = wa._bark_logger.get_recent_events(limit=99999)
            long_clips = [e for e in all_events
                          if (e.get("duration") or 0) > min_duration
                          and e.get("clip_path")
                          and os.path.isfile(e["clip_path"])]

            results = []
            for ev in long_clips:
                r = _slice_clip(wa._bark_logger, ev["id"],
                                seg_seconds=int(max_seg),
                                smart=True,
                                silence_thresh=silence_thresh)
                results.append({"event_id": ev["id"], **r})

            total_segs = sum(r.get("count", 0) for r in results)
            self._send_json({
                "processed": len(long_clips),
                "total_segments": total_segs,
                "results": results,
            })
        elif parsed.path == "/api/clips/denoise-all":
            # POST /api/clips/denoise-all — batch FFT denoise
            settings = cfg_store.get()
            nr_db = int(data.get("nr_db", settings.get("noise_reduction_db", 12)))
            all_events = wa._bark_logger.get_recent_events(limit=99999)
            processed = 0
            for ev in all_events:
                cp = ev.get("clip_path", "")
                if cp and cp.endswith(".mp3") and os.path.isfile(cp):
                    try:
                        wa._denoise_clip(cp, nr_db)
                        processed += 1
                    except Exception:
                        pass
            self._send_json({"processed": processed, "nr_db": nr_db})

        elif parsed.path == "/api/clips/filter-speech":
            # POST /api/clips/filter-speech — run VAD, delete clips with speech
            settings = cfg_store.get()
            thresh = float(data.get("speech_thresh",
                                    settings.get("speech_filter_thresh", 0.35)))
            all_events = wa._bark_logger.get_recent_events(limit=99999)
            deleted = 0
            checked = 0
            for ev in all_events:
                cp = ev.get("clip_path", "")
                if not cp or not os.path.isfile(cp):
                    continue
                checked += 1
                speech_pct = wa._has_speech(cp)
                if speech_pct > thresh:
                    # Delete clip file
                    try:
                        os.remove(cp)
                    except OSError:
                        pass
                    # Delete matching FLAC
                    flac = cp.replace(".mp3", "_4ch.flac")
                    try:
                        if os.path.isfile(flac):
                            os.remove(flac)
                    except OSError:
                        pass
                    # Delete DB event
                    wa._bark_logger.delete_event(ev["id"])
                    deleted += 1
            self._send_json({"checked": checked, "deleted": deleted,
                             "threshold": thresh})

        elif parsed.path == "/api/clips/clean-silence":
            # POST /api/clips/clean-silence — remove clips that are all silence
            settings = cfg_store.get()
            silence_thresh = int(data.get("silence_thresh",
                                          settings.get("silence_thresh_db", -35)))
            all_events = wa._bark_logger.get_recent_events(limit=99999)
            deleted = 0
            checked = 0
            for ev in all_events:
                cp = ev.get("clip_path", "")
                if not cp or not os.path.isfile(cp):
                    continue
                checked += 1
                regions = _detect_sound_regions(cp, silence_thresh)
                if not regions:
                    try:
                        os.remove(cp)
                    except OSError:
                        pass
                    flac = cp.replace(".mp3", "_4ch.flac")
                    try:
                        if os.path.isfile(flac):
                            os.remove(flac)
                    except OSError:
                        pass
                    wa._bark_logger.delete_event(ev["id"])
                    deleted += 1
            self._send_json({"checked": checked, "deleted": deleted})

        elif parsed.path == "/api/clips/import":
            # POST /api/clips/import — handled specially (multipart)
            # This is handled in do_POST with multipart parsing
            self._send_json({"error": "Use multipart upload"}, 400)

        # ── MQTT ───────────────────────────────────────────────────────────
        elif parsed.path == "/api/mqtt/test":
            result = _mqtt_mod.get_manager().test_connection(data)
            self._send_json(result)
        elif parsed.path == "/api/mqtt/configure":
            # Save MQTT settings and apply
            settings = cfg_store.update(data)
            _mqtt_mod.get_manager().configure(settings)
            self._send_json({"ok": True})

        # ── AI labeling ────────────────────────────────────────────────────
        elif parsed.path == "/api/ai/label":
            event_id = data.get("event_id")
            if not event_id:
                self._send_json({"error": "event_id required"}, 400)
                return
            settings = cfg_store.get()
            result = _ai_label_clip(wa._bark_logger, int(event_id), settings)
            if not result.get("error"):
                wa._bark_logger.set_label(int(event_id), result["label"])
                wa._bark_logger.set_ai_note(int(event_id), result.get("description", ""))
            self._send_json(result)
        elif parsed.path == "/api/ai/label-batch":
            event_ids = data.get("event_ids", [])
            if not event_ids:
                self._send_json({"error": "event_ids required"}, 400)
                return
            settings = cfg_store.get()
            results = []
            for eid in event_ids:
                r = _ai_label_clip(wa._bark_logger, int(eid), settings)
                if not r.get("error"):
                    wa._bark_logger.set_label(int(eid), r["label"])
                    wa._bark_logger.set_ai_note(int(eid), r.get("description", ""))
                results.append(r)
            self._send_json({"results": results, "count": len(results)})

        # ── Augmentation ───────────────────────────────────────────────────
        elif parsed.path == "/api/augment":
            event_ids = data.get("event_ids", [])
            augmentations = data.get("augmentations", {})
            if not event_ids:
                self._send_json({"error": "event_ids required"}, 400)
                return
            result = _augment_clips(wa._bark_logger, event_ids, augmentations)
            self._send_json(result)

        # ── Spectral bark score ─────────────────────────────────────────
        elif parsed.path == "/api/bark-score":
            from bark_score import compute_bark_score
            event_ids = data.get("event_ids", [])
            if not event_ids:
                self._send_json({"error": "event_ids required"}, 400)
                return
            results = []
            for eid in event_ids:
                ev = wa._bark_logger.get_event_by_id(int(eid))
                if ev and ev.get("clip_path"):
                    score_data = compute_bark_score(ev["clip_path"])
                    score_data["event_id"] = eid
                    results.append(score_data)
            self._send_json({"results": results})

        # ── Auto pre-label by model confidence ──────────────────────────
        elif parsed.path == "/api/auto-prelabel":
            bark_thresh = float(data.get("bark_threshold", 0.7))
            not_bark_thresh = float(data.get("not_bark_threshold", 0.2))
            event_ids = data.get("event_ids", [])
            count_bark, count_not = 0, 0
            for eid in event_ids:
                ev = wa._bark_logger.get_event_by_id(int(eid))
                if not ev or ev.get("label") is not None:
                    continue  # skip already-labeled
                prob = ev.get("bark_prob", 0)
                if prob is None:
                    continue
                if prob >= bark_thresh:
                    wa._bark_logger.set_label(int(eid), 1)
                    count_bark += 1
                elif prob <= not_bark_thresh:
                    wa._bark_logger.set_label(int(eid), 0)
                    count_not += 1
            self._send_json({
                "labeled_bark": count_bark,
                "labeled_not_bark": count_not,
                "skipped": len(event_ids) - count_bark - count_not,
            })

        else:
            self._send_json({"error": "not found"}, 404)

    def do_PATCH(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except Exception:
            self._send_json({"error": "invalid JSON"}, 400)
            return
        parts = parsed.path.strip("/").split("/")
        # PATCH /api/dogs/<dog_id>  { "name": "Rex" }
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "dogs":
            wa._bark_logger.rename_dog(parts[2], data["name"])
            self._send_json({"ok": True})
        # PATCH /api/events/<id>  { "dog_id": "Dog 2" }
        elif len(parts) == 3 and parts[0] == "api" and parts[1] == "events":
            eid = int(parts[2])
            if "dog_id" in data:
                wa._bark_logger.retag_event(eid, data["dog_id"])
            if "label" in data:
                lv = None if data["label"] is None else int(data["label"])
                wa._bark_logger.set_label(eid, lv)
            if "ai_note" in data:
                wa._bark_logger.set_ai_note(eid, data["ai_note"])
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        # DELETE /api/events/<id>
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "events":
            clip_path = wa._bark_logger.delete_event(int(parts[2]))
            if clip_path and os.path.isfile(clip_path):
                os.remove(clip_path)
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_HEAD(self):
        """Handle HEAD requests — needed for audio seeking (Range probes)."""
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/clips/"):
            filename = os.path.basename(path)
            filepath = os.path.join("./clips", filename)
            if os.path.isfile(filepath):
                ctype = "audio/mpeg" if filepath.endswith(".mp3") else "audio/wav"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(os.path.getsize(filepath)))
                self.end_headers()
                return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # ── Pages ─────────────────────────────────────────────────────────────
        if path == "/":
            self._send_html("./html/main.html")
        elif path == "/analytics":
            self._send_html("./html/analytics.html")
        elif path == "/debug":
            self._send_html("./html/debug.html")
        elif path == "/config":
            self._send_html("./html/config.html")
        elif path == "/library":
            self._send_html("./html/library.html")
        elif path == "/stream":
            self._send_html("./html/stream.html")
        elif path == "/rec":
            self._send_html("./html/record.html")
        elif path == "/train":
            self._send_html("./html/train.html")
        elif path == "/mqtt":
            self._send_html("./html/mqtt.html")
        elif path == "/augment":
            self._send_html("./html/augment.html")

        elif path == "/nav.js":
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript")
            self.end_headers()
            with open("./html/nav.js", "rb") as f:
                self.wfile.write(f.read())

        # ── Record button ──────────────────────────────────────────────────────
        elif path.startswith("/store-record"):
            button = qs.get("button", [None])[0]
            if button == "rec":
                wa.store_clip()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                count = len(list(glob.glob("./clips/*.wav"))) + len(list(glob.glob("./clips/*.mp3")))
                self.wfile.write(f"Recorded! {count} clips in storage.".encode())
            else:
                self.send_response(404)
                self.end_headers()

        # ── JSON APIs ─────────────────────────────────────────────────────────
        elif path == "/api/bark":
            json_data = wa.get_last_pred().copy()
            probs = json_data.get("bark_probability", [])
            json_data["bark_probability"] = max(probs) if probs else 0.0
            self._send_json(json_data)

        elif path == "/api/analytics":
            self._send_json({
                "dogs":   wa._bark_logger.get_dog_stats(),
                "totals": wa._bark_logger.get_analytics(),
            })

        elif path == "/api/dogs":
            self._send_json(wa._bark_logger.get_all_dogs())

        elif path == "/api/events":
            limit  = int(qs.get("limit",  ["50"])[0])
            dog_id = qs.get("dog_id", [None])[0]
            self._send_json(wa._bark_logger.get_recent_events(limit=limit, dog_id=dog_id))

        elif path == "/api/debug":
            self._send_json(wa.get_debug_info())

        elif path == "/api/ai/log":
            self._send_json({"calls": list(_ai_label_log), "total": len(_ai_label_log)})

        elif path == "/api/devices":
            self._send_json(wa.list_audio_devices())

        elif path == "/api/config":
            self._send_json(wa.get_config())

        elif path == "/api/stream":
            self._send_json(wa._streamer.get_status())

        elif path == "/api/upload":
            self._send_json(wa._uploader.get_status())

        elif path == "/api/settings":
            self._send_json(cfg_store.get_public())

        elif path == "/api/devices/video":
            # List /dev/video* devices with names from v4l2-ctl
            import glob as _g
            import subprocess as _sp
            devs = sorted(_g.glob("/dev/video*"))
            # Try to get device names
            name_map = {}
            try:
                out = _sp.check_output(
                    ["v4l2-ctl", "--list-devices"],
                    stderr=_sp.STDOUT, timeout=5
                ).decode()
                current_name = ""
                for line in out.splitlines():
                    line = line.rstrip()
                    if not line:
                        continue
                    if not line.startswith("\t") and not line.startswith(" "):
                        current_name = line.rstrip(":")
                    else:
                        dev = line.strip()
                        if dev.startswith("/dev/video"):
                            name_map[dev] = current_name
            except Exception:
                pass
            result = []
            for d in devs:
                entry = {"path": d}
                if d in name_map:
                    entry["name"] = name_map[d]
                result.append(entry)
            self._send_json(result)

        elif path == "/api/log":
            n = int(qs.get("lines", ["40"])[0])
            lines = []
            if os.path.isfile(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    lines = f.readlines()
            self._send_json({"lines": lines[-n:]})

        elif path == "/api/train":
            self._send_json(_trainer_mod.get_job().get_status())

        elif path == "/api/train/clips":
            # Return all events available for labelling
            events = wa._bark_logger.get_recent_events(limit=500)
            self._send_json(events)

        elif path == "/api/mqtt/status":
            self._send_json(_mqtt_mod.get_manager().get_status())

        # ── Camera MJPEG stream ───────────────────────────────────────────────
        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=woof_frame")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            _mjpeg.stream_to(self.wfile)

        # ── Clip audio files (Range-request aware for seeking) ────────────
        elif path.startswith("/clips/"):
            filename = os.path.basename(path)
            filepath = os.path.join("./clips", filename)
            if os.path.isfile(filepath):
                ctype = "audio/mpeg" if filepath.endswith(".mp3") else "audio/wav"
                file_size = os.path.getsize(filepath)
                range_hdr = self.headers.get("Range")

                if range_hdr and range_hdr.startswith("bytes="):
                    # Parse "bytes=START-" or "bytes=START-END"
                    range_spec = range_hdr[6:]
                    parts = range_spec.split("-", 1)
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if (len(parts) > 1 and parts[1]) else file_size - 1
                    end = min(end, file_size - 1)
                    length = end - start + 1

                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.end_headers()
                    with open(filepath, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(length))
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(file_size))
                    self.end_headers()
                    with open(filepath, "rb") as f:
                        self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


def term_handler(signum, frame):
    logger.info("Ctrl+C pressed.")
    wa.stop()
    exit(1)


signal.signal(signal.SIGINT, term_handler)
wa = Woofalytics()
_mjpeg.set_data_provider(wa)


def run_server(handler_class=RequestHandler, port=8000):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


def main():
    logger.info("Starting Woofalytics server, press Ctrl+C to stop...")
    wa.start()

    # Initialize MQTT from saved settings
    try:
        settings = cfg_store.get()
        _mqtt_mod.get_manager().configure(settings)
    except Exception as exc:
        logger.warning(f"MQTT init: {exc}")

    run_server()


if __name__ == "__main__":
    main()
