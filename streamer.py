"""
YouTubeStreamer — manages a persistent ffmpeg RTMP process.

Audio conflict avoidance: PyAudio owns the ALSA mic exclusively for bark
detection. ffmpeg uses a silent lavfi audio source so there is no device
contention. The on-screen OSD (text overlay) communicates bark state instead.
"""
import datetime
import logging
import os
import subprocess
import threading
import time

import settings as cfg_store

OSD_FILE  = "/tmp/woof_osd.txt"
RTMP_BASE = "rtmp://a.rtmp.youtube.com/live2"


class YouTubeStreamer:
    def __init__(self):
        self._logger = logging.getLogger("YouTubeStreamer")
        self._process   = None
        self._thread    = None
        self._stop_flag = True          # starts stopped; call start() to go live
        self._reset_timer = None
        self._started_at  = None
        self._last_error  = ""
        self._write_osd(barking=False)

    # ── Settings accessors (always read from live settings) ───────────────────

    def _key(self):        return cfg_store.get()["stream_key"]
    def _video_dev(self):  return cfg_store.get()["video_device"]
    def _quiet_sec(self):  return cfg_store.get()["bark_quiet_seconds"]
    def _resolution(self): return cfg_store.get()["stream_resolution"]
    def _fps(self):        return cfg_store.get()["stream_fps"]
    def _bitrate(self):    return cfg_store.get()["stream_bitrate_kbps"]

    # ── OSD helpers ───────────────────────────────────────────────────────────

    def _write_osd(self, barking=False, prob=0.0, dog_id=None):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if barking:
            dog = f" | {dog_id}" if dog_id else ""
            text = f"BARKING! Prob: {prob:.0%}{dog} | {ts}"
        else:
            text = f"Monitoring... | {ts}"
        try:
            with open(OSD_FILE, "w") as f:
                f.write(text)
        except Exception as exc:
            self._logger.warning(f"OSD write failed: {exc}")

    def set_state(self, barking: bool, prob: float = 0.0, dog_id=None):
        """Called from record.py on every bark event or quiet period."""
        self._write_osd(barking=barking, prob=prob, dog_id=dog_id)
        if barking:
            if self._reset_timer:
                self._reset_timer.cancel()
            self._reset_timer = threading.Timer(
                self._quiet_sec(), lambda: self._write_osd(barking=False)
            )
            self._reset_timer.daemon = True
            self._reset_timer.start()

    # ── ffmpeg command ────────────────────────────────────────────────────────

    def _cmd(self):
        res  = self._resolution()
        fps  = self._fps()
        kbps = self._bitrate()
        return [
            "ffmpeg", "-loglevel", "warning",
            # Video: webcam
            "-f", "v4l2", "-video_size", res, "-framerate", str(fps),
            "-i", self._video_dev(),
            # Audio: silent source (avoids ALSA device conflict with PyAudio)
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            # OSD text overlay — re-reads file every frame (reload=1)
            "-vf", (
                f"drawtext=textfile={OSD_FILE}:reload=1"
                ":fontsize=36:fontcolor=white"
                ":box=1:boxcolor=black@0.55:boxborderw=10"
                ":x=20:y=20"
            ),
            # Encode
            "-c:v", "libx264", "-preset", "veryfast",
            f"-b:v", f"{kbps}k", f"-maxrate", f"{int(kbps*1.2)}k",
            f"-bufsize", f"{kbps*2}k",
            "-pix_fmt", "yuv420p", "-g", str(fps * 2),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-map", "0:v", "-map", "1:a",
            "-f", "flv", f"{RTMP_BASE}/{self._key()}",
        ]

    # ── Process loop ──────────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop_flag:
            if not self._key():
                self._logger.warning("Stream key not set — sleeping 30 s.")
                self._last_error = "No stream key configured."
                time.sleep(30)
                continue
            self._logger.info("Starting ffmpeg → YouTube Live...")
            self._last_error = ""
            try:
                self._process = subprocess.Popen(
                    self._cmd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                for line in self._process.stdout:
                    line = line.rstrip()
                    self._logger.debug(f"ffmpeg: {line}")
                    if "error" in line.lower():
                        self._last_error = line
                    if self._stop_flag:
                        break
                self._process.wait()
                if not self._stop_flag:
                    self._logger.warning(
                        f"ffmpeg exited ({self._process.returncode}). Restarting in 5 s..."
                    )
                    self._last_error = f"ffmpeg exited with code {self._process.returncode}"
                    time.sleep(5)
            except Exception as exc:
                self._logger.error(f"ffmpeg error: {exc}")
                self._last_error = str(exc)
                time.sleep(5)
        self._started_at = None

    # ── Public start / stop ───────────────────────────────────────────────────

    def start(self):
        if not self._stop_flag:
            return  # already running
        self._stop_flag = False
        self._started_at = datetime.datetime.now().isoformat()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._logger.info("YouTubeStreamer started.")

    def stop(self):
        self._stop_flag = True
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
        if self._reset_timer:
            self._reset_timer.cancel()
        self._logger.info("YouTubeStreamer stopped.")

    def is_running(self):
        return not self._stop_flag and self._thread is not None and self._thread.is_alive()

    def get_status(self):
        return {
            "running":     self.is_running(),
            "started_at":  self._started_at,
            "last_error":  self._last_error,
            "stream_key_set": bool(self._key()),
            "video_device":   self._video_dev(),
            "resolution":     self._resolution(),
            "fps":            self._fps(),
            "bitrate_kbps":   self._bitrate(),
        }
