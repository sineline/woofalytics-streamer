"""
Settings — persistent JSON config stored in the Docker volume.
Values here survive container restarts and override docker-compose.yml defaults.
"""
import json
import os
import threading
import logging

SETTINGS_PATH = os.environ.get("SETTINGS_FILE", "./clips/settings.json")

_DEFAULT = {
    "stream_key":         os.environ.get("YOUTUBE_STREAM_KEY", ""),
    "video_device":       os.environ.get("VIDEO_DEVICE", "/dev/video2"),
    "stream_resolution":  "1280x720",
    "stream_fps":         30,
    "stream_bitrate_kbps": 2500,
    "bark_quiet_seconds": int(os.environ.get("BARK_QUIET_SECONDS", "10")),
    # auto_stream: "always" | "on_bark" | "scheduled" | "off"
    "auto_stream":        "off",
    "schedule_start":     "08:00",
    "schedule_end":       "22:00",
    "mic_device_index":   -1,
    "mic_channels":       int(os.environ.get("MIC_CHANNELS", "4")),
    "mic_sample_rate":    int(os.environ.get("MIC_SAMPLE_RATE", "16000")),
    "mic_array_spacing":  float(os.environ.get("MIC_ARRAY_SPACING", "0.1")),
    # ── Archive upload ──────────────────────────────────────────────────────
    # upload_mode: "off" | "s3" | "sftp"
    "upload_mode":          "off",
    # S3-compatible (AWS, B2, MinIO, Wasabi)
    "upload_s3_endpoint":   "",       # blank = AWS; or https://s3.us-west-002.backblazeb2.com
    "upload_s3_bucket":     "",
    "upload_s3_prefix":     "woofalytics/",
    "upload_s3_key":        "",
    "upload_s3_secret":     "",
    # SFTP
    "upload_sftp_host":     "",
    "upload_sftp_port":     22,
    "upload_sftp_user":     "",
    "upload_sftp_password": "",
    "upload_sftp_key_path": "",
    "upload_sftp_dir":      "/upload/woofalytics",
    # ── MQTT ────────────────────────────────────────────────────────────────
    "mqtt_enabled":         False,
    "mqtt_broker":          "",
    "mqtt_port":            1883,
    "mqtt_username":        "",
    "mqtt_password":        "",
    "mqtt_tls":             False,
    "mqtt_topic":           "woofalytics/bark",
    "mqtt_ha_discovery":    False,
    # ── AI labeling (Google Gemini) ─────────────────────────────────────────
    "gemini_api_key":       "",
}

_lock = threading.Lock()
_logger = logging.getLogger("Settings")


def _load() -> dict:
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
            # Merge with defaults so new keys are always present
            merged = {**_DEFAULT, **data}
            return merged
        except Exception as e:
            _logger.warning(f"Settings load failed: {e}")
    return dict(_DEFAULT)


def _save(data: dict):
    os.makedirs(os.path.dirname(os.path.abspath(SETTINGS_PATH)), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get() -> dict:
    with _lock:
        return _load()


def update(patch: dict) -> dict:
    with _lock:
        data = _load()
        data.update(patch)
        _save(data)
        return data


def get_public() -> dict:
    """Return settings safe to expose to the browser (mask secrets)."""
    s = get()
    s_pub = dict(s)
    s_pub["stream_key_set"] = bool(s_pub.get("stream_key", ""))
    s_pub["stream_key"]     = ""   # never send raw key to browser
    s_pub["upload_s3_secret_set"] = bool(s_pub.get("upload_s3_secret", ""))
    s_pub["upload_s3_secret"]     = ""
    s_pub["upload_sftp_password_set"] = bool(s_pub.get("upload_sftp_password", ""))
    s_pub["upload_sftp_password"]     = ""
    s_pub["mqtt_password_set"]        = bool(s_pub.get("mqtt_password", ""))
    s_pub["mqtt_password"]            = ""
    s_pub["gemini_api_key_set"]       = bool(s_pub.get("gemini_api_key", ""))
    s_pub["gemini_api_key"]           = ""
    return s_pub
