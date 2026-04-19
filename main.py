import logging
import json
import os
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import glob
import settings as cfg_store
from record import Woofalytics

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Main")

LOG_PATH = "./log.txt"


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
            # Apply bark_quiet_seconds immediately if changed
            if "auto_stream" in data:
                pass  # schedule handled by stream page
            self._send_json(cfg_store.get_public())
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
            wa._bark_logger.retag_event(int(parts[2]), data["dog_id"])
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
                count = len(list(glob.glob("./clips/*.wav")))
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
            # List /dev/video* devices visible in the container
            import glob as _g
            devs = sorted(_g.glob("/dev/video*"))
            self._send_json([{"path": d} for d in devs])

        elif path == "/api/log":
            n = int(qs.get("lines", ["40"])[0])
            lines = []
            if os.path.isfile(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    lines = f.readlines()
            self._send_json({"lines": lines[-n:]})

        # ── Clip audio files ──────────────────────────────────────────────────
        elif path.startswith("/clips/"):
            filename = os.path.basename(path)
            filepath = os.path.join("./clips", filename)
            if os.path.isfile(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Accept-Ranges", "bytes")
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


def run_server(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


def main():
    logger.info("Starting Woofalytics server, press Ctrl+C to stop...")
    wa.start()
    run_server()


if __name__ == "__main__":
    main()
