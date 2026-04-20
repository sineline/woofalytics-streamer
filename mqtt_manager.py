"""
mqtt_manager.py — MQTT event publisher for Woofalytics.

Publishes bark detection events to a configurable MQTT broker.
Supports optional TLS and Home Assistant MQTT auto-discovery.
"""

import json
import logging
import threading
import time
from typing import Optional

_logger = logging.getLogger("MQTT")

try:
    import paho.mqtt.client as mqtt

    _HAS_MQTT = True
except ImportError:
    _HAS_MQTT = False
    _logger.warning("paho-mqtt not installed — MQTT publishing disabled")


class MQTTManager:
    """Thread-safe MQTT client wrapper."""

    def __init__(self):
        self._lock = threading.Lock()
        self._client: Optional[object] = None
        self._connected = False
        self._config = {}
        self._last_error = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def configure(self, settings: dict):
        """Apply MQTT settings. Reconnects if broker/credentials changed."""
        with self._lock:
            new_cfg = {
                "enabled": settings.get("mqtt_enabled", False),
                "broker": settings.get("mqtt_broker", ""),
                "port": int(settings.get("mqtt_port", 1883)),
                "username": settings.get("mqtt_username", ""),
                "password": settings.get("mqtt_password", ""),
                "tls": settings.get("mqtt_tls", False),
                "topic": settings.get("mqtt_topic", "woofalytics/bark"),
                "ha_discovery": settings.get("mqtt_ha_discovery", False),
            }

            # Check if reconnect is needed
            broker_changed = (
                new_cfg["broker"] != self._config.get("broker")
                or new_cfg["port"] != self._config.get("port")
                or new_cfg["username"] != self._config.get("username")
                or new_cfg["tls"] != self._config.get("tls")
            )
            was_enabled = self._config.get("enabled", False)
            self._config = new_cfg

        if not new_cfg["enabled"]:
            self.disconnect()
            return

        if broker_changed or not was_enabled:
            self.disconnect()
            self._connect()

    def publish_bark(self, event_data: dict):
        """Publish a bark event. Non-blocking, fails silently."""
        if not self._connected or not self._config.get("enabled"):
            return
        topic = self._config.get("topic", "woofalytics/bark")
        try:
            payload = json.dumps({
                "event": "bark",
                "probability": event_data.get("bark_prob", 0),
                "peak_dbfs": event_data.get("peak_dbfs", -60),
                "avg_dbfs": event_data.get("avg_dbfs", -60),
                "doa": event_data.get("doa", 90),
                "timestamp": event_data.get("timestamp", ""),
                "clip_path": event_data.get("clip_path", ""),
            })
            if self._client:
                self._client.publish(topic, payload, qos=1)
                _logger.debug(f"Published bark to {topic}")
        except Exception as exc:
            _logger.warning(f"MQTT publish failed: {exc}")

    def test_connection(self, settings: dict) -> dict:
        """Test MQTT connection with given settings. Returns status dict."""
        if not _HAS_MQTT:
            return {"ok": False, "error": "paho-mqtt not installed"}

        broker = settings.get("mqtt_broker", "")
        port = int(settings.get("mqtt_port", 1883))
        if not broker:
            return {"ok": False, "error": "No broker host specified"}

        try:
            test_client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="woofalytics-test",
            )
            username = settings.get("mqtt_username", "")
            password = settings.get("mqtt_password", "")
            if username:
                test_client.username_pw_set(username, password)
            if settings.get("mqtt_tls", False):
                test_client.tls_set()

            result = {"ok": False, "error": "Connection timeout"}
            event = threading.Event()

            def on_connect(client, userdata, flags, rc, properties=None):
                if rc == 0:
                    result["ok"] = True
                    result["error"] = ""
                else:
                    result["error"] = f"Connection refused (rc={rc})"
                event.set()

            test_client.on_connect = on_connect
            test_client.connect_async(broker, port, keepalive=5)
            test_client.loop_start()
            event.wait(timeout=5)
            test_client.loop_stop()
            test_client.disconnect()
            return result
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def get_status(self) -> dict:
        """Return current MQTT status."""
        return {
            "available": _HAS_MQTT,
            "enabled": self._config.get("enabled", False),
            "connected": self._connected,
            "broker": self._config.get("broker", ""),
            "topic": self._config.get("topic", ""),
            "last_error": self._last_error,
        }

    def disconnect(self):
        """Disconnect from MQTT broker."""
        with self._lock:
            if self._client:
                try:
                    self._client.loop_stop()
                    self._client.disconnect()
                except Exception:
                    pass
                self._client = None
            self._connected = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self):
        """Connect to the MQTT broker."""
        if not _HAS_MQTT:
            return

        broker = self._config.get("broker", "")
        port = self._config.get("port", 1883)
        if not broker:
            return

        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="woofalytics",
            )
            username = self._config.get("username", "")
            password = self._config.get("password", "")
            if username:
                client.username_pw_set(username, password)
            if self._config.get("tls"):
                client.tls_set()

            # Set LWT (Last Will and Testament) for availability
            avail_topic = self._config.get("topic", "woofalytics") + "/availability"
            client.will_set(avail_topic, "offline", qos=1, retain=True)

            def on_connect(cl, userdata, flags, rc, properties=None):
                if rc == 0:
                    self._connected = True
                    self._last_error = ""
                    _logger.info(f"MQTT connected to {broker}:{port}")
                    # Publish availability
                    cl.publish(avail_topic, "online", qos=1, retain=True)
                    # Publish HA discovery if enabled
                    if self._config.get("ha_discovery"):
                        self._publish_ha_discovery(cl)
                else:
                    self._connected = False
                    self._last_error = f"Connection refused (rc={rc})"
                    _logger.warning(self._last_error)

            def on_disconnect(cl, userdata, flags, rc, properties=None):
                self._connected = False
                if rc != 0:
                    self._last_error = f"Unexpected disconnect (rc={rc})"
                    _logger.warning(self._last_error)

            client.on_connect = on_connect
            client.on_disconnect = on_disconnect
            client.connect_async(broker, port, keepalive=60)
            client.loop_start()

            with self._lock:
                self._client = client

        except Exception as exc:
            self._last_error = str(exc)
            _logger.error(f"MQTT connect failed: {exc}")

    def _publish_ha_discovery(self, client):
        """Publish Home Assistant MQTT auto-discovery config."""
        topic_base = self._config.get("topic", "woofalytics/bark")
        discovery_topic = "homeassistant/binary_sensor/woofalytics_bark/config"
        payload = json.dumps({
            "name": "Woofalytics Bark Detector",
            "unique_id": "woofalytics_bark_sensor",
            "state_topic": topic_base,
            "availability_topic": topic_base + "/availability",
            "device_class": "sound",
            "value_template": "{{ 'ON' if value_json.event == 'bark' else 'OFF' }}",
            "json_attributes_topic": topic_base,
            "device": {
                "identifiers": ["woofalytics"],
                "name": "Woofalytics",
                "manufacturer": "Woofalytics",
                "model": "Bark Detector v2",
            },
        })
        client.publish(discovery_topic, payload, qos=1, retain=True)
        _logger.info("Published Home Assistant discovery config")


# ── Singleton ─────────────────────────────────────────────────────────────────
_manager = MQTTManager()


def get_manager() -> MQTTManager:
    return _manager
