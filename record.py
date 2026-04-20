import logging
import os
import time
import threading
import subprocess
import datetime

import pyaudio
import wave
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import requests

from pyargus.directionEstimation import (
    gen_ula_scanning_vectors,
    corr_matrix_estimate,
    DOA_Bartlett,
    DOA_Capon,
    DOA_MEM,
)

from event_filter import EventFilter
import settings as cfg_store
from logger import BarkLogger
from streamer import YouTubeStreamer
from uploader import Uploader

# Replace with your unique event name and IFTTT Webhooks API key
IFTTT_EVENT_NAME = "woof"
IFTTT_KEY = "YOUR_IFTTT_WEBHOOKS_KEY"

last_preds = []


class Woofalytics:
    def __init__(self, clip_past_context_seconds=15, clip_future_context_seconds=15):
        self._logger = logging.getLogger("Woofalytics")
        self._recording_device_index = self.find_andrea_mic_array()

        self._sample_format = pyaudio.paInt16  # 16 bits per sample
        # MIC_CHANNELS=1 for mono webcam mics; 2 for the Andrea stereo array
        self._channels = int(os.environ.get("MIC_CHANNELS", "2"))
        # MIC_SAMPLE_RATE: use 48000 for USB webcam mics, 44100 for Andrea array
        self._fs = int(os.environ.get("MIC_SAMPLE_RATE", "44100"))
        # Chunk = 10ms of audio at the configured sample rate
        self._chunk = int(self._fs * 0.01)
        self._model_sample_rate = 16_000

        self._clip_past_context_seconds = clip_past_context_seconds
        self._clip_future_context_seconds = clip_future_context_seconds

        self._store_flag = False
        self._stop_flag = False

        self._buffer = []

        self._worker_thread = None

        self.set_mic_volume()

        self._model = torch.jit.load("./models/traced_model.pt")
        self._model.eval()
        self._logger.debug(f"Model loaded as: {self._model}")

        # Detect model version by probing input shape
        self._model_version = self._detect_model_version()
        self._logger.info(f"Model version: V{self._model_version}")

        # Pre-build the resampler once (avoids recreating it on every inference call)
        self._resampler = T.Resample(self._fs, self._model_sample_rate, dtype=torch.float32)

        if self._model_version == 2:
            self._model_window_size = 50
            self._model_window_overlap = 25
            # Need ~550ms of 16kHz audio (after resample) to produce 50 fbank frames
            # At 44100Hz with 441 chunk: each chunk is 10ms, need ~55 chunks
            self._infer_chunk_count = 55
        else:
            self._model_window_size = 6
            self._model_window_overlap = 3
            self._infer_chunk_count = 8

        self._model_last_pred = {
            "datetime": datetime.datetime.now().isoformat(),
            "bark_probability": [],
        }
        self._pred_lock = threading.Lock()

        self.ef = EventFilter()

        self._bark_prob_threshold = 0.88

        # Auto-save: save a clip for every confirmed bark, with a cooldown
        self._auto_save_cooldown = int(os.environ.get("AUTO_SAVE_COOLDOWN", "30"))
        self._last_auto_save_time = 0.0
        self._store_bark_info = {"prob": 0.0, "peak_dbfs": -60.0, "avg_dbfs": -60.0, "doa": 90.0}

        # Telemetry exposed to the debug/config pages
        self._start_time = time.time()
        self._current_audio_level = {"peak_dbfs": -60.0, "avg_dbfs": -60.0, "updated_at": ""}
        self._last_doa = {"doa1": 90, "doa2": 90, "doa3": 90}

        # Logging + streaming
        self._bark_logger = BarkLogger()
        self._uploader    = Uploader(bark_logger=self._bark_logger)
        self._streamer    = YouTubeStreamer()
        # Stream is started on-demand via /api/stream — not auto-started here.
        # It will auto-start if YOUTUBE_STREAM_KEY was set in env at boot.
        if cfg_store.get()["stream_key"]:
            self._streamer.start()

        # DOA — use all available mic channels for better angular resolution.
        # PS Eye has 4 mics (~13 mm spacing); Andrea has 2 (~40 mm spacing).
        # d is inter-element spacing as a fraction of wavelength (tune per device).
        d = float(os.environ.get("MIC_ARRAY_SPACING", "0.1"))
        M = self._channels if self._channels >= 2 else 2
        array_alignment = np.arange(0, M, 1) * d
        incident_angles = np.arange(0, 181, 1)
        self.ula_scanning_vectors = gen_ula_scanning_vectors(
            array_alignment, incident_angles
        )
        self._logger.info(
            f"DOA: {M}-element ULA, spacing={d}λ — using {self._channels} mic channel(s)"
        )

    def find_andrea_mic_array(self) -> int:
        """Find the preferred microphone.

        Searches for a device whose name contains the MIC_DEVICE_HINT env var
        (default: 'Andrea PureAudio'). Falls back to the first available input
        device so the container works with any USB microphone.
        """
        mic_hint = os.environ.get("MIC_DEVICE_HINT", "Andrea PureAudio")
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get("deviceCount")

        first_input_index = -1
        first_input_name = None
        hint_match_index = -1

        for i in range(numdevices):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxInputChannels") > 0:
                name = device_info.get("name")
                self._logger.debug(f"Device index {i}: {name}")
                if first_input_index == -1:
                    first_input_index = i
                    first_input_name = name
                if mic_hint.lower() in name.lower():
                    self._logger.info(f"Found matching mic '{name}' at index {i}")
                    hint_match_index = i
                    break

        p.terminate()

        if hint_match_index != -1:
            return hint_match_index

        if first_input_index != -1:
            self._logger.warning(
                f"No device matching '{mic_hint}' found. "
                f"Falling back to '{first_input_name}' at index {first_input_index}."
            )
            return first_input_index

        self._logger.error("No input recording devices found.")
        return -1

    def set_mic_volume(self, volume_percentage: int = 75):
        """Set capture volume via amixer. Non-fatal: logs a warning if amixer fails
        (e.g. inside a container without the right ALSA card exposed)."""
        try:
            output = subprocess.check_output("amixer get Capture".split(), text=True)
            self._logger.debug(output)
            output = subprocess.check_output(
                f"amixer set Capture {volume_percentage}% unmute".split(), text=True
            )
            self._logger.info(output)
            output = subprocess.check_output("amixer get Capture".split(), text=True)
            self._logger.debug(output)
        except Exception as exc:
            self._logger.warning(f"Could not set mic volume via amixer: {exc}")

    def start(self):
        self._worker_thread = threading.Thread(target=self._recording_worker)
        self._worker_thread.start()

    def _recording_worker(self):
        past_frames_count = int(
            self._fs / self._chunk * self._clip_past_context_seconds * self._channels
        )
        future_frames_count = int(
            self._fs / self._chunk * self._clip_future_context_seconds * self._channels
        )

        self._logger.info("Starting recording loop...")
        self._logger.debug(
            f"Clip past context seconds: {self._clip_past_context_seconds}, number of frames: {past_frames_count}"
        )
        self._logger.debug(
            f"Clip future context seconds: {self._clip_future_context_seconds}, number of frames: {future_frames_count}"
        )

        p = pyaudio.PyAudio()
        stream = p.open(
            format=self._sample_format,
            channels=self._channels,
            rate=self._fs,
            frames_per_buffer=self._chunk,
            input=True,
            input_device_index=self._recording_device_index,
        )

        self._sample_size = p.get_sample_size(self._sample_format)

        record_buffer = []
        infer_buffer = []

        # how many samples for window length of 6?
        window_len_samples = int(self._fs * self._model_window_size / 1000.0)
        window_shift_samples = int(self._fs * self._model_window_overlap / 1000.0)
        self._logger.info(
            f"Window len #samples: {window_len_samples}, overlap #samples: {window_shift_samples}"
        )

        while not self._stop_flag:
            try:
                data = stream.read(self._chunk, exception_on_overflow=False)
            except OSError as ex:
                self._logger.exception(ex)
                # Terminate the PortAudio interface
                p.terminate()

                p = pyaudio.PyAudio()
                stream = p.open(
                    format=self._sample_format,
                    channels=self._channels,
                    rate=self._fs,
                    frames_per_buffer=self._chunk,
                    input=True,
                    input_device_index=self._recording_device_index,
                )

                data = stream.read(self._chunk, exception_on_overflow=False)

            record_buffer.append(data)

            # infer:
            infer_buffer.append(data)

            if len(infer_buffer) >= self._infer_chunk_count:
                self.infer_chunk(infer_buffer.copy())
                infer_buffer = []

            # record:
            if not self._store_flag:  # we just keep past frames in buffer
                if (
                    len(record_buffer) > past_frames_count
                ):  # discard some earlier frames
                    record_buffer = record_buffer[-past_frames_count:]
            else:  # got a signal to store the frames
                if (
                    len(record_buffer) >= past_frames_count + future_frames_count
                ):  # have enought frames to dump to a file
                    info = self._store_bark_info
                    self._dump_file(
                        record_buffer.copy(),
                        bark_prob=info["prob"],
                        peak_dbfs=info["peak_dbfs"],
                        avg_dbfs=info["avg_dbfs"],
                        doa=info["doa"],
                    )
                    record_buffer = record_buffer[-past_frames_count:]

                    self._store_flag = False
                else:  # keep recording until the desired len is reached
                    pass

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

    def stop(self):
        self._stop_flag = True
        self._streamer.stop()
        if self._worker_thread:
            self._worker_thread.join()

    def store_clip(self):
        self._logger.info("Got a store request...")
        self._store_flag = True

    def _dump_file(self, frames, bark_prob=0.0, peak_dbfs=-60.0, avg_dbfs=-60.0, doa=90.0):
        t = threading.Thread(
            target=self._dump_worker, args=[frames, bark_prob, peak_dbfs, avg_dbfs, doa]
        )
        t.start()

    def _dump_worker(self, frames, bark_prob, peak_dbfs, avg_dbfs, doa):
        import subprocess
        os.makedirs("./clips", exist_ok=True)
        raw_pcm  = b"".join(frames)
        filename = f"./clips/{time.time_ns()}.mp3"
        try:
            # Pipe raw PCM → ffmpeg → MP3 (mono mix-down keeps file small)
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    # Input: interleaved S16_LE PCM from stdin
                    "-f", "s16le", "-ar", str(self._fs),
                    "-ac", str(self._channels), "-i", "pipe:0",
                    # Output: MP3, mix down to mono, VBR quality 4 (~128 kbps)
                    "-ac", "1", "-q:a", "4",
                    filename,
                ],
                input=raw_pcm,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode())
        except Exception as exc:
            # Fallback: save as WAV so no clip is ever lost
            self._logger.warning(f"MP3 encode failed ({exc}), falling back to WAV")
            filename = filename.replace(".mp3", ".wav")
            wf = wave.open(filename, "wb")
            wf.setnchannels(self._channels)
            wf.setsampwidth(self._sample_size)
            wf.setframerate(self._fs)
            wf.writeframes(raw_pcm)
            wf.close()
        self._logger.info(f"Stored {filename}")

        duration = len(frames) * self._chunk / self._fs
        event_id = self._bark_logger.log_event(
            timestamp=datetime.datetime.now().isoformat(),
            clip_path=filename,
            bark_prob=bark_prob,
            peak_dbfs=peak_dbfs,
            avg_dbfs=avg_dbfs,
            duration=duration,
            doa=doa,
        )
        # Async upload (non-blocking; mode checked inside uploader)
        if event_id and filename:
            self._uploader.enqueue(event_id, filename)

    def get_last_pred(self):
        return self._model_last_pred

    def infer_chunk(self, frames):
        t = threading.Thread(target=self.infer_worker, args=[frames])
        t.start()

    def _detect_model_version(self):
        """Probe the loaded JIT model to determine if it's V1 (MLP) or V2 (CNN)."""
        try:
            test_v2 = torch.zeros(1, 50, 80)
            with torch.no_grad():
                self._model(test_v2)
            return 2
        except Exception:
            pass
        try:
            test_v1 = torch.zeros(1, 480)
            with torch.no_grad():
                self._model(test_v1)
            return 1
        except Exception:
            self._logger.warning("Could not determine model version, defaulting to V1")
            return 1

    def infer_worker(self, frames):
        audio_array = np.copy(np.frombuffer(b"".join(frames), dtype=np.int16))
        del frames

        if self._channels >= 2:
            # Multi-channel: compute DOA across all mic channels.
            # PS Eye 4-mic array gives real directional estimates.
            audio_array = audio_array.reshape((self._channels, -1), order="F")
            try:
                corr = corr_matrix_estimate(audio_array.T, imp="fast")
                doa1 = np.argmax(DOA_Bartlett(corr, self.ula_scanning_vectors))
                doa2 = np.argmax(DOA_Capon(corr, self.ula_scanning_vectors))
                doa3 = np.argmax(DOA_MEM(corr, self.ula_scanning_vectors))
            except Exception:
                # Singular matrix or other DOA failure (common with non-array mics)
                doa1 = doa2 = doa3 = 90
            # Model only needs single channel — use channel 0
            audio_array = audio_array[0:1, :]
        else:
            # Truly mono (should not happen with PS Eye): no DOA
            audio_array = audio_array.reshape((1, -1))
            doa1 = doa2 = doa3 = 90

        audio_array_torch = torch.from_numpy(audio_array)
        audio_array_float = audio_array_torch / torch.iinfo(torch.int16).max
        # Use the pre-built resampler (avoids rebuilding on every call)
        resampled_waveform = self._resampler(audio_array_float)
        mel_spectrogram = torchaudio.compliance.kaldi.fbank(
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            waveform=resampled_waveform,
        )

        if self._model_version == 2:
            # V2 CNN: needs [1, 50, 80] with CMVN normalization
            n_frames = mel_spectrogram.shape[0]
            if n_frames < 50:
                # Pad with zeros if we don't have enough frames
                pad = torch.zeros(50 - n_frames, 80)
                mel_spectrogram = torch.cat([mel_spectrogram, pad], dim=0)
            mel_spectrogram = mel_spectrogram[:50]  # take first 50 frames
            # CMVN: zero-mean, unit-variance per mel bin
            mean = mel_spectrogram.mean(dim=0, keepdim=True)
            std  = mel_spectrogram.std(dim=0, keepdim=True).clamp(min=1e-6)
            mel_spectrogram = (mel_spectrogram - mean) / std
            model_input = mel_spectrogram.unsqueeze(0)  # [1, 50, 80]
        else:
            # V1 MLP: needs [1, 480] flattened
            mel_spectrogram = mel_spectrogram.flatten().unsqueeze(0)
            if mel_spectrogram.size()[1] != 480:
                self._logger.error(f"Wrong size for LMEL features: {mel_spectrogram.size()}")
                return
            model_input = mel_spectrogram

        with torch.no_grad():
            pred = self._model(model_input).detach().item()

        with self._pred_lock:
            if "bark_probability" not in self._model_last_pred:
                self._model_last_pred["bark_probability"] = [pred]
            else:
                while len(self._model_last_pred["bark_probability"]) > 16:
                    del self._model_last_pred["bark_probability"][0]
                self._model_last_pred["bark_probability"].append(pred)

            self._model_last_pred["datetime"] = datetime.datetime.now().isoformat()

        # Always update live audio level and DOA telemetry
        arr_np = audio_array_float.numpy()
        peak = float(np.max(np.abs(arr_np)))
        rms  = float(np.sqrt(np.mean(arr_np ** 2)))
        peak_dbfs_now = round(20 * np.log10(peak + 1e-9), 1)
        avg_dbfs_now  = round(20 * np.log10(rms  + 1e-9), 1)
        self._current_audio_level = {
            "peak_dbfs": peak_dbfs_now,
            "avg_dbfs":  avg_dbfs_now,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        self._last_doa = {"doa1": int(doa1), "doa2": int(doa2), "doa3": int(doa3)}

        if pred >= self._bark_prob_threshold:
            # Compute loudness of this audio window
            arr = audio_array_float.numpy()
            peak = float(np.max(np.abs(arr)))
            rms  = float(np.sqrt(np.mean(arr ** 2)))
            peak_dbfs = 20 * np.log10(peak + 1e-9)
            avg_dbfs  = 20 * np.log10(rms  + 1e-9)

            print(
                f"[{datetime.datetime.now().isoformat()}, {doa1:03d}, {doa2:03d}, {doa3:03d}]: "
                f"*** BARKING ***: {pred:.3f}  peak={peak_dbfs:.1f}dBFS"
            )
            with open("./log.txt", "a") as f:
                f.write(
                    f"{datetime.datetime.now().isoformat()}\t{pred}\t{doa1}\t{doa2}\t{doa3}\n"
                )
            last_preds.append(1)

            if len(last_preds) >= 6:
                del last_preds[0]

            if sum(last_preds) >= 3:
                # Update stream overlay
                self._streamer.set_state(barking=True, prob=pred)

                # Auto-save clip (with cooldown to avoid flooding disk)
                now = time.time()
                if now - self._last_auto_save_time > self._auto_save_cooldown:
                    self._last_auto_save_time = now
                    self._store_bark_info = {
                        "prob": pred, "peak_dbfs": peak_dbfs,
                        "avg_dbfs": avg_dbfs, "doa": float((doa1 + doa2 + doa3) / 3),
                    }
                    self.store_clip()

                    # Publish MQTT event
                    try:
                        import mqtt_manager
                        mqtt_manager.get_manager().publish_bark({
                            "bark_prob": pred,
                            "peak_dbfs": peak_dbfs,
                            "avg_dbfs": avg_dbfs,
                            "doa": float((doa1 + doa2 + doa3) / 3),
                            "timestamp": datetime.datetime.now().isoformat(),
                        })
                    except Exception:
                        pass  # MQTT failures should never break detection

                if self.ef.fire():
                    self.ifttt_event()
        else:
            if len(last_preds) > 0:
                del last_preds[0]
            print(
                f"[{datetime.datetime.now().isoformat()}, {doa1:03d}, {doa2:03d}, {doa3:03d}]: Not barking: {pred}\r",
                end="",
            )

    def ifttt_event(self):
        # URL for the Maker Webhooks API endpoint
        ifttt_url = (
            f"https://maker.ifttt.com/trigger/{IFTTT_EVENT_NAME}/with/key/{IFTTT_KEY}"
        )

        # Send the HTTP POST request to trigger the IFTTT applet
        response = requests.post(ifttt_url)

        # Check the response
        if response.status_code == 200:
            self._logger.info("IFTTT applet triggered successfully.")
        else:
            self._logger.warning("Failed to trigger the IFTTT applet.")

    # ── Telemetry / config API helpers ────────────────────────────────────────

    def list_audio_devices(self):
        """Return all PyAudio input devices visible in the container."""
        p = pyaudio.PyAudio()
        devices = []
        try:
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                if d.get("maxInputChannels", 0) > 0:
                    devices.append({
                        "index":               i,
                        "name":                d["name"],
                        "max_input_channels":  d["maxInputChannels"],
                        "default_sample_rate": int(d["defaultSampleRate"]),
                        "is_selected":         i == self._recording_device_index,
                    })
        finally:
            p.terminate()
        return devices

    def get_debug_info(self):
        import platform

        # Resolve selected device name
        device_name = "Unknown"
        try:
            p = pyaudio.PyAudio()
            d = p.get_device_info_by_index(self._recording_device_index)
            device_name = d["name"]
            p.terminate()
        except Exception:
            pass

        # GPU info
        gpu: dict = {"available": torch.cuda.is_available()}
        if gpu["available"]:
            try:
                gpu["name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu["memory_total_mb"] = props.total_memory // (1024 * 1024)
                gpu["memory_used_mb"]  = torch.cuda.memory_allocated(0) // (1024 * 1024)
            except Exception:
                pass

        return {
            "device": {
                "index":         self._recording_device_index,
                "name":          device_name,
                "channels":      self._channels,
                "sample_rate":   self._fs,
                "chunk_samples": self._chunk,
                "latency_ms":    round(self._chunk / self._fs * 1000, 1),
            },
            "audio_level": self._current_audio_level,
            "last_pred":   self._model_last_pred,
            "last_doa":    self._last_doa,
            "model": {
                "path":       "./models/traced_model.pt",
                "window_ms":  self._model_window_size,
                "overlap_ms": self._model_window_overlap,
                "threshold":  self._bark_prob_threshold,
                "sample_rate": self._model_sample_rate,
            },
            "threads": {
                "recording_alive":  self._worker_thread.is_alive() if self._worker_thread else False,
                "streamer_running": not self._streamer._stop_flag,
            },
            "gpu": gpu,
            "system": {
                "python":        platform.python_version(),
                "torch":         torch.__version__,
                "uptime_seconds": int(time.time() - self._start_time),
            },
        }

    def get_config(self):
        return {
            "mic": {
                "device_hint":    os.environ.get("MIC_DEVICE_HINT", "Andrea PureAudio"),
                "channels":       self._channels,
                "sample_rate":    self._fs,
                "array_spacing":  float(os.environ.get("MIC_ARRAY_SPACING", "0.1")),
                "alsa_device":    os.environ.get("MIC_ALSA_DEVICE", "hw:1,0"),
            },
            "detection": {
                "bark_threshold":    self._bark_prob_threshold,
                "auto_save_cooldown": self._auto_save_cooldown,
            },
            "stream": {
                "youtube_key_set":    bool(os.environ.get("YOUTUBE_STREAM_KEY", "")),
                "video_device":       os.environ.get("VIDEO_DEVICE", "/dev/video0"),
                "bark_quiet_seconds": int(os.environ.get("BARK_QUIET_SECONDS", "10")),
            },
            "storage": {
                "events_db":          os.environ.get("EVENTS_DB", "./clips/events.db"),
                "clip_past_seconds":  self._clip_past_context_seconds,
                "clip_future_seconds": self._clip_future_context_seconds,
            },
        }

    def set_config(self, data: dict):
        """Apply runtime-safe config changes (no restart needed)."""
        if "bark_threshold" in data:
            val = float(data["bark_threshold"])
            if 0.1 <= val <= 1.0:
                self._bark_prob_threshold = val
                self._logger.info(f"bark_threshold updated → {val}")
        if "auto_save_cooldown" in data:
            val = int(data["auto_save_cooldown"])
            if val >= 0:
                self._auto_save_cooldown = val
                self._logger.info(f"auto_save_cooldown updated → {val}s")
        return self.get_config()

