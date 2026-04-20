# woofalytics-streamer

> **A community fork of [woofalytics](https://github.com/mdoulaty/woofalytics) by [@mdoulaty](https://github.com/mdoulaty).**
> This fork was extended by an **agentic AI** ([Antigravity](https://deepmind.google/), built by Google DeepMind) working alongside the repository owner. The AI autonomously designed, implemented, debugged, and committed all additions described below.

---

## What is Woofalytics?

Woofalytics is an AI-powered dog bark detector originally built by [@mdoulaty](https://github.com/mdoulaty) to run on a Raspberry Pi with a dual-channel microphone array. It uses a compact feed-forward neural network to classify barks in real time, estimates the **direction of arrival (DOA)** of the sound, and exposes a live web dashboard.

The original motivation: distinguish a neighbour's dog from one's own and use the trigger to auto-dispense treats — breaking the bark-response cycle. The model itself is a small two-hidden-layer network with a sigmoid output over 80-dimensional log-Mel filterbank features extracted from 60 ms windows. It is fast enough for real-time use on constrained hardware.

**Original hardware:** Raspberry Pi 4 + [Andrea Electronics PureAudio USB Array Microphone](https://andreaelectronics.com/array-microphone/) (2-channel linear array).

---

## What this fork adds

This fork keeps everything from the original and layers a production-ready monitoring and management stack on top, containerised with Docker.

### 🐳 Containerisation
- `Dockerfile`, `docker-compose.yml`, `.dockerignore` — runs the full stack in a single command
- NVIDIA GPU support via the Container Toolkit (CPU fallback included)
- `privileged: true` passthrough for full USB audio visibility inside the container (required for the PS Eye 4-mic array — ALSA card registration does not transfer through `/dev/snd` alone)
- Persistent volume at `./clips/` for recorded audio and the SQLite database

### 🎙️ Extended microphone support: Sony PlayStation Eye (PS Eye)
The original code targeted a 2-channel Andrea array. This fork adds full support for the **PS Eye camera**, which exposes a **4-microphone ULA** (`bNrChannels=4` at 16 000 Hz per the USB audio descriptor).

- DOA estimation upgraded to a 4-element ULA beamformer
- `MIC_CHANNELS=4`, `MIC_SAMPLE_RATE=16000`, `MIC_ARRAY_SPACING` configurable at runtime without a rebuild

### 📊 Analytics dashboard (`/analytics`)
- Per-dog noise contribution charts (bark count, cumulative duration, peak dBFS)
- Timeline heat-map of bark activity by hour
- Filterable by dog identity and date range

### 📼 Clip Library (`/library`)
- Browse, play, retag, and delete stored WAV clips
- Per-clip audio player, probability, peak dBFS, duration, DOA badge
- Bulk select + delete
- Dog reassignment dropdown
- Search and sort (newest / oldest / loudest / longest)
- Upload status badge per clip (queued / uploaded / failed)

### 🗄️ Relational SQLite schema
- New `dogs` table joined to `events`
- **Auto-naming**: unidentified dogs are sequentially assigned "Dog 1", "Dog 2", … on first bark
- CRUD API: create dog, rename dog, retag event, delete event + file

### 📺 Stream page (`/stream`)
- Go Live / Stop controls for YouTube RTMP (ffmpeg + `libx264`)
- Auto-stream modes: manual / on-bark / always / scheduled (time window)
- Encoding quality settings: resolution, fps, bitrate, OSD reset delay
- Live stream status (duration, error display, key-set indicator)
- Quick-links to YouTube Studio and Live Dashboard

### ☁️ Archive upload — async clip backup
An alternative to live streaming: automatically upload bark clips to cloud storage for long-term archiving and audit.

| Backend | Details |
|---|---|
| **S3-compatible** | AWS S3, Backblaze B2, MinIO, Wasabi — just change the endpoint URL |
| **SFTP** | Any SSH server; password or private key auth |

- Non-blocking background queue with 3-attempt retry and exponential back-off
- Upload status (`queued` / `uploaded` / `failed`) tracked per clip in the DB
- Secrets (S3 secret, SFTP password, stream key) are **never returned to the browser**

### ⚙️ Config page (`/config`)
- **Mic device selector** — dropdown of all ALSA input devices, channels, sample rate, DOA array spacing
- **Stream key input** — masked password field with show/hide toggle, saved to the Docker volume (not in env)
- **Video device selector** — lists `/dev/video*` devices
- **Runtime sliders** — bark threshold and auto-save cooldown apply instantly without a restart
- **docker-compose.yml snippet generator** — copy-paste ready config for the current settings

### 🐞 Debug page (`/debug`)
- Real-time system stats (CPU, RAM, GPU if present)
- Live VU meter per channel
- DOA compass visualization
- Active audio device health
- Scrollable live log tail

### 🧭 Shared navigation (`/nav.js`)
- Single JS file injected on every page
- Sticky glassmorphism bar with active-link detection
- Live bark status pill (polling `/api/bark`)

### 🧠 Training & Labeling Pipeline (`/train`)
- **V2 CNN bark detector** — 1D CNN with batch normalization, 500ms context window (vs 60ms in V1)
- **CMVN normalization** — volume-independent classification (catches distant barks, ignores loud non-bark sounds)
- **Smart clip slicing** — uses `ffmpeg silencedetect` to skip silent sections when splitting long clips into short labelling segments
- **Waveform player** — interactive canvas-based player with click-to-seek, speed controls, and keyboard shortcuts
- **Per-clip labeling** — Bark / Not-Bark buttons with keyboard shortcuts (B/N)

### 🤖 AI-Assisted Labeling (Google Gemini)
- Integrates with **Gemini 2.0 Flash** to auto-classify audio clips as bark or not-bark
- Per-clip "AI Label" button and bulk "AI Label All" for rapid labeling
- Returns confidence scores and audio descriptions
- API key stored securely in `settings.json` (masked in UI)

### 📡 MQTT Integration (`/mqtt`)
- Publish bark events as JSON to any MQTT broker
- Configurable broker, port, username/password, TLS
- Home Assistant auto-discovery support (`binary_sensor` with `device_class: sound`)
- Connection test button with live status indicator
- Last Will and Testament (LWT) for online/offline availability

### 🔀 Data Augmentation (`/augment`)
- Generate training clip variations from labelled data
- **Gain variation**: ±6 dB, ±12 dB
- **Speed/pitch shift**: 0.85×, 0.9×, 1.1×, 1.15×
- **Reverb simulation**: room echo via convolution
- Augmented clips inherit the source label and integrate directly with the training pipeline

### 💾 Persistent settings
`settings.py` writes a `./clips/settings.json` file that survives container restarts and takes precedence over `docker-compose.yml` env vars for runtime-configurable fields.

---

## Hardware

| Component | Original | This fork |
|---|---|---|
| SBC | Raspberry Pi 4 | Any x86-64 Linux PC or ARM64 SBC (tested on Ubuntu) |
| Microphone | Andrea PureAudio 2-ch USB array | **Sony PS Eye** (4-mic ULA, 16 kHz) |
| GPU | — | Optional NVIDIA GPU for faster inference |

> **Running on an Odroid N2+?** The stack runs natively on Ubuntu 22.04 for ARM64. Install PyTorch with `--index-url https://download.pytorch.org/whl/cpu`. No CUDA needed — the model is small enough for CPU-only inference. For streaming, use `h264_v4l2m2m` hardware encoding instead of `libx264` to keep CPU load low.

---

## Quick start (Docker)

```bash
git clone https://github.com/sineline/woofalytics-streamer.git
cd woofalytics-streamer

# (optional) set your YouTube stream key
# edit docker-compose.yml → YOUTUBE_STREAM_KEY=xxxx

docker compose up --build
```

Open **http://localhost:8000**

> **PS Eye users:** The `privileged: true` flag in `docker-compose.yml` is required for ALSA to see the 4-channel card inside the container. Also update the USB device path if your bus/device numbers differ (`lsusb` to check):
> ```yaml
> devices:
>   - /dev/bus/usb/001/008:/dev/bus/usb/001/008  # adjust to your lsusb output
> ```

---

## Native install (no Docker)

```bash
# System dependencies (Debian/Ubuntu)
sudo apt install python3-pip python3-venv portaudio19-dev ffmpeg \
                 v4l-utils libsndfile1 alsa-utils

# Create venv
python3 -m venv .venv && source .venv/bin/activate

# PyTorch (CPU-only build — works on ARM64 and x86)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -r requirements.txt

python main.py
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MIC_DEVICE_HINT` | `USB Camera` | Substring matched against PyAudio device names |
| `MIC_CHANNELS` | `4` | `2` for Andrea array, `4` for PS Eye |
| `MIC_SAMPLE_RATE` | `16000` | Hz — PS Eye only supports 16000 |
| `MIC_ARRAY_SPACING` | `0.1` | Inter-mic spacing as fraction of wavelength (λ) |
| `AUTO_SAVE_COOLDOWN` | `30` | Seconds between auto-saved clips |
| `YOUTUBE_STREAM_KEY` | _(empty)_ | Disables streaming if unset |
| `VIDEO_DEVICE` | `/dev/video0` | V4L2 device for ffmpeg video capture |
| `BARK_QUIET_SECONDS` | `10` | OSD reset delay after last bark |
| `EVENTS_DB` | `./clips/events.db` | SQLite database path |

All of these can also be set at runtime via the **Config page** and are persisted to `./clips/settings.json`.

---

## Web pages

| Path | Description |
|---|---|
| `/` | Live dashboard — bark probability, VU meters, DOA compass |
| `/analytics` | Per-dog noise analytics and timeline |
| `/library` | Browse, play, tag, and delete recorded clips |
| `/stream` | YouTube streaming controls + archive upload config |
| `/train` | Label clips, train model, AI-assisted labeling, smart slicing |
| `/augment` | Generate training data variations (gain, speed, reverb) |
| `/mqtt` | MQTT broker configuration and event publishing |
| `/debug` | System telemetry, live log, audio device health |
| `/config` | Device selectors, stream key, runtime tuning |
| `/rec` | Manual clip recording trigger |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/bark` | Current bark probability and timestamp |
| GET | `/api/analytics` | Aggregated per-dog stats |
| GET | `/api/events?limit=N&dog_id=X` | Recent bark events |
| GET | `/api/dogs` | All dogs in the DB |
| POST | `/api/dogs` | Create a new dog `{ "name": "Rex" }` |
| PATCH | `/api/dogs/<id>` | Rename a dog `{ "name": "Buddy" }` |
| DELETE | `/api/events/<id>` | Delete event + WAV file |
| PATCH | `/api/events/<id>` | Retag event `{ "dog_id": "Dog 2" }` |
| GET | `/api/stream` | Stream status |
| POST | `/api/stream` | Start/stop stream `{ "action": "start" }` |
| GET | `/api/upload` | Upload queue status |
| GET/POST | `/api/settings` | Read / write persistent settings |
| GET/POST | `/api/config` | Read / write runtime config (threshold, cooldown) |
| GET | `/api/devices` | List PyAudio input devices |
| GET | `/api/devices/video` | List `/dev/video*` devices |
| GET | `/api/debug` | Live system telemetry |
| GET | `/api/train` | Training job status |
| GET | `/api/train/clips` | All clips available for labeling |
| POST | `/api/train` | Start a training run |
| POST | `/api/clips/<id>/slice` | Slice a clip into segments `{ "smart": true }` |
| POST | `/api/ai/label` | AI-classify a single clip `{ "event_id": 123 }` |
| POST | `/api/ai/label-batch` | AI-classify multiple clips |
| POST | `/api/augment` | Generate augmented clips |
| GET | `/api/mqtt/status` | MQTT connection status |
| POST | `/api/mqtt/test` | Test MQTT broker connection |
| POST | `/api/mqtt/configure` | Save & apply MQTT settings |

---

## Model

Two model versions are supported:

### V1 (Legacy)
```
Input: 80-dim log-Mel filterbank (60 ms window)
→ FC(480 → 64, ReLU)
→ FC(64 → 32, ReLU)
→ FC(32 → 1, Sigmoid)
Output: P(barking)
```

### V2 (CNN — current)
```
Input: [50, 80] log-Mel filterbank (500 ms window, CMVN normalized)
→ Conv1d(80→64, k=5) + BatchNorm + ReLU + MaxPool
→ Conv1d(64→128, k=3) + BatchNorm + ReLU + MaxPool
→ Conv1d(128→128, k=3) + BatchNorm + ReLU + AdaptiveAvgPool
→ FC(128→64, ReLU + Dropout)
→ FC(64→1, Sigmoid)
Output: P(barking)
```

**Key difference:** V2 uses CMVN (per-window mean/variance normalization) so it classifies by spectral shape rather than absolute volume. This catches distant barks and ignores nearby loud non-bark sounds.

The model version is auto-detected at startup. Train via the `/train` page.

---

## IFTTT integration

Unchanged from the original. Set in `record.py`:

```python
IFTTT_EVENT_NAME = "woof"
IFTTT_KEY = "YOUR_IFTTT_WEBHOOKS_KEY"
```

---

## Credits

- **Original project:** [woofalytics](https://github.com/mdoulaty/woofalytics) by [@mdoulaty](https://github.com/mdoulaty) — bark detection model, DOA estimation, core recording loop, web server, IFTTT integration.
- **This fork:** All additions (Docker, multi-page UI, relational DB, archive upload, stream controls, config/debug pages, PS Eye support) were designed and implemented by **[Antigravity](https://deepmind.google/)**, an agentic AI coding assistant built by the Google DeepMind team, working interactively with the repository owner [@sineline](https://github.com/sineline).

> No original logic was removed or broken. All new features are additive.
