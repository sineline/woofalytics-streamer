"""
trainer.py — In-process model fine-tuning from archived bark clips.

Architecture V3 (Dual-branch CNN — multi-class with spatial features):
  WoofClassifierV3(num_classes=N, spatial_dim=5)
  Audio branch:  Conv1d stack → 256-dim vector
  Spatial branch: FC(5→16) from [doa, ch1_rms, ch2_rms, ch3_rms, ch4_rms]
  Fusion: Concat(256+16) → FC(272→64) → FC(64→N)

Architecture V2 (1D CNN — binary, kept for backward compatibility):
  WoofClassifierV2(input_shape=[1, 50, 80])
  Conv1d(80→32, k=5) + BN + ReLU + MaxPool(2)
  Conv1d(32→64, k=3) + BN + ReLU + MaxPool(2)
  Conv1d(64→64, k=3) + BN + ReLU + AdaptivePool(4)
  Flatten → FC(256→64, ReLU, Dropout) → FC(64→1, Sigmoid)

Feature pipeline:
  torchaudio.compliance.kaldi.fbank(num_mel_bins=80, frame_length=25, frame_shift=10)
  → 50-frame windows (500ms) → CMVN → [1, 50, 80]

Legacy V1 architecture (kept for backward compatibility):
  WoofClassifier(input_size=480)
  FC(480→64, ReLU) → FC(64→32, ReLU) → FC(32→1, Sigmoid)

Modes:
  - fine_tune  (default): load existing model weights, fine-tune on new data
  - full       : re-initialise weights from scratch (needs sufficient data)

Training runs in a daemon thread; progress is queryable via get_status().
A timestamped backup is made before any model file is overwritten.
"""

import collections
import logging
import os
import random
import shutil
import threading
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

_logger = logging.getLogger("Trainer")

MODEL_PATH   = "./models/traced_model.pt"
BACKUP_DIR   = "./clips/model_backups"
SAMPLE_RATE  = 16_000

# ── V2 constants ──────────────────────────────────────────────────────────────
WIN_FRAMES_V2 = 50     # 50 fbank frames = 500ms context
NUM_MEL_BINS  = 80
INPUT_SIZE_V2 = WIN_FRAMES_V2 * NUM_MEL_BINS  # 4000

# ── V1 constants (legacy) ────────────────────────────────────────────────────
WIN_FRAMES_V1 = 6
INPUT_SIZE_V1 = 480


# ── CMVN (cepstral mean & variance normalization) ────────────────────────────

def apply_cmvn(fbank: torch.Tensor) -> torch.Tensor:
    """Per-utterance CMVN: zero-mean, unit-variance per mel bin.

    Input:  [T, 80]  (time × mel bins)
    Output: [T, 80]  normalized
    """
    mean = fbank.mean(dim=0, keepdim=True)
    std  = fbank.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (fbank - mean) / std


# ── V2 Model (1D CNN) ────────────────────────────────────────────────────────

class WoofClassifierV2(nn.Module):
    """Bark classifier using 1D convolutions over mel-spectrogram frames.

    Input shape: [batch, 50, 80]  (time × mel bins)
    The convolutions operate over the time axis with mel bins as channels.
    """
    def __init__(self):
        super().__init__()
        # Transpose input to [batch, 80, 50] for Conv1d (channels=mel, length=time)
        self.conv1 = nn.Conv1d(NUM_MEL_BINS, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)  # → [batch, 32, 25]

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)  # → [batch, 64, 12]

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.apool = nn.AdaptiveAvgPool1d(4)  # → [batch, 64, 4]

        self.fc1 = nn.Linear(64 * 4, 64)
        self.drop = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 80] → transpose to [batch, 80, 50]
        x = x.transpose(1, 2)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.apool(F.relu(self.bn3(self.conv3(x))))

        x = x.flatten(1)  # [batch, 256]
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return torch.sigmoid(self.output_layer(x))


# ── V3 Model (Dual-branch CNN: audio + spatial) ──────────────────────────

class WoofClassifierV3(nn.Module):
    """Multi-class sound classifier with spatial features.

    Takes a single input tensor [batch, 50, 85] where:
      - [:, :, :80] = mel spectrogram (audio features)
      - [:, 0, 80:85] = spatial features [doa_norm, ch1, ch2, ch3, ch4]
        (same spatial values replicated across time dim, only first row used)

    Output: [batch, num_classes] logits (no softmax — use CrossEntropyLoss)
    """
    def __init__(self, num_classes=2, spatial_dim=SPATIAL_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_dim = spatial_dim

        # Audio branch (same architecture as V2)
        self.conv1 = nn.Conv1d(NUM_MEL_BINS, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.apool = nn.AdaptiveAvgPool1d(4)

        # Spatial branch
        self.spatial_fc = nn.Linear(spatial_dim, 16)

        # Fusion
        self.fuse_fc1 = nn.Linear(64 * 4 + 16, 64)
        self.fuse_drop = nn.Dropout(0.3)
        self.fuse_fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 50, 85] — split into audio and spatial
        audio = x[:, :, :NUM_MEL_BINS]           # [batch, 50, 80]
        spatial = x[:, 0, NUM_MEL_BINS:]          # [batch, 5] (from first frame)

        # Audio branch: [batch, 50, 80] → [batch, 80, 50]
        a = audio.transpose(1, 2)
        a = self.pool1(F.relu(self.bn1(self.conv1(a))))
        a = self.pool2(F.relu(self.bn2(self.conv2(a))))
        a = self.apool(F.relu(self.bn3(self.conv3(a))))
        a = a.flatten(1)  # [batch, 256]

        # Spatial branch
        s = F.relu(self.spatial_fc(spatial))  # [batch, 16]

        # Fusion
        fused = torch.cat([a, s], dim=1)      # [batch, 272]
        fused = F.relu(self.fuse_fc1(fused))
        fused = self.fuse_drop(fused)
        return self.fuse_fc2(fused)            # [batch, num_classes] logits


# ── V1 Model (legacy MLP — kept for backward compat) ─────────────────────────

class WoofClassifier(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE_V1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output_layer(x))


# ── Feature extraction ───────────────────────────────────────────────────────

def _load_clip(path: str) -> Optional[torch.Tensor]:
    """Load audio clip → mono float32 waveform at SAMPLE_RATE."""
    try:
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE, dtype=torch.float32)(waveform)
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
    except Exception as exc:
        _logger.warning(f"Could not load {path}: {exc}")
        return None


def _extract_windows_v2(
    waveform: torch.Tensor,
    augment: bool = False,
) -> List[torch.Tensor]:
    """Extract 50-frame (500ms) fbank windows with CMVN.

    Returns list of [1, 50, 80] tensors, ready for WoofClassifierV2.
    """
    if augment:
        # Gain augmentation: random ±12 dB
        gain_db = random.uniform(-12, 12)
        gain_factor = 10 ** (gain_db / 20)
        waveform = waveform * gain_factor

    try:
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=waveform,
            num_mel_bins=NUM_MEL_BINS,
            frame_length=25,
            frame_shift=10,
        )  # [T, 80]
    except Exception as exc:
        _logger.warning(f"fbank failed: {exc}")
        return []

    # Apply CMVN
    fbank = apply_cmvn(fbank)

    T_frames = fbank.shape[0]
    windows = []
    step = WIN_FRAMES_V2 // 2  # 50% overlap → step=25 frames (250ms)

    for start in range(0, T_frames - WIN_FRAMES_V2 + 1, step):
        w = fbank[start:start + WIN_FRAMES_V2]  # [50, 80]
        windows.append(w.unsqueeze(0))  # [1, 50, 80]

    return windows


def _extract_windows_v1(waveform: torch.Tensor) -> List[torch.Tensor]:
    """Legacy V1 extraction: 6-frame windows → [1, 480] flat vectors."""
    try:
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=waveform,
            num_mel_bins=NUM_MEL_BINS,
            frame_length=25,
            frame_shift=10,
        )
    except Exception as exc:
        _logger.warning(f"fbank failed: {exc}")
        return []

    T_frames = fbank.shape[0]
    windows = []
    step = WIN_FRAMES_V1 // 2
    for start in range(0, T_frames - WIN_FRAMES_V1 + 1, step):
        w = fbank[start:start + WIN_FRAMES_V1].flatten().unsqueeze(0)  # [1, 480]
        if w.shape[1] == INPUT_SIZE_V1:
            windows.append(w)
    return windows


def build_dataset(
    clips: List[dict],
    model_version: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build X and y tensors from labelled clips.

    V3: X is [N, 50, 85], y is [N] (class indices)
    V2: X is [N, 50, 80], y is [N, 1]
    V1: X is [N, 480],    y is [N, 1]

    For V2/V3 training, each clip is processed twice (original + augmented).
    """
    X, y = [], []
    for item in clips:
        waveform = _load_clip(item["path"])
        if waveform is None:
            continue

        if model_version == 3:
            class_id = int(item.get("sound_class") or item.get("label", 0))
            # Spatial features: [doa_norm, ch1, ch2, ch3, ch4]
            doa = float(item.get("doa") or 90) / 180.0  # normalize to [0, 1]
            ch_e = item.get("ch_energies") or [0, 0, 0, 0]
            # Normalize channel energies: dBFS values typically -60 to 0
            # Map to roughly [-1, 1] range
            ch_norm = [(e + 30) / 30 for e in ch_e]  # -60→-1, 0→1
            spatial = [doa] + ch_norm[:4]
            # Pad to 4 channels if fewer
            while len(spatial) < SPATIAL_DIM:
                spatial.append(0.0)
            spatial_t = torch.tensor(spatial[:SPATIAL_DIM], dtype=torch.float32)

            for aug in [False, True]:
                for w in _extract_windows_v2(waveform, augment=aug):
                    # w is [1, 50, 80] — append spatial to make [1, 50, 85]
                    w = w.squeeze(0)  # [50, 80]
                    # Broadcast spatial to all frames (model reads from first row)
                    s_expand = spatial_t.unsqueeze(0).expand(50, -1)  # [50, 5]
                    combined = torch.cat([w, s_expand], dim=1)  # [50, 85]
                    X.append(combined.unsqueeze(0))  # [1, 50, 85]
                    y.append(class_id)

        elif model_version == 2:
            label = float(item.get("label", 0))
            for w in _extract_windows_v2(waveform, augment=False):
                X.append(w)
                y.append(torch.tensor([[label]], dtype=torch.float32))
            for w in _extract_windows_v2(waveform, augment=True):
                X.append(w)
                y.append(torch.tensor([[label]], dtype=torch.float32))
        else:
            label = float(item.get("label", 0))
            for w in _extract_windows_v1(waveform):
                X.append(w)
                y.append(torch.tensor([[label]], dtype=torch.float32))

    if not X:
        if model_version == 3:
            return torch.empty(0, WIN_FRAMES_V2, NUM_MEL_BINS + SPATIAL_DIM), torch.empty(0, dtype=torch.long)
        if model_version == 2:
            return torch.empty(0, WIN_FRAMES_V2, NUM_MEL_BINS), torch.empty(0, 1)
        return torch.empty(0, INPUT_SIZE_V1), torch.empty(0, 1)

    X_cat = torch.cat(X, dim=0)
    if model_version == 3:
        y_cat = torch.tensor(y, dtype=torch.long)
    else:
        y_cat = torch.cat(y, dim=0)
    return X_cat, y_cat


# ── Training job ──────────────────────────────────────────────────────────────

class TrainingJob:
    def __init__(self):
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._status = {
            "state":       "idle",   # idle | running | done | error
            "mode":        None,
            "epoch":       0,
            "total_epochs": 0,
            "loss":        None,
            "val_loss":    None,
            "accuracy":    None,
            "loss_history": [],
            "message":     "",
            "started_at":  None,
            "finished_at": None,
            "backup_path": None,
        }

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def is_running(self) -> bool:
        with self._lock:
            return self._status["state"] == "running"

    def start(
        self,
        clips: List[dict],
        mode: str = "fine_tune",
        epochs: int = 20,
        lr: float = 1e-3,
        val_split: float = 0.2,
    ):
        if self.is_running():
            return {"error": "Training already running"}

        with self._lock:
            self._status.update({
                "state": "running",
                "mode": mode,
                "epoch": 0,
                "total_epochs": epochs,
                "loss": None,
                "val_loss": None,
                "accuracy": None,
                "loss_history": [],
                "message": "Preparing dataset…",
                "started_at": time.time(),
                "finished_at": None,
                "backup_path": None,
            })

        self._thread = threading.Thread(
            target=self._run,
            args=(clips, mode, epochs, lr, val_split),
            daemon=True,
        )
        self._thread.start()
        return {"ok": True}

    # ── Internal worker ───────────────────────────────────────────────────────

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _run(self, clips, mode, epochs, lr, val_split):
        try:
            # ── Always train V2 (CNN) ────────────────────────────────────
            self._set(message="Building V2 feature dataset (500ms windows + CMVN)…")
            X, y = build_dataset(clips, model_version=2)
            n = X.shape[0]
            if n < 10:
                self._set(state="error", message=f"Too few usable windows ({n}). Need ≥10.")
                return

            # Train/val split
            perm    = torch.randperm(n)
            n_val   = max(1, int(n * val_split))
            val_idx = perm[:n_val]
            tr_idx  = perm[n_val:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            self._set(message=f"Dataset ready: {len(tr_idx)} train / {n_val} val windows (V2 CNN)")

            # ── Load / init model ──────────────────────────────────────────
            model = WoofClassifierV2()
            if mode == "fine_tune" and os.path.isfile(MODEL_PATH):
                try:
                    jit_model = torch.jit.load(MODEL_PATH)
                    # Check if it's a V2 model by probing input shape
                    test_in = torch.zeros(1, WIN_FRAMES_V2, NUM_MEL_BINS)
                    try:
                        jit_model(test_in)
                        # V2 model — extract state dict
                        sd = {}
                        for name, param in jit_model.named_parameters():
                            sd[name] = param
                        model.load_state_dict(sd, strict=False)
                        self._set(message="Loaded existing V2 model weights for fine-tuning")
                    except Exception:
                        self._set(message="Existing model is V1 (MLP). Starting fresh V2 CNN training.")
                except Exception as exc:
                    self._set(message=f"Could not load weights ({exc}), training from scratch")
            else:
                self._set(message="Initialising V2 CNN model from scratch")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = model.to(device)
            X_tr   = X_tr.to(device);  y_tr  = y_tr.to(device)
            X_val  = X_val.to(device); y_val = y_val.to(device)

            opt     = torch.optim.Adam(model.parameters(), lr=lr)
            bs      = min(32, len(tr_idx))
            history = []

            # ── Training loop ──────────────────────────────────────────────
            for ep in range(1, epochs + 1):
                model.train()
                perm_ep = torch.randperm(X_tr.shape[0], device=device)
                ep_loss = 0.0
                batches = 0
                for i in range(0, X_tr.shape[0], bs):
                    idx = perm_ep[i:i+bs]
                    out = model(X_tr[idx])
                    loss = F.binary_cross_entropy(out, y_tr[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item()
                    batches += 1

                # Validation
                model.eval()
                with torch.no_grad():
                    val_out  = model(X_val)
                    val_loss = F.binary_cross_entropy(val_out, y_val).item()
                    preds    = (val_out > 0.5).float()
                    acc      = (preds == y_val).float().mean().item()

                avg_loss = ep_loss / max(batches, 1)
                history.append({"epoch": ep, "loss": round(avg_loss, 4),
                                 "val_loss": round(val_loss, 4), "acc": round(acc, 4)})
                self._set(
                    epoch=ep, loss=round(avg_loss, 4),
                    val_loss=round(val_loss, 4), accuracy=round(acc, 4),
                    loss_history=list(history),
                    message=f"Epoch {ep}/{epochs} — loss {avg_loss:.4f} val_loss {val_loss:.4f} acc {acc:.2%}",
                )

            # ── Save model ─────────────────────────────────────────────────
            self._set(message="Saving V2 model…")
            # Backup existing model
            os.makedirs(BACKUP_DIR, exist_ok=True)
            if os.path.isfile(MODEL_PATH):
                ts          = int(time.time())
                backup_path = os.path.join(BACKUP_DIR, f"traced_model_{ts}.pt")
                shutil.copy2(MODEL_PATH, backup_path)
                self._set(backup_path=backup_path)

            # Trace and save
            model.eval().cpu()
            example = torch.zeros(1, WIN_FRAMES_V2, NUM_MEL_BINS)
            traced  = torch.jit.trace(model, example)
            torch.jit.save(traced, MODEL_PATH)

            self._set(
                state="done",
                finished_at=time.time(),
                message=f"Training complete. V2 CNN model saved to {MODEL_PATH}",
            )
            _logger.info("Training job finished successfully (V2 CNN)")

        except Exception as exc:
            _logger.exception("Training failed")
            self._set(state="error", message=str(exc), finished_at=time.time())


# ── Singleton ─────────────────────────────────────────────────────────────────
_job = TrainingJob()


def get_job() -> TrainingJob:
    return _job
