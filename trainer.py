"""
trainer.py — In-process model fine-tuning from archived bark clips.

Architecture (must match the original notebook):
  WoofClassifier(input_size=480)
  FC(480→64, ReLU) → FC(64→32, ReLU) → FC(32→1, Sigmoid)

Feature pipeline (identical to record.py inference):
  torchaudio.compliance.kaldi.fbank(num_mel_bins=80, frame_length=25, frame_shift=10)
  → 6-frame windows → flatten → [1, 480]

Modes:
  - fine_tune  (default): load existing model weights, fine-tune on new data
  - full       : re-initialise weights from scratch (needs sufficient data)

Training runs in a daemon thread; progress is queryable via get_status().
A timestamped backup is made before any model file is overwritten.
"""

import collections
import logging
import os
import shutil
import threading
import time
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

_logger = logging.getLogger("Trainer")

MODEL_PATH   = "./models/traced_model.pt"
BACKUP_DIR   = "./clips/model_backups"
SAMPLE_RATE  = 16_000
WIN_FRAMES   = 6       # fbank frames per window
INPUT_SIZE   = 480     # 80 mel bins × 6 frames


# ── Model definition (mirrors notebook WoofClassifier) ───────────────────────

class WoofClassifier(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.output_layer(x))


# ── Feature extraction (identical to record.py inference path) ───────────────

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


def _extract_windows(waveform: torch.Tensor) -> List[torch.Tensor]:
    """Extract all 6-frame fbank windows from a waveform → list of [1,480] tensors."""
    try:
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
        )  # shape: [T, 80]
    except Exception as exc:
        _logger.warning(f"fbank failed: {exc}")
        return []

    T_frames = fbank.shape[0]
    windows = []
    step = WIN_FRAMES // 2  # 50% overlap
    for start in range(0, T_frames - WIN_FRAMES + 1, step):
        w = fbank[start:start + WIN_FRAMES].flatten().unsqueeze(0)  # [1, 480]
        if w.shape[1] == INPUT_SIZE:
            windows.append(w)
    return windows


def build_dataset(
    clips: List[dict],  # list of {"path": str, "label": int}  label 1=bark 0=not-bark
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build X [N,480] and y [N,1] tensors from a list of labelled clips."""
    X, y = [], []
    for item in clips:
        waveform = _load_clip(item["path"])
        if waveform is None:
            continue
        label = float(item["label"])
        for w in _extract_windows(waveform):
            X.append(w)
            y.append(torch.tensor([[label]], dtype=torch.float32))
    if not X:
        return torch.empty(0, INPUT_SIZE), torch.empty(0, 1)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


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
            self._set(message="Building feature dataset…")
            X, y = build_dataset(clips)
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
            self._set(message=f"Dataset ready: {len(tr_idx)} train / {n_val} val windows")

            # ── Load / init model ──────────────────────────────────────────
            model = WoofClassifier()
            if mode == "fine_tune":
                try:
                    # Load weights from traced model via scripted copy
                    jit_model = torch.jit.load(MODEL_PATH)
                    # Extract state dict from JIT model
                    sd = {k: v for k, v in jit_model.named_parameters()}
                    model.fc1.weight.data          = sd["fc1.weight"]
                    model.fc1.bias.data            = sd["fc1.bias"]
                    model.fc2.weight.data          = sd["fc2.weight"]
                    model.fc2.bias.data            = sd["fc2.bias"]
                    model.output_layer.weight.data = sd["output_layer.weight"]
                    model.output_layer.bias.data   = sd["output_layer.bias"]
                    self._set(message="Loaded existing model weights for fine-tuning")
                except Exception as exc:
                    self._set(message=f"Could not load weights ({exc}), training from scratch")
            else:
                self._set(message="Initialising model from scratch")

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
            self._set(message="Saving model…")
            # Backup existing model
            os.makedirs(BACKUP_DIR, exist_ok=True)
            if os.path.isfile(MODEL_PATH):
                ts          = int(time.time())
                backup_path = os.path.join(BACKUP_DIR, f"traced_model_{ts}.pt")
                shutil.copy2(MODEL_PATH, backup_path)
                self._set(backup_path=backup_path)

            # Trace and save
            model.eval().cpu()
            example = torch.zeros(1, INPUT_SIZE)
            traced  = torch.jit.trace(model, example)
            torch.jit.save(traced, MODEL_PATH)

            self._set(
                state="done",
                finished_at=time.time(),
                message=f"Training complete. Model saved to {MODEL_PATH}",
            )
            _logger.info("Training job finished successfully")

        except Exception as exc:
            _logger.exception("Training failed")
            self._set(state="error", message=str(exc), finished_at=time.time())


# ── Singleton ─────────────────────────────────────────────────────────────────
_job = TrainingJob()


def get_job() -> TrainingJob:
    return _job
