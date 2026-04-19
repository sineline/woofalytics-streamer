"""
Uploader — async background clip upload for archiving.

Clips are saved locally first, then queued here for background upload.
Supports:
  - s3:   AWS S3, Backblaze B2, MinIO, Wasabi (any S3-compatible endpoint)
  - sftp: any SSH/SFTP server
  - off:  no upload

Upload status is written back to the events DB so the Library page can
show per-clip status badges.
"""
import logging
import os
import queue
import threading
import time
from typing import Optional

import settings as cfg_store

_logger = logging.getLogger("Uploader")


# ── Backend implementations ───────────────────────────────────────────────────

def _upload_s3(local_path: str, remote_name: str, s) -> str:
    """Upload to any S3-compatible endpoint. Returns public/presigned URL or key."""
    try:
        import boto3
        from botocore.client import Config
    except ImportError:
        raise RuntimeError("boto3 not installed — run: pip install boto3")

    endpoint = s.get("upload_s3_endpoint") or None   # None = AWS default
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=s["upload_s3_key"],
        aws_secret_access_key=s["upload_s3_secret"],
        config=Config(signature_version="s3v4"),
    )
    bucket = s["upload_s3_bucket"]
    prefix = s.get("upload_s3_prefix", "woofalytics/").rstrip("/") + "/"
    key = prefix + remote_name

    client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _upload_sftp(local_path: str, remote_name: str, s) -> str:
    """Upload to an SFTP server."""
    try:
        import paramiko
    except ImportError:
        raise RuntimeError("paramiko not installed — run: pip install paramiko")

    host   = s["upload_sftp_host"]
    port   = int(s.get("upload_sftp_port", 22))
    user   = s["upload_sftp_user"]
    pwd    = s.get("upload_sftp_password", "")
    keyf   = s.get("upload_sftp_key_path", "")
    remote_dir = s.get("upload_sftp_dir", "/upload/woofalytics")

    t = paramiko.Transport((host, port))
    if keyf and os.path.isfile(keyf):
        key = paramiko.RSAKey.from_private_key_file(keyf)
        t.connect(username=user, pkey=key)
    else:
        t.connect(username=user, password=pwd)

    sftp = paramiko.SFTPClient.from_transport(t)
    try:
        sftp.mkdir(remote_dir)
    except IOError:
        pass  # already exists
    remote_path = remote_dir.rstrip("/") + "/" + remote_name
    sftp.put(local_path, remote_path)
    t.close()
    return f"sftp://{host}{remote_path}"


# ── Worker ────────────────────────────────────────────────────────────────────

class Uploader:
    def __init__(self, bark_logger=None):
        self._logger     = logging.getLogger("Uploader")
        self._queue: queue.Queue = queue.Queue()
        self._bark_logger = bark_logger  # for writing status back to DB
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._stats = {"queued": 0, "ok": 0, "failed": 0, "last_error": ""}

    def enqueue(self, event_id: int, local_path: str):
        """Non-blocking: add a clip to the upload queue."""
        remote_name = os.path.basename(local_path)
        self._queue.put((event_id, local_path, remote_name))
        self._stats["queued"] += 1
        if self._bark_logger:
            self._bark_logger.set_upload_status(event_id, "queued")

    def _worker(self):
        while True:
            try:
                event_id, local_path, remote_name = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            s = cfg_store.get()
            mode = s.get("upload_mode", "off")
            if mode == "off":
                self._queue.task_done()
                continue

            for attempt in range(3):
                try:
                    if mode == "s3":
                        url = _upload_s3(local_path, remote_name, s)
                    elif mode == "sftp":
                        url = _upload_sftp(local_path, remote_name, s)
                    else:
                        self._queue.task_done()
                        break

                    self._logger.info(f"Uploaded {remote_name} → {url}")
                    self._stats["ok"] += 1
                    self._stats["queued"] = max(0, self._stats["queued"] - 1)
                    if self._bark_logger:
                        self._bark_logger.set_upload_status(event_id, "uploaded", url)
                    break

                except Exception as exc:
                    self._logger.warning(
                        f"Upload attempt {attempt+1}/3 failed for {remote_name}: {exc}"
                    )
                    self._stats["last_error"] = str(exc)
                    if attempt == 2:
                        self._stats["failed"] += 1
                        self._stats["queued"] = max(0, self._stats["queued"] - 1)
                        if self._bark_logger:
                            self._bark_logger.set_upload_status(event_id, "failed")
                    else:
                        time.sleep(5 * (attempt + 1))

            self._queue.task_done()

    def get_status(self) -> dict:
        s = cfg_store.get()
        return {
            "mode":       s.get("upload_mode", "off"),
            "queued":     self._stats["queued"],
            "uploaded":   self._stats["ok"],
            "failed":     self._stats["failed"],
            "last_error": self._stats["last_error"],
            "queue_depth": self._queue.qsize(),
            "thread_alive": self._thread.is_alive(),
        }
