import json
import sqlite3
import threading
import datetime
import logging
import os

DB_PATH = os.environ.get("EVENTS_DB", "./clips/events.db")


class BarkLogger:
    def __init__(self):
        self._logger = logging.getLogger("BarkLogger")
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(DB_PATH)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS dogs (
                    dog_id  TEXT PRIMARY KEY,
                    name    TEXT NOT NULL,
                    color   TEXT,
                    created TEXT NOT NULL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    clip_path TEXT,
                    bark_prob REAL,
                    peak_dbfs REAL,
                    avg_dbfs  REAL,
                    duration  REAL,
                    doa       REAL,
                    dog_id    TEXT REFERENCES dogs(dog_id)
                )
            """)
            # Sound classes table for multi-class classification
            c.execute("""
                CREATE TABLE IF NOT EXISTS sound_classes (
                    class_id  INTEGER PRIMARY KEY,
                    name      TEXT NOT NULL,
                    icon      TEXT DEFAULT '🔊',
                    color     TEXT DEFAULT '#64748b'
                )
            """)
            # Seed default classes if empty
            existing = c.execute("SELECT COUNT(*) FROM sound_classes").fetchone()[0]
            if existing == 0:
                c.executemany(
                    "INSERT INTO sound_classes (class_id, name, icon, color) VALUES (?,?,?,?)",
                    [
                        (0, "Other",               "🔇", "#64748b"),
                        (1, "Bark (Unknown Dog)",   "🐕", "#22c55e"),
                    ],
                )
            # Migrations for existing DBs
            for col_def in ["doa REAL", "avg_dbfs REAL",
                            "upload_status TEXT", "upload_url TEXT",
                            "label INTEGER",
                            "ch_energies TEXT",
                            "sound_class INTEGER",
                            "ai_note TEXT",
                            "raw_clip_path TEXT"]:
                try:
                    c.execute(f"ALTER TABLE events ADD COLUMN {col_def}")
                except Exception:
                    pass
            # Migrate existing binary labels → sound_class
            c.execute("""
                UPDATE events SET sound_class = 1
                WHERE label = 1 AND sound_class IS NULL
            """)
            c.execute("""
                UPDATE events SET sound_class = 0
                WHERE label = 0 AND sound_class IS NULL
            """)
            # Backfill raw_clip_path for existing events with _4ch.flac files
            rows = c.execute(
                "SELECT id, clip_path FROM events WHERE raw_clip_path IS NULL AND clip_path IS NOT NULL"
            ).fetchall()
            for eid, cp in rows:
                if cp and cp.endswith(".mp3"):
                    flac = cp.rsplit(".", 1)[0] + "_4ch.flac"
                    if os.path.isfile(flac):
                        c.execute("UPDATE events SET raw_clip_path=? WHERE id=?", (flac, eid))

    # ── Dog identity helpers ──────────────────────────────────────────────────

    def _next_dog_id(self, conn):
        """Return the next sequential auto-name: Dog 1, Dog 2, …"""
        rows = conn.execute("SELECT dog_id FROM dogs").fetchall()
        existing = {r[0] for r in rows}
        n = 1
        while f"Dog {n}" in existing:
            n += 1
        return f"Dog {n}"

    def get_all_dogs(self):
        with self._conn() as c:
            cur = c.execute("SELECT dog_id, name, color, created FROM dogs ORDER BY created ASC")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def rename_dog(self, dog_id, new_name):
        """Rename a dog. Creates a new dog_id row if needed."""
        with self._lock:
            with self._conn() as c:
                c.execute("UPDATE dogs SET name=? WHERE dog_id=?", (new_name, dog_id))

    def create_dog(self, name=None):
        """Create a new dog with an auto or explicit name. Returns dog_id."""
        with self._lock:
            with self._conn() as c:
                dog_id = name or self._next_dog_id(c)
                c.execute(
                    "INSERT OR IGNORE INTO dogs (dog_id, name, created) VALUES (?,?,?)",
                    (dog_id, dog_id, datetime.datetime.now().isoformat()),
                )
            return dog_id

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_event(self, timestamp, clip_path, bark_prob, peak_dbfs, avg_dbfs,
                  duration, doa=90.0, dog_id=None, ch_energies=None, raw_clip_path=None):
        with self._lock:
            try:
                ch_json = json.dumps(ch_energies) if ch_energies else None
                with self._conn() as c:
                    # Auto-assign to first dog, creating "Dog 1" if none exist
                    if not dog_id:
                        row = c.execute(
                            "SELECT dog_id FROM dogs ORDER BY created ASC LIMIT 1"
                        ).fetchone()
                        if row:
                            dog_id = row[0]
                        else:
                            dog_id = self._next_dog_id(c)
                            c.execute(
                                "INSERT OR IGNORE INTO dogs (dog_id, name, created) VALUES (?,?,?)",
                                (dog_id, dog_id, timestamp),
                            )
                    cur = c.execute(
                        "INSERT INTO events "
                        "(timestamp,clip_path,bark_prob,peak_dbfs,avg_dbfs,duration,doa,dog_id,ch_energies,raw_clip_path) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (timestamp, clip_path, bark_prob, peak_dbfs, avg_dbfs, duration, doa, dog_id, ch_json, raw_clip_path),
                    )
                    event_id = cur.lastrowid
                self._logger.info(
                    f"Logged bark: [{dog_id}] prob={bark_prob:.2f} "
                    f"peak={peak_dbfs:.1f}dBFS doa={doa:.0f}°"
                )
                return event_id
            except Exception as exc:
                self._logger.error(f"DB write failed: {exc}")
                return None

    def set_upload_status(self, event_id: int, status: str, url: str = None):
        """Called by Uploader to record 'queued'/'uploaded'/'failed' per clip."""
        with self._lock:
            with self._conn() as c:
                c.execute(
                    "UPDATE events SET upload_status=?, upload_url=? WHERE id=?",
                    (status, url, event_id),
                )

    def set_label(self, event_id: int, label):
        """Manually label an event: 1=bark, 0=not-bark, None=clear."""
        with self._lock:
            with self._conn() as c:
                c.execute("UPDATE events SET label=?, sound_class=? WHERE id=?",
                          (label, label, event_id))

    def set_ai_note(self, event_id: int, note: str):
        """Store an AI-generated description on an event."""
        with self._lock:
            with self._conn() as c:
                c.execute("UPDATE events SET ai_note=? WHERE id=?",
                          (note, event_id))

    def set_sound_class(self, event_id: int, class_id: int):
        """Set the sound class for an event (multi-class labeling)."""
        with self._lock:
            with self._conn() as c:
                # Also set legacy label for backward compat (0=other, 1+=bark variant)
                legacy_label = 0 if class_id == 0 else 1
                c.execute("UPDATE events SET sound_class=?, label=? WHERE id=?",
                          (class_id, legacy_label, event_id))

    def get_labelled_clips(self) -> list:
        """Return all events that have an explicit label set, for training."""
        with self._conn() as c:
            cur = c.execute(
                "SELECT id, clip_path, label, sound_class, doa, ch_energies FROM events "
                "WHERE (label IS NOT NULL OR sound_class IS NOT NULL) AND clip_path IS NOT NULL"
            )
            rows = [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
            # Parse ch_energies JSON
            for r in rows:
                if r.get("ch_energies"):
                    try:
                        r["ch_energies"] = json.loads(r["ch_energies"])
                    except Exception:
                        r["ch_energies"] = None
            return rows

    # ── Sound classes ─────────────────────────────────────────────────────────

    def get_sound_classes(self) -> list:
        """Return all sound classes."""
        with self._conn() as c:
            cur = c.execute("SELECT class_id, name, icon, color FROM sound_classes ORDER BY class_id")
            return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]

    def create_sound_class(self, name: str, icon: str = "🔊", color: str = "#64748b") -> int:
        """Create a new sound class. Returns the class_id."""
        with self._lock:
            with self._conn() as c:
                # Auto-assign next class_id
                max_id = c.execute("SELECT MAX(class_id) FROM sound_classes").fetchone()[0]
                new_id = (max_id or 0) + 1
                c.execute(
                    "INSERT INTO sound_classes (class_id, name, icon, color) VALUES (?,?,?,?)",
                    (new_id, name, icon, color),
                )
                return new_id

    def delete_sound_class(self, class_id: int):
        """Delete a sound class. Cannot delete class 0 (Other)."""
        if class_id == 0:
            return False
        with self._lock:
            with self._conn() as c:
                # Clear sound_class on events that had this class
                c.execute("UPDATE events SET sound_class=NULL WHERE sound_class=?", (class_id,))
                c.execute("DELETE FROM sound_classes WHERE class_id=?", (class_id,))
        return True

    def delete_event(self, event_id):
        """Delete an event record. Returns clip_path so caller can delete the file."""
        with self._lock:
            with self._conn() as c:
                row = c.execute(
                    "SELECT clip_path FROM events WHERE id=?", (event_id,)
                ).fetchone()
                clip_path = row[0] if row else None
                c.execute("DELETE FROM events WHERE id=?", (event_id,))
        return clip_path

    def retag_event(self, event_id, dog_id):
        """Change which dog is assigned to an event."""
        with self._lock:
            with self._conn() as c:
                c.execute("UPDATE events SET dog_id=? WHERE id=?", (dog_id, event_id))

    def get_event_by_id(self, event_id):
        """Return a single event dict by ID, or None."""
        with self._conn() as c:
            cols_sel = (
                "id,timestamp,clip_path,bark_prob,peak_dbfs,avg_dbfs,"
                "duration,doa,dog_id,upload_status,upload_url,label,"
                "ch_energies,sound_class"
            )
            cur = c.execute(f"SELECT {cols_sel} FROM events WHERE id=?", (event_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            ev = dict(zip(cols, row))
            if ev.get("ch_energies"):
                try:
                    ev["ch_energies"] = json.loads(ev["ch_energies"])
                except Exception:
                    ev["ch_energies"] = None
            return ev

    # ── Queries ───────────────────────────────────────────────────────────────


    def get_recent_events(self, limit=50, dog_id=None):
        with self._conn() as c:
            cols_sel = (
                "id,timestamp,clip_path,bark_prob,peak_dbfs,avg_dbfs,"
                "duration,doa,dog_id,upload_status,upload_url,label,"
                "ch_energies,sound_class,ai_note,raw_clip_path"
            )
            if dog_id:
                cur = c.execute(
                    f"SELECT {cols_sel} FROM events WHERE dog_id=? ORDER BY id DESC LIMIT ?",
                    (dog_id, limit),
                )
            else:
                cur = c.execute(
                    f"SELECT {cols_sel} FROM events ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            for r in rows:
                if r.get("ch_energies"):
                    try:
                        r["ch_energies"] = json.loads(r["ch_energies"])
                    except Exception:
                        r["ch_energies"] = None
            return rows

    def get_dog_stats(self):
        with self._conn() as c:
            cur = c.execute(
                "SELECT d.dog_id, d.name, d.color, "
                "COUNT(e.id) as bark_count, "
                "AVG(e.bark_prob) as avg_prob, "
                "AVG(e.peak_dbfs) as avg_peak_dbfs, "
                "AVG(e.doa) as avg_doa, "
                "MAX(e.timestamp) as last_seen "
                "FROM dogs d LEFT JOIN events e ON d.dog_id = e.dog_id "
                "GROUP BY d.dog_id ORDER BY bark_count DESC"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_analytics(self):
        with self._conn() as c:
            hourly = c.execute(
                "SELECT strftime('%H', timestamp) as hour, COUNT(*) as count "
                "FROM events WHERE timestamp >= datetime('now', '-7 days') "
                "GROUP BY hour ORDER BY hour"
            ).fetchall()
            daily = c.execute(
                "SELECT date(timestamp) as day, COUNT(*) as count "
                "FROM events WHERE timestamp >= datetime('now', '-30 days') "
                "GROUP BY day ORDER BY day"
            ).fetchall()
            doa_rows = c.execute(
                "SELECT COALESCE(dog_id,'Unknown') as dog_id, doa "
                "FROM events WHERE doa IS NOT NULL"
            ).fetchall()
            today_count = c.execute(
                "SELECT COUNT(*) FROM events WHERE date(timestamp) = date('now')"
            ).fetchone()[0]
        return {
            "hourly":      [{"hour": int(r[0]), "count": r[1]} for r in hourly],
            "daily":       [{"date": r[0], "count": r[1]} for r in daily],
            "doa_points":  [{"dog_id": r[0], "doa": r[1]} for r in doa_rows],
            "today_count": today_count,
        }

    def update_dog_id(self, event_id, dog_id):
        with self._lock:
            with self._conn() as c:
                c.execute("UPDATE events SET dog_id=? WHERE id=?", (dog_id, event_id))
