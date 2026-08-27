"""
Shared logging layer.

Both the live-camera backend (app.py) and the standalone history
viewer (logs_server.py) import this module and point at the same
scan_log.db SQLite file.

Because the log lives in a plain file on disk -- not in either
process's memory -- the history server keeps working (and keeps
showing every past detection) even if the camera/inference server
(app.py) is stopped, restarted, or was never started at all --
AS LONG AS both processes are on the same computer.

If you want to view history from a DIFFERENT laptop than the one
running app3.py, this file alone can't help -- a local SQLite file
isn't visible over the network. See logs_server.py, which can run
in "remote" mode and pull the same data over HTTP from app3.py
instead of reading this file directly.
"""
import os
import sqlite3
import threading
from datetime import datetime

DB_PATH = os.environ.get("SCAN_DB_PATH", "scan_log.db")
_lock = threading.Lock()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            iso_time TEXT NOT NULL,
            type TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def log_scan(scan_type: str, confidence: float, ts=None):
    """Called by app.py right after a decision is made -- this is the
    'somehow send it to another backend/database' step. It just writes
    a row to the shared SQLite file."""
    ts = ts if ts is not None else datetime.now().timestamp()
    iso_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO scans (timestamp, iso_time, type, confidence) VALUES (?, ?, ?, ?)",
            (ts, iso_time, scan_type, confidence),
        )
        conn.commit()
        conn.close()


def get_history(limit=100, type_filter=None):
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        if type_filter in ("PLASTIC", "NON-PLASTIC"):
            rows = conn.execute(
                "SELECT id, iso_time, type, confidence FROM scans "
                "WHERE type = ? ORDER BY id DESC LIMIT ?",
                (type_filter, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, iso_time, type, confidence FROM scans "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_stats():
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        plastic = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE type = 'PLASTIC'"
        ).fetchone()[0]
        conn.close()
    return {"total": total, "plastic": plastic, "non_plastic": total - plastic}


def get_all_rows():
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT iso_time, type, confidence FROM scans ORDER BY id ASC"
        ).fetchall()
        conn.close()
    return rows


def clear_history():
    """Wipes every row from the scans table. Used by the 'Clear History'
    button. This is permanent -- there is no undo."""
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM scans")
        # Reset the autoincrement counter so new scans start at id 1 again.
        conn.execute("DELETE FROM sqlite_sequence WHERE name='scans'")
        conn.commit()
        conn.close()


# Make sure the table exists the moment either process imports this module.
init_db()
