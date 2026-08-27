from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import onnxruntime as ort
import numpy as np
import cv2
import threading
import time
import sqlite3
import csv
import io
from datetime import datetime

app = FastAPI()

# Load the new ResNet ONNX model
session = ort.InferenceSession("plastic_sorter_resnet.onnx")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

latest_frame = None
last_scan_data = {"type": "WAITING", "prob": 0, "timestamp": 0}

# ---------------------------------------------------------------------------
# SQLite logging (records every scan with timestamp, type, confidence)
# ---------------------------------------------------------------------------
DB_PATH = "scan_log.db"
db_lock = threading.Lock()


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


def log_scan(scan_type: str, confidence: float, ts: float):
    iso_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO scans (timestamp, iso_time, type, confidence) VALUES (?, ?, ?, ?)",
            (ts, iso_time, scan_type, confidence),
        )
        conn.commit()
        conn.close()


init_db()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def capture_frames():
    global latest_frame
    target_index = 1
    cap = cv2.VideoCapture(target_index)

    if cap.isOpened():
        print(f"Success! Connected to external webcam at index {target_index}")
    else:
        print(f"External webcam not found at index {target_index}. Falling back to laptop camera (0)...")
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
        time.sleep(0.03)


threading.Thread(target=capture_frames, daemon=True).start()


def generate_video_stream():
    global latest_frame
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Vision Unit · Plastic Sort Scanner</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0c0f; --panel: #14171c; --panel-raised: #1a1e24;
            --hairline: rgba(255,255,255,0.08); --grid: rgba(124,138,153,0.06);
            --steel: #7c8a99; --steel-dim: #4a5560; --ink: #eef1f4;
            --cyan: #2dd4bf; --amber: #ffb020; --green: #00e08a; --coral: #ff5a5f;
            --mono: 'JetBrains Mono', ui-monospace, monospace;
            --sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }

        html, body {
            margin: 0; height: 100%; background: var(--bg); color: var(--ink);
            font-family: var(--sans);
        }

        body {
            display: flex; flex-direction: column; align-items: center;
            min-height: 100vh; padding: 20px; position: relative;
            background-image:
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid) 1px, transparent 1px);
            background-size: 28px 28px;
        }

        .header { text-align: center; margin-bottom: 22px; }
        .eyebrow {
            font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.22em;
            color: var(--cyan); text-transform: uppercase; margin: 0 0 8px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .eyebrow::before, .eyebrow::after { content: ''; width: 14px; height: 1px; background: var(--steel-dim); }
        h1 {
            font-family: var(--mono); font-weight: 800; font-size: 1.35rem;
            letter-spacing: 0.03em; margin: 0; color: var(--ink);
        }
        .subhead { font-family: var(--mono); font-size: 0.72rem; color: var(--steel); margin-top: 6px; letter-spacing: 0.05em; }

        .viewfinder {
            position: relative; width: min(85vw, 340px); aspect-ratio: 1/1;
            border-radius: 4px; overflow: hidden; background: #000;
            box-shadow: 0 20px 60px rgba(0,0,0,0.6);
        }
        img#webcam { width: 100%; height: 100%; object-fit: cover; display: block; filter: saturate(1.05) contrast(1.02); }

        .corner {
            position: absolute; width: 26px; height: 26px; pointer-events: none;
            border-color: var(--cyan); transition: border-color 0.3s ease;
        }
        .corner.tl { top: 10px; left: 10px; border-top: 2px solid; border-left: 2px solid; }
        .corner.tr { top: 10px; right: 10px; border-top: 2px solid; border-right: 2px solid; }
        .corner.bl { bottom: 10px; left: 10px; border-bottom: 2px solid; border-left: 2px solid; }
        .corner.br { bottom: 10px; right: 10px; border-bottom: 2px solid; border-right: 2px solid; }
        .viewfinder.locked .corner { border-color: var(--green); }
        .viewfinder.warn .corner { border-color: var(--amber); }

        .crop-guide {
            position: absolute; top: 6.25%; left: 6.25%; width: 87.5%; height: 87.5%;
            border: 1px dashed rgba(255,255,255,0.18); border-radius: 6px; pointer-events: none;
        }

        .scanline {
            position: absolute; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
            box-shadow: 0 0 12px 1px var(--cyan);
            top: 0; opacity: 0.85; pointer-events: none;
            animation: sweep 2.6s ease-in-out infinite;
        }
        @keyframes sweep { 0% { top: 6%; opacity: 0; } 8% { opacity: 0.85; } 50% { top: 94%; opacity: 0.85; } 92% { opacity: 0; } 100% { top: 6%; opacity: 0; } }
        .viewfinder.locked .scanline, .viewfinder.warn .scanline { animation-play-state: paused; opacity: 0; }

        .flash { position: absolute; inset: 0; background: #fff; opacity: 0; pointer-events: none; }
        .flash.fire { animation: flash-fire 0.35s ease-out; }
        @keyframes flash-fire { 0% { opacity: 0.55; } 100% { opacity: 0; } }

        .readout {
            width: min(85vw, 340px); margin-top: 16px; background: var(--panel);
            border: 1px solid var(--hairline); border-radius: 10px; padding: 14px 16px;
        }
        .readout-row { display: flex; align-items: baseline; justify-content: space-between; }
        .readout-label { font-family: var(--mono); font-size: 0.68rem; color: var(--steel); letter-spacing: 0.15em; text-transform: uppercase; }
        #status { font-family: var(--mono); font-weight: 600; font-size: 0.95rem; color: var(--ink); margin-top: 4px; min-height: 22px; text-align: left; letter-spacing: 0.02em; }

        .meter { margin-top: 12px; display: flex; gap: 3px; height: 14px; }
        .meter-seg { flex: 1; background: var(--panel-raised); border-radius: 1px; transition: background 0.15s ease; }
        .meter-seg.on { background: var(--steel); }
        .meter-seg.on.plastic { background: var(--green); }
        .meter-seg.on.nonplastic { background: var(--amber); }

        /* --- Stats row --- */
        .stats-row {
            width: min(85vw, 340px); margin-top: 12px; display: flex; gap: 8px;
        }
        .stat-chip {
            flex: 1; background: var(--panel); border: 1px solid var(--hairline);
            border-radius: 10px; padding: 10px 6px; text-align: center;
        }
        .stat-chip .stat-val { font-family: var(--mono); font-weight: 700; font-size: 1.1rem; color: var(--ink); }
        .stat-chip .stat-label {
            font-family: var(--mono); font-size: 0.58rem; color: var(--steel);
            letter-spacing: 0.12em; text-transform: uppercase; margin-top: 3px;
        }
        .stat-chip.plastic .stat-val { color: var(--green); }
        .stat-chip.nonplastic .stat-val { color: var(--amber); }

        /* --- Log panel --- */
        .log-panel {
            width: min(85vw, 340px); margin-top: 12px; background: var(--panel);
            border: 1px solid var(--hairline); border-radius: 10px;
            padding: 10px 14px 12px; display: flex; flex-direction: column;
        }
        .log-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .log-title { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.15em; color: var(--steel); text-transform: uppercase; }
        .log-export {
            font-family: var(--mono); font-size: 0.6rem; color: var(--cyan);
            text-decoration: none; border: 1px solid var(--hairline);
            padding: 4px 8px; border-radius: 5px; transition: border-color 0.2s ease;
        }
        .log-export:hover { border-color: var(--cyan); }
        .log-list { max-height: 190px; overflow-y: auto; font-family: var(--mono); font-size: 0.72rem; }
        .log-list::-webkit-scrollbar { width: 5px; }
        .log-list::-webkit-scrollbar-thumb { background: var(--steel-dim); border-radius: 4px; }
        .log-row {
            display: grid; grid-template-columns: 1fr auto auto; gap: 10px;
            padding: 6px 0; border-bottom: 1px solid var(--hairline); align-items: center;
        }
        .log-row:last-child { border-bottom: none; }
        .log-time { color: var(--steel); }
        .log-type { font-weight: 600; text-align: right; }
        .log-type.plastic { color: var(--green); }
        .log-type.nonplastic { color: var(--amber); }
        .log-conf { color: var(--steel-dim); text-align: right; min-width: 34px; }
        .log-empty { color: var(--steel-dim); text-align: center; padding: 14px 0; }
    </style>
    </head>
    <body>
        <div class="header">
            <p class="eyebrow">Material Recovery · Vision Unit</p>
            <h1>PLASTIC SORT SCANNER</h1>
            <p class="subhead">Center item in frame — hold steady</p>
        </div>

        <div class="viewfinder" id="viewfinder">
            <img id="webcam" src="/video_feed" />
            <div class="crop-guide"></div>
            <div class="scanline" id="scanline"></div>
            <div class="corner tl"></div>
            <div class="corner tr"></div>
            <div class="corner bl"></div>
            <div class="corner br"></div>
            <div class="flash" id="flash"></div>
        </div>

        <div class="readout">
            <div class="readout-row">
                <span class="readout-label">Status</span>
                <span class="readout-label" id="modelTag">BACKEND ONLINE</span>
            </div>
            <div id="status" aria-live="polite">Waiting for ESP32 trigger...</div>
            <div class="meter" id="meter"></div>
        </div>

        <div class="stats-row">
            <div class="stat-chip">
                <div class="stat-val" id="statTotal">0</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-chip plastic">
                <div class="stat-val" id="statPlastic">0</div>
                <div class="stat-label">Plastic</div>
            </div>
            <div class="stat-chip nonplastic">
                <div class="stat-val" id="statNonPlastic">0</div>
                <div class="stat-label">Non-Plastic</div>
            </div>
        </div>

        <div class="log-panel">
            <div class="log-header">
                <span class="log-title">Scan Log</span>
                <a class="log-export" href="/export/csv" download>Export CSV</a>
            </div>
            <div class="log-list" id="logList">
                <div class="log-empty">No scans yet</div>
            </div>
        </div>

        <script>
            const METER_SEGMENTS = 10;
            const meterEl = document.getElementById('meter');
            const statusEl = document.getElementById('status');
            const viewfinder = document.getElementById('viewfinder');
            const flashEl = document.getElementById('flash');
            let lastTimestamp = 0;

            for (let i = 0; i < METER_SEGMENTS; i++) {
                const seg = document.createElement('div');
                seg.className = 'meter-seg';
                meterEl.appendChild(seg);
            }
            const segments = Array.from(meterEl.children);

            function updateUI(data) {
                // Flash animation
                flashEl.classList.remove('fire');
                void flashEl.offsetWidth;
                flashEl.classList.add('fire');

                viewfinder.classList.remove('locked', 'warn');

                let isPlastic = data.type === 'PLASTIC';
                let typeCls = isPlastic ? 'plastic' : 'nonplastic';
                let color = isPlastic ? 'var(--green)' : 'var(--amber)';

                viewfinder.classList.add(isPlastic ? 'locked' : 'warn');
                statusEl.style.color = color;
                statusEl.textContent = `${data.type} — ${Math.round(data.prob * 100)}%`;

                // Update Meter
                const filled = Math.round(data.prob * METER_SEGMENTS);
                segments.forEach((seg, i) => {
                    seg.classList.remove('on', 'plastic', 'nonplastic');
                    if (i < filled) {
                        seg.classList.add('on', typeCls);
                    }
                });

                // A new scan just happened — refresh stats/log right away
                refreshStats();
                refreshHistory();

                // Reset view after delay
                setTimeout(() => {
                    viewfinder.classList.remove('locked', 'warn');
                    statusEl.textContent = 'Waiting for ESP32 trigger...';
                    statusEl.style.color = 'var(--ink)';
                    segments.forEach(seg => seg.classList.remove('on', 'plastic', 'nonplastic'));
                }, 2000);
            }

            async function refreshStats() {
                try {
                    const res = await fetch('/stats');
                    const s = await res.json();
                    document.getElementById('statTotal').textContent = s.total;
                    document.getElementById('statPlastic').textContent = s.plastic;
                    document.getElementById('statNonPlastic').textContent = s.non_plastic;
                } catch (e) { /* ignore transient errors */ }
            }

            async function refreshHistory() {
                try {
                    const res = await fetch('/history?limit=15');
                    const rows = await res.json();
                    const listEl = document.getElementById('logList');
                    if (!rows.length) {
                        listEl.innerHTML = '<div class="log-empty">No scans yet</div>';
                        return;
                    }
                    listEl.innerHTML = rows.map(r => {
                        const cls = r.type === 'PLASTIC' ? 'plastic' : 'nonplastic';
                        const timePart = r.iso_time.split(' ')[1] || r.iso_time;
                        return `<div class="log-row">
                            <span class="log-time">${timePart}</span>
                            <span class="log-type ${cls}">${r.type}</span>
                            <span class="log-conf">${Math.round(r.confidence * 100)}%</span>
                        </div>`;
                    }).join('');
                } catch (e) { /* ignore transient errors */ }
            }

            // Poll backend for updates from ESP32
            setInterval(async () => {
                try {
                    let res = await fetch('/scan_status');
                    let data = await res.json();

                    if (data.timestamp > lastTimestamp && lastTimestamp !== 0) {
                        updateUI(data);
                    }
                    lastTimestamp = data.timestamp;
                } catch (e) { console.error("Error fetching status"); }
            }, 500);

            // Periodic refresh in case another client triggers a scan
            setInterval(() => { refreshStats(); refreshHistory(); }, 4000);

            refreshStats();
            refreshHistory();
        </script>
    </body>
    </html>
    """


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/scan_status")
def scan_status():
    return last_scan_data


@app.get("/history")
def get_history(limit: int = 50):
    """Return the most recent scan records, newest first."""
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT iso_time, type, confidence FROM scans ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


@app.get("/stats")
def get_stats():
    """Return running totals for the readout chips."""
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        plastic = conn.execute("SELECT COUNT(*) FROM scans WHERE type = 'PLASTIC'").fetchone()[0]
        conn.close()
    return {"total": total, "plastic": plastic, "non_plastic": total - plastic}


@app.get("/export/csv")
def export_csv():
    """Download the full scan history as a CSV file."""
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT iso_time, type, confidence FROM scans ORDER BY id ASC"
        ).fetchall()
        conn.close()

    def generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "type", "confidence"])
        yield buf.getvalue()
        for row in rows:
            buf.seek(0)
            buf.truncate(0)
            writer.writerow(row)
            yield buf.getvalue()

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scan_log.csv"},
    )


@app.get("/scan")
def scan_item():
    global latest_frame, last_scan_data
    if latest_frame is None:
        return {"error": "Camera initializing"}

    frame = latest_frame.copy()

    # --- Accurate Cropping Math + INTER_AREA Downscaling ---
    h, w = frame.shape[:2]

    # 1. Calculate the exact square crop region (mimicking JS logic)
    min_dim = min(h, w)
    crop_ratio = 224 / 256
    crop_size = int(min_dim * crop_ratio)

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2

    # 2. Extract the large square from the original high-res frame
    cropped_img = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # 3. Shrink to 224x224 using INTER_AREA (Critical for accuracy when downscaling)
    img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- ResNet Normalization ---
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    img = np.transpose(img, (2, 0, 1))
    input_data = np.expand_dims(img, axis=0)

    # --- ResNet Inference ---
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})

    logits = result[0][0]
    probs = softmax(logits)

    prob_plastic = probs[0]
    prob_non_plastic = probs[1]

    # Decision logic
    is_plastic = prob_plastic >= 0.40

    scan_type = "PLASTIC" if is_plastic else "NON-PLASTIC"
    confidence = float(prob_plastic if is_plastic else prob_non_plastic)
    ts = time.time()

    # Update global state for the UI to consume
    last_scan_data = {
        "type": scan_type,
        "prob": confidence,
        "timestamp": ts,
    }

    # Persist the reading (timestamp + detected type) for the log/history views
    log_scan(scan_type, confidence, ts)

    if is_plastic:
        print(f"Plastic! (Confidence: {prob_plastic*100:.1f}%)")
        return {"angle": 0}
    else:
        print(f"Non-Plastic! (Confidence: {prob_non_plastic*100:.1f}%)")
        return {"angle": 180}
