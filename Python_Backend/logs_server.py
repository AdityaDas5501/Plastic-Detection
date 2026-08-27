"""
Standalone scan-history server.

Run this as its own process, on its own port, completely independent
of app3.py (the camera/inference server):

    uvicorn logs_server:app --host 0.0.0.0 --port 8001

TWO MODES, controlled by the APP3_HOST environment variable:

1. LOCAL MODE (default -- APP3_HOST not set)
   Use this when logs_server.py runs on the SAME laptop as app3.py.
   It reads scan_log.db straight off disk via db.py, so it keeps
   working -- and keeps showing every past detection -- even if
   app3.py is stopped, crashed, or was never started.

2. REMOTE MODE (set APP3_HOST)
   Use this when logs_server.py runs on a DIFFERENT laptop than
   app3.py. A local SQLite file can't be seen from another machine,
   so in this mode logs_server.py instead calls app3.py's HTTP API
   (added to app3.py) over the network. This means app3.py DOES need
   to be running and reachable for this mode to show anything.

   To use it, find the IP address of the laptop running app3.py on
   your local network (e.g. 192.168.1.42), then start logs_server.py
   like this:

       # macOS/Linux
       APP3_HOST=http://192.168.1.42:8000 uvicorn logs_server:app --host 0.0.0.0 --port 8001

       # Windows (PowerShell)
       $env:APP3_HOST="http://192.168.1.42:8000"; uvicorn logs_server:app --host 0.0.0.0 --port 8001

   Also make sure port 8000 is reachable from the other laptop (same
   Wi-Fi network, and no firewall blocking it).
"""
import os
import csv
import io

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

APP3_HOST = os.environ.get("APP3_HOST", "").rstrip("/")
REMOTE_MODE = bool(APP3_HOST)

if REMOTE_MODE:
    import requests
else:
    import db  # shared SQLite logging module -- only needed in local mode

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    if REMOTE_MODE:
        mode_note = f"Remote mode — pulling scan history live from app3.py at {APP3_HOST}"
    else:
        mode_note = "Local mode — reads scan_log.db directly and does not depend on the camera server being online."
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scan History · Vision Unit</title>
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
        * { box-sizing: border-box; }
        html, body {
            margin: 0; min-height: 100%; background: var(--bg); color: var(--ink);
            font-family: var(--sans);
        }
        body {
            display: flex; flex-direction: column; align-items: center;
            padding: 28px 16px 60px;
            background-image:
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid) 1px, transparent 1px);
            background-size: 28px 28px;
        }
        .wrap { width: 100%; max-width: 760px; }

        .header { text-align: center; margin-bottom: 20px; }
        .eyebrow {
            font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.22em;
            color: var(--cyan); text-transform: uppercase; margin: 0 0 8px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
        }
        .eyebrow::before, .eyebrow::after { content: ''; width: 14px; height: 1px; background: var(--steel-dim); }
        h1 { font-family: var(--mono); font-weight: 800; font-size: 1.35rem; margin: 0; }
        .subhead { font-family: var(--mono); font-size: 0.7rem; color: var(--steel); margin-top: 6px; }

        .stats-row { display: flex; gap: 10px; margin-bottom: 16px; }
        .stat-chip {
            flex: 1; background: var(--panel); border: 1px solid var(--hairline);
            border-radius: 10px; padding: 14px 8px; text-align: center;
        }
        .stat-chip .stat-val { font-family: var(--mono); font-weight: 700; font-size: 1.4rem; }
        .stat-chip .stat-label { font-family: var(--mono); font-size: 0.62rem; color: var(--steel); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 4px; }
        .stat-chip.plastic .stat-val { color: var(--green); }
        .stat-chip.nonplastic .stat-val { color: var(--amber); }

        .toolbar {
            display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
            margin-bottom: 12px;
        }
        .filter-btn, .action-btn {
            font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.05em;
            background: var(--panel); color: var(--steel); border: 1px solid var(--hairline);
            padding: 7px 12px; border-radius: 7px; cursor: pointer; text-decoration: none;
        }
        .filter-btn.active { color: var(--cyan); border-color: var(--cyan); }
        .action-btn { color: var(--cyan); margin-left: auto; }
        .action-btn:hover, .filter-btn:hover { border-color: var(--steel); }
        .last-updated { font-family: var(--mono); font-size: 0.62rem; color: var(--steel-dim); margin-left: 4px; }

        .table-panel {
            background: var(--panel); border: 1px solid var(--hairline);
            border-radius: 10px; overflow: hidden;
        }
        table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.8rem; }
        thead th {
            text-align: left; font-size: 0.62rem; letter-spacing: 0.12em; color: var(--steel);
            text-transform: uppercase; padding: 12px 16px; background: var(--panel-raised);
            position: sticky; top: 0;
        }
        thead th.num { text-align: right; }
        tbody td { padding: 10px 16px; border-top: 1px solid var(--hairline); }
        tbody td.num { text-align: right; color: var(--steel); }
        tbody tr:hover { background: rgba(255,255,255,0.02); }
        .badge { font-weight: 700; padding: 2px 8px; border-radius: 5px; font-size: 0.72rem; }
        .badge.plastic { color: var(--green); background: rgba(0,224,138,0.1); }
        .badge.nonplastic { color: var(--amber); background: rgba(255,176,32,0.1); }
        .table-scroll { max-height: 60vh; overflow-y: auto; }
        .empty-state { text-align: center; padding: 40px 20px; color: var(--steel-dim); font-family: var(--mono); font-size: 0.8rem; }
        .offline-note {
            font-family: var(--mono); font-size: 0.64rem; color: var(--steel-dim);
            text-align: center; margin-top: 14px; letter-spacing: 0.02em;
        }
        .action-btn.danger { color: var(--coral); }
        .action-btn.danger:hover { border-color: var(--coral); }
        .modal-backdrop {
            display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6);
            align-items: center; justify-content: center; z-index: 10; padding: 20px;
        }
        .modal-backdrop.show { display: flex; }
        .modal {
            background: var(--panel); border: 1px solid var(--hairline); border-radius: 10px;
            padding: 20px; max-width: 340px; width: 100%; text-align: center;
        }
        .modal p { font-family: var(--mono); font-size: 0.78rem; color: var(--ink); margin: 0 0 16px; }
        .modal-buttons { display: flex; gap: 10px; }
        .modal-buttons button {
            flex: 1; font-family: var(--mono); font-size: 0.72rem; padding: 9px;
            border-radius: 7px; cursor: pointer; letter-spacing: 0.05em;
        }
        .modal-buttons .confirm { background: var(--coral); color: #1a0d0e; border: none; font-weight: 700; }
        .modal-buttons .cancel { background: transparent; color: var(--steel); border: 1px solid var(--hairline); }
    </style>
    </head>
    <body>
        <div class="wrap">
            <div class="header">
                <p class="eyebrow">Material Recovery · Vision Unit</p>
                <h1>SCAN HISTORY</h1>
                <p class="subhead">Persistent log — works whether or not the camera server is running</p>
            </div>

            <div class="stats-row">
                <div class="stat-chip">
                    <div class="stat-val" id="statTotal">–</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-chip plastic">
                    <div class="stat-val" id="statPlastic">–</div>
                    <div class="stat-label">Plastic</div>
                </div>
                <div class="stat-chip nonplastic">
                    <div class="stat-val" id="statNonPlastic">–</div>
                    <div class="stat-label">Non-Plastic</div>
                </div>
            </div>

            <div class="toolbar">
                <button class="filter-btn active" data-filter="">All</button>
                <button class="filter-btn" data-filter="PLASTIC">Plastic</button>
                <button class="filter-btn" data-filter="NON-PLASTIC">Non-Plastic</button>
                <span class="last-updated" id="lastUpdated"></span>
                <a class="action-btn" href="/export/csv" download>⬇ Export CSV</a>
                <button class="action-btn danger" id="clearBtn">🗑 Clear</button>
            </div>

            <div class="table-panel">
                <div class="table-scroll">
                    <table>
                        <thead>
                            <tr>
                                <th class="num">#</th>
                                <th>Timestamp</th>
                                <th>Detected</th>
                                <th class="num">Confidence</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                    <div class="empty-state" id="emptyState" style="display:none;">No scans recorded yet.</div>
                </div>
            </div>

            <p class="offline-note" id="modeNote">__MODE_NOTE__</p>
        </div>

        <div class="modal-backdrop" id="clearModal">
            <div class="modal">
                <p>Delete all scan history? This can't be undone.</p>
                <div class="modal-buttons">
                    <button class="cancel" id="clearCancel">Cancel</button>
                    <button class="confirm" id="clearConfirm">Delete all</button>
                </div>
            </div>
        </div>

        <script>
            let currentFilter = "";
            const tableBody = document.getElementById('tableBody');
            const emptyState = document.getElementById('emptyState');
            const lastUpdatedEl = document.getElementById('lastUpdated');
            const clearBtn = document.getElementById('clearBtn');
            const clearModal = document.getElementById('clearModal');

            clearBtn.addEventListener('click', () => clearModal.classList.add('show'));
            document.getElementById('clearCancel').addEventListener('click', () => clearModal.classList.remove('show'));
            document.getElementById('clearConfirm').addEventListener('click', async () => {
                clearModal.classList.remove('show');
                try {
                    const res = await fetch('/api/clear', { method: 'POST' });
                    if (!res.ok) throw new Error('clear failed');
                    refreshAll();
                } catch (e) {
                    lastUpdatedEl.textContent = 'clear failed — is app3.py reachable?';
                }
            });

            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    currentFilter = btn.dataset.filter;
                    refreshAll();
                });
            });

            async function refreshStats() {
                try {
                    const res = await fetch('/api/stats');
                    const s = await res.json();
                    document.getElementById('statTotal').textContent = s.total;
                    document.getElementById('statPlastic').textContent = s.plastic;
                    document.getElementById('statNonPlastic').textContent = s.non_plastic;
                } catch (e) { /* server unreachable momentarily, ignore */ }
            }

            async function refreshHistory() {
                try {
                    const url = '/api/history?limit=200' + (currentFilter ? '&type=' + encodeURIComponent(currentFilter) : '');
                    const res = await fetch(url);
                    const rows = await res.json();

                    if (!rows.length) {
                        tableBody.innerHTML = '';
                        emptyState.style.display = 'block';
                        return;
                    }
                    emptyState.style.display = 'none';

                    tableBody.innerHTML = rows.map((r, i) => {
                        const cls = r.type === 'PLASTIC' ? 'plastic' : 'nonplastic';
                        return `<tr>
                            <td class="num">${rows.length - i}</td>
                            <td>${r.iso_time}</td>
                            <td><span class="badge ${cls}">${r.type}</span></td>
                            <td class="num">${Math.round(r.confidence * 100)}%</td>
                        </tr>`;
                    }).join('');

                    lastUpdatedEl.textContent = 'updated ' + new Date().toLocaleTimeString();
                } catch (e) {
                    lastUpdatedEl.textContent = 'connection lost — retrying...';
                }
            }

            function refreshAll() { refreshStats(); refreshHistory(); }

            refreshAll();
            setInterval(refreshAll, 3000);
        </script>
    </body>
    </html>
    """
    return html.replace("__MODE_NOTE__", mode_note)


if REMOTE_MODE:
    # ---- Remote mode: proxy every call over HTTP to app3.py's machine ----

    @app.get("/api/history")
    def api_history(limit: int = 200, type: str = None):
        try:
            params = {"limit": limit}
            if type:
                params["type"] = type
            r = requests.get(f"{APP3_HOST}/api/history", params=params, timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return JSONResponse(
                status_code=503,
                content={"error": f"Can't reach app3.py at {APP3_HOST}. Is it running and on the same network?"},
            )

    @app.get("/api/stats")
    def api_stats():
        try:
            r = requests.get(f"{APP3_HOST}/api/stats", timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return JSONResponse(
                status_code=503,
                content={"error": f"Can't reach app3.py at {APP3_HOST}"},
            )

    @app.get("/export/csv")
    def export_csv():
        try:
            r = requests.get(f"{APP3_HOST}/export/csv", timeout=10)
            r.raise_for_status()
        except requests.RequestException:
            return Response(content="Could not reach app3.py to export history.", status_code=503)
        return Response(
            content=r.content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=scan_log.csv"},
        )

    @app.post("/api/clear")
    def api_clear():
        try:
            r = requests.post(f"{APP3_HOST}/api/clear", timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return JSONResponse(
                status_code=503,
                content={"error": f"Can't reach app3.py at {APP3_HOST}"},
            )

else:
    # ---- Local mode: read scan_log.db directly, same machine as app3.py ----

    @app.get("/api/history")
    def api_history(limit: int = 200, type: str = None):
        return db.get_history(limit=limit, type_filter=type)

    @app.get("/api/stats")
    def api_stats():
        return db.get_stats()

    @app.get("/export/csv")
    def export_csv():
        rows = db.get_all_rows()

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

    @app.post("/api/clear")
    def api_clear():
        db.clear_history()
        return {"status": "cleared"}
