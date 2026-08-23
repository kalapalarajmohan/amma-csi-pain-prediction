#!/usr/bin/env python3
"""
CSI SIMPLE DASHBOARD - one file, one run. Built for a plain-language read,
not a signal-processing read.

Instead of a raw subcarrier heatmap (meaningless unless you already know
CSI), this shows:
  - One big word: CALM / MOVEMENT / NO SIGNAL
  - A simple line graph of "activity level" over the last few minutes -
    up means more movement, flat/low means still
  - A plain-English sentence explaining the current reading

Reads the same live_amp.log file amma_40nights.py is already writing to.
Does not touch or interrupt the running collection service.

Run on the Jetson:
    python3 csi_dashboard_simple.py

Then open, from any browser on the same network:
    http://<jetson-ip>:8000

If port 8000 is already taken by a previous run, free it first:
    sudo fuser -k 8000/tcp
"""

import json
import os
import socketserver
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

# ---------------------------------------------------------------- CONFIG ---
LOG_FILE = "/home/rajmohan/csi_sciatica/data/live_amp.log"
HTTP_PORT = 8000

# How many recent activity readings to keep for the line graph.
# One reading is computed roughly every 0.5s, so 240 ~= last 2 minutes.
TREND_LEN = 240

# -------------------------------------------------------------- STATE -----
_lock = threading.Lock()
_recent_rows = deque(maxlen=60)      # raw amplitude vectors, short window for "now" reading
_trend = deque(maxlen=TREND_LEN)     # (timestamp, activity_level) for the line graph
_stats = {
    "connected": False,
    "last_line_ts": 0.0,
    "line_count": 0,
    "line_rate": 0.0,
    "baseline": None,
    "log_error": None,
}

# ---------------------------------------------------------- LOG PARSING ---
def parse_line(line):
    """Expected shape: '<unix_timestamp> <v1>,<v2>,...,<vN>\\n'"""
    line = line.strip()
    if not line:
        return None
    parts = line.split(" ", 1)
    if len(parts) != 2:
        return None
    _ts_str, vals_str = parts
    try:
        vals = [float(x) for x in vals_str.split(",")]
    except ValueError:
        return None
    if len(vals) < 4:
        return None
    return vals

def row_energy(vals):
    return sum(vals) / len(vals)

def tail_file():
    while not os.path.exists(LOG_FILE):
        with _lock:
            _stats["log_error"] = f"waiting for {LOG_FILE} to exist"
        time.sleep(1.0)

    with _lock:
        _stats["log_error"] = None

    f = open(LOG_FILE, "r", errors="replace")
    f.seek(0, os.SEEK_END)

    last_rate_check = time.time()
    count_since_check = 0
    last_trend_ts = 0.0

    while True:
        line = f.readline()
        if not line:
            time.sleep(0.05)
            with _lock:
                if time.time() - _stats["last_line_ts"] > 3.0:
                    _stats["connected"] = False
            continue

        vals = parse_line(line)
        if vals is None:
            continue

        now = time.time()
        with _lock:
            _recent_rows.append(vals)
            _stats["connected"] = True
            _stats["last_line_ts"] = now
            _stats["line_count"] += 1
        count_since_check += 1

        # sample the trend line at ~2Hz instead of every packet - keeps the
        # graph smooth and readable instead of a jittery mess
        if now - last_trend_ts >= 0.5:
            with _lock:
                recent = list(_recent_rows)[-10:]
            if recent:
                energy = row_energy([row_energy(r) for r in recent] if False else recent[-1])
                # activity = short-term variability of energy, more forgiving than a single row
                energies = [row_energy(r) for r in recent]
                mean_e = sum(energies) / len(energies)
                variability = (sum((e - mean_e) ** 2 for e in energies) / len(energies)) ** 0.5
                with _lock:
                    _trend.append((now, variability))
            last_trend_ts = now

        if now - last_rate_check >= 1.0:
            with _lock:
                _stats["line_rate"] = count_since_check / (now - last_rate_check)
            count_since_check = 0
            last_rate_check = now

# --------------------------------------------------------- FEATURE CALC ---
def compute_readout():
    with _lock:
        recent = list(_recent_rows)[-20:]
        trend = list(_trend)
        baseline = _stats["baseline"]
        connected = _stats["connected"]
        line_rate = _stats["line_rate"]
        log_error = _stats["log_error"]

    if log_error:
        return {
            "status": "NO SIGNAL", "status_color": "bad",
            "message": "Waiting for the collection service to start writing data.",
            "trend": [], "connected": False, "line_rate": 0.0,
        }

    if len(recent) < 5:
        return {
            "status": "WAITING", "status_color": "dim",
            "message": "Warming up - collecting the first few readings.",
            "trend": [], "connected": connected, "line_rate": round(line_rate, 1),
        }

    energies = [row_energy(r) for r in recent]
    mean_e = sum(energies) / len(energies)
    variability = (sum((e - mean_e) ** 2 for e in energies) / len(energies)) ** 0.5

    # Simple, honest thresholds - not a trained model, just "how much is this
    # bouncing around compared to a moment ago". Movement makes it bounce more.
    if baseline is not None:
        rel = variability / (baseline + 1e-6)
        if rel > 1.8:
            status, color = "MOVEMENT", "warn"
            msg = "Signal is changing a lot right now - likely someone moving in the room."
        elif rel > 1.15:
            status, color = "SOME ACTIVITY", "warn"
            msg = "Signal is changing more than the quiet baseline - small movement or shifting."
        else:
            status, color = "CALM", "good"
            msg = "Signal is steady, close to the quiet baseline you set."
    else:
        if variability > 3.0:
            status, color = "MOVEMENT", "warn"
            msg = "Signal is changing a lot right now - likely someone moving in the room."
        elif variability > 1.2:
            status, color = "SOME ACTIVITY", "warn"
            msg = "Signal shows some change - small movement or shifting."
        else:
            status, color = "CALM", "good"
            msg = "Signal is steady right now. Tip: click 'Set quiet baseline' when the room is empty for a clearer read."

    trend_out = [{"t": t, "v": round(v, 4)} for t, v in trend]

    return {
        "status": status,
        "status_color": color,
        "message": msg,
        "trend": trend_out,
        "connected": connected,
        "line_rate": round(line_rate, 1),
    }

# --------------------------------------------------------------- HTML -----
PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Amma CSI Monitor</title>
<style>
  :root {
    --bg: #0b0f14; --panel: #121821; --line: #232b36;
    --fg: #e7edf5; --dim: #7c8a9c; --accent: #4fd1c5;
    --warn: #f6ad55; --bad: #fc8181; --good: #68d391;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    padding: 24px; max-width: 720px; margin: 0 auto;
  }
  h1 { font-size: 16px; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 24px; text-align: center;}

  .status-card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 36px 24px; text-align: center; margin-bottom: 16px;
  }
  .status-word { font-size: 42px; font-weight: 800; letter-spacing: 0.02em; margin-bottom: 10px; }
  .status-word.good { color: var(--good); }
  .status-word.warn { color: var(--warn); }
  .status-word.bad { color: var(--bad); }
  .status-word.dim { color: var(--dim); }
  .status-message { color: var(--dim); font-size: 15px; line-height: 1.5; max-width: 480px; margin: 0 auto; }

  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 20px; margin-bottom: 16px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); margin: 0 0 14px; font-weight: 600; }
  canvas { width: 100%; display: block; }

  .footer-row { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--dim); }
  .dot-row { display: flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--bad); }
  .dot.on { background: var(--good); }

  button {
    background: var(--accent); border: none; color: #06201d;
    font-weight: 600; padding: 12px 16px; border-radius: 10px;
    cursor: pointer; font-size: 14px; width: 100%; margin-top: 4px;
  }
  button:hover { opacity: 0.9; }

  .hint { font-size: 12px; color: var(--dim); text-align: center; margin-top: 8px; line-height: 1.5; }
</style>
</head>
<body>
  <h1>Amma &mdash; Room Activity Monitor</h1>

  <div class="status-card">
    <div id="statusWord" class="status-word dim">&mdash;</div>
    <div id="statusMessage" class="status-message">Connecting&hellip;</div>
  </div>

  <div class="panel">
    <h2>Activity over the last 2 minutes</h2>
    <canvas id="trend" width="640" height="160"></canvas>
  </div>

  <div class="panel">
    <button id="markBaseline">Set quiet baseline (click when room is empty/still)</button>
    <div class="hint">This tells the system what "calm" looks like in this room, so readings are more accurate.</div>
  </div>

  <div class="footer-row">
    <div class="dot-row"><div id="dot" class="dot"></div><span id="connText">Waiting for data&hellip;</span></div>
    <span id="rate">&mdash;</span>
  </div>
</body>
<script>
const canvas = document.getElementById('trend');
const ctx = canvas.getContext('2d');

function drawTrend(points) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // grid baseline
  ctx.strokeStyle = '#232b36';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H - 20);
  ctx.lineTo(W, H - 20);
  ctx.stroke();

  if (!points || points.length < 2) return;

  const vals = points.map(p => p.v);
  const vmax = Math.max(...vals, 0.5);
  const pad = 10;

  ctx.strokeStyle = '#4fd1c5';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = (i / (points.length - 1)) * W;
    const y = H - pad - (p.v / vmax) * (H - pad * 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // soft fill under the line
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = 'rgba(79, 209, 197, 0.08)';
  ctx.fill();
}

async function poll() {
  try {
    const res = await fetch('/data');
    const d = await res.json();

    document.getElementById('dot').className = 'dot' + (d.connected ? ' on' : '');
    document.getElementById('connText').textContent = d.connected ? 'Live' : 'No new data';
    document.getElementById('rate').textContent = d.line_rate + ' readings/sec';

    const w = document.getElementById('statusWord');
    w.textContent = d.status;
    w.className = 'status-word ' + d.status_color;
    document.getElementById('statusMessage').textContent = d.message;

    drawTrend(d.trend);
  } catch (e) {
    document.getElementById('connText').textContent = 'Dashboard unreachable';
  }
  setTimeout(poll, 500);
}

document.getElementById('markBaseline').addEventListener('click', async () => {
  const btn = document.getElementById('markBaseline');
  btn.textContent = 'Baseline set ✓';
  await fetch('/mark_baseline', { method: 'POST' });
  setTimeout(() => { btn.textContent = 'Set quiet baseline (click when room is empty/still)'; }, 2000);
});

poll();
</script>
</html>
"""

# ------------------------------------------------------------- SERVER -----
class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="text/html"):
        body_b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body_b)))
        self.end_headers()
        self.wfile.write(body_b)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE)
        elif self.path == "/data":
            readout = compute_readout()
            self._send(200, json.dumps(readout), "application/json")
        else:
            self._send(404, "not found")

    def do_POST(self):
        if self.path == "/mark_baseline":
            with _lock:
                recent = list(_recent_rows)[-30:]
            if recent:
                energies = [row_energy(r) for r in recent]
                mean_e = sum(energies) / len(energies)
                variability = (sum((e - mean_e) ** 2 for e in energies) / len(energies)) ** 0.5
                with _lock:
                    _stats["baseline"] = variability
                self._send(200, json.dumps({"ok": True}), "application/json")
            else:
                self._send(200, json.dumps({"ok": False, "error": "no data yet"}), "application/json")
        else:
            self._send(404, "not found")

def main():
    print(f"[csi_dashboard_simple] numpy available: {HAVE_NUMPY}")
    print(f"[csi_dashboard_simple] tailing: {LOG_FILE}")
    t = threading.Thread(target=tail_file, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[http] serving on http://0.0.0.0:{HTTP_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown]")

if __name__ == "__main__":
    main()
