#!/usr/bin/env python3
"""
CSI LIVE DASHBOARD (file-tail version) - one file, one run.

Why this version exists: the ESP32/collection pipeline (esp32_collect.py,
signal_collect_v2.py, etc.) is already running and already owns UDP port
5005. Rather than compete for that port, this dashboard TAILS the log
file esp32_collect.py is already writing to, live, and visualizes it.
Zero interference with the running collection.

Run on the Jetson:
    python3 csi_dashboard_tail.py

Then open, from any browser on the same network:
    http://<jetson-ip>:8000

Only dependency: numpy (already installed via apt). Everything else stdlib.
Works on Python 3.6+ (Jetson's default).
"""

import ast
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
# Edit this if your log file lives somewhere else.
LOG_FILE = "/home/rajmohan/amma-project/data/esp32_data.txt"
HTTP_PORT = 8000

# -------------------------------------------------------------- STATE -----
HISTORY_LEN = 150
_lock = threading.Lock()
_history = deque(maxlen=HISTORY_LEN)   # each item: list of floats (amplitude per subcarrier)
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
    """
    Expected shape: '<date> <time> [optional rssi] b'<bytes literal>'\n'
    We don't rely on exact field count - just find where the bytes
    literal starts and parse from there.
    """
    idx = line.find("b'")
    if idx == -1:
        idx = line.find('b"')
    if idx == -1:
        return None
    literal = line[idx:].strip()
    try:
        raw = ast.literal_eval(literal)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(raw, (bytes, bytearray)) or len(raw) < 4:
        return None

    # Convert byte pairs into a pseudo amplitude signal.
    # We don't assume an exact header length - just take signed byte pairs
    # across the whole payload so this works even if header framing shifts.
    n_pairs = len(raw) // 2
    if n_pairs < 4:
        return None
    vals = []
    for i in range(n_pairs):
        b0 = raw[2 * i]
        b1 = raw[2 * i + 1]
        # interpret as signed
        s0 = b0 - 256 if b0 > 127 else b0
        s1 = b1 - 256 if b1 > 127 else b1
        amp = (s0 * s0 + s1 * s1) ** 0.5
        vals.append(amp)
    return vals

def tail_file():
    while not os.path.exists(LOG_FILE):
        with _lock:
            _stats["log_error"] = f"waiting for {LOG_FILE} to exist"
        time.sleep(1.0)

    with _lock:
        _stats["log_error"] = None

    f = open(LOG_FILE, "r", errors="replace")
    f.seek(0, os.SEEK_END)  # start at the end - only show new data going forward

    last_rate_check = time.time()
    count_since_check = 0

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
            _history.append(vals)
            _stats["connected"] = True
            _stats["last_line_ts"] = now
            _stats["line_count"] += 1
        count_since_check += 1

        if now - last_rate_check >= 1.0:
            with _lock:
                _stats["line_rate"] = count_since_check / (now - last_rate_check)
            count_since_check = 0
            last_rate_check = now

# --------------------------------------------------------- FEATURE CALC ---
def compute_readout():
    with _lock:
        hist = list(_history)
        baseline = _stats["baseline"]
        connected = _stats["connected"]
        line_rate = _stats["line_rate"]
        log_error = _stats["log_error"]

    if log_error:
        return {"movement": 0.0, "energy": 0.0, "presence": log_error,
                "connected": False, "line_rate": 0.0}

    if len(hist) < 5:
        return {"movement": 0.0, "energy": 0.0, "presence": "waiting for data",
                "connected": connected, "line_rate": round(line_rate, 1)}

    recent = hist[-30:] if len(hist) >= 30 else hist
    # rows may have different lengths across packets - trim to shortest for safety
    min_len = min(len(r) for r in recent)
    recent = [r[:min_len] for r in recent]

    if HAVE_NUMPY:
        arr = np.array(recent)
        energy_per_row = arr.mean(axis=1)
        movement = float(np.std(energy_per_row))
        energy = float(arr.mean())
    else:
        energy_per_row = [sum(row) / len(row) for row in recent]
        mean_e = sum(energy_per_row) / len(energy_per_row)
        movement = (sum((e - mean_e) ** 2 for e in energy_per_row) / len(energy_per_row)) ** 0.5
        energy = mean_e

    presence = "unknown"
    if baseline is not None:
        delta = abs(energy - baseline)
        presence = "activity detected" if delta > 0.15 * (baseline + 1e-6) else "quiet / near-baseline"

    return {
        "movement": round(movement, 4),
        "energy": round(energy, 4),
        "presence": presence,
        "connected": connected,
        "line_rate": round(line_rate, 1),
    }

# --------------------------------------------------------------- HTML -----
PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CSI Live Dashboard</title>
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
    padding: 24px;
  }
  h1 { font-size: 18px; font-weight: 600; letter-spacing: 0.02em; margin: 0 0 4px; }
  .sub { color: var(--dim); font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  .panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--dim); margin: 0 0 12px; font-weight: 600; }
  canvas { width: 100%; display: block; border-radius: 6px; background: #060a0f; }
  .status-row { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--bad); }
  .dot.on { background: var(--good); box-shadow: 0 0 8px var(--good); }
  .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
  .stat:last-child { border-bottom: none; }
  .stat .label { color: var(--dim); }
  .stat .val { font-weight: 600; font-variant-numeric: tabular-nums; }
  button { background: var(--accent); border: none; color: #06201d; font-weight: 600; padding: 10px 14px; border-radius: 7px; cursor: pointer; font-size: 13px; width: 100%; margin-top: 10px; }
  button:hover { opacity: 0.9; }
  .presence { font-size: 15px; font-weight: 600; margin-top: 4px; }
  .presence.quiet { color: var(--dim); }
  .presence.active { color: var(--warn); }
  .legend { display:flex; justify-content:space-between; font-size:11px; color:var(--dim); margin-top:6px; }
  .note { font-size: 11px; color: var(--dim); margin-top: 14px; line-height: 1.5; }
</style>
</head>
<body>
  <h1>CSI LIVE DASHBOARD</h1>
  <div class="sub">Reading the running collection pipeline's log file live &mdash; not competing for the UDP port</div>

  <div class="grid">
    <div class="panel">
      <h2>Waterfall (time &darr; scrolling, byte-pairs &rarr;)</h2>
      <canvas id="heatmap" width="900" height="420"></canvas>
      <div class="legend"><span>low amplitude</span><span>high amplitude</span></div>
    </div>

    <div class="panel">
      <h2>Live Status</h2>
      <div class="status-row">
        <div id="dot" class="dot"></div>
        <span id="connText">Waiting for log data&hellip;</span>
      </div>

      <div class="stat"><span class="label">Lines/sec</span><span class="val" id="rate">&mdash;</span></div>
      <div class="stat"><span class="label">Movement (std of energy)</span><span class="val" id="movement">&mdash;</span></div>
      <div class="stat"><span class="label">Mean energy</span><span class="val" id="energy">&mdash;</span></div>

      <div style="margin-top:14px;">
        <div class="label" style="color:var(--dim); font-size:12px;">Presence read</div>
        <div id="presence" class="presence quiet">&mdash;</div>
      </div>

      <button id="markBaseline">Mark room as EMPTY (set baseline)</button>

      <div class="note">Reading: existing collection pipeline's log file, tailed live.
      This does not touch or interrupt the running collection process.</div>
    </div>
  </div>

<script>
const canvas = document.getElementById('heatmap');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;

function amplitudeToColor(v, vmin, vmax) {
  let t = (v - vmin) / (Math.max(vmax - vmin, 1e-6));
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(20 + t * 235);
  const g = Math.round(40 + t * 170);
  const b = Math.round(60 + (1 - t) * 120);
  return `rgb(${r},${g},${b})`;
}

async function poll() {
  try {
    const res = await fetch('/data');
    const d = await res.json();

    document.getElementById('dot').className = 'dot' + (d.stats.connected ? ' on' : '');
    document.getElementById('connText').textContent = d.stats.connected
      ? 'Reading live data' : 'No new data';
    document.getElementById('rate').textContent = d.stats.line_rate + ' /s';
    document.getElementById('movement').textContent = d.readout.movement;
    document.getElementById('energy').textContent = d.readout.energy;

    const p = document.getElementById('presence');
    p.textContent = d.readout.presence;
    p.className = 'presence ' + (d.readout.presence === 'activity detected' ? 'active' : 'quiet');

    if (d.history && d.history.length > 0) {
      const rows = d.history;
      const nRows = rows.length, nCols = rows[0].length;
      let vmin = Infinity, vmax = -Infinity;
      for (const row of rows) for (const v of row) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }

      ctx.clearRect(0, 0, W, H);
      const cellW = W / nCols, cellH = H / nRows;
      for (let r = 0; r < nRows; r++) {
        for (let c = 0; c < nCols; c++) {
          ctx.fillStyle = amplitudeToColor(rows[r][c], vmin, vmax);
          ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
        }
      }
    }
  } catch (e) {
    document.getElementById('connText').textContent = 'Dashboard server unreachable';
  }
  setTimeout(poll, 250);
}

document.getElementById('markBaseline').addEventListener('click', async () => {
  await fetch('/mark_baseline', { method: 'POST' });
});

poll();
</script>
</body>
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
            with _lock:
                hist = list(_history)[-60:]
                stats = dict(_stats)
            readout = compute_readout()
            payload = {"history": hist, "stats": {
                "connected": stats["connected"],
                "line_rate": round(stats["line_rate"], 1),
            }, "readout": readout}
            self._send(200, json.dumps(payload), "application/json")
        else:
            self._send(404, "not found")

    def do_POST(self):
        if self.path == "/mark_baseline":
            with _lock:
                hist = list(_history)
            if hist:
                min_len = min(len(r) for r in hist[-30:])
                recent = [r[:min_len] for r in hist[-30:]]
                if HAVE_NUMPY:
                    energy = float(np.mean([sum(r) / len(r) for r in recent]))
                else:
                    energy = sum(sum(r) / len(r) for r in recent) / len(recent)
                with _lock:
                    _stats["baseline"] = energy
                self._send(200, json.dumps({"ok": True, "baseline": energy}), "application/json")
            else:
                self._send(200, json.dumps({"ok": False, "error": "no data yet"}), "application/json")
        else:
            self._send(404, "not found")

def main():
    print(f"[csi_dashboard_tail] numpy available: {HAVE_NUMPY}")
    print(f"[csi_dashboard_tail] tailing: {LOG_FILE}")
    t = threading.Thread(target=tail_file, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[http] serving on http://0.0.0.0:{HTTP_PORT}  (open this from your laptop/phone browser)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown]")

if __name__ == "__main__":
    main()
