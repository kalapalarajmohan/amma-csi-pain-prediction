#!/usr/bin/env python3
"""
CSI LIVE DASHBOARD - one file, one run.

What this does:
  - Listens for raw CSI UDP packets from the ESP32 (same format as esp32_receiver.py)
  - Serves a live web page with a scrolling subcarrier heatmap + movement/presence readout
  - No Flask, no sklearn, no scipy. Only numpy (already installed via apt) + Python stdlib.

Run on the Jetson:
    python3 csi_dashboard.py

Then open, from any browser on the same network:
    http://<jetson-ip>:8000

If numpy isn't available it still runs in a degraded pure-python mode.

Config: edit the CONFIG block below if your port / subcarrier count differ
from what's in config.yaml. It will try to read config.yaml first and only
falls back to these defaults if that file is missing or unreadable.
"""

import json
import os
import socket
import struct
import threading
import time
from collections import deque

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

# ---------------------------------------------------------------- CONFIG ---
DEFAULTS = {"udp_port": 5005, "n_subcarriers": 52, "http_port": 8000}

def load_config():
    cfg = dict(DEFAULTS)
    for path in ("config.yaml", "/home/rajmohan/csi_sciatica/config.yaml"):
        if os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    raw = yaml.safe_load(f)
                cfg["udp_port"] = raw.get("network", {}).get("esp32_udp_port", cfg["udp_port"])
                cfg["n_subcarriers"] = raw.get("esp32", {}).get("n_subcarriers", cfg["n_subcarriers"])
            except Exception:
                pass
            break
    return cfg

CFG = load_config()
UDP_PORT = CFG["udp_port"]
N_SC = CFG["n_subcarriers"]
HTTP_PORT = CFG["http_port"]

# -------------------------------------------------------------- STATE -----
HISTORY_LEN = 150           # rows of history kept for the heatmap
_lock = threading.Lock()
_history = deque(maxlen=HISTORY_LEN)   # each item: list of N_SC floats (amplitude)
_stats = {
    "connected": False,
    "last_packet_ts": 0.0,
    "pkt_count": 0,
    "pkt_rate": 0.0,
    "baseline": None,       # noise-floor amplitude vector, set by "mark empty" button
}

# ---------------------------------------------------------- UDP LISTENER --
def _parse_packet(data):
    expected = 2 * N_SC
    if len(data) < expected:
        return None
    try:
        raw = struct.unpack(f"{2*N_SC}b", data[:expected])
    except struct.error:
        return None
    if HAVE_NUMPY:
        imag = np.array(raw[0::2], dtype=np.float32)
        real = np.array(raw[1::2], dtype=np.float32)
        amp = np.sqrt(real**2 + imag**2)
        return amp.tolist()
    else:
        imag = raw[0::2]
        real = raw[1::2]
        return [ (r*r + i*i) ** 0.5 for r, i in zip(real, imag) ]

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[udp] listening on 0.0.0.0:{UDP_PORT}, expecting {N_SC} subcarriers")

    last_rate_check = time.time()
    count_since_check = 0

    while True:
        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            with _lock:
                if time.time() - _stats["last_packet_ts"] > 3.0:
                    _stats["connected"] = False
            continue

        amp = _parse_packet(data)
        if amp is None:
            continue

        now = time.time()
        with _lock:
            _history.append(amp)
            _stats["connected"] = True
            _stats["last_packet_ts"] = now
            _stats["pkt_count"] += 1
        count_since_check += 1

        if now - last_rate_check >= 1.0:
            with _lock:
                _stats["pkt_rate"] = count_since_check / (now - last_rate_check)
            count_since_check = 0
            last_rate_check = now

# --------------------------------------------------------- FEATURE CALC ---
def compute_readout():
    """Cheap, dependency-free movement/presence estimate from recent history."""
    with _lock:
        hist = list(_history)
        baseline = _stats["baseline"]
        connected = _stats["connected"]
        pkt_rate = _stats["pkt_rate"]

    if len(hist) < 5:
        return {"movement": 0.0, "energy": 0.0, "presence": "unknown",
                "connected": connected, "pkt_rate": round(pkt_rate, 1)}

    recent = hist[-30:] if len(hist) >= 30 else hist
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
        "pkt_rate": round(pkt_rate, 1),
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
  .grid {
    display: grid; grid-template-columns: 2fr 1fr; gap: 16px;
  }
  .panel {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 16px;
  }
  .panel h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--dim); margin: 0 0 12px; font-weight: 600;
  }
  canvas { width: 100%; display: block; border-radius: 6px; background: #060a0f; }
  .status-row { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--bad); }
  .dot.on { background: var(--good); box-shadow: 0 0 8px var(--good); }
  .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
  .stat:last-child { border-bottom: none; }
  .stat .label { color: var(--dim); }
  .stat .val { font-weight: 600; font-variant-numeric: tabular-nums; }
  .big { font-size: 28px; font-weight: 700; margin: 6px 0 2px; }
  button {
    background: var(--accent); border: none; color: #06201d;
    font-weight: 600; padding: 10px 14px; border-radius: 7px;
    cursor: pointer; font-size: 13px; width: 100%; margin-top: 10px;
  }
  button:hover { opacity: 0.9; }
  .presence { font-size: 15px; font-weight: 600; margin-top: 4px; }
  .presence.quiet { color: var(--dim); }
  .presence.active { color: var(--warn); }
  .legend { display:flex; justify-content:space-between; font-size:11px; color:var(--dim); margin-top:6px; }
</style>
</head>
<body>
  <h1>CSI LIVE DASHBOARD</h1>
  <div class="sub">Subcarrier amplitude waterfall &mdash; watch it change when someone moves in the room</div>

  <div class="grid">
    <div class="panel">
      <h2>Waterfall (time &darr; scrolling, subcarriers &rarr;)</h2>
      <canvas id="heatmap" width="900" height="420"></canvas>
      <div class="legend"><span>low amplitude</span><span>high amplitude</span></div>
    </div>

    <div class="panel">
      <h2>Live Status</h2>
      <div class="status-row">
        <div id="dot" class="dot"></div>
        <span id="connText">Waiting for ESP32&hellip;</span>
      </div>

      <div class="stat"><span class="label">Packet rate</span><span class="val" id="rate">&mdash;</span></div>
      <div class="stat"><span class="label">Movement (std of energy)</span><span class="val" id="movement">&mdash;</span></div>
      <div class="stat"><span class="label">Mean energy</span><span class="val" id="energy">&mdash;</span></div>

      <div style="margin-top:14px;">
        <div class="label" style="color:var(--dim); font-size:12px;">Presence read</div>
        <div id="presence" class="presence quiet">&mdash;</div>
      </div>

      <button id="markBaseline">Mark room as EMPTY (set baseline)</button>
    </div>
  </div>

<script>
const canvas = document.getElementById('heatmap');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;

function amplitudeToColor(v, vmin, vmax) {
  let t = (v - vmin) / (Math.max(vmax - vmin, 1e-6));
  t = Math.max(0, Math.min(1, t));
  // dark teal -> cyan -> warm amber, so movement visually "lights up"
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
      ? 'ESP32 connected' : 'No signal';
    document.getElementById('rate').textContent = d.stats.pkt_rate + ' pkt/s';
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
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean

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
                "pkt_rate": round(stats["pkt_rate"], 1),
            }, "readout": readout}
            self._send(200, json.dumps(payload), "application/json")
        else:
            self._send(404, "not found")

    def do_POST(self):
        if self.path == "/mark_baseline":
            with _lock:
                hist = list(_history)
            if hist:
                if HAVE_NUMPY:
                    energy = float(np.mean([sum(r) / len(r) for r in hist[-30:]]))
                else:
                    recent = hist[-30:]
                    energy = sum(sum(r) / len(r) for r in recent) / len(recent)
                with _lock:
                    _stats["baseline"] = energy
                self._send(200, json.dumps({"ok": True, "baseline": energy}), "application/json")
            else:
                self._send(200, json.dumps({"ok": False, "error": "no data yet"}), "application/json")
        else:
            self._send(404, "not found")

def main():
    print(f"[csi_dashboard] numpy available: {HAVE_NUMPY}")
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[http] serving on http://0.0.0.0:{HTTP_PORT}  (open this from your laptop/phone browser)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutdown]")

if __name__ == "__main__":
    main()
