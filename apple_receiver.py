import os, json, time, yaml, threading, logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify

def load_config():
    for path in ["config.yaml", "/home/rajmohan/csi_sciatica/config.yaml"]:
        if os.path.exists(path):
            with open(path) as f: return yaml.safe_load(f)
    return {"network": {"apple_health_port": 5050}, "paths": {"groundtruth": "/tmp/groundtruth"}}

cfg    = load_config()
GT_DIR = cfg["paths"]["groundtruth"]
PORT   = cfg["network"]["apple_health_port"]
os.makedirs(GT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [APPLE] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("apple_receiver")

_store_lock = threading.Lock()
_store = []
app = Flask(__name__)

@app.route("/groundtruth", methods=["POST"])
def receive_groundtruth():
    try:
        raw = request.get_json(force=True, silent=True)
        if raw is None:
            return jsonify({"status": "error", "message": "No JSON"}), 400
        records = _parse_health_export(raw)
        if not records:
            return jsonify({"status": "error", "message": "No valid records"}), 400
        with _store_lock: _store.extend(records)
        log.info(f"Received {len(records)} record(s) — total={len(_store)}")
        return jsonify({"status": "ok", "received": len(records)}), 200
    except Exception as e:
        log.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/groundtruth/latest", methods=["GET"])
def get_latest():
    with _store_lock:
        if not _store: return jsonify({"status": "empty"}), 200
        return jsonify(_store[-1]), 200

@app.route("/groundtruth/count", methods=["GET"])
def get_count():
    with _store_lock: return jsonify({"count": len(_store)}), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "alive", "store_size": len(_store)}), 200

def _parse_health_export(raw):
    records = []
    now_ts  = time.time()
    if any(k in raw for k in ("hrv", "heart_rate", "breathing_rate", "pain_start")):
        records.append({
            "timestamp":      raw.get("timestamp", now_ts),
            "hrv":            _safe_float(raw.get("hrv")),
            "heart_rate":     _safe_float(raw.get("heart_rate")),
            "breathing_rate": _safe_float(raw.get("breathing_rate")),
            "pain_start":     raw.get("pain_start"),
            "pain_end":       raw.get("pain_end"),
            "source": "flat",
        })
        return records
    metric_map = {
        "Heart Rate Variability": "hrv",
        "Heart Rate":             "heart_rate",
        "Respiratory Rate":       "breathing_rate",
        "Breathing Rate":         "breathing_rate",
    }
    all_points = {}
    for metric_name, field_name in metric_map.items():
        if metric_name in raw:
            for point in raw[metric_name].get("data", []):
                date_str = point.get("date", "")
                qty      = _safe_float(point.get("qty"))
                if qty is not None:
                    if date_str not in all_points:
                        all_points[date_str] = {"timestamp": _parse_ts(date_str)}
                    all_points[date_str][field_name] = qty
    for date_str, values in all_points.items():
        records.append({**values, "source": "health_auto_export"})
    return records

def _safe_float(val):
    try: return float(val)
    except (TypeError, ValueError): return None

def _parse_ts(date_str):
    for fmt in ["%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]:
        try: return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError: continue
    return time.time()

def _autosave_loop():
    while True:
        time.sleep(60)
        with _store_lock:
            if not _store: continue
            snapshot = list(_store)
        fname = os.path.join(GT_DIR, f"gt_{int(time.time())}.json")
        with open(fname, "w") as f: json.dump(snapshot, f)
        log.info(f"Saved {len(snapshot)} records -> {fname}")

def get_store_snapshot():
    with _store_lock: return list(_store)

def get_store_size():
    with _store_lock: return len(_store)

def start(host="0.0.0.0", port=PORT, blocking=True):
    threading.Thread(target=_autosave_loop, daemon=True).start()
    log.info(f"Apple Health receiver on port {port}")
    log.info(f"iPhone -> POST to http://192.168.29.232:{port}/groundtruth")
    if blocking:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    else:
        threading.Thread(
            target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
            daemon=True
        ).start()

if __name__ == "__main__":
    start()
