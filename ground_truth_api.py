import json, time, os, logging, threading
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GT] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ground_truth_api")

app      = Flask(__name__)
LOG_FILE = "data/pain_labels.jsonl"
os.makedirs("data", exist_ok=True)

_labels_lock = threading.Lock()
_labels      = []

@app.route("/groundtruth", methods=["POST"])
def log_pain():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON"}), 400
        event = data.get("event")
        if event not in ("pain_start", "pain_end"):
            return jsonify({"status": "error", "message": "event must be pain_start or pain_end"}), 400
        entry = {
            "timestamp":     data.get("timestamp", time.time()),
            "readable_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event":         event,
            "intensity":     data.get("intensity", 5),
            "source":        "explicit",
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        with _labels_lock:
            _labels.append(entry)
        log.info(f"{event} recorded | intensity={entry['intensity']}")
        return jsonify({"status": "recorded", "entry": entry}), 200
    except Exception as e:
        log.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/labels/recent", methods=["GET"])
def get_recent():
    with _labels_lock: return jsonify(_labels[-20:]), 200

@app.route("/labels/count", methods=["GET"])
def get_count():
    with _labels_lock: return jsonify({"count": len(_labels)}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive"}), 200

def get_labels_snapshot():
    with _labels_lock: return list(_labels)

def get_labels_count():
    with _labels_lock: return len(_labels)

def start(host="0.0.0.0", port=5051, blocking=True):
    log.info(f"Ground truth API on port {port}")
    log.info("Siri shortcut -> POST {event: pain_start, intensity: 7}")
    if blocking:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    else:
        threading.Thread(
            target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
            daemon=True
        ).start()

if __name__ == "__main__":
    start()
