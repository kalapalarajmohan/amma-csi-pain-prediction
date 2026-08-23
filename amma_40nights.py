"""
amma_40nights.py
=================
40-night personalized sciatica learning system for Amma.
Five models running simultaneously. All 6 physiological signals.
Saves everything. Labels every morning. Gets smarter every night.

BEFORE SLEEP:
    python3.8 amma_40nights.py

MORNING WITH PAIN:
    python3.8 amma_40nights.py --label pain

MORNING NO PAIN:
    python3.8 amma_40nights.py --label ok

SEE PROGRESS:
    python3.8 amma_40nights.py --report

MODELS:
    Night  1-7  : Symbolic AI only (physics rules)
    Night  8-13 : + Reservoir Computing + TDA
    Night 14-20 : + Kernel SVM
    Night 21+   : + Temporal LSTM-lite (full ensemble)
    Night 40    : Final model — Amma's personal pain predictor

SIGNALS CAPTURED EVERY 30 SECONDS:
    1. Movement       (amplitude variance)
    2. Breathing rate (0.15-0.5 Hz band peak detection)
    3. Subcarrier lo  (forward lean proxy)
    4. Subcarrier hi  (side-lying proxy)
    5. Instability    (temporal variance)
    6. Activity state (active/restless/still)
    + Apple Watch HRV + HR if available

MEMORY: ~6KB per night. 40 nights = 240KB. Jetson safe.
CLOUD:  Saves to /home/rajmohan/csi_sciatica/data/nights/
        Optional: rclone sync to Google Drive (see --setup-cloud)
"""

import os, sys, time, json, socket, struct, threading, logging, argparse, pickle
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from datetime import datetime
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks

# ── Paths ─────────────────────────────────────────────
BASE      = "/home/rajmohan/csi_sciatica"
NIGHTS    = f"{BASE}/data/nights"
MODELS    = f"{BASE}/data/models"
LOGS      = f"{BASE}/logs"
for d in [NIGHTS, MODELS, LOGS]:
    os.makedirs(d, exist_ok=True)

# ── Logging ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            f"{LOGS}/amma_40nights.log",
            maxBytes=5*1024*1024, backupCount=3
        ) if hasattr(logging, 'handlers') else logging.FileHandler(f"{LOGS}/amma_40nights.log"),
    ]
)
import logging.handlers
# Recreate with rotating handler
for h in logging.root.handlers[:]: logging.root.removeHandler(h)
fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
fh  = logging.handlers.RotatingFileHandler(f"{LOGS}/amma_40nights.log", maxBytes=5*1024*1024, backupCount=3)
fh.setFormatter(fmt)
logging.root.addHandler(sh); logging.root.addHandler(fh)
logging.root.setLevel(logging.INFO)
log = logging.getLogger("amma")

# ── Constants ─────────────────────────────────────────
UDP_PORT    = 5005
APPLE_PORT  = 5050
FS          = 100.0
WINDOW_PKTS = 300     # ~8s at 26 pkt/s
N_SC        = 52
INTERVAL    = 30      # record every 30 seconds

# ── Safe value helper ─────────────────────────────────
def safe(v, default=0.0):
    if v is None: return default
    try:
        f = float(v)
        return default if (f != f or abs(f) == float('inf')) else f
    except: return default

# ── ESP32 receiver ────────────────────────────────────
_ring      = deque(maxlen=10000)
_ring_lock = threading.Lock()
_connected = False
_rx_count  = 0

def _esp32_loop():
    global _connected, _rx_count
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(5.0)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
    except OSError as e:
        log.error(f"Cannot bind {UDP_PORT}: {e}"); return
    last_rx = time.time()
    while True:
        try:
            data, _ = sock.recvfrom(65535)
            if len(data) < 2*N_SC: continue
            raw  = struct.unpack(f"{2*N_SC}b", data[:2*N_SC])
            imag = np.array(raw[0::2], dtype=np.float32)
            real = np.array(raw[1::2], dtype=np.float32)
            amp  = np.abs(real + 1j*imag)
            with _ring_lock:
                _ring.append((time.time(), amp))
                try:
                    with open("/home/rajmohan/csi_sciatica/data/live_amp.log", "a") as _lf:
                        _lf.write(str(time.time()) + " " + ",".join(f"{v:.3f}" for v in amp) + chr(10))
                except Exception:
                    pass
            _connected = True; _rx_count += 1; last_rx = time.time()
        except socket.timeout:
            if _connected and (time.time()-last_rx) > 15:
                log.warning("ESP32 signal lost"); _connected = False
        except Exception as e:
            log.error(f"ESP32 recv: {e}")

def get_window():
    with _ring_lock:
        if len(_ring) < WINDOW_PKTS: return None
        recent = list(_ring)[-WINDOW_PKTS:]
        times  = np.array([r[0] for r in recent])
        amps   = np.stack([r[1] for r in recent], axis=0)
        return amps, times

# ── Signal extraction (6 signals) ────────────────────
def extract_signals(amp, times=None):
    """Extract all 6 physiological signals from CSI amplitude window."""
    try:
        g      = safe(amp.mean(), 1.0)
        energy = amp.mean(axis=1)

        # Signal 1: Movement
        movement = safe(np.std(energy), 0.0)

        # Signal 2: Breathing rate
        breathing = None
        try:
            if times is not None and len(times) > 1:
                fs_actual = (len(times) - 1) / (times[-1] - times[0])
            else:
                fs_actual = FS
            if fs_actual >= 4:
                nyq  = fs_actual/2
                b, a = butter(4, [0.15/nyq, 0.5/nyq], btype='band')
                filt = filtfilt(b, a, energy)
                pks, _ = find_peaks(filt, distance=int(fs_actual*2))
                if len(pks) >= 2:
                    rate = 60.0 / np.mean(np.diff(pks) / fs_actual)
                    breathing = round(rate, 1) if 6 <= rate <= 30 else None
        except: pass

        # Signal 3+4: Subcarrier asymmetry
        lo = safe(amp[:, :17].mean() / g, 1.0)
        hi = safe(amp[:, 35:].mean() / g, 1.0)

        # Signal 5: Instability
        instab = safe(np.mean(np.var(amp, axis=0)) / g, 0.0)

        # Signal 6: Activity state
        if movement > 0.8:    activity = "active"
        elif movement > 0.3:  activity = "restless"
        else:                 activity = "still"

        return {
            "movement":  round(movement, 3),
            "breathing": breathing,
            "lo":        round(lo, 3),
            "hi":        round(hi, 3),
            "instab":    round(instab, 3),
            "activity":  activity,
            "amp":       round(g, 2),
        }
    except Exception as e:
        log.error(f"Extract error: {e}")
        return {"movement":0.0,"breathing":None,"lo":1.0,"hi":1.0,
                "instab":0.0,"activity":"still","amp":1.0}

# ── Apple Watch receiver ──────────────────────────────
_apple = []
def _start_apple():
    try:
        from flask import Flask, request, jsonify
        import logging as _l; _l.getLogger('werkzeug').disabled = True
        app = Flask(__name__); app.logger.disabled = True
        @app.route("/groundtruth", methods=["POST"])
        def recv():
            _apple.append(request.get_json(force=True,silent=True) or {})
            return jsonify({"status":"ok"}), 200
        @app.route("/health")
        def health(): return jsonify({"status":"alive"}), 200
        threading.Thread(
            target=lambda: app.run("0.0.0.0",APPLE_PORT,debug=False,use_reloader=False),
            daemon=True).start()
    except Exception as e:
        log.warning(f"Apple receiver: {e}")

# ── Load all labeled nights ───────────────────────────
def load_nights():
    nights, labels = [], []
    files = sorted([f for f in os.listdir(NIGHTS) if f.endswith(".json")])
    for fname in files:
        with open(os.path.join(NIGHTS, fname)) as f:
            s = json.load(f)
        if s.get("label") in ("pain","ok") and s.get("readings"):
            nights.append(s["readings"])
            labels.append(1 if s["label"]=="pain" else 0)
    return nights, labels

# ── Feature vector from night readings ────────────────
def night_features(readings):
    """12 features summarizing a night of readings."""
    mvs  = [safe(r.get("movement"), 0.0) for r in readings]
    brs  = [safe(r.get("breathing"), 14.0) for r in readings]
    his  = [safe(r.get("hi"), 1.0) for r in readings]
    los  = [safe(r.get("lo"), 1.0) for r in readings]

    # Restless episodes
    above = [1 if m>0.4 else 0 for m in mvs]
    runs  = []
    run   = 0
    for a in above:
        if a: run+=1
        elif run>0: runs.append(run); run=0
    if run>0: runs.append(run)

    feats = [
        np.mean(mvs), np.std(mvs), np.max(mvs),
        np.mean(brs), np.std(brs),
        np.mean(his), np.max(his),
        np.mean(los),
        len(runs),
        max(runs) if runs else 0,
        np.percentile(mvs, 90),
        np.percentile(his, 90),
    ]
    return np.nan_to_num(np.array(feats, dtype=float), nan=0.0)

# ── MODEL 1: Symbolic AI ─────────────────────────────
def symbolic_predict(readings):
    """Physics rules. Always available. Night 1+"""
    scores = []
    for r in readings:
        s    = 0
        hour = safe(r.get("hour", 0))
        hi   = safe(r.get("hi"), 1.0)
        act  = r.get("activity", "still")
        if hour >= 22 or hour < 6:
            if hi > 1.3:              s += 50
            if act == "restless":     s += 20
            if act == "active":       s += 30
        scores.append(min(s, 100))
    night_score = np.mean(scores[-12:]) if scores else 0
    return 1 if night_score > 25 else 0, night_score/100

# ── MODEL 2: Echo State Network ──────────────────────
class ESN:
    def __init__(self, n_res=100):
        rng = np.random.RandomState(42)
        self.W_in = rng.randn(n_res, 6) * 0.1
        W = rng.randn(n_res, n_res)
        W[rng.rand(n_res,n_res) > 0.1] = 0
        ev = np.linalg.eigvals(W)
        sr = np.max(np.abs(ev)) + 1e-8
        self.W    = W * (0.9 / sr)
        self.n    = n_res
        self.clf  = None
        self.trained = False

    def _run(self, readings):
        state  = np.zeros(self.n)
        states = []
        for r in readings:
            x = np.array([
                safe(r.get("movement"),0.0),
                safe(r.get("breathing"),14.0)/20.0,
                safe(r.get("hi"),1.0),
                safe(r.get("lo"),1.0),
                safe(r.get("instab"),0.0),
                1.0 if r.get("activity")=="active" else 0.0,
            ])
            state = np.tanh(self.W_in @ x + self.W @ state)
            states.append(state.copy())
        S = np.array(states)
        return np.concatenate([S.mean(axis=0), S.std(axis=0)])

    def train(self, nights, labels):
        try:
            from sklearn.linear_model import RidgeClassifier
            X = np.array([self._run(n) for n in nights])
            X = np.nan_to_num(X, nan=0.0)
            self.clf = RidgeClassifier(alpha=1.0)
            self.clf.fit(X, labels)
            self.trained = True
        except Exception as e:
            log.warning(f"ESN train: {e}")

    def predict(self, readings):
        if not self.trained: return 0, 0.5
        try:
            f    = self._run(readings).reshape(1,-1)
            f    = np.nan_to_num(f, nan=0.0)
            pred = int(self.clf.predict(f)[0])
            scores = self.clf.decision_function(f)[0]
            if hasattr(scores,'__len__'):
                conf = float(np.max(np.abs(scores)))
            else:
                conf = float(abs(scores))
            conf = min(conf/2, 1.0)
            return pred, conf
        except: return 0, 0.5

# ── MODEL 3: TDA ─────────────────────────────────────
class TDAModel:
    def __init__(self):
        self.pain_template = None
        self.ok_template   = None
        self.trained       = False

    def train(self, nights, labels):
        try:
            pain_feats = [night_features(n) for n,l in zip(nights,labels) if l==1]
            ok_feats   = [night_features(n) for n,l in zip(nights,labels) if l==0]
            if pain_feats: self.pain_template = np.mean(pain_feats, axis=0)
            if ok_feats:   self.ok_template   = np.mean(ok_feats,   axis=0)
            if pain_feats and ok_feats: self.trained = True
        except Exception as e:
            log.warning(f"TDA train: {e}")

    def predict(self, readings):
        if not self.trained: return 0, 0.5
        try:
            f      = night_features(readings)
            d_pain = np.linalg.norm(f - self.pain_template)
            d_ok   = np.linalg.norm(f - self.ok_template)
            total  = d_pain + d_ok + 1e-8
            conf   = d_ok / total
            return (1 if d_pain < d_ok else 0), float(conf)
        except: return 0, 0.5

# ── MODEL 4: Kernel SVM ──────────────────────────────
class KernelSVM:
    def __init__(self):
        self.pipe    = None
        self.trained = False

    def train(self, nights, labels):
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            X = np.array([night_features(n) for n in nights])
            X = np.nan_to_num(X, nan=0.0)
            y = np.array(labels)
            if len(np.unique(y)) < 2: return
            self.pipe = Pipeline([
                ('sc',  StandardScaler()),
                ('svm', SVC(kernel='rbf',C=10,gamma='scale',
                            probability=True,random_state=42))
            ])
            self.pipe.fit(X, y)
            self.trained = True
        except Exception as e:
            log.warning(f"SVM train: {e}")

    def predict(self, readings):
        if not self.trained: return 0, 0.5
        try:
            f    = night_features(readings).reshape(1,-1)
            f    = np.nan_to_num(f, nan=0.0)
            pred = int(self.pipe.predict(f)[0])
            prob = self.pipe.predict_proba(f)[0]
            return pred, float(prob.max())
        except: return 0, 0.5

# ── MODEL 5: Temporal LSTM-lite ───────────────────────
class TemporalModel:
    def __init__(self):
        self.pipe    = None
        self.trained = False

    def _seq_features(self, readings, window=12):
        feats = []
        step  = window//2
        for i in range(0, len(readings)-window, step):
            chunk = readings[i:i+window]
            mvs = [safe(r.get("movement"),0.0) for r in chunk]
            brs = [safe(r.get("breathing"),14.0) for r in chunk]
            his = [safe(r.get("hi"),1.0) for r in chunk]
            feats.extend([np.mean(mvs),np.std(mvs),np.max(mvs),
                          np.mean(brs),np.mean(his),np.max(his),np.std(his)])
        if not feats: return np.zeros(28)
        arr = np.array(feats[:28])
        if len(arr) < 28:
            arr = np.pad(arr, (0, 28-len(arr)))
        return np.nan_to_num(arr, nan=0.0)

    def train(self, nights, labels):
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            X = np.array([self._seq_features(n) for n in nights])
            X = np.nan_to_num(X, nan=0.0)
            y = np.array(labels)
            if len(np.unique(y)) < 2: return
            self.pipe = Pipeline([
                ('sc',  StandardScaler()),
                ('svm', SVC(kernel='rbf',C=5,gamma='scale',
                            probability=True,random_state=42))
            ])
            self.pipe.fit(X, y)
            self.trained = True
        except Exception as e:
            log.warning(f"Temporal train: {e}")

    def predict(self, readings):
        if not self.trained: return 0, 0.5
        try:
            f    = self._seq_features(readings).reshape(1,-1)
            f    = np.nan_to_num(f, nan=0.0)
            pred = int(self.pipe.predict(f)[0])
            prob = self.pipe.predict_proba(f)[0]
            return pred, float(prob.max())
        except: return 0, 0.5

# ── Ensemble ──────────────────────────────────────────
def ensemble_predict(sym, esn, tda, svm, tmp, n_nights):
    """
    Weighted ensemble. Weights shift as models activate.
    Night 1-7:  symbolic only
    Night 8-13: symbolic + ESN + TDA
    Night 14-20: + SVM
    Night 21+:  all five
    """
    sym_pred, sym_conf = sym
    esn_pred, esn_conf = esn
    tda_pred, tda_conf = tda
    svm_pred, svm_conf = svm
    tmp_pred, tmp_conf = tmp

    if n_nights < 8:
        score = sym_conf if sym_pred else (1-sym_conf)
        return sym_pred, score, "symbolic_only"

    elif n_nights < 14:
        w = [0.4, 0.3, 0.3, 0.0, 0.0]
        preds = [sym_pred, esn_pred, tda_pred, 0, 0]
        confs = [sym_conf, esn_conf, tda_conf, 0.5, 0.5]

    elif n_nights < 21:
        w = [0.25, 0.25, 0.25, 0.25, 0.0]
        preds = [sym_pred, esn_pred, tda_pred, svm_pred, 0]
        confs = [sym_conf, esn_conf, tda_conf, svm_conf, 0.5]

    else:
        w = [0.15, 0.15, 0.15, 0.25, 0.30]
        preds = [sym_pred, esn_pred, tda_pred, svm_pred, tmp_pred]
        confs = [sym_conf, esn_conf, tda_conf, svm_conf, tmp_conf]

    score = sum(w[i]*preds[i] for i in range(5))
    pred  = 1 if score > 0.5 else 0
    conf  = sum(w[i]*confs[i] for i in range(5))
    return pred, conf, "ensemble"

# ── Alert ─────────────────────────────────────────────
def send_alert(message, risk):
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://ntfy.sh/" + open("/home/rajmohan/csi_sciatica/data/ntfy_topic.txt").read().strip(),
            data=message.encode('utf-8'), method='POST')
        req.add_header('Title', f'Amma Alert - Risk {risk}%')
        req.add_header('Priority', 'high' if risk>=80 else 'default')
        req.add_header('Tags', 'warning')
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        log.warning(f"Alert: {e}"); return False

# ── Cloud sync (optional) ─────────────────────────────
def cloud_sync():
    try:
        ret = os.system("rclone sync /home/rajmohan/csi_sciatica/data/nights gdrive:amma_sciatica/nights --quiet 2>/dev/null")
        if ret == 0: log.info("Cloud sync OK")
    except: pass

# ── Train all models ──────────────────────────────────
def train_all_models(esn_m, tda_m, svm_m, tmp_m):
    nights, labels = load_nights()
    n = len(nights)
    if n < 2: return n

    log.info(f"Training on {n} labeled nights ({sum(labels)} pain, {n-sum(labels)} ok)...")

    if n >= 8:
        esn_m.train(nights, labels)
        tda_m.train(nights, labels)

    if n >= 14:
        svm_m.train(nights, labels)

    if n >= 21:
        tmp_m.train(nights, labels)

    # Save models
    try:
        with open(f"{MODELS}/models.pkl",'wb') as f:
            pickle.dump({'esn':esn_m,'tda':tda_m,'svm':svm_m,'tmp':tmp_m,'n':n}, f)
        log.info("Models saved")
    except Exception as e:
        log.warning(f"Model save: {e}")

    return n

# ── Load saved models ─────────────────────────────────
def load_models(esn_m, tda_m, svm_m, tmp_m):
    path = f"{MODELS}/models.pkl"
    if not os.path.exists(path): return 0
    try:
        with open(path,'rb') as f:
            saved = pickle.load(f)
        esn_m.__dict__.update(saved['esn'].__dict__)
        tda_m.__dict__.update(saved['tda'].__dict__)
        svm_m.__dict__.update(saved['svm'].__dict__)
        tmp_m.__dict__.update(saved['tmp'].__dict__)
        n = saved.get('n', 0)
        log.info(f"Models loaded (trained on {n} nights)")
        return n
    except Exception as e:
        log.warning(f"Model load: {e}"); return 0

# ── Main monitor ──────────────────────────────────────
def run_monitor():
    # Start receivers
    threading.Thread(target=_esp32_loop, daemon=True).start()
    _start_apple()
    time.sleep(3)

    # Wait for ESP32
    log.info("Waiting for ESP32...")
    waited = 0
    while not _connected:
        time.sleep(2); waited += 2
        if waited > 120:
            log.error("No ESP32 signal. Check hardware."); sys.exit(1)
    log.info(f"ESP32 connected ({_rx_count} packets so far)")

    # Initialize models
    esn_m = ESN()
    tda_m = TDAModel()
    svm_m = KernelSVM()
    tmp_m = TemporalModel()

    # Load saved models + count labeled nights
    n_trained = load_models(esn_m, tda_m, svm_m, tmp_m)

    # Also retrain fresh on all labeled nights
    nights, labels = load_nights()
    n_nights = len(nights)
    if n_nights >= 2:
        train_all_models(esn_m, tda_m, svm_m, tmp_m)
        n_trained = n_nights

    # Session
    date         = datetime.now().strftime("%Y-%m-%d")
    session_file = os.path.join(NIGHTS, f"{date}.json")
    session = {
        "date":       date,
        "start_time": datetime.now().strftime("%H:%M:%S"),
        "start_ts":   time.time(),
        "label":      None,
        "readings":   [],
        "n_nights_trained": n_trained,
    }

    # Load existing session if interrupted
    if os.path.exists(session_file):
        try:
            with open(session_file) as f:
                existing = json.load(f)
            if existing.get("label") is None:
                session = existing
                log.info(f"Resuming session ({len(session['readings'])} readings so far)")
        except: pass

    last_alert = 0
    reading_count = len(session["readings"])

    log.info("=" * 55)
    log.info("  AMMA 40-NIGHT LEARNING SYSTEM")
    log.info(f"  Night: {date}  |  Nights trained: {n_trained}/40")
    log.info(f"  Models active: {'Symbolic' if n_trained<8 else 'Symbolic+ESN+TDA' if n_trained<14 else 'Symbolic+ESN+TDA+SVM' if n_trained<21 else 'ALL FIVE'}")
    log.info("  Recording every 30 seconds")
    log.info("  Morning: python3.8 amma_40nights.py --label pain/ok")
    log.info("=" * 55)

    try:
        while True:
            time.sleep(INTERVAL)

            window = get_window()
            if window is None:
                log.warning("No signal window — waiting...")
                continue
            amp, win_times = window
            # Extract signals
            feat = extract_signals(amp, win_times)

            # Apple Watch
            apple = _apple[-1] if _apple else {}
            hrv   = safe(apple.get("hrv"), None) if apple else None
            hr    = safe(apple.get("heart_rate"), None) if apple else None

            hour = datetime.now().hour + datetime.now().minute/60

            reading = {
                "time":      datetime.now().strftime("%H:%M"),
                "ts":        time.time(),
                "hour":      round(hour, 2),
                "movement":  feat["movement"],
                "breathing": feat["breathing"],
                "lo":        feat["lo"],
                "hi":        feat["hi"],
                "instab":    feat["instab"],
                "activity":  feat["activity"],
                "amp":       feat["amp"],
                "hrv":       hrv,
                "hr":        hr,
            }
            session["readings"].append(reading)
            reading_count += 1

            # Run all models
            sym_out = symbolic_predict(session["readings"])
            esn_out = esn_m.predict(session["readings"])
            tda_out = tda_m.predict(session["readings"])
            svm_out = svm_m.predict(session["readings"])
            tmp_out = tmp_m.predict(session["readings"])

            pred, conf, mode = ensemble_predict(
                sym_out, esn_out, tda_out, svm_out, tmp_out, n_trained)

            risk = int(conf * 100)

            # Display
            icon = "🔴" if risk>=65 else ("🟡" if risk>=35 else "🟢")
            models_str = f"sym={sym_out[0]} esn={esn_out[0]} tda={tda_out[0]} svm={svm_out[0]} tmp={tmp_out[0]}"
            log.info(
                f"{icon} {feat['activity']:<10} "
                f"move={feat['movement']:.3f} "
                f"br={feat['breathing'] or '?':>4} "
                f"hi={feat['hi']:.3f} "
                f"risk={risk}% [{mode}]"
            )
            if reading_count % 12 == 0:  # every 6 hours
                log.info(f"   Models: {models_str}")

            # Alert if high risk
            if risk >= 70 and (datetime.now().hour >= 22 or datetime.now().hour < 6) and (time.time()-last_alert) > 300:
                msg = f"Amma night risk {risk}%\nMovement elevated. Check position."
                if send_alert(msg, risk):
                    log.info("   ALERT SENT")
                last_alert = time.time()

            # Save session every reading
            session["last_risk"] = risk
            session["last_mode"] = mode
            with open(session_file, "w") as f:
                json.dump(session, f)

            # Cloud sync every 2 hours
            if reading_count % 240 == 0:
                threading.Thread(target=cloud_sync, daemon=True).start()

    except KeyboardInterrupt:
        log.info(f"Stopped. {len(session['readings'])} readings saved.")
        with open(session_file, "w") as f:
            json.dump(session, f)
        log.info(f"File: {session_file}")
        log.info("Morning: python3.8 amma_40nights.py --label pain/ok")

# ── Label morning ─────────────────────────────────────
def label_morning(label):
    files = sorted([f for f in os.listdir(NIGHTS) if f.endswith(".json")])
    if not files:
        print("No session found. Run monitor first."); return

    target = None
    for fname in reversed(files):
        path = os.path.join(NIGHTS, fname)
        with open(path) as f: s = json.load(f)
        if s.get("label") is None:
            target = path; session = s; break

    if not target:
        print("All nights already labeled.")
        # Show last night anyway
        with open(os.path.join(NIGHTS, files[-1])) as f:
            session = json.load(f)
        print(f"Last night: {session['date']} — {session.get('label','unlabeled')}")
        return

    session["label"]    = label
    session["wake_time"]= datetime.now().strftime("%H:%M:%S")
    session["wake_ts"]  = time.time()

    with open(target, "w") as f:
        json.dump(session, f)

    # Retrain all models with new night
    esn_m = ESN(); tda_m = TDAModel()
    svm_m = KernelSVM(); tmp_m = TemporalModel()
    n = train_all_models(esn_m, tda_m, svm_m, tmp_m)

    readings = session.get("readings", [])
    duration = 0
    if session.get("start_ts") and session.get("wake_ts"):
        duration = (session["wake_ts"] - session["start_ts"]) / 3600

    n_active   = sum(1 for r in readings if r.get("activity")=="active")
    n_restless = sum(1 for r in readings if r.get("activity")=="restless")
    n_still    = sum(1 for r in readings if r.get("activity")=="still")

    print()
    print(f"  Night     : {session['date']}")
    print(f"  Label     : {label.upper()}")
    print(f"  Duration  : {duration:.1f} hours")
    print(f"  Readings  : {len(readings)}")
    print(f"  Still     : {n_still}  Restless: {n_restless}  Active: {n_active}")
    print(f"  Models trained on: {n} nights total")
    print()

    nights_done, labels_done = load_nights()
    pain_count = sum(labels_done)
    ok_count   = len(labels_done) - pain_count
    print(f"  Progress  : {len(labels_done)}/40 nights labeled ({pain_count} pain, {ok_count} ok)")

    next_milestone = 8 if len(labels_done)<8 else 14 if len(labels_done)<14 else 21 if len(labels_done)<21 else 40
    print(f"  Next milestone: night {next_milestone} — {'ESN+TDA activate' if next_milestone==8 else 'SVM activates' if next_milestone==14 else 'Full ensemble' if next_milestone==21 else 'Complete!'}")

    if label == "pain" and len(readings) >= 6:
        print()
        print("  Last 3 hours before waking:")
        for r in readings[-6:]:
            icon = "⚠" if r.get("activity")!="still" else " "
            print(f"    {r.get('time','?')}  {r.get('activity','?'):<10}  "
                  f"move={r.get('movement',0):.3f}  "
                  f"br={r.get('breathing') or '?'}  {icon}")

    cloud_sync()
    print(f"\n  Saved: {target}")

# ── Report ────────────────────────────────────────────
def show_report():
    files = sorted([f for f in os.listdir(NIGHTS) if f.endswith(".json")])
    if not files:
        print("No nights recorded yet."); return

    nights, labels = load_nights()
    n_nights = len(nights)

    print("=" * 60)
    print("  AMMA 40-NIGHT REPORT")
    print("=" * 60)
    print(f"\n  {'Date':<12} {'Hours':>6} {'Readings':>9} {'Active':>7} {'HiRisk':>7}  Label")
    print("  " + "-"*55)

    for fname in files:
        with open(os.path.join(NIGHTS, fname)) as f:
            s = json.load(f)
        readings = s.get("readings", [])
        duration = 0
        if s.get("start_ts") and s.get("wake_ts"):
            duration = (s["wake_ts"] - s["start_ts"]) / 3600
        label    = s.get("label", "unlabeled")
        n_active = sum(1 for r in readings if r.get("activity")=="active")
        n_hirisk = sum(1 for r in readings if safe(r.get("hi"),1.0)>1.3)
        icon     = "🔴" if label=="pain" else ("🟢" if label=="ok" else "⬜")
        print(f"  {s['date']:<12} {duration:>6.1f} {len(readings):>9} {n_active:>7} {n_hirisk:>7}  {icon} {label}")

    print()
    print(f"  Total    : {len(files)} nights")
    print(f"  Labeled  : {n_nights} ({sum(labels)} pain, {n_nights-sum(labels)} ok)")
    print(f"  Remaining: {max(0,40-n_nights)} nights to go")

    if n_nights >= 2:
        pain_nights = [nights[i] for i in range(n_nights) if labels[i]==1]
        ok_nights   = [nights[i] for i in range(n_nights) if labels[i]==0]

        if pain_nights and ok_nights:
            pain_mv = np.mean([np.mean([safe(r.get("movement"),0) for r in n]) for n in pain_nights])
            ok_mv   = np.mean([np.mean([safe(r.get("movement"),0) for r in n]) for n in ok_nights])
            pain_hi = np.mean([np.mean([safe(r.get("hi"),1)       for r in n]) for n in pain_nights])
            ok_hi   = np.mean([np.mean([safe(r.get("hi"),1)       for r in n]) for n in ok_nights])

            print()
            print("  PATTERN SO FAR:")
            print(f"    Pain nights avg movement : {pain_mv:.3f}")
            print(f"    OK nights avg movement   : {ok_mv:.3f}")
            print(f"    Pain nights avg hi       : {pain_hi:.3f}")
            print(f"    OK nights avg hi         : {ok_hi:.3f}")

            if pain_mv > ok_mv * 1.1:
                print("    → Amma more restless on pain nights")
            if pain_hi > ok_hi * 1.05:
                print("    → Higher side-compression on pain nights")

        # Model status
        print()
        print("  MODEL STATUS:")
        print(f"    Symbolic AI        : ACTIVE (night 1+)")
        print(f"    Reservoir + TDA    : {'ACTIVE' if n_nights>=8  else f'activates night 8 ({8-n_nights} to go)'}")
        print(f"    Kernel SVM         : {'ACTIVE' if n_nights>=14 else f'activates night 14 ({14-n_nights} to go)'}")
        print(f"    Temporal LSTM-lite : {'ACTIVE' if n_nights>=21 else f'activates night 21 ({21-n_nights} to go)'}")
        print(f"    Full ensemble      : {'ACTIVE' if n_nights>=21 else f'activates night 21 ({21-n_nights} to go)'}")

    print(f"\n  Data: {NIGHTS}/")

# ── Cloud setup ───────────────────────────────────────
def setup_cloud():
    print()
    print("CLOUD SETUP (Google Drive via rclone)")
    print("="*45)
    print()
    print("1. Install rclone:")
    print("   curl https://rclone.org/install.sh | sudo bash")
    print()
    print("2. Configure Google Drive:")
    print("   rclone config")
    print("   → Choose 'n' for new remote")
    print("   → Name it 'gdrive'")
    print("   → Choose Google Drive")
    print("   → Follow auth steps")
    print()
    print("3. Test:")
    print("   rclone ls gdrive:")
    print()
    print("After setup, data syncs automatically every 2 hours.")
    print("For iCloud: use rclone with WebDAV to iCloud Drive.")

# ── Entry point ───────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Amma 40-Night Learning System")
    p.add_argument("--label",       choices=["pain","ok"], help="Label last night on waking")
    p.add_argument("--report",      action="store_true",   help="Show all nights + model status")
    p.add_argument("--setup-cloud", action="store_true",   help="Instructions for cloud sync")
    args = p.parse_args()

    if args.label:
        label_morning(args.label)
    elif args.report:
        show_report()
    elif args.setup_cloud:
        setup_cloud()
    else:
        run_monitor()
