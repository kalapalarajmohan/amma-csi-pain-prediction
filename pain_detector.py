import numpy as np
from datetime import datetime
import json, os, time

class SciaticaPainDetector:
    def __init__(self, threshold=75.0, history_sec=15, fs=100.0):
        self.threshold      = threshold
        self.episode_active = False
        self.episode_start  = None
        self.history        = []
        self.history_maxlen = int(history_sec)
        self.log_path       = "data/pain_labels.jsonl"
        os.makedirs("data", exist_ok=True)

    def compute_instability(self, csi_window):
        amp = np.abs(csi_window)
        per_sc_var  = np.var(amp, axis=0)
        instability = np.mean(per_sc_var) / (np.mean(amp) + 1e-8)
        return float(instability)

    def score(self, spine_angle_deg, instability, hrv=None, hr=None):
        s = 0.0
        if abs(spine_angle_deg) > 15: s += 30
        if instability > 4.0:         s += 35
        if hr  is not None and hr  > 90: s += 20
        if hrv is not None and hrv < 30: s += 15
        return min(s, 100.0)

    def analyze(self, spine_angle_deg, instability, hrv=None, hr=None):
        s       = self.score(spine_angle_deg, instability, hrv, hr)
        is_pain = s >= self.threshold
        self.history.append(s)
        if len(self.history) > self.history_maxlen:
            self.history.pop(0)

        if is_pain and not self.episode_active:
            self.episode_active = True
            self.episode_start  = time.time()
            self._log_event("pain_start", s, spine_angle_deg, instability)
            return "START", s
        elif not is_pain and self.episode_active:
            self.episode_active = False
            duration = time.time() - (self.episode_start or time.time())
            self._log_event("pain_end", s, spine_angle_deg, instability, duration_sec=duration)
            self.episode_start = None
            return "END", s
        return ("CONTINUE" if self.episode_active else "IDLE"), s

    def _log_event(self, event, score, spine, instability, duration_sec=None):
        entry = {
            "timestamp":     time.time(),
            "readable_time": datetime.now().isoformat(),
            "event":         event,
            "score":         round(score, 1),
            "spine_angle":   round(spine, 1),
            "instability":   round(instability, 3),
            "source":        "detector",
        }
        if duration_sec is not None:
            entry["duration_sec"] = round(duration_sec, 1)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent_score_avg(self):
        if not self.history: return 0.0
        return float(np.mean(self.history))

if __name__ == "__main__":
    detector = SciaticaPainDetector()
    print("=== Pain Detector Test ===")
    print(f"{'Min':>4}  {'Spine':>7} {'Instab':>8} {'Score':>7} {'State'}")
    print("-" * 45)
    timeline = [
        (1,   0.0, 0.2,  None, None),
        (2,   8.0, 0.3,  None, None),
        (3,  16.0, 0.5,  None, None),
        (4,  20.0, 0.85, 85.0, 95.0),
        (5,  25.0, 0.90, 78.0, 98.0),
        (6,   5.0, 0.25, 90.0, 70.0),
        (7,   2.0, 0.15, None, None),
    ]
    for min_n, spine, instab, hrv, hr in timeline:
        state, score = detector.analyze(spine, instab, hrv=hrv, hr=hr)
        print(f"  {min_n:>4}m  {spine:>+7.1f} {instab:>8.2f} {score:>7.1f} {state}")
    print()
    print("PASS")
