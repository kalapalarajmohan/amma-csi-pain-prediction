import sys, json
import numpy as np

def load_amp_log(path):
    ts_list, amp_list = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                ts = float(parts[0])
                amps = np.array([float(x) for x in parts[1].split(",")])
            except ValueError:
                continue
            ts_list.append(ts)
            amp_list.append(amps)
    return np.array(ts_list), np.array(amp_list)

def estimate_fs_actual(timestamps):
    if len(timestamps) < 2:
        return None
    dt = np.diff(timestamps)
    dt = dt[(dt > 0) & (dt < 5)]
    if len(dt) == 0:
        return None
    return 1.0 / np.median(dt)

def simple_find_peaks(x, distance=1):
    peaks = []
    i = 1
    while i < len(x) - 1:
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if not peaks or (i - peaks[-1]) >= distance:
                peaks.append(i)
        i += 1
    return np.array(peaks)

def estimate_breathing_bpm(amp_window, fs, window_ts=None):
    from scipy.signal import butter, filtfilt
    energy = amp_window.mean(axis=1)
    if window_ts is not None and len(window_ts) > 1:
        span = window_ts[-1] - window_ts[0]
        if span > 0:
            fs = (len(window_ts) - 1) / span
    if fs < 4 or len(energy) < 8:
        return None
    nyq = fs / 2
    try:
        b, a = butter(4, [0.15/nyq, 0.5/nyq], btype='band')
        filt = filtfilt(b, a, energy)
        pks = simple_find_peaks(filt, distance=int(fs*2))
        if len(pks) >= 2:
            rate = 60.0 / np.mean(np.diff(pks) / fs)
            return round(rate, 1) if 6 <= rate <= 30 else None
    except Exception:
        return None
    return None

def load_logged_breathing(json_path):
    with open(json_path) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("readings", [])
    return [(r["ts"], r.get("breathing")) for r in records if "ts" in r]

def main():
    amp_path, json_path = sys.argv[1], sys.argv[2]
    logged = load_logged_breathing(json_path)
    if not logged:
        print("No records with 'ts' found in JSON."); sys.exit(1)
    ts_min, ts_max = logged[0][0], logged[-1][0]
    print(f"Night covers ts {ts_min:.0f} to {ts_max:.0f} ({(ts_max-ts_min)/3600:.1f} hours)")
    all_ts, all_amp = load_amp_log(amp_path)
    mask = (all_ts >= ts_min - 60) & (all_ts <= ts_max + 60)
    matched_ts, matched_amp = all_ts[mask], all_amp[mask]
    if len(matched_ts) < 50:
        print(f"Only found {len(matched_ts)} matching packets -- check overlap."); sys.exit(1)
    fs = estimate_fs_actual(matched_ts)
    print(f"Found {len(matched_ts)} matching packets, real fs_actual = {fs:.2f} Hz")
    print(f"\n{'time':>8} {'logged_bpm':>12} {'computed_bpm':>14} {'diff':>8}")
    checked = 0
    for logged_ts, logged_bpm in logged[::5]:
        if logged_bpm is None:
            continue
        wmask = (matched_ts >= logged_ts - 15) & (matched_ts <= logged_ts + 15)
        window = matched_amp[wmask]
        if len(window) < 10:
            continue
        window_ts_local = matched_ts[wmask]
        computed = estimate_breathing_bpm(window, fs, window_ts_local)
        if computed is None:
            continue
        print(f"{logged_ts % 86400:8.0f} {logged_bpm:12.1f} {computed:14.1f} {computed-logged_bpm:+8.1f}")
        checked += 1
        if checked >= 30:
            break
    print(f"\nCompared {checked} windows.")

if __name__ == "__main__":
    main()
