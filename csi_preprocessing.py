import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import PCA
import yaml, os

def load_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

def sanitize_input(csi):
    csi = csi.copy()
    for sc in range(csi.shape[1]):
        col = csi[:, sc]
        bad = ~np.isfinite(col)
        if bad.any():
            col_mean = np.nanmean(col) if np.isfinite(col).any() else 0.0
            col[bad] = col_mean
            csi[:, sc] = col
    return csi

def reject_outliers(csi, window=50, sigma=2.5):
    csi_clean = csi.copy()
    packet_energy = np.mean(np.abs(csi), axis=1)
    for i in range(len(packet_energy)):
        lo = max(0, i - window // 2)
        hi = min(len(packet_energy), i + window // 2)
        local_mean = np.mean(packet_energy[lo:hi])
        local_std  = np.std(packet_energy[lo:hi]) + 1e-8
        if abs(packet_energy[i] - local_mean) > sigma * local_std:
            csi_clean[i] = np.mean(csi[lo:hi], axis=0)
    return csi_clean

def smooth_amplitude(csi, window=11, poly=3):
    amplitude = np.abs(csi)
    smoothed  = np.zeros_like(amplitude)
    win = min(window, len(csi) if len(csi) % 2 == 1 else len(csi) - 1)
    win = max(win, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    for sc in range(amplitude.shape[1]):
        smoothed[:, sc] = savgol_filter(amplitude[:, sc], window_length=win, polyorder=poly)
    return smoothed

def sanitize_phase(csi_complex):
    raw_phase   = np.angle(csi_complex)
    clean_phase = np.zeros_like(raw_phase)
    x = np.arange(raw_phase.shape[1])
    for i in range(raw_phase.shape[0]):
        slope, intercept = np.polyfit(x, raw_phase[i], 1)
        clean_phase[i] = raw_phase[i] - (slope * x + intercept)
    return clean_phase

def lowpass_filter(csi, cutoff_hz=5.0, fs_hz=100.0, order=4):
    nyq = fs_hz / 2.0
    cutoff_hz = min(cutoff_hz, nyq * 0.95)
    cutoff_hz = max(cutoff_hz, 0.01)
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)
    filtered = np.zeros_like(csi)
    for sc in range(csi.shape[1]):
        filtered[:, sc] = filtfilt(b, a, csi[:, sc])
    return filtered

def subtract_baseline(csi, baseline=None, baseline_packets=200):
    n = min(baseline_packets, len(csi))
    if baseline is None:
        baseline = np.mean(csi[:n], axis=0, keepdims=True)
    return csi - baseline, baseline

def pca_denoise(csi, n_components=10):
    max_components = min(csi.shape[0], csi.shape[1])
    n_components = min(n_components, max_components)
    pca = PCA(n_components=n_components)
    compressed    = pca.fit_transform(csi)
    reconstructed = pca.inverse_transform(compressed)
    print(f"  [PCA] {n_components} components -> {np.sum(pca.explained_variance_ratio_)*100:.1f}% variance explained")
    return reconstructed

def zscore_normalize(csi, eps=1e-8):
    mean = np.mean(csi, axis=0, keepdims=True)
    std  = np.std(csi, axis=0, keepdims=True) + eps
    return (csi - mean) / std, mean, std

def run_csi_pipeline(csi_complex, fs_hz=100.0, baseline=None, pca_components=10, lowpass_cutoff=5.0):
    print("=== CSI Filter Pipeline ===")
    print("[0/7] Sanitizing input...")
    csi_complex = sanitize_input(csi_complex)
    print("[1/7] Rejecting outliers...")
    csi_amp = reject_outliers(csi_complex)
    print("[2/7] Smoothing amplitude...")
    csi_amp = smooth_amplitude(csi_amp)
    print("[3/7] Sanitizing phase...")
    phase_clean = sanitize_phase(csi_complex)
    print("[4/7] Low-pass filter...")
    csi_amp = lowpass_filter(csi_amp, cutoff_hz=lowpass_cutoff, fs_hz=fs_hz)
    print("[5/7] Baseline subtraction...")
    csi_amp, baseline_used = subtract_baseline(csi_amp, baseline=baseline)
    print("[6/7] PCA denoising...")
    csi_amp = pca_denoise(csi_amp, n_components=pca_components)
    amplitude_pre_norm = csi_amp.copy()
    print("[7/7] Z-score normalization...")
    csi_amp, mean, std = zscore_normalize(csi_amp)
    print(f"=== Done | shape={csi_amp.shape} | NaNs={np.isnan(csi_amp).any()} ===")
    return {
        "clean":     csi_amp,
        "amplitude": amplitude_pre_norm,
        "baseline":  baseline_used,
        "mean":      mean,
        "std":       std,
        "phase":     phase_clean,
    }

if __name__ == "__main__":
    print("Testing pipeline...")
    t = np.linspace(0, 30, 3000)
    raw = (np.sin(2*np.pi*0.1*t)[:,None]*np.random.randn(1,52) + 0.5*np.random.randn(3000,52)) * np.exp(1j*np.random.uniform(-np.pi,np.pi,(3000,52)))
    r = run_csi_pipeline(raw)
    print(f"Mean={r['clean'].mean():.4f}  Std={r['clean'].std():.4f}")
    print("PASS" if abs(r['clean'].mean()) < 0.01 else "FAIL")
