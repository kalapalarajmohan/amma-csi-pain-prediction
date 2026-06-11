import os, sys, time, logging, logging.handlers, argparse, traceback
import numpy as np

def load_config():
    for path in ["config.yaml", "/home/rajmohan/csi_sciatica/config.yaml"]:
        if os.path.exists(path):
            import yaml
            with open(path) as f: return yaml.safe_load(f)
    return {}

cfg     = load_config()
LOG_DIR = cfg.get("paths", {}).get("logs", "/home/rajmohan/csi_sciatica/logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
fh  = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "orchestrator.log"), maxBytes=10*1024*1024, backupCount=3)
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(fh)
root.addHandler(ch)
log = logging.getLogger("orchestrator")

try:
    import esp32_receiver
    import apple_receiver
    import ground_truth_api
    from csi_preprocessing import run_csi_pipeline
    from pain_detector import SciaticaPainDetector
    log.info("All modules imported OK")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def wait_for_esp32(synthetic=False, timeout=60):
    log.info("Waiting for ESP32...")
    start = time.time()
    while time.time() - start < timeout:
        if esp32_receiver.get_buffer_size() > 100:
            log.info(f"ESP32 live — buf={esp32_receiver.get_buffer_size()}")
            return True
        time.sleep(1)
    log.error("No ESP32 data")
    return False

def _health_loop():
    import threading
    def _loop():
        while True:
            time.sleep(60)
            s = esp32_receiver.get_stats()
            log.info(f"[HEALTH] connected={esp32_receiver.is_connected()} "
                     f"rx={s['packets_received']} drop={s['packets_dropped']} "
                     f"apple={apple_receiver.get_store_size()} "
                     f"labels={ground_truth_api.get_labels_count()}")
    threading.Thread(target=_loop, daemon=True).start()

def run_live(synthetic=False):
    detector = SciaticaPainDetector()
    fs       = cfg.get("esp32", {}).get("packet_rate_hz", 100.0)
    window   = int(fs * 5)  # 5 second windows

    log.info("=== LIVE MONITORING STARTED ===")
    log.info("Trigger pain labels via:")
    log.info(f"  curl -X POST http://192.168.29.232:5051/groundtruth -H 'Content-Type: application/json' -d '{{\"event\":\"pain_start\",\"intensity\":7}}'")

    while True:
        ts_list, raw_csi = esp32_receiver.get_packets(window)
        if raw_csi is None:
            time.sleep(1)
            continue

        try:
            result    = run_csi_pipeline(raw_csi, fs_hz=fs)
            amplitude = result["amplitude"]

            # Pose from subcarrier groups
            amp_var = np.var(amplitude, axis=0)
            lo  = amp_var[:17].mean()
            mid = amp_var[17:35].mean()
            hi  = amp_var[35:].mean()
            ref = amp_var.mean() + 1e-8
            # Calibrated from real data: leaning gives lo-hi ~ -2.4, upright ~ +7.9
            # Map this range to degrees: upright=0, forward lean=+30
            raw_signal = (lo - hi) / ref
            spine = float(np.clip((0.598 - raw_signal) * 20, -45, 45))
            hip   = float(np.clip((mid - hi) / ref * 10, -1, 1))

            # Instability
            instability = detector.compute_instability(amplitude)

            # Get latest Apple Watch data
            gt      = apple_receiver.get_store_snapshot()
            latest  = gt[-1] if gt else {}
            hrv     = latest.get("hrv")
            hr      = latest.get("heart_rate")

            # Pain detection
            state, score = detector.analyze(spine, instability, hrv=hrv, hr=hr)

            log.info(f"spine={spine:+.1f}deg  hip={hip:+.2f}  "
                     f"instab={instability:.2f}  score={score:.0f}  state={state}  "
                     f"hrv={hrv}  hr={hr}")

            if state == "START":
                log.info("!!! PAIN EPISODE DETECTED !!!")
            elif state == "END":
                log.info("--- Pain episode ended ---")

        except Exception as e:
            log.error(f"Pipeline error: {e}")
            log.error(traceback.format_exc())

        time.sleep(5)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="Use synthetic ESP32 data")
    args = p.parse_args()

    log.info("=" * 50)
    log.info("  CSI SCIATICA PIPELINE STARTING")
    log.info(f"  synthetic={args.synthetic}")
    log.info("=" * 50)

    esp32_receiver.start(synthetic=args.synthetic)
    apple_receiver.start(blocking=False)
    ground_truth_api.start(blocking=False, port=5051)
    time.sleep(2)

    _health_loop()

    if not wait_for_esp32(synthetic=args.synthetic, timeout=60):
        if not args.synthetic:
            log.error("No ESP32. Try: python3.8 orchestrator.py --synthetic")
            sys.exit(1)

    run_live(synthetic=args.synthetic)

if __name__ == "__main__":
    main()
