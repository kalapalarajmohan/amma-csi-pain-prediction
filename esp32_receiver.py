import os, socket, struct, threading, logging, time, yaml
import numpy as np
from collections import deque

def load_config():
    for path in ["config.yaml", "/home/rajmohan/csi_sciatica/config.yaml"]:
        if os.path.exists(path):
            with open(path) as f: return yaml.safe_load(f)
    return {"network": {"esp32_udp_port": 5005, "buffer_size": 65535},
            "esp32":   {"packet_rate_hz": 100, "n_subcarriers": 52}}

cfg           = load_config()
UDP_PORT      = cfg["network"]["esp32_udp_port"]
BUFFER_SIZE   = cfg["network"]["buffer_size"]
N_SUBCARRIERS = cfg["esp32"]["n_subcarriers"]
PACKET_RATE   = cfg["esp32"]["packet_rate_hz"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ESP32] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("esp32_receiver")

RING_SIZE  = PACKET_RATE * 60 * 5
_ring_lock = threading.Lock()
_ring      = deque(maxlen=RING_SIZE)
_stats     = {"packets_received": 0, "packets_dropped": 0, "last_packet_ts": 0.0, "connected": False}
_stop_event    = threading.Event()
_listen_thread = None

def _parse_packet(data):
    expected = 2 * N_SUBCARRIERS
    if len(data) < expected: return None
    try:
        raw  = struct.unpack(f"{2*N_SUBCARRIERS}b", data[:expected])
        imag = np.array(raw[0::2], dtype=np.float32)
        real = np.array(raw[1::2], dtype=np.float32)
        return real + 1j * imag
    except struct.error:
        return None

def _make_synthetic_packet(t):
    amp = (1.0 + 0.3*np.sin(2*np.pi*0.05*t) + 0.1*np.sin(2*np.pi*0.25*t)
           + 0.05*np.random.randn(N_SUBCARRIERS))
    phase = np.linspace(-np.pi, np.pi, N_SUBCARRIERS) + 0.05*np.random.randn(N_SUBCARRIERS)
    return amp * np.exp(1j * phase)

def _udp_listen_loop(synthetic=False):
    if synthetic:
        log.warning("SYNTHETIC MODE")
        interval = 1.0 / PACKET_RATE
        t = 0.0
        while not _stop_event.is_set():
            ts = time.time()
            csi = _make_synthetic_packet(t)
            with _ring_lock: _ring.append((ts, csi))
            _stats["packets_received"] += 1
            _stats["last_packet_ts"]   = ts
            _stats["connected"]        = True
            t += interval
            time.sleep(interval)
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(5.0)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        log.info(f"Listening on UDP port {UDP_PORT}")
    except OSError as e:
        log.error(f"Cannot bind port {UDP_PORT}: {e}"); return

    while not _stop_event.is_set():
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            if not _stats["connected"]:
                log.info(f"ESP32 connected from {addr[0]}")
            ts  = time.time()
            csi = _parse_packet(data)
            if csi is None:
                _stats["packets_dropped"] += 1; continue
            with _ring_lock: _ring.append((ts, csi))
            _stats["packets_received"] += 1
            _stats["last_packet_ts"]   = ts
            _stats["connected"]        = True
            if _stats["packets_received"] % 1000 == 0:
                log.info(f"rx={_stats['packets_received']} dropped={_stats['packets_dropped']} buf={len(_ring)}")
        except socket.timeout:
            elapsed = time.time() - _stats["last_packet_ts"]
            if _stats["connected"] and elapsed > 10:
                log.warning(f"No packets for {elapsed:.0f}s")
                _stats["connected"] = False
        except Exception as e:
            log.error(f"Error: {e}")
    sock.close()

def start(synthetic=False):
    global _listen_thread
    _stop_event.clear()
    _listen_thread = threading.Thread(target=_udp_listen_loop, args=(synthetic,), daemon=True)
    _listen_thread.start()
    log.info(f"ESP32 receiver started (synthetic={synthetic})")

def stop():
    _stop_event.set()
    if _listen_thread: _listen_thread.join(timeout=6)

def get_packets(n):
    with _ring_lock:
        if len(_ring) < n: return None, None
        recent = list(_ring)[-n:]
    return [r[0] for r in recent], np.stack([r[1] for r in recent], axis=0)

def get_buffer_size():
    with _ring_lock: return len(_ring)

def is_connected():
    if _stats["last_packet_ts"] == 0: return False
    return (time.time() - _stats["last_packet_ts"]) < 10.0

def get_stats():
    return dict(_stats)

if __name__ == "__main__":
    print("Testing ESP32 receiver in synthetic mode...")
    start(synthetic=True)
    time.sleep(8)
    ts, csi = get_packets(500)
    if csi is not None:
        print(f"Shape: {csi.shape}  Mean amp: {np.abs(csi).mean():.3f}")
        print(f"Stats: {get_stats()}")
        print("PASS")
    else:
        print("FAIL - no data")
    stop()
