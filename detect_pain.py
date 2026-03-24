cat > ~/amma-project/detect_pain.py << 'EOF'
import time
import subprocess
import os

# CONFIGURATION
BASELINE = -80
THRESHOLD = 15
WINDOW = 60
SPIKE_COUNT = 10
LOG_FILE = "/home/rajmohan/amma-project/data/amma_data.txt"

def get_latest_signal():
    try:
        # Compatibility fix: using stdout=subprocess.PIPE for Python 3.6
        result = subprocess.run(['tail', '-n', '1', LOG_FILE],
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              universal_newlines=True)
        
        line = result.stdout.strip()
        if line:
            parts = line.split()
            # Grabbing the last element ensures we get the RSSI number
            if len(parts) > 0:
                return int(parts[-1])
    except Exception:
        pass
    return None

spikes = []
print("--- Monitoring Started (Stable Version) ---")

while True:
    signal = get_latest_signal()
    
    if signal is not None:
        now = time.time()
        
        # Check if the signal deviates significantly from the baseline
        if abs(signal - BASELINE) > THRESHOLD:
            spikes.append(now)
            # Optional: print(f"Spike detected: {signal}") 
            
        # Sliding Window: Keep only spikes from the last 60 seconds
        spikes = [s for s in spikes if now - s < WINDOW]
        
        # Alert Logic
        if len(spikes) >= SPIKE_COUNT:
            print(f"ALERT: High Restlessness Detected! ({len(spikes)} spikes in {WINDOW}s)")
            # Reset after alert to avoid continuous spamming
            spikes = [] 
            
    time.sleep(1)
EOF
