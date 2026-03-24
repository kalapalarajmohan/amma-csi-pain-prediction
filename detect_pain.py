import time
import subprocess

BASELINE = -80
THRESHOLD = 15
WINDOW = 60
SPIKE_COUNT = 10

def get_latest_signal():
    result = subprocess.run(['tail', '-1', '/home/rajmohan/amma-project/data/amma_data.txt'],
capture_output=True, text=True)
line = result.stdout.strip()
if line:
parts = line.split()
if len(parts) >= 3:
return int(parts[2])
return None

spikes = []
print("Monitoring started...")

while True:
signal = get_latest_signal()
if signal:
now = time.time()
if abs(signal - BASELINE) > THRESHOLD:
spikes.append(now)
spikes = [s for s in spikes if now - s < WINDOW]
if len(spikes) >= SPIKE_COUNT:
print(f"ALERT: Pain detected! {len(spikes)} spikes in 60 seconds")
spikes = []
time.sleep(1)
