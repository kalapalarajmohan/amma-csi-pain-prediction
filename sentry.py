import subprocess
import time
import datetime
import os
import signal as sig_handler

from test_plug import get_token, control_plug

def cleanup(signum, frame):
    os.remove(LOCK)
    exit()

sig_handler.signal(sig_handler.SIGTERM, cleanup)
sig_handler.signal(sig_handler.SIGINT, cleanup)

LOCK = "/tmp/sentry.lock"
if os.path.exists(LOCK):
    print("Already running. Exit.")
    exit()
open(LOCK, 'w').close()

DATA = "/home/rajmohan/amma-project/data/amma_data.txt"
ALERTS = "/home/rajmohan/amma-project/data/alerts.txt"

BASELINE = -59
THRESHOLD = 20
SPIKE_COUNT =5 
WINDOW = 60
SPIKES = []

def read_signal():
    try:
        r = subprocess.run(['tail','-n','1',DATA],capture_output=True,text=True)
        if not r.stdout.strip():
            return None
        return int(r.stdout.strip().split()[-1])
    except:
        return None

print("Sentry Active - Monitoring Amma")

while True:
    sig = read_signal()
    if sig is not None:
        print(f"Signal: {sig}")
        now = time.time()
        if abs(sig - BASELINE) > THRESHOLD:
            SPIKES.append(now)
        SPIKES[:] = [s for s in SPIKES if now -s < 60]
        if len(SPIKES) >= 5:
            t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(ALERTS,'a') as f:
                f.write(t+ 'ALERT sig=' + str(sig) + '\n')
                token = get_token()
                control_plug(token, True)
            SPIKES.clear()
    time.sleep(1)
