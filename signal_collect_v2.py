import subprocess
import time
import datetime

DATA = "/home/rajmohan/amma-project/data/amma_data.txt"

print("Collector running...")

while True:
    r = subprocess.run(['iwconfig','wlan0'],capture_output=True,text=True)
    for line in r.stdout.split('\n'):
        if 'Signal level' in line:
            sig = line.split('Signal level=')[1].split(' ')[0]
            t = datetime.datetime.now().strftime('%H:%M:%S')
            with open(DATA,'a') as f:
                f.write(t+ ' ' + sig + '\n')
    time.sleep(1)
