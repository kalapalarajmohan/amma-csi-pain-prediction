import subprocess
import time
import datetime

DATA = "/home/rajmohan/amma-project/data/amma_data_wlan1.txt"

while True:
    try:
        r = subprocess.run(['/sbin/iwconfig','wlan1'],capture_output=True,text=True)
        if 'Signal level' not in r.stdout:
            r = subprocess.run(['/sbin/iwconfig','wlan0'],capture_output=True,text=True)
    except:
        r = subprocess.run(['/sbin/iwconfig','wlan0'],capture_output=True,text=True)
    for line in r.stdout.split('\n'):
        if 'Signal level' in line:
            sig = line.split('Signal level=')[1].split(' ')[0]
            t = datetime.datetime.now().strftime('%H:%M:%S')
            with open(DATA,'a') as f:
                f.write(t + ' ' + sig +'\n')
    time.sleep(1)

