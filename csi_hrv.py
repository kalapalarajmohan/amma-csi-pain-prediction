import datetime
import time
import math
import numpy as np
import subprocess


DATA= "/home/rajmohan/amma-project/data/esp32_data.txt"


result = subprocess.run(['tail', '-600', DATA], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

amplitude_buffer = []

for line in lines:
    try:
        parts = line.strip().split(' ', 2)
        t = parts[0] + ' '+ parts[1] 
        raw = parts[2]
        b = eval(raw)
        ints = list(b)
        amps = [math.sqrt(ints[i]**2 + ints[i+1]**2) for i in range(0, len(ints)-1, 2)]
        amplitude_buffer.append(amps)
    except:
        pass


data_array = np.array(amplitude_buffer)
fft_result = np.fft.fft(data_array, axis=0)
fft_magnitude = np.abs(fft_result)
freqs = np.fft.fftfreq(len(amplitude_buffer), d=0.1)


#look for breathing frequency(0.1 to 0.5 Hz)
breathing_mask = (freqs > 0.1) & (freqs < 0.5)
breathing_power = fft_magnitude[breathing_mask, :]
heart_mask = (freqs > 0.8) & (freqs < 2.0)
heart_power = fft_magnitude[heart_mask, :]


best_subcarrier = np.argmax(breathing_power.mean(axis=0))

breathing_peak_idx = np.argmax(breathing_power[:, best_subcarrier])
breathing_freq = abs(freqs[breathing_mask][breathing_peak_idx])
breathing_rate = breathing_freq * 60

heart_peak_idx = np.argmax(heart_power[:, best_subcarrier])
heart_freqs = abs(freqs[heart_mask][heart_peak_idx])
heart_rate = heart_freqs * 60

t_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
if 8 <= breathing_rate <= 20 and 40 <= heart_rate <= 90:
    print(f"AMMA {t_now} BR :{breathing_rate:.1f} HR:{heart_rate:.1f}")
else:
    print(f"NOISE{t_now} BR:{breathing_rate:.1f} HR:{heart_rate:.1f}")
