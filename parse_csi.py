import ast
import math
DATA = "/home/rajmohan/amma-project/data/esp32_data.txt"
with open(DATA, 'r') as f:
    lines = f.readlines()[-20:]
for line in f:
    try:
        parts= line.strip().split(' ',1)
        t= parts[0]
        csi_raw = parts[1]
        csi_bytes = eval(csi_raw)
        print("bytes:", len(csi_bytes))
        print(t, amplitudes[:5])
        csi_ints = [int(b) for b in csi_bytes]
        amplitude = []
        for i in range(0, len(csi_ints)-1,2):
            real = csi_ints[i]
            imag = csi_ints[i+1]
            amp = math.sqrt(real**2 + imag**2)
            amplitudes.append(amp)
    except Exception as e:
        print(e)
