import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = []
with open("data/readings.txt", "r") as f:
    for line in f:
        if "Signal level" in line:
            val = line.split("Signal level=")[1].split(" ")[0]
            data.append(int(val))

plt.figure(figsize=(12,4))
plt.plot(data[:1000])
plt.title("Amma Sleep Signal - First 1000 readings")
plt.xlabel("Time (seconds)")
plt.ylabel("Signal dBm")
plt.savefig("data/amma_signal.png")
print("Plot saved!")
