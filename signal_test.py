import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

time=np.linspace(0,10,1000)
signal=np.sin(2 * np.pi * time) + np.random.normal(0,0.1,1000)

plt.figure()
plt.plot(time, signal)
plt.title("simulated WiFi signal")
plt.xlabel("time(seconds)")
plt.ylabel("signal strength")
plt.savefig("data/signal.png")
print("signal plot saved to data/signal.png")
