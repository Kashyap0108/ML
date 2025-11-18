import math
import matplotlib.pyplot as plt
import numpy as np
threshold = 0.1
Poutref = 0.63
x = []
P_out_ray = []
for i in np.arange(0.01,0.9,0.01):
  x.append(i)
  P_out_ray.append(1 - math.exp(-threshold/i))

plt.plot(x,P_out_ray)
plt.xlabel("Average SNR")
plt.ylabel("Outage probability")
plt.title("Outage probability for rayleigh fading channel")
plt.axhline(Poutref,color="green",linestyle="--")


import math
import matplotlib.pyplot as plt
import numpy as np

threshold = 0.1
Poutref = 0.63
k = 3
x = []
P_out_rician = []
for i in np.arange(0.01, 0.9, 0.01):
  x.append(i)
  gamma_avg = i
  P_out = (1 - math.exp(-threshold/gamma_avg))*math.exp(-k)
  P_out_rician.append(P_out)
plt.plot(x, P_out_rician)
plt.xlabel("Average SNR")
plt.ylabel("Outage probability")
plt.title("Outage probability for rician fading channel")
plt.axhline(Poutref, color="green", linestyle="--")