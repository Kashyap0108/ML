import numpy as np
import matplotlib.pyplot as plt
import math
num_samples = 100000
h_real = np.random.randn(num_samples)
h_imag = np.random.randn(num_samples)
h = h_real + 1j*h_imag
channel_gain = np.abs(h)**2
threshold_SNR_db = 5
threshold_SNR_linear = 10**(threshold_SNR_db/10)
average_SNR_db = 15
average_SNR_linear = 10**(average_SNR_db/10)
gamma = average_SNR_linear*channel_gain
count = 0
for i in range(num_samples):
  if gamma[i] < threshold_SNR_linear:
    count += 1
outage_probability = count/num_samples
print("Outage probability:", outage_probability)
Poutref = 1 - math.exp(-threshold_SNR_linear/gamma[i])
print("Poutref:", Poutref)