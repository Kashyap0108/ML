import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_samples = 10000
delays = np.array([0, 20, 40, 60, 80])
powers_db = np.array([0, -3, -6, -9, -12])

# Power
powers_lin = 10 ** (powers_db / 10.0)
powers_lin /= np.sum(powers_lin)

h = np.zeros((num_samples, len(delays)), dtype=complex)
for i, P in enumerate(powers_lin):
    real = np.random.normal(0, np.sqrt(P/2), num_samples)
    imag = np.random.normal(0, np.sqrt(P/2), num_samples)
    h[:, i] = real + 1j * imag

pdp = np.mean(np.abs(h) ** 2, axis=0)
pdp /= np.sum(pdp)

plt.stem(delays, pdp, basefmt=" ")
plt.xlabel("Delay (ns)")
plt.ylabel("Normalized PDP")
plt.title("Power Delay Profile (PDP)")
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

L = np.array([1, 2, 3, 4, 5, 6])  # Diversity orders
SNR_dB = np.linspace(0, 25, 1000)  # SNR from 0 to 25 dB
SNR_linear = 10 ** (SNR_dB / 10)

plt.figure(figsize=(8, 6))

for l in L:
    # Calculate BER for BPSK with L-branch MRC Rayleigh fading
    # Formula: Pb = 0.5 * (1 - sqrt(SNR/(1+SNR)))^L
    ber = 0.5 * (1 - np.sqrt(SNR_linear / (1 + SNR_linear))) ** l
    plt.semilogy(SNR_dB, ber, label=f"L={l}")

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("SNR vs BER")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.ylim(1e-6, 1)
plt.show()
