import numpy as np
import matplotlib.pyplot as plt

# Parameters
EbN0_dB = np.arange(0, 21, 2)   # Eb/N0 range in dB
N_bits_list = [1000, 10000, 100000, 1000000]
ber_curves = []

plt.figure(figsize=(10, 6))

for N_bits in N_bits_list:
    ber_sim = []

    for ebn0 in EbN0_dB:
        bits = np.random.randint(0, 2, N_bits)
        x = 2 * bits - 1   # BPSK: 0→-1, 1→+1
        h = (np.random.randn(N_bits) + 1j * np.random.randn(N_bits)) / np.sqrt(2)
        EbN0 = 10 ** (ebn0 / 10)
        N0 = 1 / EbN0
        noise = (np.random.randn(N_bits) + 1j * np.random.randn(N_bits)) * np.sqrt(N0 / 2)
        y = h * x + noise
        y_eq = y / h
        bits_rx = np.real(y_eq) >= 0
        errors = np.sum(bits != bits_rx)
        ber_sim.append(errors / N_bits)
    ber_curves.append(ber_sim)
    plt.semilogy(EbN0_dB, ber_sim, marker='o', label=f"N_bits={N_bits}")

plt.grid(True, which='both')
plt.xlabel('$E_b/N_0$ (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER of BPSK (0→-1, 1→+1) over Rayleigh SISO for Different Number of Bits')
plt.legend()
plt.show()
