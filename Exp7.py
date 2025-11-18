import numpy as np
import matplotlib.pyplot as plt

def simulate_alamouti_ber(num_symbols, snr_db_range):
    """Simulate BER for Alamouti STBC over Rayleigh fading channel"""
    ber_values = []

    for snr_db in snr_db_range:
        snr_linear = 10**(snr_db / 10)
        noise_var = 1 / snr_linear

        # Generate bits and BPSK modulate
        bits = np.random.randint(0, 2, num_symbols * 2)
        symbols = 2 * bits - 1  # BPSK: 0→-1, 1→+1
        s = symbols.reshape(-1, 2)  # [s1, s2] pairs

        # Rayleigh fading channels (quasi-static)
        h1 = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols)) / np.sqrt(2)
        h2 = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols)) / np.sqrt(2)

        # Complex Gaussian noise
        noise = np.sqrt(noise_var/2) * (np.random.randn(num_symbols, 2) +
                                        1j*np.random.randn(num_symbols, 2))

        # Alamouti transmission (2 time slots)
        r1 = h1*s[:, 0] + h2*s[:, 1] + noise[:, 0]
        r2 = h1*(-np.conj(s[:, 1])) + h2*np.conj(s[:, 0]) + noise[:, 1]

        # Alamouti combining
        y1 = np.conj(h1)*r1 + h2*np.conj(r2)
        y2 = np.conj(h2)*r1 - h1*np.conj(r2)

        # Equalization
        channel_gain = np.abs(h1)**2 + np.abs(h2)**2
        s1_est = y1 / channel_gain
        s2_est = y2 / channel_gain

        # BPSK decision
        detected_symbols = np.stack([np.sign(np.real(s1_est)),
                                     np.sign(np.real(s2_est))], axis=1).flatten()
        detected_bits = ((detected_symbols + 1) / 2).astype(int)

        # Calculate BER
        ber = np.sum(detected_bits != bits) / len(bits)
        ber_values.append(ber)

    return np.array(ber_values)

# Configuration
sample_sizes = [10000, 100000, 1000000]
snr_range = np.arange(0, 11, 1)

plt.figure(figsize=(10, 5))

for num_samples in sample_sizes:
    print(f"Simulating {num_samples} symbols...")
    ber = simulate_alamouti_ber(num_samples, snr_range)
    plt.semilogy(snr_range, ber, '-o', label=f"Samples = {num_samples}")

plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("BER Performance of 2x1 MISO with Alamouti STBC (Rayleigh Fading)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
