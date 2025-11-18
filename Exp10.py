import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Modulation
def modulate(bits, mod_type):
    if mod_type == 'BPSK':
        return 2*bits - 1
    elif mod_type == 'QPSK':
        return ((2*bits[0::2]-1) + 1j*(2*bits[1::2]-1)) / np.sqrt(2)

# Demodulation
def demodulate(symbols, mod_type):
    if mod_type == 'BPSK':
        return (np.real(symbols) > 0).astype(int)
    elif mod_type == 'QPSK':
        bits = np.zeros(2*len(symbols), int)
        bits[0::2] = np.real(symbols) > 0
        bits[1::2] = np.imag(symbols) > 0
        return bits

# Maximum Likelihood Detector
def ml_detector(received, channel, constellation):
    Nt = channel.shape[1]
    # All possible symbol combinations
    all_combos = np.array(list(product(constellation, repeat=Nt)))
    # Expected received signals
    expected = channel @ all_combos.T
    # Find minimum distance
    distances = np.sum(np.abs(received[:, None] - expected)**2, axis=0)
    return all_combos[np.argmin(distances)]

# Simulation
def simulate_ber(Nt, Nr, mod_type, snr_db, num_symbols=200):
    bits_per_symbol = {'BPSK': 1, 'QPSK': 2}[mod_type]

    # Generate constellation points
    constellation_bits = np.array(list(product([0, 1], repeat=bits_per_symbol))).flatten()
    constellation = np.unique(modulate(constellation_bits, mod_type))

    ber_list = []
    for snr in snr_db:
        snr_linear = 10**(snr/10)
        noise_var = Nt / snr_linear
        errors, total = 0, 0

        for _ in range(num_symbols):
            # Generate bits and modulate
            bits = np.random.randint(0, 2, Nt * bits_per_symbol)
            symbols = modulate(bits, mod_type)

            # Rayleigh channel + noise
            H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)
            noise = (np.random.randn(Nr) + 1j*np.random.randn(Nr)) * np.sqrt(noise_var/2)
            received = H @ symbols + noise

            # Detect and demodulate
            detected = ml_detector(received, H, constellation)
            detected_bits = demodulate(detected, mod_type)

            # Count errors
            errors += np.sum(bits != detected_bits)
            total += len(bits)

        ber_list.append(errors / total)
    return np.array(ber_list)

# Run simulation
snr_range = np.arange(0, 21, 2)
configs = [('BPSK', 2, 2), ('BPSK', 4, 4), ('QPSK', 2, 2), ('QPSK', 4, 4)]

plt.figure(figsize=(10, 6))
for mod_type, Nt, Nr in configs:
    print(f"Running {mod_type} {Nt}x{Nr}...")
    ber = simulate_ber(Nt, Nr, mod_type, snr_range)
    plt.semilogy(snr_range, ber, marker='o', label=f"{mod_type} {Nt}x{Nr}")

plt.title("MIMO BER Performance with ML Detection")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

