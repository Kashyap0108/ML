import numpy as np
import matplotlib.pyplot as plt

# Modulation (QPSK)
def modulate(bits):
    bits = bits.reshape((-1, 2))
    symbols = (1 - 2*bits[:, 0]) + 1j*(1 - 2*bits[:, 1])
    return symbols / np.sqrt(2)

# Demodulation (QPSK)
def demodulate(symbols):
    bit0 = (symbols.real < 0).astype(int)
    bit1 = (symbols.imag < 0).astype(int)
    return np.stack((bit0, bit1), axis=1).reshape(-1)

# Simulation function
def simulate_linear_detectors(Nt, Nr, snr_db, num_symbols=10000):
    bits_per_symbol = 2  # QPSK
    ber_zf, ber_mmse = [], []

    for snr in snr_db:
        # Calculate noise variance
        snr_linear = 10**(snr/10)
        N0 = 1 / (snr_linear * bits_per_symbol)
        noise_std = np.sqrt(N0 / 2)

        errors_zf, errors_mmse, total_bits = 0, 0, 0

        for _ in range(num_symbols):
            # Generate bits and modulate
            bits = np.random.randint(0, 2, Nt * bits_per_symbol)
            symbols = modulate(bits).reshape(Nt, 1)

            # Rayleigh fading channel
            H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)

            # Add noise
            noise = noise_std * (np.random.randn(Nr, 1) + 1j*np.random.randn(Nr, 1))
            received = H @ symbols + noise

            # Zero Forcing (ZF) Detection
            W_zf = np.linalg.pinv(H)
            detected_zf = W_zf @ received
            bits_zf = demodulate(detected_zf.flatten())

            # MMSE Detection
            H_hermitian = np.conjugate(H.T)
            W_mmse = np.linalg.inv(H_hermitian @ H + N0*np.eye(Nt)) @ H_hermitian
            detected_mmse = W_mmse @ received
            bits_mmse = demodulate(detected_mmse.flatten())

            # Count errors
            errors_zf += np.sum(bits != bits_zf)
            errors_mmse += np.sum(bits != bits_mmse)
            total_bits += len(bits)

        ber_zf.append(errors_zf / total_bits)
        ber_mmse.append(errors_mmse / total_bits)

    return np.array(ber_zf), np.array(ber_mmse)

# Configuration
configs = [(2, 2), (2, 3), (4, 4), (4, 3), (4, 6)]
snr_range = np.arange(0, 21, 2)

plt.figure(figsize=(10, 7))

# Run simulations
for Nt, Nr in configs:
    print(f"Simulating {Nt}x{Nr}...")
    ber_zf, ber_mmse = simulate_linear_detectors(Nt, Nr, snr_range)

    plt.semilogy(snr_range, ber_zf, 'o--', label=f'ZF {Nt}x{Nr}')
    plt.semilogy(snr_range, ber_mmse, 's-', label=f'MMSE {Nt}x{Nr}')

plt.title('BER of ZF & MMSE Linear Detectors in MIMO (QPSK)')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend()
plt.tight_layout()
plt.show()
