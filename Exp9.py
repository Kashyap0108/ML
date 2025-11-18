import numpy as np
import matplotlib.pyplot as plt

# Calculate channel capacity
def calculate_capacity(H, snr_linear, nt, nr):
    if nt == 1 and nr == 1:  # SISO
        return np.log2(1 + snr_linear * np.abs(H)**2)
    elif nt == 1 or nr == 1:  # SIMO or MISO
        return np.log2(1 + snr_linear * np.linalg.norm(H)**2)
    else:  # MIMO
        H_hermitian = H.conj().T
        I = np.eye(nr)
        det = np.linalg.det(I + (snr_linear / nt) * H @ H_hermitian)
        return np.real(np.log2(det))

# Simulation function
def simulate_capacity(nt, nr, snr_db_range, num_samples=2000):
    capacities = []

    for snr_db in snr_db_range:
        snr_linear = 10**(snr_db / 10)
        capacity_samples = []

        for _ in range(num_samples):
            # Generate random Rayleigh fading channel
            H = (np.random.randn(nr, nt) + 1j*np.random.randn(nr, nt)) / np.sqrt(2)
            capacity = calculate_capacity(H, snr_linear, nt, nr)
            capacity_samples.append(capacity)

        # Average capacity (ergodic capacity)
        capacities.append(np.mean(capacity_samples))

    return np.array(capacities)

# Configuration
configs = {
    'SISO (1x1)': (1, 1),
    'SIMO (1x2)': (1, 2),
    'MISO (2x1)': (2, 1),
    'MIMO (2x2)': (2, 2),
}

snr_range = np.arange(-10, 21, 1)

# Run simulations and plot
plt.figure(figsize=(10, 7))

for name, (nt, nr) in configs.items():
    print(f"Calculating {name}...")
    capacity = simulate_capacity(nt, nr, snr_range)
    plt.plot(snr_range, capacity, marker='o', markersize=4, linestyle='--', label=name)

plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Ergodic Channel Capacity (bits/s/Hz)', fontsize=12)
plt.title('Ergodic Channel Capacity vs. SNR', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, which='both', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()
