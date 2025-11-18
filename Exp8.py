import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

Nt_values = [1, 2, 4]
Nr_values = [1, 2, 4]
SNR_dB_values = [0, 10, 15, 20, 25]
bandwidth = 1

capacity_results = {}
table_data = []

for Nt in Nt_values:
    for Nr in Nr_values:
        capacity_results[f'{Nt}x{Nr}'] = []
        for SNR_dB in SNR_dB_values:

            SNR_linear = 10**(SNR_dB / 10)
            N0 = 1
            P_total_over_N0 = SNR_linear
            H = (np.random.randn(Nr, Nt) + 1j * np.random.randn(Nr, Nt)) / np.sqrt(2)
            HH_conj_transpose = np.conj(H).T
            term = (P_total_over_N0 / Nt) * np.dot(H, HH_conj_transpose)
            Inr = np.eye(Nr)
            matrix_for_determinant = Inr + term
            det_matrix = np.linalg.det(matrix_for_determinant)
            C = bandwidth * math.log2(np.maximum(np.real(det_matrix), 1e-12))
            table_data.append([SNR_dB, Nt, Nr, f"{C:.4f}"])

df_capacity = pd.DataFrame(table_data, columns=["SNR (dB)", "Nt", "Nr", "Capacity (bits/s/Hz)"])
display(df_capacity)