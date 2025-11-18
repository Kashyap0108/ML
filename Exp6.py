import numpy as np, matplotlib.pyplot as plt
np.random.seed(0)
SNRdB = np.arange(0,21,2); SNR = 10**(SNRdB/10)
N = 100000                      # adjust for accuracy/speed
Nr_list = [1,2,3,4,5]
methods = ["MRC","EGC","SC"]

bits = np.random.randint(0,2,size=N); s = 2*bits-1

for method in methods:
    plt.figure(figsize=(6,4))
    for Nr in Nr_list:
        bers = []
        for snr in SNR:
            sigma = np.sqrt(1/(2*snr))
            h = (np.random.randn(N,Nr)+1j*np.random.randn(N,Nr))/np.sqrt(2)
            n = (np.random.randn(N,Nr)+1j*np.random.randn(N,Nr))*sigma
            r = h * s[:,None] + n

            if method=="MRC":
                y = np.sum(np.conj(h)*r,axis=1)
            elif method=="EGC":
                w = np.conj(h)/(np.abs(h)+1e-12)
                y = np.sum(w*r,axis=1)
            else:  # SC
                idx = np.argmax(np.abs(h)**2,axis=1)
                h_sc = h[np.arange(N),idx]; r_sc = r[np.arange(N),idx]
                y = np.conj(h_sc)*r_sc

            dec = (np.real(y)>0).astype(int)
            bers.append(np.mean(dec!=bits))
        plt.semilogy(SNRdB, bers, marker='o', label=f'Nr={Nr}')
    plt.title(f'{method} - SIMO BPSK Rayleigh'); plt.xlabel('SNR (dB)'); plt.ylabel('BER')
    plt.grid(True,which='both'); plt.legend(); plt.tight_layout()
plt.show()
