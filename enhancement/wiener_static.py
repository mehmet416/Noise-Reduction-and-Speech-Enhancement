import numpy as np
from core.stft import stft, istft

def wiener_filter_static(
    noisy,
    fs,
    win_len_sec=0.020,
    hop_len_sec=0.010,
    nfft=512,
    alpha=0.98,           # decision-directed smoothing
    init_sec=0.8,         # initial noise estimation window
    init_quantile=0.2,    # quantile for initial noise PSD estimation
    psd_smooth=0.8,
    gain_floor=0.05,
    eps=1e-10,
):
    """
    Static (stationary) STFT-domain Wiener filter with Decision-Directed a priori SNR.
    - No adaptive noise tracking: Phi_v is estimated once and kept fixed.
    - Does NOT assume initial silence. Uses low-quantile of early frames to init noise PSD.
    - Robust STFT/ISTFT settings to avoid NOLA/frame mismatch issues.
    - Output length == input length.
    """

    x = np.asarray(noisy, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    x = x - np.mean(x)
    N = len(x)
    if N == 0:
        return x

    nperseg = int(round(win_len_sec * fs))
    hop = int(round(hop_len_sec * fs))
    nperseg = max(32, min(nperseg, N))
    hop = max(1, min(hop, nperseg - 1))

    Y = stft(
        x,
        n_fft=nfft,
        hop=hop,
        window="hann"
    )
    K, M = Y.shape
    P = np.abs(Y) ** 2
    # --- Noise PSD init (fixed) via quantile over early frames ---
    init_frames = int(max(5, min(M, round((init_sec * fs) / hop))))
    P_s = np.zeros(K, dtype=np.float64)
    P_hist = np.zeros((K, init_frames), dtype=np.float64)

    for m in range(init_frames):
        P_s = psd_smooth * P_s + (1.0 - psd_smooth) * P[:, m]
        P_hist[:, m] = P_s

    Phi_v = np.quantile(P_hist, init_quantile, axis=1)
    Phi_v = np.maximum(Phi_v, eps)

    # --- Decision-Directed Wiener (Phi_v fixed) ---
    S_hat = np.zeros_like(Y, dtype=np.complex128)
    prev_S_hat = np.zeros(K, dtype=np.complex128)

    for m in range(M):
        Ykm = Y[:, m]
        Ypow = P[:, m]

        gamma = Ypow / Phi_v

        if m == 0:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            xi = alpha * (np.abs(prev_S_hat) ** 2 / Phi_v) + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)

        xi = np.maximum(xi, 0.0)

        G = xi / (1.0 + xi)
        if gain_floor is not None and gain_floor > 0:
            G = np.maximum(G, gain_floor)

        Sh = G * Ykm
        S_hat[:, m] = Sh
        prev_S_hat = Sh

    enh = istft(
        S_hat,
        hop=hop,
        window="hann"
    )

    enh = np.asarray(enh, dtype=np.float64)

    # Output length fix
    if len(enh) < N:
        enh = np.pad(enh, (0, N - len(enh)))
    else:
        enh = enh[:N]

    return np.nan_to_num(enh, nan=0.0, posinf=0.0, neginf=0.0)
