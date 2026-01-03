import numpy as np
from scipy.signal import stft, istft


def wiener_dd(
    noisy,
    fs,
    win_len_sec=0.02,
    hop_len_sec=0.01,
    nfft=512,
    alpha=0.98,            # decision-directed smoothing
    init_sec=0.5,
    init_quantile=0.2,
    psd_smooth=0.8,
    beta_noise_fast=0.90,  # fast adaptation (noise-only)
    beta_noise_slow=0.995, # slow adaptation (speech present)
    gamma_th=3.0,
    gain_floor=0.05,
    eps=1e-10,
):
    """
    Adaptive STFT-domain Wiener filter.
    - Noise PSD tracked online
    - Soft speech presence probability
    - Suitable for non-stationary noise
    """

    x = np.asarray(noisy, dtype=np.float64)
    x -= np.mean(x)
    N = len(x)

    nperseg = int(win_len_sec * fs)
    hop = int(hop_len_sec * fs)
    noverlap = nperseg - hop

    _, _, Y = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
    )

    K, M = Y.shape
    P = np.abs(Y) ** 2

    # --- Noise PSD initialization (robust) ---
    init_frames = min(M, int(init_sec * fs / hop))
    Phi_v = np.quantile(P[:, :init_frames], init_quantile, axis=1)
    Phi_v = np.maximum(Phi_v, eps)

    S_hat = np.zeros_like(Y, dtype=np.complex128)
    prev_S_hat = np.zeros(K, dtype=np.complex128)
    P_s = Phi_v.copy()

    for m in range(M):
        Ykm = Y[:, m]
        Ypow = P[:, m]

        # smoothed periodogram
        P_s = psd_smooth * P_s + (1 - psd_smooth) * Ypow

        # a-posteriori SNR
        gamma = Ypow / (Phi_v + eps)

        # decision-directed a-priori SNR
        if m == 0:
            xi = np.maximum(gamma - 1, 0)
        else:
            xi = alpha * (np.abs(prev_S_hat)**2 / (Phi_v + eps)) \
                 + (1 - alpha) * np.maximum(gamma - 1, 0)

        # Wiener gain
        G = xi / (1 + xi)
        G = np.maximum(G, gain_floor)

        Sh = G * Ykm
        S_hat[:, m] = Sh
        prev_S_hat = Sh

        # ---- Adaptive noise tracking ----
        speech_prob = np.clip((gamma - 1) / gamma_th, 0, 1)
        beta = beta_noise_fast * (1 - speech_prob) + beta_noise_slow * speech_prob

        Phi_v = beta * Phi_v + (1 - beta) * P_s
        Phi_v = np.maximum(Phi_v, eps)

    _, enh = istft(
        S_hat,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    enh = enh[:N]
    return np.nan_to_num(enh)
