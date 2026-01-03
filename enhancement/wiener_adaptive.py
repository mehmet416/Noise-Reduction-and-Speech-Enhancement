# enhancement/wiener_static.py
import numpy as np
from scipy.signal import stft, istft


def wiener_dd(
    noisy,
    fs,
    noise_type="stationary",
    win_len_sec=0.020,
    hop_len_sec=0.010,
    nfft=512,
    alpha=0.98,
    init_sec=0.8,          # sessizlik yoksa: ilk init_sec saniyeden quantile ile init
    init_quantile=0.2,     # 0.1~0.3 arası deneyebilirsin
    psd_smooth=0.8,        # periodogram smoothing
    gamma_th=2.0,          # speech presence gating eşiği (≈3 dB)
    gain_floor=0.05,       # çok agresif çökmesin
    eps=1e-10,
):
    """
    STFT-domain Wiener filter with Decision-Directed (DD) a priori SNR.
    No reference noise required. No initial silence required.

    Robustness:
    - Noise PSD init via low-quantile of early frames (speech may exist).
    - Noise tracking uses speech-presence gating (gamma_th) to avoid learning speech as noise.
    - Uses boundary='zeros', padded=True to keep STFT/ISTFT consistent (avoid NOLA issues).
    - Output is trimmed/padded to exactly len(noisy).
    """

    x = np.asarray(noisy, dtype=np.float64)

    # If stereo -> mono
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # Remove DC
    x = x - np.mean(x)
    N = len(x)
    if N == 0:
        return x

    nperseg = int(round(win_len_sec * fs))
    hop = int(round(hop_len_sec * fs))
    nperseg = max(32, min(nperseg, N))
    hop = max(1, min(hop, nperseg - 1))
    noverlap = nperseg - hop

    # --- STFT ---
    f, t, Y = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True,
        return_onesided=True,
    )

    K, M = Y.shape
    P = np.abs(Y) ** 2

    # --- noise tracking speed ---
    nt = (noise_type or "").lower()
    if "stationary" in nt:
        beta_noise = 0.995
    elif "slow" in nt:
        beta_noise = 0.985
    elif "fast" in nt:
        beta_noise = 0.97
    else:
        beta_noise = 0.985

    # --- init noise PSD via quantile over early frames ---
    init_frames = int(max(5, min(M, round((init_sec * fs) / hop))))
    P_s = np.zeros(K, dtype=np.float64)
    P_hist = np.zeros((K, init_frames), dtype=np.float64)

    for m in range(init_frames):
        P_s = psd_smooth * P_s + (1.0 - psd_smooth) * P[:, m]
        P_hist[:, m] = P_s

    Phi_v = np.quantile(P_hist, init_quantile, axis=1)
    Phi_v = np.maximum(Phi_v, eps)

    # --- DD loop ---
    S_hat = np.zeros_like(Y, dtype=np.complex128)
    prev_S_hat = np.zeros(K, dtype=np.complex128)

    for m in range(M):
        Ykm = Y[:, m]
        Ypow = P[:, m]

        # smooth periodogram
        P_s = psd_smooth * P_s + (1.0 - psd_smooth) * Ypow

        gamma = Ypow / Phi_v  # a-posteriori SNR

        if m == 0:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            xi = alpha * (np.abs(prev_S_hat) ** 2 / Phi_v) + (1.0 - alpha) * np.maximum(gamma - 1.0, 0.0)

        # Wiener gain (with floor)
        G = xi / (1.0 + xi)
        if gain_floor is not None and gain_floor > 0:
            G = np.maximum(G, gain_floor)

        Sh = G * Ykm
        S_hat[:, m] = Sh
        prev_S_hat = Sh

        # --- Noise PSD update (speech presence gating) ---
        # speech present if gamma is high => don't update noise there
        noise_mask = gamma < gamma_th
        if np.any(noise_mask):
            Phi_v[noise_mask] = beta_noise * Phi_v[noise_mask] + (1.0 - beta_noise) * P_s[noise_mask]

        # Safety clamps
        Phi_v = np.minimum(Phi_v, np.maximum(P_s, eps))  # PSD shouldn't exceed observed smoothed power too much
        Phi_v = np.maximum(Phi_v, eps)

    # --- iSTFT ---
    _, enh = istft(
        S_hat,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        input_onesided=True,
        boundary=True,
    )

    enh = np.asarray(enh, dtype=np.float64)

    # Make output exactly same length as input
    if len(enh) < N:
        enh = np.pad(enh, (0, N - len(enh)))
    else:
        enh = enh[:N]

    # Final safety
    enh = np.nan_to_num(enh, nan=0.0, posinf=0.0, neginf=0.0)
    return enh
