import numpy as np
from scipy.signal import stft, istft

def wiener_dd(
    noisy,
    fs,
    noise_type="stationary",  
    win_len_sec=0.02,
    hop_len_sec=0.01,
    nfft=512,
    alpha=0.98,               # A-priori SNR smoothing factor
    init_sec=0.2,             # Duration (seconds) assumed to be noise-only
    gain_floor=0.01,          # Maximum attenuation (-40 dB)
    oversubtraction=1.0,      # Noise overestimation factor to increase suppression
    eps=1e-10,
):
    """
    Optimized Adaptive Wiener Filter.
    Supports the 'noise_type' argument for compatibility,
    but internally relies on adaptive noise tracking and VAD logic.
    """

    x = np.asarray(noisy, dtype=np.float64)
    x -= np.mean(x)
    N = len(x)

    # STFT parameters
    nperseg = int(win_len_sec * fs)
    hop = int(hop_len_sec * fs)
    noverlap = nperseg - hop

    # 1. STFT
    _, _, Y = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=True
    )
    
    K, M = Y.shape
    P_noisy = np.abs(Y) ** 2

    # 2. Noise initialization (assume initial frames are noise-only)
    init_frames = max(1, int(init_sec * fs / hop))
    init_frames = min(init_frames, M)
    
    # Initial noise PSD estimate
    # Using a quantile instead of the mean for robustness
    Phi_noise = np.quantile(P_noisy[:, :init_frames], 0.2, axis=1)
    Phi_noise = np.maximum(Phi_noise, eps)

    # Output STFT matrix
    S_hat = np.zeros_like(Y, dtype=np.complex128)
    
    # Previous gain magnitude squared (for decision-directed approach)
    prev_G_mag_sq = np.ones(K)

    # Noise update smoothing factor
    # Stationary noise changes slowly, non-stationary noise adapts faster
    alpha_noise_base = 0.99 if noise_type == "stationary" else 0.95

    # --- MAIN LOOP ---
    for m in range(M):
        Y_frame = Y[:, m]
        P_frame = P_noisy[:, m]

        # A) A-posteriori SNR
        # Oversubtraction intentionally overestimates noise to increase suppression
        gamma = P_frame / (oversubtraction * Phi_noise + eps)
        
        # B) A-priori SNR (Decision-Directed approach, Ephraim & Malah)

        xi = alpha * (
                prev_G_mag_sq *
                (np.abs(Y[:, m-1])**2 if m > 0 else P_frame) /
                (Phi_noise + eps)
             ) + (1 - alpha) * np.maximum(gamma - 1, 0)
             
        # C) Wiener gain computation
        G = xi / (1 + xi)
        G = np.maximum(G, gain_floor)
        
        prev_G_mag_sq = G**2

        # D) Apply gain to noisy STFT coefficients
        S_hat[:, m] = G * Y_frame

        # E) Adaptive noise update (Noise tracking with simple VAD)
        # If the gain is low, speech is likely absent → update noise estimate
        # If the gain is high, speech is likely present → freeze noise estimate
        
        # Speech absence probability (1 = no speech, 0 = speech present)
        speech_absent_prob = 1.0 - np.clip(G * 2.0, 0, 1)
        
        # Time-varying noise smoothing factor
        alpha_t = alpha_noise_base + (1 - alpha_noise_base) * (1 - speech_absent_prob)
        
        # Update noise PSD only in speech-absent regions
        Phi_noise = alpha_t * Phi_noise + (1 - alpha_t) * P_frame
        Phi_noise = np.maximum(Phi_noise, eps)

    # 3. ISTFT
    _, enh = istft(
        S_hat,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )
    
    return np.nan_to_num(enh[:N])
