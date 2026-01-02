import numpy as np
from core.stft import stft

# --------------------------------------------------
# Log-Spectral Distance (LSD)
# --------------------------------------------------
def log_spectral_distance(clean, estimate, frame_len=512, hop_len=128):
    L = min(len(clean), len(estimate))
    clean = clean[:L]
    estimate = estimate[:L]

    X = stft(clean, n_fft=frame_len, hop=hop_len)
    Y = stft(estimate, n_fft=frame_len, hop=hop_len)

    mag_X = np.abs(X) + 1e-12
    mag_Y = np.abs(Y) + 1e-12

    lsd = np.sqrt(
        np.mean((20 * np.log10(mag_X) - 20 * np.log10(mag_Y))**2)
    )

    return lsd
