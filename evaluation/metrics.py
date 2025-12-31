import numpy as np

# --------------------------------------------------
# Mean Squared Error
# --------------------------------------------------
def mse(clean, estimate):
    L = min(len(clean), len(estimate))
    return np.mean((clean[:L] - estimate[:L])**2)

# --------------------------------------------------
# Signal-to-Noise Ratio
# --------------------------------------------------
def snr(clean, estimate):
    L = min(len(clean), len(estimate))
    noise = clean[:L] - estimate[:L]
    return 10 * np.log10(
        np.sum(clean[:L]**2) / (np.sum(noise**2) + 1e-12)
    )

# --------------------------------------------------
# SNR Improvement
# --------------------------------------------------
def snr_improvement(clean, noisy, enhanced):
    return snr(clean, enhanced) - snr(clean, noisy)

# --------------------------------------------------
# Segmental SNR (speech standard)
# --------------------------------------------------
def segmental_snr(clean, estimate, frame_len=256):
    eps = 1e-12
    L = min(len(clean), len(estimate))
    clean = clean[:L]
    estimate = estimate[:L]

    num_frames = L // frame_len
    snrs = []

    for i in range(num_frames):
        c = clean[i*frame_len:(i+1)*frame_len]
        e = estimate[i*frame_len:(i+1)*frame_len]
        snrs.append(
            10 * np.log10(
                np.sum(c**2) / (np.sum((c - e)**2) + eps)
            )
        )

    return np.mean(snrs)
