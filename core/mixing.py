import numpy as np
from core.audio_io import match_length

# --------------------------------------------------
# Mix clean speech with noise at desired SNR
# --------------------------------------------------
def mix_with_snr(clean, noise, snr_db):
    clean, noise = match_length(clean, noise)

    Ps = np.mean(clean**2)
    Pn = np.mean(noise**2) + 1e-12

    alpha = np.sqrt(Ps / (Pn * 10**(snr_db / 10)))
    noisy = clean + alpha * noise

    return noisy.astype(np.float32), (alpha * noise).astype(np.float32)
