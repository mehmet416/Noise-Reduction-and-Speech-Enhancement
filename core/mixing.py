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

def mix_sources_multichannel(sources, A):
    """
    Mixes multiple mono sources into multi-channel mixtures.
    sources: List of 1D arrays
    A: Mixing matrix (Num_Sensors x Num_Sources)
    """

    # Truncate all sources to the same minimum length before stacking
    lengths = [len(s) for s in sources]
    L = min(lengths)
    S_list = [np.asarray(s, dtype=np.float64)[:L] for s in sources]

    # Stack sources (Num_Sources x Length)
    S = np.vstack(S_list)
    
    # Force C-contiguous memory layout
    S = np.ascontiguousarray(S)
    A = np.ascontiguousarray(A.astype(np.float64))

    # Validate mixing matrix shape
    if A.shape[1] != S.shape[0]:
        raise ValueError(f"Mixing matrix A must have shape (Num_Sensors, Num_Sources); got A.shape={A.shape}, Num_Sources={S.shape[0]}")

    # Mix
    X = np.dot(A, S)

    # Normalize (prevent clipping)
    max_val = np.max(np.abs(X))
    if max_val > 0:
        X /= (max_val + 1e-8)

    return X


def error_signal(clean, enhanced):
    """
    Compute the error signal between clean and enhanced signals.
    """
    clean, enhanced = match_length(clean, enhanced)
    error = clean - enhanced
    return error.astype(np.float32)