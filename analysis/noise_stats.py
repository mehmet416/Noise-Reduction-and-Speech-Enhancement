import numpy as np
from core.stft import stft

# --------------------------------------------------
# Spectral flatness (noise vs tone)
# --------------------------------------------------
def spectral_flatness(X):
    eps = 1e-12
    geo_mean = np.exp(np.mean(np.log(np.abs(X) + eps), axis=0))
    arith_mean = np.mean(np.abs(X) + eps, axis=0)
    return geo_mean / arith_mean

# --------------------------------------------------
# Spectral centroid
# --------------------------------------------------
def spectral_centroid(X, fs):
    freqs = np.linspace(0, fs / 2, X.shape[0])
    magnitude = np.abs(X)
    return np.sum(freqs[:, None] * magnitude, axis=0) / (
        np.sum(magnitude, axis=0) + 1e-12
    )

# --------------------------------------------------
# Short-time spectral entropy
# --------------------------------------------------
def spectral_entropy(X):
    power = np.abs(X)**2
    power /= np.sum(power, axis=0, keepdims=True) + 1e-12
    entropy = -np.sum(power * np.log(power + 1e-12), axis=0)
    return entropy

# --------------------------------------------------
# Energy variance over time
# --------------------------------------------------
def energy_variance(energies_db):
    return np.var(energies_db)

# --------------------------------------------------
# Spectral flatness variance
# --------------------------------------------------
def flatness_variance(flatness):
    return np.var(flatness)

# --------------------------------------------------
# Nonstationarity index (simple & effective)
# --------------------------------------------------
def nonstationarity_index(energies_db, flatness):
    return energy_variance(energies_db) + flatness_variance(flatness)
