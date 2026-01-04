import numpy as np
from scipy.signal import convolve2d
from core.stft import stft, istft

def spectral_subtraction(
    x,
    fs,
    frame_len=512,
    hop_len=128,
    alpha=2.0,       # Oversubtraction
    beta=0.01,       # Spectral Floor
    smoothing_width=3 # Frequency smoothing window (bins)
):
    # Ensure float32 [-1, 1] range
    x = x.astype(np.float32)
    if np.max(np.abs(x)) > 1.1:
        x /= 32768.0  # Normalize if 16-bit PCM

    # 1. STFT
    X = stft(x, n_fft=frame_len, hop=hop_len)
    mag = np.abs(X)
    phase = np.angle(X)
    
    # 2. Noise Estimation (using initial silence)
    n_init = 6
    noise_est = np.mean(mag[:, :n_init], axis=1, keepdims=True)
    
    # 3. Frequency Smoothing (Optional but recommended)
    if smoothing_width > 1:
        kernel = np.ones((smoothing_width, 1)) / smoothing_width
        noise_est = convolve2d(noise_est, kernel, mode='same', boundary='symm')

    # 4. Subtraction with Flooring
    subtracted_mag = mag - (alpha * noise_est)
    
    # Apply Spectral Floor
    floor_val = beta * noise_est
    clean_mag = np.maximum(subtracted_mag, floor_val)

    # 5. Reconstruction
    X_clean = clean_mag * np.exp(1j * phase)
    x_hat = istft(X_clean, hop=hop_len)
    
    # 6. Length Fix (ISTFT often adds padding)
    if len(x_hat) > len(x):
        x_hat = x_hat[:len(x)]
    elif len(x_hat) < len(x):
        x_hat = np.pad(x_hat, (0, len(x) - len(x_hat)))

    # 7. Energy Recovery / Gain Normalization
    scale_factor = np.max(np.abs(x)) / (np.max(np.abs(x_hat)) + 1e-8)
    x_hat = x_hat * scale_factor

    return x_hat