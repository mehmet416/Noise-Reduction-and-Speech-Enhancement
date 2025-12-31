import librosa
import numpy as np

# --------------------------------------------------
# STFT wrapper
# --------------------------------------------------
def stft(x, n_fft=512, hop=128, window="hann"):
    return librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=True
    )

# --------------------------------------------------
# ISTFT wrapper
# --------------------------------------------------
def istft(X, hop=128, window="hann"):
    return librosa.istft(
        X,
        hop_length=hop,
        window=window,
        center=True
    )
