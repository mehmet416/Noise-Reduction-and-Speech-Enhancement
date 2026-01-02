import numpy as np
import librosa
import soundfile as sf
from IPython.display import Audio

# --------------------------------------------------
# Load audio (mono, float32)
# --------------------------------------------------
def load_audio(path, sr=16000):
    x, fs = librosa.load(path, sr=sr, mono=True)

    x = x.astype(np.float32)

    # Remove NaN / Inf if any
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard normalization
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val

    return x, fs


# --------------------------------------------------
# Save audio
# --------------------------------------------------
def save_audio(path, x, fs):
    sf.write(path, x, fs)

# --------------------------------------------------
# Play audio (for debugging)
# --------------------------------------------------
def play_audio(x, fs=16000):
    x = np.nan_to_num(x)
    return Audio(x, rate=fs)


# --------------------------------------------------
# Signal utilities
# --------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-12)

def match_length(x, y):
    L = min(len(x), len(y))
    return x[:L], y[:L]
