import numpy as np
import librosa
import soundfile as sf
from IPython.display import Audio

# --------------------------------------------------
# Load audio (mono, float32)
# --------------------------------------------------
def load_audio(path, sr=16000):
    x, fs = librosa.load(path, sr=sr, mono=True)
    return x.astype(np.float32), fs

# --------------------------------------------------
# Save audio
# --------------------------------------------------
def save_audio(path, x, fs):
    sf.write(path, x, fs)

# --------------------------------------------------
# Play audio (for debugging)
# --------------------------------------------------
def play_audio(x, fs):
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
