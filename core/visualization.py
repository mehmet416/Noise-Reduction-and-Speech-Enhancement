import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# --------------------------------------------------
# Time-domain plot (compact)
# --------------------------------------------------
def plot_waveforms(signals, labels, fs, title):
    t = np.arange(len(signals[0])) / fs
    plt.figure(figsize=(8, 3))  

    for sig, lab in zip(signals, labels):
        plt.plot(t, sig, label=lab, alpha=0.8, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title, fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Spectrogram plot (compact)
# --------------------------------------------------
def plot_spectrogram(x, fs, title, n_fft=512, hop=128):
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    X_db = librosa.amplitude_to_db(np.abs(X), ref=np.max)

    plt.figure(figsize=(8, 3))  
    librosa.display.specshow(
        X_db,
        sr=fs,
        hop_length=hop,
        x_axis="time",
        y_axis="hz"
    )
    plt.colorbar(format="%+2.0f dB", pad=0.01)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.show()
