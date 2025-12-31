import numpy as np
from core.framing import frame_signal
from analysis.vad import vad_energy
from analysis.noise_stats import spectral_flatness, nonstationarity_index
from core.stft import stft

# --------------------------------------------------
# Noise type classification
# --------------------------------------------------
def classify_noise(x, fs, frame_len=512, hop_len=128):
    frames = frame_signal(x, frame_len, hop_len)

    if len(frames) < 5:
        return "unknown"

    energies = np.sum(frames**2, axis=1) + 1e-12
    energies_db = 10 * np.log10(energies)

    # Always select lowest-energy frames
    num_noise_frames = max(5, int(0.2 * len(energies_db)))
    idx = np.argsort(energies_db)[:num_noise_frames]
    noise_frames = frames[idx]

    flatness_vals = []
    for frame in noise_frames:
        X = stft(frame, n_fft=frame_len, hop=frame_len)
        flatness_vals.append(np.mean(spectral_flatness(X)))

    flatness_vals = np.array(flatness_vals)

    # Noise stationarity index (frequency-domain)
    nsi = np.var(flatness_vals)

    # Thresholds tuned for speech frame sizes
    if nsi < 1e-3:
        return "stationary"
    elif nsi < 1e-2:
        return "slowly_varying"
    else:
        return "highly_nonstationary"


def classify_noise_DEBUG(x, fs, frame_len=512, hop_len=128):
    frames = frame_signal(x, frame_len, hop_len)
    print("Total frames:", len(frames))

    vad_mask, energies_db = vad_energy(frames, percentile=70)

    print("Energy dB range:", energies_db.min(), energies_db.max())

    noise_idx = energies_db < np.percentile(energies_db, 30)
    print("Noise frame count:", np.sum(noise_idx))

    if np.sum(noise_idx) < 5:
        print("⚠️ Not enough noise frames")
        return "unknown"

    noise_frames = frames[noise_idx]

    flatness_vals = []
    for frame in noise_frames:
        X = stft(frame, n_fft=frame_len, hop=frame_len)
        flatness_vals.append(np.mean(spectral_flatness(X)))

    flatness_vals = np.array(flatness_vals)

    nsi = np.var(energies_db) + np.var(flatness_vals)
    print("Nonstationarity index:", nsi)

    if nsi < 2:
        return "stationary"
    elif nsi < 10:
        return "slowly_varying"
    else:
        return "highly_nonstationary"
def classify_noise_DEBUG(x, fs, frame_len=512, hop_len=128):
    from core.framing import frame_signal
    from analysis.vad import vad_energy
    from analysis.noise_stats import spectral_flatness
    from core.stft import stft
    import numpy as np

    frames = frame_signal(x, frame_len, hop_len)
    print("Total frames:", len(frames))

    if len(frames) == 0:
        print("❌ No frames created")
        return "unknown"

    vad_mask, energies_db = vad_energy(frames, percentile=70)
    print("Energy dB range:", energies_db.min(), energies_db.max())

    noise_idx = energies_db < np.percentile(energies_db, 30)
    print("Noise frame count:", np.sum(noise_idx))

    if np.sum(noise_idx) < 5:
        print("⚠️ Not enough noise frames")
        return "unknown"

    noise_frames = frames[noise_idx]

    flatness_vals = []
    for frame in noise_frames:
        X = stft(frame, n_fft=frame_len, hop=frame_len)
        flatness_vals.append(np.mean(spectral_flatness(X)))

    flatness_vals = np.array(flatness_vals)

    nsi = np.var(energies_db) + np.var(flatness_vals)
    print("Nonstationarity index:", nsi)

    if nsi < 2:
        return "stationary"
    elif nsi < 10:
        return "slowly_varying"
    else:
        return "highly_nonstationary"
