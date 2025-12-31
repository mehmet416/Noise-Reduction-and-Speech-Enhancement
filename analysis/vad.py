import numpy as np

# --------------------------------------------------
# Energy-based VAD (frame-wise)
# --------------------------------------------------
def vad_energy(frames, percentile=70):
    eps = 1e-12
    energies = np.sum(frames**2, axis=1) + eps
    energies_db = 10 * np.log10(energies)

    thresh = np.percentile(energies_db, percentile)
    vad_mask = energies_db > thresh

    return vad_mask.astype(int), energies_db

