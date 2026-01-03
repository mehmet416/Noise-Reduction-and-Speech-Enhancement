import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import hankel, svd
from scipy.io import wavfile

# =============================================================================
# CLASS: PCA DENOISER
# =============================================================================
class PCADenoiser:
    def __init__(self, embedding_dim=200, k_components=20):
        """
        embedding_dim (L): Window size. Increased to 200 for 5 seconds of audio.
        k_components (K): Number of principal components to keep.
        """
        self.L = embedding_dim
        self.K = k_components
        self.singular_values = None
        
    def reconstruct_from_hankel(self, matrix):
        rows, cols = matrix.shape
        n_samples = rows + cols - 1
        reconstructed = np.zeros(n_samples)
        count = np.zeros(n_samples)
        
        for r in range(rows):
            # Vectorized addition (loop optimized for speed)
            # Diagonal Averaging
            reconstructed[r:r+cols] += matrix[r, :]
            count[r:r+cols] += 1
            
        return reconstructed / count

    def fit_transform(self, noisy_signal):
        print(f"[PCA] Creating matrix (Embedding L={self.L})...")
        N = len(noisy_signal)
        
        # 1. Trajectory Matrix (Hankel)
        # Converting the signal into an [L x M] matrix.
        first_col = noisy_signal[:self.L]
        last_row = noisy_signal[self.L-1:]
        H = hankel(first_col, last_row)
        
        print(f"[PCA] Calculating SVD (Matrix Size: {H.shape})... Please wait.")
        
        # 2. SVD (Singular Value Decomposition)
        # full_matrices=False, saves RAM.
        U, Sigma, Vt = svd(H, full_matrices=False)
        
        self.singular_values = Sigma
        
        # 3. Thresholding (Noise Removal)
        Sigma_clean = np.zeros_like(Sigma)
        Sigma_clean[:self.K] = Sigma[:self.K] # Keep only the strongest K components
        
        energy_ratio = np.sum(Sigma[:self.K]**2) / np.sum(Sigma**2)
        print(f"[PCA] {energy_ratio*100:.2f}% of energy retained (Signal Subspace).")
        print(f"[PCA] Remaining {(1-energy_ratio)*100:.2f}% discarded as noise.")
        
        # 4. Reconstruction
        H_clean = U @ np.diag(Sigma_clean) @ Vt
        clean_signal = self.reconstruct_from_hankel(H_clean)
        
        return clean_signal
