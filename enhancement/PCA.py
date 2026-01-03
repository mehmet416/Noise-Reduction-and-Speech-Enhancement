import numpy as np
from scipy.linalg import hankel, svd

class PCADenoiser:
    def __init__(self, embedding_dim=300, k_components=None, energy_threshold=None):
        self.L = embedding_dim
        self.K = k_components
        self.energy_th = energy_threshold
        self.singular_values = None
        self.used_k = 0 
        
    def reconstruct_from_hankel(self, matrix):
        rows, cols = matrix.shape
        n_samples = rows + cols - 1
        reconstructed = np.zeros(n_samples)
        count = np.zeros(n_samples)
        
        for r in range(rows):
            reconstructed[r:r+cols] += matrix[r, :]
            count[r:r+cols] += 1
            
        count[count == 0] = 1
        return reconstructed / count

    def fit_transform(self, noisy_signal):
        # Koruma
        if len(noisy_signal) < self.L: return noisy_signal

        # 1. Hankel
        first_col = noisy_signal[:self.L]
        last_row = noisy_signal[self.L-1:]
        H = hankel(first_col, last_row)
        
        # 2. SVD
        U, S, Vt = svd(H, full_matrices=False)
        self.singular_values = S
        
        # 3. K Seçimi
        if self.energy_th is not None:
            total_energy = np.sum(S**2)
            cumulative = np.cumsum(S**2) / total_energy
            
            # Eşiği geçen nokta
            k_idx = np.searchsorted(cumulative, self.energy_th)
            current_k = k_idx + 1
            
            # --- DÜZELTME BURADA ---
            # Eskiden *0.5 ile sınırlandırmıştık, şimdi kaldırıyoruz.
            # Sadece matris boyutunu aşmasın yeter.
            current_k = max(1, min(current_k, len(S) - 1))
            self.used_k = current_k
            
        else:
            current_k = self.K if self.K is not None else 20
            self.used_k = current_k

        # 4. Thresholding
        S_clean = np.zeros_like(S)
        S_clean[:self.used_k] = S[:self.used_k]
        
        # 5. Reconstruction
        H_clean = U @ np.diag(S_clean) @ Vt
        return self.reconstruct_from_hankel(H_clean)