import numpy as np
from scipy.signal import stft, istft

def wiener_dd(
    noisy,
    fs,
    noise_type="stationary", # Benchmark kodunla uyum için bu parametreyi tuttuk
    win_len_sec=0.02,
    hop_len_sec=0.01,
    nfft=512,
    alpha=0.98,            # A-priori SNR smoothing
    init_sec=0.2,          # İlk kaç saniyeyi gürültü kabul edelim?
    gain_floor=0.01,       # Maksimum bastırma (-40dB)
    oversubtraction=1.5,   # SNR ARTIRICI HAMLE: Gürültüyü agresif çıkar
    eps=1e-10,
):
    """
    Optimized Adaptive Wiener Filter.
    Supports 'noise_type' argument for compatibility but uses internal VAD logic.
    """

    x = np.asarray(noisy, dtype=np.float64)
    x -= np.mean(x)
    N = len(x)

    # STFT Ayarları
    nperseg = int(win_len_sec * fs)
    hop = int(hop_len_sec * fs)
    noverlap = nperseg - hop

    # 1. STFT
    _, _, Y = stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary="zeros", padded=True)
    
    K, M = Y.shape
    P_noisy = np.abs(Y) ** 2

    # 2. Gürültü Başlatma (İlk sessiz olduğu varsayılan kısım)
    init_frames = max(1, int(init_sec * fs / hop))
    init_frames = min(init_frames, M)
    
    # Başlangıç gürültü profili (Ortalama yerine Quantile daha güvenlidir)
    Phi_noise = np.quantile(P_noisy[:, :init_frames], 0.2, axis=1)
    Phi_noise = np.maximum(Phi_noise, eps)

    # Çıktı matrisi
    S_hat = np.zeros_like(Y, dtype=np.complex128)
    
    # Decision-Directed için önceki gain
    prev_G_mag_sq = np.ones(K) 

    # Noise Update Katsayısı (Gürültü tipine göre hafif ayar)
    # Eğer stationary ise gürültü yavaş değişir (0.99), değilse hızlı (0.95)
    alpha_noise_base = 0.99 if noise_type == "stationary" else 0.95

    # --- ANA DÖNGÜ ---
    for m in range(M):
        Y_frame = Y[:, m]
        P_frame = P_noisy[:, m]

        # A) A-Posteriori SNR (Oversubtraction burada devreye giriyor)
        # Gürültüyü olduğundan büyük (1.5x) varsayıyoruz ki bastırma artsın.
        gamma = P_frame / (oversubtraction * Phi_noise + eps)
        
        # B) A-Priori SNR (Decision Directed - Ephraim Malah)
        # Önceki filtrelenmiş sinyalden gelen bilgi + şimdiki ölçüm
        xi = alpha * (prev_G_mag_sq * (np.abs(Y[:, m-1])**2 if m > 0 else P_frame) / (Phi_noise + eps)) + \
             (1 - alpha) * np.maximum(gamma - 1, 0)
             
        # C) Wiener Gain
        G = xi / (1 + xi)
        G = np.maximum(G, gain_floor)
        
        prev_G_mag_sq = G**2

        # D) Sinyali Filtrele
        S_hat[:, m] = G * Y_frame

        # E) Gürültü Güncelleme (Adaptive Noise Tracking)
        # Basit ama etkili bir VAD (Voice Activity Detection):
        # Eğer hesaplanan Gain düşükse (G < 0.3), orada konuşma yok demektir -> Gürültüyü güncelle.
        # Gain yüksekse konuşma vardır -> Gürültüyü güncelleme.
        
        # Olasılık maskesi: Konuşma yoksa 1, varsa 0
        speech_absent_prob = 1.0 - np.clip(G * 2.0, 0, 1) 
        
        # Sadece gürültü olduğu düşünülen yerlerde güncelleme yap
        alpha_t = alpha_noise_base + (1 - alpha_noise_base) * (1 - speech_absent_prob)
        Phi_noise = alpha_t * Phi_noise + (1 - alpha_t) * P_frame
        Phi_noise = np.maximum(Phi_noise, eps)

    # 3. ISTFT
    _, enh = istft(S_hat, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    return np.nan_to_num(enh[:N])