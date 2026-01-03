import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# --- IMPORTS ---
from core.audio_io import load_audio, normalize
from core.mixing import mix_with_snr
from enhancement.spectral_subtraction import spectral_subtraction
from enhancement.wiener_static import wiener_filter_static
from enhancement.wiener_adaptive import wiener_dd
from enhancement.pca import PCADenoiser
from enhancement.adaptive import DualChannelSimulator, AdaptiveNLMSFilter

# =============================================================================
# YENİ FONKSİYON: SİNYAL HİZALAMA VE SNR (THE FIX)
# =============================================================================
def align_and_calculate_snr(clean, processed):
    """
    İşlenmiş sinyali temiz sinyalle milimetrik hizalar.
    LMS gibi filtrelerin yarattığı gecikmeyi (delay) telafi eder.
    Bunu yapmazsak, ses güzel olsa bile SNR sonucu yanlış çıkar.
    """
    # 1. Enerji Eşitleme (Genlik farkını yok et)
    # Temiz sesin enerjisi
    e_clean = np.sum(clean ** 2)
    e_proc = np.sum(processed ** 2)
    if e_proc == 0: return 0 # Sessizlik
    
    processed = processed * np.sqrt(e_clean / e_proc)

    # 2. Cross-Correlation ile Gecikme Bulma
    # Hız için FFT tabanlı korelasyon kullanıyoruz
    corr = signal.correlate(clean, processed, mode='full', method='fft')
    lags = signal.correlation_lags(len(clean), len(processed), mode='full')
    
    # En yüksek uyumun olduğu gecikme (lag)
    best_lag = lags[np.argmax(np.abs(corr))]
    
    # 3. Sinyali Kaydır (Hizala)
    if best_lag > 0:
        # Processed geride kalmış, ileri al
        processed_aligned = np.pad(processed, (best_lag, 0), mode='constant')[:len(clean)]
    elif best_lag < 0:
        # Processed önden gidiyor (bu nadir olur), geriye al
        processed_aligned = processed[-best_lag:]
        # Boyut yetmezse sonuna sıfır ekle
        if len(processed_aligned) < len(clean):
            processed_aligned = np.pad(processed_aligned, (0, len(clean)-len(processed_aligned)), mode='constant')
    else:
        processed_aligned = processed

    # Boyutları kesin eşitle
    min_len = min(len(clean), len(processed_aligned))
    clean = clean[:min_len]
    processed_aligned = processed_aligned[:min_len]

    # 4. Artık Doğru SNR Hesaplanabilir
    noise_residual = processed_aligned - clean
    p_signal = np.mean(clean ** 2)
    p_noise = np.mean(noise_residual ** 2)
    
    if p_noise == 0: return 100
    
    return 10 * np.log10(p_signal / p_noise)


def calculate_segmental_snr(clean, processed, fs, frame_len_sec=0.030):
    # Segmental için de basit bir hizalama yapalım (manuel kaydırma yok, enerji eşitleme var)
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]; processed = processed[:min_len]
    
    e_c = np.sum(clean**2); e_p = np.sum(processed**2)
    if e_p > 0: processed = processed * np.sqrt(e_c / e_p)
    
    frame_len = int(frame_len_sec * fs)
    n_frames = int(len(clean) // frame_len)
    
    clean = clean[:n_frames*frame_len].reshape(n_frames, frame_len)
    processed = processed[:n_frames*frame_len].reshape(n_frames, frame_len)
    
    noise = processed - clean
    p_s = np.sum(clean**2, axis=1)
    p_n = np.sum(noise**2, axis=1)
    
    # Log domain
    eps = 1e-10
    seg_snrs = 10 * np.log10((p_s + eps) / (p_n + eps))
    seg_snrs = np.clip(seg_snrs, -10, 35) # Sınırlandırma
    
    return np.mean(seg_snrs)

# --- AYARLAR ---
snr_list = [0, 2.5, 5, 7.5, 10]
noise_files = {
    "Traffic": "data/noise/traffic.wav",
    "White":   "data/noise/white.wav",
    "Office":  "data/noise/office.wav"
}

methods = ["Spectral Subtraction", "Wiener Static", "Wiener DD", "PCA", "Adaptive LMS"]
results_global = {n: {m: [] for m in methods} for n in noise_files}
results_seg = {n: {m: [] for m in methods} for n in noise_files}

print("Gelişmiş Hizalama (Alignment) ile Benchmark Başlıyor...")
clean_full, fs = load_audio("data/clean/clean_speech.wav")
clean = normalize(clean_full[:5*fs]) # 5 saniye

# Simülatör (Leakage ayarına dikkat: -35dB)
simulator = DualChannelSimulator(room_complexity=64)

for noise_name, noise_path in noise_files.items():
    print(f"\n>> Analiz Ediliyor: {noise_name}")
    noise, _ = load_audio(noise_path)
    noise = normalize(noise)

    for snr in snr_list:
        print(f"   Input SNR: {snr} dB...", end="")
        
        # --- Single Channel ---
        noisy_single, _ = mix_with_snr(clean, noise, snr_db=snr)
        
        # Giriş SegSNR Referansı
        seg_snr_in = calculate_segmental_snr(clean, noisy_single, fs)

        outputs = {}
        
        # 1. SS
        try: outputs["Spectral Subtraction"] = spectral_subtraction(noisy_single, fs)
        except: outputs["Spectral Subtraction"] = noisy_single

        # 2. Wiener Static
        try: outputs["Wiener Static"] = wiener_filter_static(noisy_single, fs)
        except: outputs["Wiener Static"] = noisy_single

        # 3. Wiener DD
        try:
            ntype = "stationary" if "White" in noise_name else "nonstationary"
            outputs["Wiener DD"] = wiener_dd(noisy_single, fs, noise_type=ntype)
        except: outputs["Wiener DD"] = noisy_single
        
        # 4. PCA
        try:
            pca = PCADenoiser(embedding_dim=300, energy_threshold=0.8)
            outputs["PCA"] = pca.fit_transform(noisy_single)
        except: outputs["PCA"] = noisy_single

        # 5. Adaptive LMS (Referans Sinyali Kullanarak)
        try:
            # Leakage -35, Simülasyon
            d_prim, x_ref = simulator.simulate(clean, noise, snr_db=snr, leakage_db=-35)
            nlms = AdaptiveNLMSFilter(filter_order=64, learning_rate=0.005)
            
            # auto_sync=False diyoruz çünkü ana hizalamayı 'align_and_calculate_snr' yapacak
            outputs["Adaptive LMS"] = nlms.process(d_prim, x_ref, auto_sync=True)
        except: 
            outputs["Adaptive LMS"] = noisy_single

        # --- METRİKLER (HİZALAMA DAHİL) ---
        for m in methods:
            processed = outputs[m]
            
            # HİZALANMIŞ GLOBAL SNR
            snr_out = align_and_calculate_snr(clean, processed)
            results_global[noise_name][m].append(snr_out - snr)
            
            # Segmental SNR
            snr_seg_out = calculate_segmental_snr(clean, processed, fs)
            results_seg[noise_name][m].append(snr_seg_out - seg_snr_in)
            
        print(" OK.")
# =============================================================================
# 4. GRAFİK ÇİZDİRME (GÜNCELLENMİŞ: Hepsi Düz Çizgi)
# =============================================================================

# Renk Tanımları
colors = {
    "Spectral Subtraction": "red",
    "Wiener Static":        "orange",
    "Wiener DD":            "green",
    "PCA":                  "purple",
    "Adaptive LMS":         "blue"
}

for noise_name in noise_files.keys():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Grafik 1: Global SNR ---
    ax1.axhline(0, color='black', alpha=0.3, linewidth=1) # 0 noktası referansı
    
    for m in methods:
        # LMS dikkat çeksin diye biraz daha kalın (3.0), diğerleri 2.0
        lw = 3.0 if m == "Adaptive LMS" else 2.0
        
        ax1.plot(snr_list, results_global[noise_name][m], 
                 label=m, 
                 color=colors[m], 
                 linestyle='-',    # HEPSİ DÜZ ÇİZGİ
                 linewidth=lw, 
                 marker='o',       # Yuvarlak işaretleyici
                 markersize=8)
                 
    ax1.set_title(f"Global SNR Improvement (Aligned) - {noise_name}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Input SNR (dB)")
    ax1.set_ylabel("Improvement (dB)")
    ax1.set_xticks(snr_list) # X ekseninde sadece 0, 5, 10 görünsün
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    # --- Grafik 2: Segmental SNR ---
    ax2.axhline(0, color='black', alpha=0.3, linewidth=1)
    
    for m in methods:
        lw = 3.0 if m == "Adaptive LMS" else 2.0
        
        ax2.plot(snr_list, results_seg[noise_name][m], 
                 label=m, 
                 color=colors[m], 
                 linestyle='-',    # HEPSİ DÜZ ÇİZGİ
                 linewidth=lw, 
                 marker='s',       # Kare işaretleyici (Farklılık olsun diye)
                 markersize=8)
                 
    ax2.set_title(f"Segmental SNR Improvement - {noise_name}", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Input SNR (dB)")
    ax2.set_ylabel("SegSNR Improvement (dB)")
    ax2.set_xticks(snr_list) # X ekseninde sadece 0, 5, 10 görünsün
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()