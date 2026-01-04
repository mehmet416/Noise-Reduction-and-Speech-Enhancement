import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# --- IMPORTS ---
from core.audio_io import load_audio, normalize
from core.mixing import mix_with_snr
from enhancement.spectral_subtraction import spectral_subtraction
from enhancement.wiener_static import wiener_filter_static
from enhancement.wiener_adaptive import wiener_dd # Güncellediğimiz dosya
from enhancement.pca import PCADenoiser
from enhancement.adaptive import DualChannelSimulator, AdaptiveNLMSFilter

# =============================================================================
# HİZALAMA VE SNR (Senin kodun - Dokunulmadı)
# =============================================================================
def align_and_calculate_snr(clean, processed):
    e_clean = np.sum(clean ** 2)
    e_proc = np.sum(processed ** 2)
    if e_proc == 0: return 0 
    
    processed = processed * np.sqrt(e_clean / e_proc)

    corr = signal.correlate(clean, processed, mode='full', method='fft')
    lags = signal.correlation_lags(len(clean), len(processed), mode='full')
    best_lag = lags[np.argmax(np.abs(corr))]
    
    if best_lag > 0:
        processed_aligned = np.pad(processed, (best_lag, 0), mode='constant')[:len(clean)]
    elif best_lag < 0:
        processed_aligned = processed[-best_lag:]
        if len(processed_aligned) < len(clean):
            processed_aligned = np.pad(processed_aligned, (0, len(clean)-len(processed_aligned)), mode='constant')
    else:
        processed_aligned = processed

    min_len = min(len(clean), len(processed_aligned))
    clean = clean[:min_len]
    processed_aligned = processed_aligned[:min_len]

    noise_residual = processed_aligned - clean
    p_signal = np.mean(clean ** 2)
    p_noise = np.mean(noise_residual ** 2)
    
    if p_noise == 0: return 100
    return 10 * np.log10(p_signal / p_noise)


def calculate_segmental_snr(clean, processed, fs, frame_len_sec=0.030):
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
    
    eps = 1e-10
    seg_snrs = 10 * np.log10((p_s + eps) / (p_n + eps))
    seg_snrs = np.clip(seg_snrs, -10, 35)
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
clean = normalize(clean_full[:5*fs]) 

# Simülatör (Leakage ayarına dikkat: -25dB - SENİN KODUNDAKİ GİBİ)
simulator = DualChannelSimulator(room_complexity=64)

for noise_name, noise_path in noise_files.items():
    print(f"\n>> Analiz Ediliyor: {noise_name}")
    noise, _ = load_audio(noise_path)
    noise = normalize(noise)

    for snr in snr_list:
        print(f"   Input SNR: {snr} dB...", end="")
        
        # --- Single Channel ---
        noisy_single, _ = mix_with_snr(clean, noise, snr_db=snr)
        
        seg_snr_in = calculate_segmental_snr(clean, noisy_single, fs)

        outputs = {}
        
        # 1. SS
        try: outputs["Spectral Subtraction"] = spectral_subtraction(noisy_single, fs)
        except: outputs["Spectral Subtraction"] = noisy_single

        # 2. Wiener Static
        try: outputs["Wiener Static"] = wiener_filter_static(noisy_single, fs)
        except: outputs["Wiener Static"] = noisy_single

        # 3. Wiener DD (Burası yeni yazdığımız fonksiyonu çağırıyor)
        try:
            ntype = "stationary" if "White" in noise_name else "nonstationary"
            # Fonksiyonu noise_type parametresiyle çağırmaya devam ediyoruz (bozulmasın diye)
            outputs["Wiener DD"] = wiener_dd(noisy_single, fs, noise_type=ntype)
        except: outputs["Wiener DD"] = noisy_single
        
        # 4. PCA
        try:
            pca = PCADenoiser(embedding_dim=300, energy_threshold=0.8)
            outputs["PCA"] = pca.fit_transform(noisy_single)
        except: outputs["PCA"] = noisy_single

        # 5. Adaptive LMS (Referans Sinyali Kullanarak)
        try:
            # Leakage -25dB (Senin kodun)
            d_prim, x_ref = simulator.simulate(clean, noise, snr_db=snr, leakage_db=-25)
            nlms = AdaptiveNLMSFilter(filter_order=64, learning_rate=0.005)
            
            outputs["Adaptive LMS"] = nlms.process(d_prim, x_ref, auto_sync=True)
        except: 
            outputs["Adaptive LMS"] = noisy_single

        # --- METRİKLER ---
        for m in methods:
            processed = outputs[m]
            
            # Global
            snr_out = align_and_calculate_snr(clean, processed)
            results_global[noise_name][m].append(snr_out - snr)
            
            # Segmental
            snr_seg_out = calculate_segmental_snr(clean, processed, fs)
            results_seg[noise_name][m].append(snr_seg_out - seg_snr_in)
            
        print(" OK.")

# --- ÇİZDİRME (Düz Çizgiler) ---
colors = {
    "Spectral Subtraction": "red",
    "Wiener Static":        "orange",
    "Wiener DD":            "green",
    "PCA":                  "purple",
    "Adaptive LMS":         "blue"
}

for noise_name in noise_files.keys():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Global SNR
    ax1.axhline(0, color='black', alpha=0.3, linewidth=1)
    for m in methods:
        lw = 3.0 if m == "Adaptive LMS" else 2.0
        ax1.plot(snr_list, results_global[noise_name][m], label=m, 
                 color=colors[m], linestyle='-', linewidth=lw, marker='o', markersize=8)
    ax1.set_title(f"Global SNR Improvement (Aligned) - {noise_name}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Input SNR (dB)"); ax1.set_ylabel("Improvement (dB)")
    ax1.set_xticks(snr_list); ax1.grid(True, alpha=0.4); ax1.legend()

    # Segmental SNR
    ax2.axhline(0, color='black', alpha=0.3, linewidth=1)
    for m in methods:
        lw = 3.0 if m == "Adaptive LMS" else 2.0
        ax2.plot(snr_list, results_seg[noise_name][m], label=m, 
                 color=colors[m], linestyle='-', linewidth=lw, marker='s', markersize=8)
    ax2.set_title(f"Segmental SNR Improvement - {noise_name}", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Input SNR (dB)"); ax2.set_ylabel("SegSNR Improvement (dB)")
    ax2.set_xticks(snr_list); ax2.grid(True, alpha=0.4); ax2.legend()
    
    plt.tight_layout()
    plt.show()