import matplotlib.pyplot as plt
import numpy as np

# --- IMPORTS ---
from core.audio_io import load_audio, normalize
from core.mixing import mix_with_snr
from enhancement.spectral_subtraction import spectral_subtraction
from enhancement.wiener_static import wiener_filter_static
from enhancement.wiener_adaptive import wiener_dd
from enhancement.PCA import PCADenoiser
from enhancement.adaptive import DualChannelSimulator, AdaptiveNLMSFilter

# =============================================================================
# YENİ FONKSİYON: SEGMENTAL SNR HESAPLAMA
# =============================================================================
def calculate_segmental_snr(clean, processed, fs, frame_len_sec=0.030):
    """
    Sinyali 30ms'lik çerçevelere böler ve Segmental SNR hesaplar.
    Global SNR'dan farklı olarak insan kulağının algısına daha yakındır.
    """
    # 1. Boyut ve Enerji Eşitleme
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    
    # Enerji ölçekleme (Genlik farkını yok saymak için)
    e_clean = np.sum(clean**2)
    e_proc = np.sum(processed**2)
    if e_proc > 0:
        processed = processed * np.sqrt(e_clean / e_proc)
    
    # 2. Çerçeveleme (Framing)
    frame_len = int(frame_len_sec * fs)
    n_frames = int(len(clean) / frame_len)
    
    # Tam bölünebilmesi için kırp
    clean = clean[:n_frames*frame_len]
    processed = processed[:n_frames*frame_len]
    
    # (N_frames, Frame_len) formatına dönüştür
    clean_frames = clean.reshape(n_frames, frame_len)
    processed_frames = processed.reshape(n_frames, frame_len)
    
    # 3. Gürültü (Error) Hesapla
    noise_frames = processed_frames - clean_frames
    
    # 4. Güç Hesapla
    signal_energy = np.sum(clean_frames**2, axis=1)
    noise_energy = np.sum(noise_frames**2, axis=1)
    
    # 5. Logaritmik Hesaplama ve Sınırlama (Clamping)
    # Sessiz kısımlarda sonsuz değer çıkmaması için epsilon ve min/max sınırlar
    eps = 1e-10
    
    # Her çerçevenin SNR'ı (dB)
    segment_snrs = 10 * np.log10((signal_energy + eps) / (noise_energy + eps))
    
    # Segmental SNR genelde -10dB ile 35dB arasına sıkıştırılır (Literatür standardı)
    segment_snrs = np.clip(segment_snrs, -10, 35)
    
    # Ortalama al
    return np.mean(segment_snrs)

# Standart Global SNR
def calculate_global_snr(clean, processed):
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]; processed = processed[:min_len]
    
    # Enerji Eşitleme
    e_c = np.sum(clean**2); e_p = np.sum(processed**2)
    if e_p > 0: processed = processed * np.sqrt(e_c / e_p)
        
    noise = processed - clean
    p_s = np.mean(clean**2); p_n = np.mean(noise**2)
    if p_n == 0: return 100
    return 10 * np.log10(p_s / p_n)

# --- AYARLAR ---
snr_list = [0, 5, 10, 15, 20]
noise_files = {
    "Traffic": "data/noise/traffic.wav",
    "White":   "data/noise/white.wav",
    "Office":  "data/noise/office.wav"
}

methods = ["Spectral Subtraction", "Wiener Static", "Wiener DD", "PCA", "Adaptive LMS"]

# Sonuçları saklamak için iki ayrı sözlük
results_global = {n: {m: [] for m in methods} for n in noise_files}
results_seg = {n: {m: [] for m in methods} for n in noise_files}

print("Benchmark ve Segmental SNR Analizi Başlıyor...")
clean_full, fs = load_audio("data/clean/clean_speech.wav")
clean = normalize(clean_full[:5*fs]) # 5 saniye

simulator = DualChannelSimulator(room_complexity=64)

# --- ANA DÖNGÜ ---
for noise_name, noise_path in noise_files.items():
    print(f"\n>> Analiz Ediliyor: {noise_name}")
    noise, _ = load_audio(noise_path)
    noise = normalize(noise)

    for snr in snr_list:
        print(f"   Input SNR: {snr} dB...", end="")
        
        # Tek Kanal Giriş
        noisy_single, _ = mix_with_snr(clean, noise, snr_db=snr)
        
        # Girişin Segmental SNR değerini hesapla (Improvement için referans)
        seg_snr_input = calculate_segmental_snr(clean, noisy_single, fs)
        
        # Fonksiyonları çalıştırmak için bir sözlük yapısı
        outputs = {}
        
        # 1. Spectral Subtraction
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
            pca = PCADenoiser(embedding_dim=200, k_components=20)
            outputs["PCA"] = pca.fit_transform(noisy_single)
        except: outputs["PCA"] = noisy_single

        # 5. Adaptive LMS (Optimize edilmiş)
        try:
            d_prim, x_ref = simulator.simulate(clean, noise, snr_db=snr, leakage_db=-35)
            nlms = AdaptiveNLMSFilter(filter_order=64, learning_rate=0.005)
            outputs["Adaptive LMS"] = nlms.process(d_prim, x_ref, auto_sync=True)
        except: outputs["Adaptive LMS"] = noisy_single

        # --- METRİK HESAPLAMA ---
        for m in methods:
            processed = outputs[m]
            
            # Global Improvement
            imp_glob = calculate_global_snr(clean, processed) - snr
            results_global[noise_name][m].append(imp_glob)
            
            # Segmental Improvement (Çıkış SegSNR - Giriş SegSNR)
            seg_out = calculate_segmental_snr(clean, processed, fs)
            imp_seg = seg_out - seg_snr_input
            results_seg[noise_name][m].append(imp_seg)
            
        print(" OK.")

# --- ÇİZDİRME (SIDE-BY-SIDE PLOTS) ---
styles = {
    "Spectral Subtraction": {"c": "red", "s": "--"},
    "Wiener Static":        {"c": "orange", "s": "--"},
    "Wiener DD":            {"c": "green", "s": "-"},
    "PCA":                  {"c": "purple", "s": "-."},
    "Adaptive LMS":         {"c": "blue", "s": "-", "lw": 2.5}
}

for noise_name in noise_files.keys():
    # 1 Satır, 2 Sütunluk grafik alanı
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Grafik 1: Global SNR ---
    ax1.axhline(0, color='black', alpha=0.3)
    for m in methods:
        y = results_global[noise_name][m]
        st = styles.get(m, {})
        ax1.plot(snr_list, y, label=m, color=st.get("c"), ls=st.get("s"), lw=st.get("lw", 1.5), marker='o')
    
    ax1.set_title(f"Global SNR Improvement ({noise_name})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Input SNR (dB)")
    ax1.set_ylabel("Improvement (dB)")
    ax1.set_xticks(snr_list)
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    # --- Grafik 2: Segmental SNR ---
    ax2.axhline(0, color='black', alpha=0.3)
    for m in methods:
        y = results_seg[noise_name][m]
        st = styles.get(m, {})
        ax2.plot(snr_list, y, label=m, color=st.get("c"), ls=st.get("s"), lw=st.get("lw", 1.5), marker='s')
    
    ax2.set_title(f"Segmental SNR Improvement ({noise_name})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Input SNR (dB)")
    ax2.set_ylabel("SegSNR Improvement (dB)")
    ax2.set_xticks(snr_list)
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()