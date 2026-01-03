import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal

# --- PROJE IMPORTLARI ---
from core.audio_io import load_audio, normalize
from core.mixing import mix_with_snr
from enhancement.pca import PCADenoiser

# --- HİZALANMIŞ SNR (Hata payını yok eder) ---
def get_aligned_snr_improvement(clean, processed, input_snr):
    e_c = np.sum(clean**2); e_p = np.sum(processed**2)
    if e_p > 0: processed = processed * np.sqrt(e_c/e_p)
    
    corr = signal.correlate(clean, processed, mode='full', method='fft')
    lag = signal.correlation_lags(len(clean), len(processed), mode='full')[np.argmax(np.abs(corr))]
    
    if lag > 0: proc_align = processed[lag:]
    elif lag < 0: proc_align = processed[:len(processed)+lag]
    else: proc_align = processed
    
    m = min(len(clean), len(proc_align))
    c = clean[:m]; p = proc_align[:m]
    
    noise_res = p - c
    p_n = np.mean(noise_res**2)
    if p_n == 0: return 0
    return 10*np.log10(np.mean(c**2)/p_n) - input_snr

# --- ÇİFT PARAMETRE OPTİMİZASYONU ---
# Hem Pencere Boyutunu (L) hem Enerji Eşiğini (Th) aynı anda deniyoruz.
L_list = [50, 100, 200, 300, 400, 500]
Th_list = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

noise_path = "data/noise/white.wav" # Önce White noise'da en iyiyi bulalım
target_snr = 5

print(f"--- ULTIMATE PCA GRID SEARCH ({target_snr}dB White Noise) ---")

clean_full, fs = load_audio("data/clean/clean_speech.wav")
clean = normalize(clean_full[:3*fs]) # 3 saniye

noise, _ = load_audio(noise_path)
noise = normalize(noise)
noisy, _ = mix_with_snr(clean, noise, snr_db=target_snr)

results = np.zeros((len(L_list), len(Th_list)))

for i, L in enumerate(L_list):
    for j, th in enumerate(Th_list):
        try:
            # PCA Çalıştır
            pca = PCADenoiser(embedding_dim=L, energy_threshold=th)
            out = pca.fit_transform(noisy)
            
            # Puanla
            imp = get_aligned_snr_improvement(clean, out, target_snr)
            results[i, j] = imp
            # print(f"L={L}, Th={th} -> {imp:.2f} dB") # Çok kalabalık olmasın diye kapattım
        except:
            results[i, j] = 0

# --- SONUÇ ANALİZİ ---
best_idx = np.unravel_index(np.argmax(results), results.shape)
best_L = L_list[best_idx[0]]
best_th = Th_list[best_idx[1]]
best_score = results[best_idx]

print("\n" + "="*50)
print(f"🚀 EN OPTİMAL PARAMETRELER BULUNDU!")
print(f"   Embedding Dimension (L) : {best_L}")
print(f"   Energy Threshold (Th)   : {best_th}")
print(f"   Max Improvement         : {best_score:.2f} dB")
print("="*50)

# Isı Haritası (Hangi ayarın daha iyi olduğunu gözle gör)
plt.figure(figsize=(10, 6))
sns.heatmap(results, annot=True, fmt=".2f", cmap="RdYlGn", 
            xticklabels=Th_list, yticklabels=L_list)
plt.title("PCA Performance Heatmap (L vs Threshold)")
plt.xlabel("Energy Threshold")
plt.ylabel("Embedding Dimension (L)")
plt.show()