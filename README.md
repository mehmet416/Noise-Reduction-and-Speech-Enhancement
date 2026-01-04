# EE473 – Noise Reduction and Speech Enhancement

This repository contains the implementation and evaluation of classical and adaptive **speech enhancement and noise reduction algorithms** developed as part of the **EE473 Digital Signal Processing** course project.

The main objective of the project is to investigate and compare multiple signal processing techniques for reducing background noise and enhancing speech quality under both **synthetic** and **real-world** noisy conditions.

---

## Implemented Methods

The following methods are implemented and evaluated:

1. Spectral Subtraction  
2. Static Wiener Filtering  
3. Adaptive Wiener Filtering (Decision-Directed)  
4. LMS Adaptive Noise Cancellation  
5. Principal Component Analysis (PCA)  
6. Independent Component Analysis (ICA)

Both single-channel and multi-channel scenarios are considered where applicable.

---

## Datasets

- **Clean Speech**  
  LibriSpeech ASR Clean Dataset  
  https://www.kaggle.com/datasets/bernardoolisan/librispeech-asr-clean-in-wavs

- **Noise Signals**  
  DEMAND: Diverse Environments Multichannel Acoustic Noise Database  
  https://www.kaggle.com/datasets/chrisfilo/demand

- **Real-World Recordings**  
  Self-recorded speech samples collected in traffic and environmental noise conditions.

---

## Experiments

### Synthetic Experiments
- Controlled mixing of clean speech and noise
- Availability of ground-truth signals
- Quantitative evaluation using SNR, correlation, and leakage metrics
- ICA-based source separation with additional sensor noise modeling

### Real-World Experiments
- Single- and multi-microphone recordings
- No clean reference signals available
- Subjective evaluation via listening tests and spectrogram analysis
- Robustness analysis under reverberation and non-ideal acoustic conditions

---

## Evaluation Metrics

- Signal-to-Noise Ratio (SNR)
- Segmental SNR
- Mean Squared Error (MSE)
- Correlation Coefficient
- Leakage Energy (for ICA)
- Time–frequency (spectrogram) analysis

<h2>Repository Structure</h2>

<pre><code>.
├── core/               # STFT, ISTFT, mixing, audio I/O utilities
├── enhancement/        # Spectral subtraction, Wiener, LMS methods, PCA 
├── separation/         # ICA implementations
├── evaluation/         # Evaluation metrics and benchmarking
├── data/               # Speech and noise datasets
├── figures/            # Spectrograms and result plots
└── report/             # Project report (LaTeX / PDF)
</code></pre>



## Notes on ICA

ICA performs well under controlled synthetic conditions with instantaneous linear mixing. However, its performance degrades significantly on real-world recordings due to reverberation, echo, and microphone mismatch, which violate ICA assumptions. To improve perceptual quality, post-processing using static Wiener filtering is applied to the separated signals.

---

## Requirements

- Python 3.x  
- NumPy  
- SciPy  
- scikit-learn  
- matplotlib  

---

## Authors

**Enes Kuzuoğlu**   
**Mehmet Emin Algül**  
**Nurullah Efe Küçük**  

*Department of Electrical & Electronics Engineering*  
*Boğaziçi University*

