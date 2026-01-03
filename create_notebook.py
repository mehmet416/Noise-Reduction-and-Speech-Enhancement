
import json

def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code
    }

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

cell_1_code = """
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from core.audio_io import load_audio, write_audio
from core.mixing import mix_with_snr
from core.visualization import plot_waveforms, plot_spectrogram
from evaluation.metrics import snr, snr_improvement
from enhancement.adaptive import AdaptiveNLMSFilter

%matplotlib inline
"""
notebook['cells'].append(create_code_cell(cell_1_code))

cell_2_code = """
# Load signals
clean_speech, fs = load_audio('data/clean/clean_speech.wav')
noise, fs_noise = load_audio('data/noise/white.wav')

# Plot
plot_waveforms([clean_speech, noise], ['Clean Speech', 'Noise'], fs, 'Original Signals')
"""
notebook['cells'].append(create_code_cell(cell_2_code))

cell_3_code = """
# Mix signals at 5 dB SNR
snr_db = 5
noisy_speech, noise_component = mix_with_snr(clean_speech, noise, snr_db)

# Plot
plot_waveforms([noisy_speech], ['Noisy Speech (5dB SNR)'], fs, 'Mixed Signal')
"""
notebook['cells'].append(create_code_cell(cell_3_code))

cell_4_code = """
# Apply adaptive filter
# The NLMS filter tries to predict the noise in the noisy signal
# d = noisy_speech (primary input)
# x = noise (reference input)
# The output 'e' is the error signal, which is the estimated clean speech
lms = AdaptiveNLMSFilter(filter_order=512, learning_rate=0.1)
enhanced_speech = lms.process(noisy_speech, noise_component)

# Plot
plot_waveforms([enhanced_speech], ['Enhanced Speech'], fs, 'Adaptive Filter Output')
"""
notebook['cells'].append(create_code_cell(cell_4_code))

cell_5_code = """
# --- Evaluation ---
initial_snr = snr(clean_speech, noisy_speech)
final_snr = snr(clean_speech, enhanced_speech)
improvement = snr_improvement(clean_speech, noisy_speech, enhanced_speech)

print(f"Initial SNR: {initial_snr:.2f} dB")
print(f"Final SNR: {final_snr:.2f} dB")
print(f"SNR Improvement: {improvement:.2f} dB")
"""
notebook['cells'].append(create_code_cell(cell_5_code))

cell_6_code = """
# --- Comparison ---
plot_waveforms(
    [clean_speech, noisy_speech, enhanced_speech],
    ['Clean', 'Noisy', 'Enhanced'],
    fs,
    'Waveform Comparison'
)

plot_spectrogram(clean_speech, fs, 'Clean Spectrogram')
plot_spectrogram(noisy_speech, fs, 'Noisy Spectrogram')
plot_spectrogram(enhanced_speech, fs, 'Enhanced Spectrogram')
"""
notebook['cells'].append(create_code_cell(cell_6_code))


with open('exp_02_adaptive.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Successfully created notebook exp_02_adaptive.ipynb")
