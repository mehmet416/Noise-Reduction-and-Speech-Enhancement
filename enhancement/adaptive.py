import numpy as np
from scipy import signal

class DualChannelSimulator:
    """
    Dual Channel Acoustic Simulator for Adaptive Filtering.
    """
    def __init__(self, room_complexity=64):
        # Random seed for reproducibility
        np.random.seed(42)
        # Room impulse response (FIR filter)
        self.room_impulse = signal.firwin(room_complexity, 0.5)

    def simulate(self, clean_speech, noise_source, snr_db=0, leakage_db=-30):
        """
        leakage_db: -30 dB means that the speech leaks into the reference mic at 30 dB lower power.
        """
        # Ensure numpy arrays
        min_len = min(len(clean_speech), len(noise_source))
        s = clean_speech[:min_len].astype(float)
        n = noise_source[:min_len].astype(float)
        
        # 1. Room Impulse Response
        # Noise reflects in the room and arrives at the primary microphone
        noise_at_primary = signal.lfilter(self.room_impulse, 1.0, n)
        
        # 2. SNR Adjustment (Primary Microphone)
        s_power = np.mean(s**2) + 1e-10
        n_power = np.mean(noise_at_primary**2) + 1e-10
        target_n_power = s_power / (10**(snr_db/10))
        scale = np.sqrt(target_n_power / n_power)
        noise_at_primary = noise_at_primary * scale
        
        primary_mic = s + noise_at_primary
        
        # 3. Leakage into Reference Microphone
        # Reference microphone hears noise clearly (n), but speech also leaks in (leakage)
        raw_n_power = np.mean(n**2) + 1e-10
        target_leakage_power = raw_n_power / (10**(abs(leakage_db)/10))
        leak_scale = np.sqrt(target_leakage_power / s_power)
        
        reference_mic = n + (s * leak_scale)
        
        # Normalizasyon
        primary_mic /= (np.max(np.abs(primary_mic)) + 1e-9)
        reference_mic /= (np.max(np.abs(reference_mic)) + 1e-9)
        
        return primary_mic, reference_mic

class AdaptiveNLMSFilter:
    """
    Standard Normalized Least Mean Squares (NLMS) Filter.
    """
    def __init__(self, filter_order=64, learning_rate=0.01):
        self.M = filter_order
        self.mu = learning_rate
        self.w = np.zeros(self.M)
        
    def process(self, primary, reference, auto_sync=True):
        d = np.array(primary, dtype=float)
        x = np.array(reference, dtype=float)
        
        # Simple Synchronization (Finding delay between d and x)
        if auto_sync:
            corr = signal.correlate(d, x, mode='full', method='fft')
            lags = signal.correlation_lags(len(d), len(x), mode='full')
            lag = lags[np.argmax(np.abs(corr))]
            if lag > 0:
                d = d[lag:]
                x = x[:len(d)]
            elif lag < 0:
                x = x[abs(lag):]
                d = d[:len(x)]

        n_samples = len(d)
        e = np.zeros(n_samples)
        input_buffer = np.zeros(self.M)
        
        # Stability constant
        eps = 1e-6
        
        # LMS Loop
        for n in range(n_samples):
            # Shift buffer and add new sample
            input_buffer = np.roll(input_buffer, 1)
            input_buffer[0] = x[n]
            
            # Prediction (Noise estimate)
            y = np.dot(self.w, input_buffer)
            
            # Error (Enhanced signal)
            e[n] = d[n] - y
            
            # NLMS Weight Update
            # Normalize by norm to make independent of signal power
            x_norm = np.dot(input_buffer, input_buffer)
            step = self.mu / (x_norm + eps)
            
            self.w = self.w + step * e[n] * input_buffer
            
        return e