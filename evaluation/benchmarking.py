from evaluation.metrics import mse, snr, snr_improvement, segmental_snr
from evaluation.spectral_metrics import log_spectral_distance

# --------------------------------------------------
# Evaluate one enhancement method
# --------------------------------------------------
def evaluate_method(clean, noisy, enhanced):
    results = {
        "SNR_in (dB)": float(snr(clean, noisy)),
        "SNR_out (dB)": float(snr(clean, enhanced)),
        "SNR_improvement (dB)": float(snr_improvement(clean, noisy, enhanced)),
        "Segmental_SNR (dB)": float(segmental_snr(clean, enhanced)),
        "MSE": float(mse(clean, enhanced)),
        "LSD": float(log_spectral_distance(clean, enhanced))
    }
    return results


def print_results(results, precision=3):
    for k, v in results.items():
        if abs(v) < 1e-3:
            print(f"{k:22s}: {v:.2e}")
        else:
            print(f"{k:22s}: {v:.{precision}f}")
