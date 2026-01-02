import numpy as np

def bss_metrics(reference, estimate, eps=1e-10):
    """
    reference, estimate: 1D signals
    returns: SDR, SIR, SAR
    """
    ref_energy = np.sum(reference**2)
    err = estimate - reference

    SDR = 10 * np.log10(ref_energy / (np.sum(err**2) + eps))

    # For single-source evaluation, SIR ≈ SDR
    SIR = SDR

    # Artifact power
    SAR = 10 * np.log10(
        ref_energy / (np.sum((estimate - reference)**2) + eps)
    )

    return SDR, SIR, SAR
