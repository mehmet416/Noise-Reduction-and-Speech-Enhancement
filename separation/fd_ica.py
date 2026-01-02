import numpy as np
from sklearn.decomposition import FastICA
from core.stft import stft, istft

def stft_multichannel(X, fs, n_fft=1024, hop=256):
    """
    X: (M, N) multichannel signal
    returns: X_f (M, F, T)
    """
    M = X.shape[0]
    X_f = []

    for m in range(M):
        Xm = stft(X[m], n_fft=n_fft, hop=hop)
        X_f.append(Xm)

    return np.stack(X_f, axis=0)

def ica_one_frequency(Xf, n_sources=2, eps=1e-8):
    """
    Xf: (M, T) complex STFT at one frequency
    """

    # Energy check (CRITICAL)
    if np.mean(np.abs(Xf)**2) < eps:
        # Not enough information → return mixture
        return Xf[:n_sources]

    X_stack = np.vstack([np.real(Xf), np.imag(Xf)])

    # Remove mean (important for numerical stability)
    X_stack -= np.mean(X_stack, axis=1, keepdims=True)

    try:
        ica = FastICA(
            n_components=n_sources,
            whiten="unit-variance",
            max_iter=300,
            tol=1e-4
        )
        S_stack = ica.fit_transform(X_stack.T).T
    except Exception:
        # ICA failed → fallback
        return Xf[:n_sources]

    half = S_stack.shape[0] // 2
    Sf = S_stack[:half] + 1j * S_stack[half:]

    return Sf


def frequency_domain_ica(X_f, n_sources=2):
    """
    X_f: (M, F, T)
    returns: S_f (n_sources, F, T)
    """
    M, F, T = X_f.shape
    S_f = np.zeros((n_sources, F, T), dtype=np.complex64)

    for f in range(F):
        S_f[:, f, :] = ica_one_frequency(X_f[:, f, :], n_sources)

    return S_f

def align_permutation(S_f):
    """
    Align permutation across frequency bins using magnitude correlation
    S_f: (n_sources, F, T)
    """
    n_sources, F, T = S_f.shape

    for f in range(1, F):
        prev_mag = np.abs(S_f[:, f-1, :])
        curr_mag = np.abs(S_f[:, f, :])

        corr = np.abs(np.corrcoef(prev_mag, curr_mag)[:n_sources, n_sources:])

        if corr[0, 0] < corr[0, 1]:
            S_f[:, f, :] = S_f[::-1, f, :]

    return S_f

def istft_sources(S_f, hop=256):
    """
    S_f: (n_sources, F, T)
    returns: time-domain sources (n_sources, N)
    """
    sources = []
    for i in range(S_f.shape[0]):
        s = istft(S_f[i], hop=hop)
        sources.append(s)
    return np.array(sources)

def fd_ica_separate(X, fs, n_sources=2, n_fft=1024, hop=256):
    """
    Full frequency-domain ICA separation
    """
    X_f = stft_multichannel(X, fs, n_fft, hop)
    S_f = frequency_domain_ica(X_f, n_sources)
    S_f = align_permutation(S_f)
    S = istft_sources(S_f, hop)
    return S
