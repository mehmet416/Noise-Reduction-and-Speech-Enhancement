import numpy as np
from sklearn.decomposition import FastICA

def fastica_separate(X, n_sources=2, random_state=0):
    """
    X : (M x N) multichannel signal
    returns: separated sources (n_sources x N)
    """
    # Centering X
    X = X - np.mean(X, axis=1, keepdims=True)

    #FastICA
    ica = FastICA(
        n_components=n_sources,
        whiten="unit-variance",
        max_iter=1000,
        tol=1e-5,
        random_state=random_state
    )

    S_hat = ica.fit_transform(X.T).T
    return S_hat

