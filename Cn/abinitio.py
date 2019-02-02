import numpy as np
import finufftpy

def cryo_pft(p, n_r, n_theta):
    """
    Compute the polar Fourier transform of projections with resolution n_r in the radial direction
    and resolution n_theta in the angular direction.
    :param p:
    :param n_r: Number of samples along each ray (in the radial direction).
    :param n_theta: Angular resolution. Number of Fourier rays computed for each projection.
    :return:
    """
    if n_theta % 2 == 1:
        raise ValueError('n_theta must be even')

    n_projs = p.shape[2]
    omega0 = 2 * np.pi / (2 * n_r - 1)
    dtheta = 2 * np.pi / n_theta

    freqs = np.zeros((2, n_r * n_theta // 2))
    for i in range(n_theta // 2):
        freqs[0, i * n_r: (i + 1) * n_r] = np.arange(n_r) * np.sin(i * dtheta)
        freqs[1, i * n_r: (i + 1) * n_r] = np.arange(n_r) * np.cos(i * dtheta)

    freqs *= omega0
    # finufftpy require it to be aligned in fortran order
    pf = np.empty((n_r * n_theta // 2, n_projs), dtype='complex128', order='F')
    finufftpy.nufft2d2many(freqs[0], freqs[1], pf, 1, 1e-15, p)
    pf = pf.reshape((n_r, n_theta // 2, n_projs), order='F')
    pf = np.concatenate((pf, pf.conj()), axis=1).copy()
    return pf, freqs