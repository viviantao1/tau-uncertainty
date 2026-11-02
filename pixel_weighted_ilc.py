"""
Global pixel-space ILC with the theoretical CMB covariance.
Sudevan & Saha (2017), arXiv:1712.09804, Method 2 (Sec. 5.3.2).

C = A A^T with A[p, j] = sqrt(Cl B_l^2 p_l^2) Y_j(p) over real harmonics j = (l, m), 2 <= l <= lmax.
Then pinv(C) (or (C + reg I)^-1, via Woodbury) only needs the K x K Gram G = A^T A and the
projections X A, both from spherical-harmonic transforms, so no npix x npix matrix is ever built.

spin=0: one map per frequency (T), cl = Cl_TT.
spin=2: (Q, U) per frequency, cl = (Cl_EE, Cl_BB); the j run over E and B modes, and Q and U are
cleaned together as one vector with one set of weights.
"""
import healpy as hp
import numpy as np


def cmb_spectrum(nside, cl, fwhm_deg=9.0, lmax=None, cutoff=0.0, spin=0):
    """Beam- and pixel-window-smoothed Cl, shape (1, L) for spin 0 or (2, L) = (EE, BB) for spin 2.
    With cutoff > 0 it is truncated where it drops far below the pinv cutoff, since pinv would
    discard those modes anyway."""
    lmax = 2 * nside if lmax is None else lmax
    cl = np.atleast_2d(np.asarray(cl, dtype=float))[:, : lmax + 1]
    if spin == 0:
        beam = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax) * hp.pixwin(nside, lmax=lmax)
    else:
        b = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax, pol=True)
        beam = np.array([b[:, 1], b[:, 2]]) * hp.pixwin(nside, lmax=lmax, pol=True)[1]
    assert len(cl) == len(np.atleast_2d(beam)), "spin=2 needs cl = (Cl_EE, Cl_BB)"
    # c = cl * beam**2
    c = cl
    c[:, :2] = 0  # no monopole/dipole
    return c[:, : np.nonzero(c.max(axis=0) >= 1e-3 * cutoff * c.max())[0].max() + 1]


def _analysis(x, lmax):
    """Pixel sum Omega * sum_p x(p) Y*_lm(p) for each component (no iterations, so exact)."""
    if len(x) == 1:
        return [hp.map2alm(x[0], lmax=lmax, iter=0)]
    return hp.map2alm_spin(x, 2, lmax=lmax)


def _synthesis(alms, nside, lmax):
    if len(alms) == 1:
        return hp.alm2map(alms[0], nside, lmax=lmax)[None]
    return np.array(hp.alm2map_spin(alms, nside, 2, lmax))


def _project(maps, c, nside):
    """X A: dot product of each (n_comp, npix) map with every column of A, via map2alm."""
    lmax = c.shape[1] - 1
    l, m = hp.Alm.getlm(lmax)
    keep = l >= 2
    mpos = m[keep] > 0
    out = []
    for x in maps:
        row = []
        for a, ck in zip(_analysis(x, lmax), c):
            a = a[keep] * np.sqrt(ck[l[keep]]) / hp.nside2pixarea(nside)
            row += [a[~mpos].real, np.sqrt(2) * a[mpos].real, np.sqrt(2) * a[mpos].imag]
        out.append(np.concatenate(row))
    return np.array(out)


def gram(mask, c, nside):
    """G = A_m^T A_m for the pixels where mask > 0.5: one synthesis + analysis per mode."""
    lmax = c.shape[1] - 1
    l, m = hp.Alm.getlm(lmax)
    m0, mpos = np.nonzero((l >= 2) & (m == 0))[0], np.nonzero((l >= 2) & (m > 0))[0]
    # same column order as _project: per component, m = 0, then Re of m > 0, then Im of m > 0
    modes = [(i, 1.0) for i in m0] + [(i, 1 / np.sqrt(2)) for i in mpos] + [(i, 1j / np.sqrt(2)) for i in mpos]
    m_pix = (mask > 0.5).astype(float)
    alms = np.zeros((len(c), hp.Alm.getsize(lmax)), dtype=complex)
    cols = []
    for k in range(len(c)):
        for i, a in modes:
            alms[k, i] = a * np.sqrt(c[k, l[i]])
            cols.append(_project([m_pix * _synthesis(alms, nside, lmax)], c, nside)[0])
            alms[k, i] = 0
    return np.array(cols).T


def mode_inverse(G, cutoff=1e-7, reg=None):
    """M such that x^T C^-1 y = (x A) M (A^T y)            for reg=None (C^-1 = pinv(C), rcond=cutoff),
    or x^T C^-1 y = (x.y - (x A) M (A^T y)) / reg          for C -> C + reg I (Woodbury)."""
    if reg is not None:
        return np.linalg.inv(G + reg * np.eye(len(G)))
    lam, V = np.linalg.eigh(G)                  # nonzero eigenvalues of G = those of C
    keep = lam > cutoff * lam.max()
    return (V[:, keep] / lam[keep]**2) @ V[:, keep].T


def ilc(X, XA, M, reg=None):
    """One ILC pass: X (n_freq, ...) zero outside the pixels used, XA = X A, M from mode_inverse."""
    X = X.reshape(len(X), -1)
    A = XA @ M @ XA.T if reg is None else (X @ X.T - XA @ M @ XA.T) / reg   # Eq. 4
    w = np.linalg.pinv(A).sum(axis=0)           # e A^+
    w /= w.sum()                                # Eq. 5, weights sum to 1
    return w


def setup(mask, nside, cl, fwhm_deg=9.0, cutoff=1e-7, lmax=None, reg=None, spin=0):
    """Everything that depends only on the mask and Cl; compute once and reuse across simulations.
    reg=None uses pinv(C) with rcond=cutoff; otherwise (C + reg I)^-1 with reg a per-pixel variance
    in the units of cl (e.g. white-noise variance, as in Anping's 2 noise^2 / pixel_area + reg_sigma^2)."""
    c = cmb_spectrum(nside, cl, fwhm_deg, lmax, cutoff if reg is None else 0.0, spin)
    m = mask > 0.5
    return dict(c=c, m=m, nside=nside, reg=reg, spin=spin,
                M1=mode_inverse(gram(m, c, nside), cutoff, reg),
                M2=mode_inverse(gram(np.ones_like(m), c, nside), cutoff, reg))


def method2_foreground_cleaning(maps, mask=None, nside=None, cl=None, fwhm_deg=9.0, cutoff=1e-7, lmax=None,
                                reg=None, spin=0, pre=None):
    """maps RING, (n_freq, npix) for spin 0 or (n_freq, 2, npix) = (Q, U) for spin 2; mask 1 = clean
    region cleaned first. Returns (full-sky map, w1, w2); the map is (npix,) or (2, npix).
    Pass pre=setup(...) to skip the (mask-only) Gram computation on repeated calls."""
    pre = setup(mask, nside, cl, fwhm_deg, cutoff, lmax, reg, spin) if pre is None else pre
    c, m, nside, reg, spin = pre["c"], pre["m"], pre["nside"], pre["reg"], pre["spin"]
    X = np.array(maps, dtype=float).reshape(len(maps), len(c), -1)
    if spin == 0:
        X -= X.mean(axis=2, keepdims=True)      # monopole; Q/U have no l < 2 part to remove
    Xm = X * m
    w1 = ilc(Xm, _project(Xm, c, nside), pre["M1"], reg)
    X[:, :, m] = np.tensordot(w1, X[:, :, m], 1)  # same cleaned map in every channel
    w2 = ilc(X, _project(X, c, nside), pre["M2"], reg)
    y = np.tensordot(w2, X, 1)
    return (y[0] if spin == 0 else y), w1, w2
