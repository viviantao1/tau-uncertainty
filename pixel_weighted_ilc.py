"""
Global pixel-space ILC with the theoretical CMB covariance.
Sudevan & Saha (2017), arXiv:1712.09804, Method 2 (Sec. 5.3.2).

C = A A^T with A[p, j] = sqrt(Cl) Y_j(p) over real harmonics j = (l, m), 2 <= l <= lmax.
Then pinv(C) (or (C + reg I)^-1, via Woodbury) only needs the K x K Gram G = A^T A and the
projections X A, both from spherical-harmonic transforms, so no npix x npix matrix is ever built.

spin=0: one map per frequency (T), cl = Cl_TT.
spin=2: (Q, U) per frequency, cl = (Cl_EE, Cl_BB); the j run over E and B modes, and Q and U are
cleaned together as one vector with one set of weights.
"""
import healpy as hp
import numpy as np


def cmb_spectrum(nside, cl, fwhm_deg=9.0, lmax=None, cutoff=0.0, spin=0):
    """Cl with monopole and dipole zeroed, shape (1, L) for spin 0 or (2, L) = (EE, BB) for spin 2.
    With cutoff > 0 it is truncated where it drops far below the pinv cutoff, since pinv would
    discard those modes anyway."""
    lmax = 2 * nside if lmax is None else lmax
    cl = np.atleast_2d(np.array(cl, dtype=float))[:, : lmax + 1]
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


def _cov(X, XA, M, reg):
    """X C^-1 X^T for X (n_freq, ...) zero outside the pixels used, XA = X A, M from mode_inverse."""
    X = X.reshape(len(X), -1)
    return XA @ M @ XA.T if reg is None else (X @ X.T - XA @ M @ XA.T) / reg   # Eq. 4


def ilc(X, XA, M, reg=None, Y=None, YA=None):
    """One ILC pass. With half-difference maps Y (and YA = Y A), their noise term Y C^-1 Y^T is
    subtracted so the weights minimise the foreground residual only, not foreground + noise."""
    A = _cov(X, XA, M, reg)
    if Y is not None:
        A = A - _cov(Y, YA, M, reg)
        T = np.vstack([-np.ones(len(A)-1), np.eye(len(A)-1)])
        if np.linalg.eigvalsh(T.T @ A @ T)[0] <= 0:
            raise RuntimeError("debiased ILC cost not positive definite: noise too large or foregrounds degenerate")
    w = np.linalg.pinv(A).sum(axis=0)           # e A^+
    w /= w.sum()                                # Eq. 5, weights sum to 1
    return w


def setup(mask, nside, cl, fwhm_deg=9.0, cutoff=1e-7, lmax=None, reg=None, spin=0):
    """Everything that depends only on the mask and Cl; compute once and reuse across simulations.
    reg=None uses pinv(C) with rcond=cutoff; otherwise (C + reg I)^-1 with reg a per-pixel variance
    in the units of cl (e.g. white-noise variance, 2 noise^2 / pixel_area + reg_sigma^2)."""
    c = cmb_spectrum(nside, cl, fwhm_deg, lmax, cutoff if reg is None else 0.0, spin)
    m = mask > 0.5
    G1 = gram(m, c, nside)   # the expensive part; independent of reg
    return dict(c=c, m=m, nside=nside, reg=reg, spin=spin, cutoff=cutoff, G1=G1,
                M1=mode_inverse(G1, cutoff, reg))


def with_reg(pre, reg):
    """Copy of pre for a new reg: reuses the cached Gram matrix, redoes only the inverse.
    pre must come from setup(..., reg=<something>) so that c was built without the pinv cutoff."""
    assert pre["reg"] is not None and reg is not None, "with_reg needs setup(reg=...) and a reg"
    return dict(pre, reg=reg, M1=mode_inverse(pre["G1"], pre["cutoff"], reg))


def method2_foreground_cleaning(maps, mask=None, nside=None, cl=None, fwhm_deg=9.0, cutoff=1e-7, lmax=None,
                                reg=None, spin=0, pre=None, half_diff=None):
    """maps RING, (n_freq, npix) for spin 0 or (n_freq, 2, npix) = (Q, U) for spin 2; mask 1 = clean
    region where the weights are fit. Returns (full-sky map, w1): w1 from the ILC on the masked
    sky, applied to the full sky; the map is (npix,) or (2, npix).
    Pass pre=setup(...) to skip the (mask-only) Gram computation on repeated calls.
    half_diff: optional (split1 - split2) / 2 maps, same shape as maps; pure noise, used to remove
    the noise term from the ILC covariance so only the foreground residual is minimised. With
    half_diff = (x1 - x2) / 2, maps must be the split mean (x1 + x2) / 2. Use reg (not reg=None)
    with half_diff."""
    pre = setup(mask, nside, cl, fwhm_deg, cutoff, lmax, reg, spin) if pre is None else pre
    c, m, nside, reg, spin = pre["c"], pre["m"], pre["nside"], pre["reg"], pre["spin"]
    X = np.array(maps, dtype=float).reshape(len(maps), len(c), -1)
    if spin == 0:
        X -= X.mean(axis=2, keepdims=True)      # monopole; Q/U have no l < 2 part to remove
    Xm, Ym = X * m, None
    if half_diff is not None:
        Y = np.array(half_diff, dtype=float).reshape(X.shape)
        if spin == 0:
            Y -= Y.mean(axis=2, keepdims=True)
        Ym = Y * m
    w1 = ilc(Xm, _project(Xm, c, nside), pre["M1"], reg, Ym, None if Ym is None else _project(Ym, c, nside))
    y = np.tensordot(w1, X, 1)
    return (y[0] if spin == 0 else y), w1
