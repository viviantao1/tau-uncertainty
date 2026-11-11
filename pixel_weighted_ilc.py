"""
Global pixel-space ILC with the theoretical CMB covariance.
Sudevan & Saha (2017), arXiv:1712.09804, Method 2 (Sec. 5.3.2).

C = A A^T with A[p, j] = sqrt(Cl) Y_j(p) over real harmonics j = (l, m), 2 <= l <= lmax
Then or (C + reg I)^-1, via Woodbury only needs the Gram G = A^T A and the projections X A, both from spherical-harmonic transforms, so no npix x npix matrix is ever built

Q and U are cleaned together as one vector with one set of weights, half difference only changes noise (same weights)
"""
import healpy as hp
import numpy as np


def cmb_spectrum(nside, cl, fwhm_deg=9.0, lmax=None, cutoff=0.0, spin=0):
    # prepare Cl for pixel space

    lmax = 2 * nside if lmax is None else lmax
    cl = np.atleast_2d(np.array(cl, dtype=float))[:, : lmax + 1] # get into shape (2, lmax) = (EE, BB) for spin 2
    c = cl
    c[:, :2] = 0 # no monopole/dipole terms
    return c[:, : np.nonzero(c.max(axis=0) >= 1e-3 * cutoff * c.max())[0].max() + 1] # trim where pinv would discard anyway

def _project(maps, c, nside):
    # A is the sqrt(Cl) Y_lm for each component (one row per pixel, one (l,m) per column) 
    # so X A = sum_p x(p) A(p) = sum_p x(p) sqrt(Cl) Y_lm(p) (dot product of each map with each (l,m))
    lmax = c.shape[1] - 1
    l, m = hp.Alm.getlm(lmax)
    keep = l >= 2
    mpos = m[keep] > 0
    out = []
    for x in maps:
        row = []
        for a, ck in zip(hp.map2alm_spin(x, 2, lmax=lmax), c): # pairs coefficients with the corresponding Cl for each component (E, B), map to alm, with sqrt(Cl) weighting
            a = a[keep] * np.sqrt(ck[l[keep]]) / hp.nside2pixarea(nside) # weight by sqrt(Cl) and pixel area
            row += [a[~mpos].real, np.sqrt(2) * a[mpos].real, np.sqrt(2) * a[mpos].imag] # dot product with each (l,m) mode: m=0, Re(m>0), Im(m>0)
        out.append(np.concatenate(row))
    return np.array(out)

def gram(mask, c, nside):
    # G = A_m^T A_m for the pixels with mask
    lmax = c.shape[1] - 1
    l, m = hp.Alm.getlm(lmax)
    m0, mpos = np.nonzero((l >= 2) & (m == 0))[0], np.nonzero((l >= 2) & (m > 0))[0]
    # same column order as _project: per component, m = 0, then Re of m > 0, then Im of m > 0
    modes = [(i, 1.0) for i in m0] + [(i, 1 / np.sqrt(2)) for i in mpos] + [(i, 1j / np.sqrt(2)) for i in mpos]
    m_pix = (mask > 0.5).astype(float)
    alms = np.zeros((len(c), hp.Alm.getsize(lmax)), dtype=complex)
    cols = []
    for k in range(len(c)):
        for i, a in modes: # iterate through each (l,m) mode, compute the dot product of the map with that mode, weighted by sqrt(Cl)
            alms[k, i] = a * np.sqrt(c[k, l[i]])
            cols.append(_project([m_pix * np.array(hp.alm2map_spin(alms, nside, 2, lmax))], c, nside)[0]) 
            alms[k, i] = 0
    return np.array(cols).T

def mode_inverse(G, cutoff=1e-7, reg=None):
    if reg is not None:
        return np.linalg.inv(G + reg * np.eye(len(G))) # Woodbury: (G + reg I)^-1 = (reg I)^-1 - (reg I)^-1 A (G + reg I)^-1 A^T (reg I)^-1
    lam, V = np.linalg.eigh(G) # nonzero eigenvalues of G = those of C
    keep = lam > cutoff * lam.max()
    return (V[:, keep] / lam[keep]**2) @ V[:, keep].T

def _cov(X, XA, M, reg):
    # X C^-1 X^T for X zero outside the pixels used, XA = X A, M from mode_inverse
    X = X.reshape(len(X), -1) # combine Q and U into one vector, X is (number of maps, 2Npix)
    return XA @ M @ XA.T if reg is None else (X @ X.T - XA @ M @ XA.T) / reg   # Eq 4

def ilc(X, XA, M, reg=None, Y=None, YA=None):
    A = _cov(X, XA, M, reg)
    if Y is not None: # half-difference maps Y (and YA = Y A): noise term Y C^-1 Y^T is subtracted so the weights minimise the foreground residual
        A = A - _cov(Y, YA, M, reg)
        T = np.vstack([-np.ones(len(A)-1), np.eye(len(A)-1)])
        if np.linalg.eigvalsh(T.T @ A @ T)[0] <= 0:
            raise RuntimeError("debiased ILC cost not positive definite: noise too large or foregrounds degenerate")
    w = np.linalg.pinv(A).sum(axis=0) # inverse of e A^+ is A^+ e
    w /= w.sum() # Eq 5, weights sum to 1
    return w

def setup(mask, nside, cl, fwhm_deg=9.0, cutoff=1e-7, lmax=None, reg=None, spin=0):
    # meant to be computed once and reused for many simulations with the same mask, nside, and Cl
    c = cmb_spectrum(nside, cl, fwhm_deg, lmax, cutoff if reg is None else 0.0, spin) # builds spectrum
    m = mask > 0.5
    G1 = gram(m, c, nside) # G = A_m^T A_m for the pixels with mask, only on the mask and Cl (no noise)
    return dict(c=c, m=m, nside=nside, reg=reg, spin=spin, cutoff=cutoff, G1=G1,
                M1=mode_inverse(G1, cutoff, reg))

def with_reg(pre, reg):
    assert pre["reg"] is not None and reg is not None, "with_reg needs setup(reg=...) and a reg" # DOES NOT WORK with pre from pinv mode
    return dict(pre, reg=reg, M1=mode_inverse(pre["G1"], pre["cutoff"], reg)) # copies pre and replaces reg and M1, reuses cached Gram G1

def method2_foreground_cleaning(maps, mask=None, nside=None, cl=None, fwhm_deg=9.0, cutoff=1e-7, lmax=None,
                                reg=None, spin=2, pre=None, half_diff=None):
    # Pass pre=setup() to skip the (mask only) Gram computation for repeat simulations
    # half_diff: optional (split1 - split2) / 2 maps, same shape as maps, maps must be the split mean (x1 + x2) / 2. Use reg (not reg=None) with half_diff
    pre = setup(mask, nside, cl, fwhm_deg, cutoff, lmax, reg, spin) if pre is None else pre # reuse precomputed Gram and mode_inverse
    c, m, nside, reg, spin = pre["c"], pre["m"], pre["nside"], pre["reg"], pre["spin"]
    X = np.array(maps, dtype=float).reshape(len(maps), len(c), -1)
    Xm, Ym = X * m, None # applies mask
    if half_diff is not None: # processing half difference maps int he same way as the main maps to compute the noise covariance
        Y = np.array(half_diff, dtype=float).reshape(X.shape)
        if spin == 0:
            Y -= Y.mean(axis=2, keepdims=True)
        Ym = Y * m
    w1 = ilc(Xm, _project(Xm, c, nside), pre["M1"], reg, Ym, None if Ym is None else _project(Ym, c, nside)) # compute Xm A, Ym A, and solve for weights
    y = np.tensordot(w1, X, 1) # applies weights to full maps --> cleaned maps
    return y, w1
