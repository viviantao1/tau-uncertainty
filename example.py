# precaculate 77 modes at lower ell to speed up 
def build_ilc_basis(cl_fid, lmax_pre=8):
    basis = np.empty((2 * mask_ilc.sum(), (lmax_pre + 1)**2 - 4), order="F")
    alms = np.zeros((2, hp.Alm.getsize(lmax_pre)), dtype=complex)
    column = 0

    for ell in range(2, lmax_pre + 1):
        for m in range(ell + 1):
            index = hp.Alm.getidx(lmax_pre, ell, m)
            amplitudes = ((1.,) if m == 0 else (1 / np.sqrt(2), 1j / np.sqrt(2)))
            
            for amplitude in amplitudes:
                alms[0, index] = amplitude * np.sqrt(cl_fid[1, ell])
                qu = hp.alm2map_spin(alms, nside_sim, 2, lmax_pre)
                basis[:, column] = np.concatenate([q[mask_ilc] for q in qu])
                alms[0, index] = 0
                column += 1

    with threadpool_limits(limits=nthreads, user_api="blas"):
        gram = basis.T @ basis

    return basis, gram

def make_ilc_inverse(variance, ilc_model):
    cl_fid, ilc_basis, ilc_gram = ilc_model
    size = 2 * mask_ilc.sum()
    spectra = cl_fid[1:3, :lmax_sim + 1] / hp.nside2pixarea(nside_sim)
    qu = np.zeros((2, hp.nside2npix(nside_sim)))

    # apply full cov. w/out store dense pixel matrix
    def covariance_product(v):
        qu[:, mask_ilc] = v.reshape(2, -1)

        # eq. 2
        alms = hp.map2alm_spin(qu, 2, lmax=lmax_sim)
        signal = hp.alm2map_spin([hp.almxfl(a, cl) for a, cl in zip(alms, spectra)], nside_sim, 2, lmax_sim)
        return (np.concatenate([q[mask_ilc] for q in signal]) + variance * v.ravel())

    # low ell approx. only to accelerate convergence
    factor = cho_factor(ilc_gram + variance * np.eye(len(ilc_gram)))

    def precondition(v):
        a = cho_solve(factor, ilc_basis.T @ v, check_finite=False)
        return (v - ilc_basis @ a) / variance

    covariance = LinearOperator((size, size), matvec=covariance_product, dtype=float)
    preconditioner = LinearOperator((size, size), matvec=precondition, dtype=float)

    def solve(rhs):
        result, info = cg(
            covariance, rhs,
            M=preconditioner,
            tol=1e-8,
            atol=0.0,
            maxiter=1000)
        if info != 0:
            raise RuntimeError(f"inverse x converge: {info}")
        return result

    return solve

def make_ilc_solver(noise, ilc_model):
    variance = 2 * noise[1]**2 / pixel_area + (0.02 * nside_sim / nside_analysis)**2
    solve = make_ilc_inverse(variance, ilc_model)
    varying = (noise[0] > 0) | (noise[1:] > 0)
    inverse_D = None

    def get_weights(mean_maps, half_difference):
        nonlocal inverse_D
        X = mean_maps[:, 1:, mask_ilc].reshape(3, -1).T
        D = X[:, 1:] - X[:, [0]]

        if inverse_D is None:
            inverse_D = np.column_stack([solve(d) for d in D.T])
        else:
            for k in np.flatnonzero(varying):
                inverse_D[:, k] = solve(D[:, k])

        H = D.T @ inverse_D
        rhs = -inverse_D.T @ X[:, 0]

        if np.any(noise):
            Y = half_difference[:, 1:, mask_ilc].reshape(3, -1).T
            E = Y[:, 1:] - Y[:, [0]]
            inverse_E = np.column_stack([
                solve(E[:, k]) if varying[k] else np.zeros(len(E))
                for k in range(2)])
            H -= E.T @ inverse_E
            rhs += inverse_E.T @ Y[:, 0]

        H = (H + H.T) / 2
        if np.linalg.eigvalsh(H)[0] <= 0:
            raise RuntimeError("fit x pos. def.")

        coefficients = np.linalg.solve(H, rhs)
        return np.r_[1 - coefficients.sum(), coefficients]

    return get_weights