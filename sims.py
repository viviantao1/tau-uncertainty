import numpy as np
import healpy as hp
import os
import camb
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from threadpoolctl import threadpool_limits
from minkasi import util as mt
import xqml
import pixel_weighted_ilc as pilc
import argparse

# ---------------------------------------------------------------- constants
pars = camb.CAMBparams()

lmax_sim = 32
nside_sim = 128
lmax_xqml = 32
nside_xqml = 16
ls_sim = np.arange(2, lmax_sim+1)
ls_xqml = np.arange(2, lmax_xqml+1)

true_tau = 0.054
grid_min, grid_max, grid_size = 0, 0.15, 500
DELL = 3
reg_sigma = 0.02 * nside_sim / nside_xqml

filename = "/data/users/class/mapmaking_in/input_maps/mask/SRoll2_lowell_f060_nside16_C.fits"
filename128 = "/data/users/class/mapmaking_in/input_maps/mask/SRoll2_lowell_f060_nside128_C.fits"
CACHE = "cache"   # delete this folder to force a recompute

# ---------------------------------------------------------------- theory spectra
def get_Clhat(tau, lmax, raw_cl=True,
              H0=67, Ombh2=0.024, Omch2=0.12,
              As=2.1e-9, ns=0.965, ks=0.05,
              kt=0.05, Omk=0, TCMB=2.7255,
              mnu=0.06, r=0, YHe=0.245400,):
    '''
    Specify tau, and use other known parameters
    '''
    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Omch2, mnu=mnu, omk=Omk, tau=tau, TCMB=TCMB, YHe=YHe)
    pars.InitPower.set_params(As=As, ns=ns, r=r, pivot_scalar=ks, pivot_tensor=kt)
    results = camb.get_results(pars)
    powers = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)
    TT, EE, BB, TE = powers.T
    return np.array([TT, EE, BB, TE])

true_spectra_sim = get_Clhat(true_tau, lmax=lmax_sim) # ensemble average power spectrum for higher resolution maps
true_spectra_xqml = get_Clhat(true_tau, lmax=lmax_xqml) # ensemble average power spectrum for xqml

# ---------------------------------------------------------------- foregrounds (cached)
os.makedirs(CACHE, exist_ok=True)
fg_file = f"{CACHE}/foregrounds_nside{nside_sim}.npz"
if os.path.exists(fg_file):
    d = np.load(fg_file)
    total23, total93, total353 = d["total23"], d["total93"], d["total353"]
else:
    import pysm3
    import pysm3.units as u

    sky = pysm3.Sky(nside=nside_sim, preset_strings=["s0", "d0"], output_unit=u.uK_CMB)

    # Convert foregrounds to CMB units
    total23 = sky.get_emission(23 * u.GHz)
    total93 = sky.get_emission(93 * u.GHz)
    total353 = sky.get_emission(353 * u.GHz)

    total23 = np.array(total23)
    total93 = np.array(total93)
    total353 = np.array(total353)

    # Rotate foregrounds from galactic to celestial (equatorial) coordinates.
    # rotate_map_alms handles the (I, Q, U) maps via spin-2 harmonics, so the
    # polarization angles are rotated along with the pixels.
    rot_gal2cel = hp.Rotator(coord=["G", "C"])
    total23 = rot_gal2cel.rotate_map_alms(total23)
    total93 = rot_gal2cel.rotate_map_alms(total93)
    total353 = rot_gal2cel.rotate_map_alms(total353)
    np.savez(fg_file, total23=total23, total93=total93, total353=total353)

# ---------------------------------------------------------------- EE grid for xQML (cached)
tau_coarse_grid = np.linspace(grid_min, grid_max, 100)
ee_file = f"{CACHE}/EE_coarse_grid_lmax{lmax_xqml}.npz"
if os.path.exists(ee_file):
    EE_coarse_grid = np.load(ee_file)["EE_coarse_grid"]
else:
    EE_coarse_grid = np.array([
        get_Clhat(t, lmax=lmax_xqml)[1]
        for t in tau_coarse_grid
    ])
    np.savez(ee_file, EE_coarse_grid=EE_coarse_grid)

EE_cubicspline = CubicSpline(
    tau_coarse_grid,
    EE_coarse_grid,
    axis=0
)

tau_grid = np.linspace(grid_min, grid_max, grid_size)

EE_grid = EE_cubicspline(tau_grid)

lmins = np.arange(2, lmax_xqml + 1 - DELL + 1, DELL)[:(lmax_xqml - 1) // DELL]
lmaxs = lmins + DELL
lmaxs[-1] = lmax_xqml + 1  # fromdeltal drops leftover ells, unmodelled power leaks into all bins
bins = xqml.Bins(lmins, lmaxs)
ell_xqml = np.arange(2, lmax_xqml+1)
ell_bins_xqml = bins.lbin
ell_index_xqml = np.array([np.where(ls_xqml == L)[0][0] for L in ell_xqml])
ell_grid = np.arange(EE_grid.shape[1])
EE_grid_binned_xqml = bins.bin_spectra(EE_grid, Dl=True) # no weighting is done since function assumes input and output are both Dl

EE_input_xqml = np.array(true_spectra_xqml[1][:lmax_xqml + 1], copy=True)
EE_input_xqml[ell_xqml] = EE_cubicspline(0.054)[ell_xqml]
clth_xqml = np.array([
    true_spectra_xqml[0][:lmax_xqml + 1],
    EE_input_xqml,
    true_spectra_xqml[2][:lmax_xqml + 1],
    true_spectra_xqml[3][:lmax_xqml + 1]])

# ---------------------------------------------------------------- masks
def create_mask (filename, class_mask=True, nside=nside_xqml):
    m = hp.read_map(filename)
    if class_mask:
        mask_class = mt.dec_mask(nside, [-74, 28], coord='C')
        return mask_class & m
    else:
        return m

m16 = hp.read_map(filename)
m128 = hp.read_map(filename128)
# mask = create_mask(filename, class_mask=True, nside=nside_xqml)

# ---------------------------------------------------------------- ILC + map generation
def noise_calcs(noise, nside):
    npix = hp.nside2npix(nside)
    solid_angle = hp.nside2pixarea(nside)
    # map sensitivity
    sigma_pix = noise * (np.pi / 180 / 60)/ np.sqrt(solid_angle)
    Nl = 4*np.pi * sigma_pix**2 / npix

    return sigma_pix, npix, Nl

def f(a, maps):
    if len(a) == 1:
        coeffs = np.array([a[0], 1-a[0]])
    else:
        coeffs = np.concatenate((a, [1-sum(a)]))
    chat = coeffs.dot(maps)
    return chat.var()

def original_ilc(nside, s, sigma_pix, npix, Nl):
    n = np.random.randn(3, npix) * sigma_pix * np.array([[0.], [1.], [1.]])  # independent Q,U noise; T = 0

    combined23 = s + total23
    combined93 = s + total93 + n
    combined353 = s + total353

    chat = np.empty((3, npix))
    chat[0] = np.zeros(npix)
    Nl_eff = np.zeros(3)

    for j in (1,2):
        temp_arr_auto = np.array([combined23[j], combined93[j], combined353[j]])
        x0 = np.zeros(len(temp_arr_auto)-1)
        sol = minimize(f, x0, args=(temp_arr_auto))
        a = np.concatenate((sol.x, np.array([1-sum(sol.x)])))
        chat[j] = a.dot(temp_arr_auto)

        Nl_eff[j] = a[1]**2*Nl

    return chat, Nl_eff

def ilc(nside, s, sigma_pix, npix, Nl):
    # two noise splits; one set of weights, fit on their mean with the noise term removed via the half-difference
    n1 = np.random.randn(3, npix) * sigma_pix * np.array([[0.], [1.], [1.]])  # independent Q,U noise; T = 0
    n2 = np.random.randn(3, npix) * sigma_pix * np.array([[0.], [1.], [1.]])

    combined23 = s + total23
    combined93 = s + total93 + (n1 + n2) / 2 # make n1, n2 into array for noise in different channels
    combined353 = s + total353

    mean_clean = np.zeros((3, npix))
    maps = np.array([combined23, combined93, combined353])
    qu = maps[:, 1:, :]

    half_diff = np.zeros_like(qu)
    half_diff[1] = (n1[1:] - n2[1:]) / 2  # only 93 GHz has noise half_diff = (n1[:, 1:] - n2[:, 1:]) / 2

    mean_clean[1:], w1 = pilc.method2_foreground_cleaning(qu, pre=pre, half_diff=half_diff)
    noise_clean = w1[1] * (n1 - n2) / 2 # noise_clean = np.tensordot(w1, (n1 - n2) / 2, 1)

    return mean_clean + noise_clean, mean_clean - noise_clean, w1 #Nl is wrong here

def generate_maps(noise, nside, foregrounds=True):
    # generates sims with foregrounds
    sigma_pix, npix, Nl = noise_calcs(noise, nside)
    s = hp.synfast(true_spectra_sim, nside, lmax=lmax_sim, new=True, pol=True)

    if foregrounds:
        return ilc(nside, s, sigma_pix, npix, Nl)
    else:
        n1 = np.random.randn(3, npix) * sigma_pix * np.array([[0.], [1.], [1.]])  # independent Q,U noise; T = 0
        n2 = np.random.randn(3, npix) * sigma_pix * np.array([[0.], [1.], [1.]])  # independent Q,U noise; T = 0

        map1 = s + n1
        map2 = s + n2

        return map1, map2, Nl

# ---------------------------------------------------------------- downgrading
def downgrade_map_with_smoothing(m, fwhm, nside_out=nside_xqml,  **kwargs): # adds pixel window function
    '''
    fwhm should be in degrees
    '''

    fwhm_rad = np.radians(fwhm)  # Convert fwhm from degrees to radians
    # Antialias a with a Gaussian beam
    m_smooth = hp.smoothing(m, fwhm=fwhm_rad, **kwargs)

    # downgrade it to nside_out
    return hp.ud_grade(m_smooth, nside_out)

def downgrade(maps, fwhm_deg, nside_out):       
    qu = maps[1:3]
    lmax = 3 * nside_out - 1                      # 47
    alm = hp.map2alm_spin(qu, 2, lmax=lmax)       # no T
    b = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax, pol=True)
    alm = [hp.almxfl(a, b[:, 1 + i]) for i, a in enumerate(alm)]   # E uses column 1, B column 2
    return np.vstack([np.zeros(hp.nside2npix(nside_out)), hp.alm2map_spin(alm, nside_out, 2, lmax)])

# ---------------------------------------------------------------- xQML simulations
def simulation_setup(noise, mask, nside=nside_xqml, fwhm=0):
    sigma_pix, npix, _ = noise_calcs(noise, nside)

    unmasked_pix = np.count_nonzero(mask)
    NoiseVar = (sigma_pix**2 * np.eye(2*unmasked_pix))
    spec = ['EE','BB']

    esti = xqml.xQML(
        mask,
        mask,
        bins,
        clth_xqml,
        NA=NoiseVar,
        NB=NoiseVar,
        lmax=lmax_xqml,
        fwhm=fwhm,
        spec=spec,
        pixwin=False
    )

    covariance_full = esti.get_covariance()
    covariance_EE = covariance_full[0:len(ell_bins_xqml), 0:len(ell_bins_xqml)]
    icovariance_EE = np.linalg.inv(covariance_EE)
    return esti, icovariance_EE

def run_single_simulation(noise, nside_sim, nside_xqml, fwhm, icovariance_EE, esti, mask, xqml=True):
    # generate sims and foregrounds, then does ILC
    map1, map2, w1 = generate_maps(noise, nside=nside_sim, foregrounds=True)

    # downgrade maps to nside_xqml with smoothing
    map1_downgraded = downgrade(map1, fwhm, nside_out=nside_xqml)
    map2_downgraded = downgrade(map2, fwhm, nside_out=nside_xqml)

    # apply mask to downgraded maps
    map1_masked = map1_downgraded * mask
    map2_masked = map2_downgraded * mask

    if xqml:
        Clhat_cross = esti.get_spectra(map1_masked, map2_masked)[0]

        residual = Clhat_cross[None, :] - EE_grid_binned_xqml        # (ntau, nbins)
        chi2_cross = np.einsum('ti,ij,tj->t', residual, icovariance_EE, residual)  # no logdet: M is fixed

    return tau_grid[np.argmin(chi2_cross)], Clhat_cross

def run_full_simulations(noise, nside_sim, nside_xqml, fwhm, mask, num_simulations):
    tau_estimates, cls = [], []

    esti, icovariance_EE = simulation_setup(noise, mask, nside_xqml, fwhm)

    for i in range(num_simulations):
        tau_estimate, cl = run_single_simulation(noise, nside_sim, nside_xqml, fwhm, icovariance_EE, esti, mask, xqml=True)
        tau_estimates.append(tau_estimate)
        cls.append(cl)

    return np.array(tau_estimates), np.array(cls)   # (nsim,), (nsim, nbins)

# ---------------------------------------------------------------- sigma_tau plot drivers
def sigma_tau_plot(noises, nsim):
    sigma_tau = []
    bias_tau = []

    for n in noises:
        auto, _ = run_full_simulations(n, nside_sim, nside_xqml, 9, mask, nsim)
        sigma_tau.append(auto.std())
        bias_tau.append(auto.mean() - 0.054)
        print(n)

    return np.array(sigma_tau), np.array(bias_tau)

def run_noise(args):
    n, seed, nsim, threads = args
    global pre                                   # ilc() reads the global `pre`
    np.random.seed(seed)
    sigma_pix, _, _ = noise_calcs(n, nside_sim)
    pre = pilc.with_reg(pre0, sigma_pix**2 + reg_sigma**2)
    with threadpool_limits(limits=threads):
        auto, cls = run_full_simulations(n, nside_sim, nside_xqml, 9, mask, nsim)
    return auto.std(), auto.mean() - 0.054, cls.mean(axis=0), cls.std(axis=0)

def sigma_tau_plot_parallel(noises, nsim, threads=2):
    # runs one noise level at a time: forking after CAMB/pilc.setup used threads can hang
    seeds = np.random.SeedSequence(1234).spawn(len(noises))
    jobs = [(n, s.generate_state(1)[0], nsim, threads) for n, s in zip(noises, seeds)]
    res = [run_noise(job) for job in jobs]
    return [np.array([r[i] for r in res]) for i in range(4)]   # sigma, bias, cl_mean, cl_std

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of the output .npz file")
    parser.add_argument("--save-spectra", action="store_true", help="also save mean/std of the binned EE spectrum per noise level")
    parser.add_argument("--class-mask", action="store_true", help="add the CLASS declination cut")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.filename) or ".", exist_ok=True)  # before the long run, so a bad path fails right away

    mask = create_mask(filename, class_mask=args.class_mask, nside=nside_xqml)

    # Gram matrices built once and reused for every noise level
    pre0 = pilc.setup(m128, nside_sim, true_spectra_sim[1:3], lmax=32, spin=2, reg=1.0)  # reg is a placeholder

    noises = np.arange(0,110,10, dtype=float)
    noises[0] = 0.1  # avoid zero noise for xQML
    sigma, bias, cl_mean, cl_std = sigma_tau_plot_parallel(noises, 200, threads=8)
    extra = dict(ell_bins=ell_bins_xqml, cl_mean=cl_mean, cl_std=cl_std) if args.save_spectra else {}
    np.savez(args.filename, noises=noises, sigma=sigma, bias=bias, **extra)


# usage: OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 python sims.py results.npz --save-spectra --class-mask