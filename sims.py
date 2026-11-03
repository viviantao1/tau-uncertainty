import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import os
import camb
from camb import model, initialpower
from scipy.optimize import minimize
from corner import corner
from minkasi.null_test.estimators import polspice
from minkasi import util as mt
import xqml
from astropy.io import fits

pars = camb.CAMBparams()

lmax_sim = 32
nside_sim = 128
lmax_xqml = 32
nside_xqml = 16
ls_sim = np.arange(2, lmax_sim+1)
ls_xqml = np.arange(2, lmax_xqml+1)

def noise_calcs(noise, nside):
    npix = hp.nside2npix(nside)
    solid_angle = hp.nside2pixarea(nside)
    # map sensitivity
    sigma_pix = noise * (np.pi / 180 / 60)/ np.sqrt(solid_angle)
    Nl = 4*np.pi * sigma_pix**2 / npix
    
    return sigma_pix, npix, Nl

def original_ilc(nside, s, sigma_pix, npix, Nl):
    n = np.random.randn(npix) * sigma_pix

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
    n = np.random.randn(npix) * sigma_pix

    combined23 = s + total23
    combined93 = s + total93 + n
    combined353 = s + total353

    chat = np.empty((3, npix))
    chat[0] = np.zeros(npix)
    Nl_eff = np.zeros(3)

    maps = np.array([combined23, combined93, combined353])
    qu = maps[:, 1:, :]

    (Q, U), w1, w2 = pilc.method2_foreground_cleaning(qu, pre=pre)
    chat[1] = Q
    chat[2] = U

    return chat, Nl #Nl is wrong here

def generate_maps(noise, nside, foregrounds=True):
    # generates sims with foregrounds
    sigma_pix, npix, Nl = noise_calcs(noise, nside)
    s = hp.synfast(true_spectra_sim, nside, lmax=lmax_sim, new=True, pol=True)

    if foregrounds:
        m1, Nl1 = ilc(nside, s, sigma_pix, npix, Nl)
        m2, Nl2 = ilc(nside, s, sigma_pix, npix, Nl)

        return m1, m2, Nl1, Nl2
    else:
        n1 = np.random.randn(npix) * sigma_pix
        n2 = np.random.randn(npix) * sigma_pix 

        map1 = s + n1
        map2 = s + n2

        return map1, map2, Nl, Nl

def f(a, maps):
    if len(a) == 1:
        coeffs = np.array([a[0], 1-a[0]])
    else:
        coeffs = np.concatenate((a, [1-sum(a)]))
    chat = coeffs.dot(maps)
    return chat.var()

def downgrade_map_with_smoothing(m, fwhm, nside_out=nside_xqml,  **kwargs):
    '''
    fwhm should be in degrees
    '''

    fwhm_rad = np.radians(fwhm)  # Convert fwhm from degrees to radians
    # Antialias a with a Gaussian beam
    m_smooth = hp.smoothing(m, fwhm=fwhm_rad, **kwargs)

    # downgrade it to nside_out
    return hp.ud_grade(m_smooth, nside_out)

def downgrade_qu(qu, fwhm_deg, nside_out):       # qu: (2, npix); T is all zeros, so skip it
    lmax = 3 * nside_out - 1                      # 47
    alm = hp.map2alm_spin(qu, 2, lmax=lmax)       # no T, no l up to 383
    b = hp.gauss_beam(np.radians(fwhm_deg), lmax=lmax, pol=True)
    alm = [hp.almxfl(a, b[:, 1 + i]) for i, a in enumerate(alm)]   # E uses column 1, B column 2
    return np.array(hp.alm2map_spin(alm, nside_out, 2, lmax))

def simulation_setup(noise, nside=nside_xqml):
    sigma_pix, npix, _ = noise_calcs(noise, nside)

    unmasked_pix = np.count_nonzero(m16)
    NoiseVar = (sigma_pix**2 * np.eye(2*unmasked_pix))
    spec = ['EE','BB']

    esti = xqml.xQML(
        m16,
        m16,
        bins,
        clth_xqml,
        NA=NoiseVar,
        NB=NoiseVar,
        lmax=lmax_xqml,
        fwhm=0,
        spec=spec,
        pixwin=False
    )

    covariance_full = esti.get_covariance() 
    covariance_EE = covariance_full[0:len(ell_bins_xqml), 0:len(ell_bins_xqml)]
    icovariance_EE = np.linalg.inv(covariance_EE)
    return esti, icovariance_EE

def create_mask (filename, class_mask=True, nside=nside_xqml):
    m = hp.read_map(filename)
    if class_mask:
        mask_class = mt.dec_mask(nside, [-74, 28], coord='C')
        return mask_class & m
    else:
        return m

def simulation_setup(noise, nside=nside_xqml):
    sigma_pix, npix, _ = noise_calcs(noise, nside)

    unmasked_pix = np.count_nonzero(m16)
    NoiseVar = (sigma_pix**2 * np.eye(2*unmasked_pix))
    spec = ['EE','BB']

    esti = xqml.xQML(
        m16,
        m16,
        bins,
        clth_xqml,
        NA=NoiseVar,
        NB=NoiseVar,
        lmax=lmax_xqml,
        fwhm=0,
        spec=spec,
        pixwin=False
    )

    covariance_full = esti.get_covariance() 
    covariance_EE = covariance_full[0:len(ell_bins_xqml), 0:len(ell_bins_xqml)]
    icovariance_EE = np.linalg.inv(covariance_EE)
    return esti, icovariance_EE

def create_mask (filename, class_mask=True, nside=nside_xqml):
    m = hp.read_map(filename)
    if class_mask:
        mask_class = mt.dec_mask(nside, [-74, 28], coord='C')
        return mask_class & m
    else:
        return m

def run_full_simulations(noise, nside_sim, nside_xqml, fwhm, mask, num_simulations):
    tau_estimates = []

    esti, icovariance_EE = simulation_setup(noise, nside_xqml)

    for i in range(num_simulations):
        tau_estimate = run_single_simulation(noise, nside_sim, nside_xqml, fwhm, icovariance_EE, esti, mask, xqml=True)
        tau_estimates.append(tau_estimate)

    return np.array(tau_estimates)