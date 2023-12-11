# Copyright 20220320 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib

import os
import numpy as np
from astropy.table import Table
from scipy.interpolate import griddata

def _nan_array(n):
    """Creates an NaN array."""
    out = np.empty(n)
    out.fill(np.nan)
    return out

def grid_interpolate_1d(x, z, eval_x):
    """This is a utility for interpolating a 1D function z(x) linearly to
    values eval_x, but also enabling extrapolation beyond the (x) bounds using
    the nearest neighbor method.
    """
    result = griddata(x, z, eval_x, method="linear")
    nn_result = griddata(x, z, eval_x, method="nearest")
    mask = np.isnan(result)
    result[mask] = nn_result[mask]
    return result


def grid_interpolate_2d(x, y, z, eval_x, eval_y):
    """This is a utility for interpolating a 2D function z(x, y) linearly to
    values (x, y) = (eval_x, eval_y), but also enabling extrapolation beyond
    the (x, y) bounds using the nearest neighbor method.
    """
    result = griddata((x, y), z, (eval_x, eval_y), method="linear")
    nn_result = griddata((x, y), z, (eval_x, eval_y), method="nearest")
    mask = np.isnan(result)
    result[mask] = nn_result[mask]
    return result


def m_func(x, b, c, d, e):
    """Empirically-motivated model we are trying to fit for m(SNR, res).

    Args:
        x (ndarray):            x[0,:] --- SNR, x[1,:] --- resolution.
    Returns:
        model (ndarray):        multiplicative bias
    """
    model = (x[0, :] / 20.0) ** d
    model *= b * ((x[1, :] / 0.5) ** c)
    return model + e


def mwt_func(x, b, c, d, e):
    """Empirically-motivated model we are trying to fit for selection bias m(SNR, res).

    Args:
        x (ndarray):            x[0,:] --- SNR, x[1,:] --- resolution.
        b,c,d,e (float):        fitting params
    Returns:
        model (ndarray):        multiplicative weight bias
    """
    model = b + (c + (x[0, :] / 20.0) ** e) / (x[1, :] + d)
    return model


def a_func(x, b, c, d):
    """Empirically-motivated model we are trying to fit for a(SNR, res).

    Args:
        x (ndarray):            x[0,:] --- SNR, x[1,:] --- resolution.
    Returns:
        model (ndarray):        fractional additive bias
    """
    model = b * (x[1, :] - c)
    model *= (x[0, :] / 20.0) ** d
    return model


def get_snr(catalog):
    """This utility computes the S/N for each object in the catalog, based on
    cmodel_flux. It does not impose any cuts and returns NaNs for invalid S/N
    values.
    """
    if "snr" in catalog.dtype.names:
        return catalog["snr"]
    elif "i_cmodel_fluxsigma" in catalog.dtype.names:  # s18
        snr = catalog["i_cmodel_flux"] / catalog["i_cmodel_fluxsigma"]
    elif "iflux_cmodel" in catalog.dtype.names:  # s15
        snr = catalog["iflux_cmodel"] / catalog["iflux_cmodel_err"]
    elif "i_cmodel_fluxerr" in catalog.dtype.names:  # s19
        snr = catalog["i_cmodel_flux"] / catalog["i_cmodel_fluxerr"]
    elif "modelfit_CModel_instFlux" in catalog.dtype.names:  # pipe 7
        snr = (
            catalog["modelfit_CModel_instFlux"] / catalog["modelfit_CModel_instFluxErr"]
        )
    else:
        snr = _nan_array(len(catalog))
    return snr


def get_photo_z(catalog, method_name):
    """Returns the best photon-z estimation

    Args:
        catalog (ndarray):  input catlog
        method_name:        name of the photo-z method (mizuki, dnn, demp)
    Returns:
        z (ndarray):        photo-z best estimation
    """
    if method_name == "mizuki":
        if "mizuki_photoz_best" in catalog.dtype.names:
            z = catalog["mizuki_photoz_best"]
        elif "mizuki_Z" in catalog.dtype.names:
            z = catalog["mizuki_Z"]
        else:
            raise ValueError("Cannot get photo-z: %s" % method_name)
    elif method_name == "dnn":
        if "dnnz_photoz_best" in catalog.dtype.names:
            z = catalog["dnnz_photoz_best"]
        elif "dnn_Z" in catalog.dtype.names:
            z = catalog["dnn_Z"]
        else:
            raise ValueError("Cannot get photo-z: %s" % method_name)
    elif method_name == "demp":
        if "dempz_photoz_best" in catalog.dtype.names:
            z = catalog["dempz_photoz_best"]
        elif "demp_Z" in catalog.dtype.names:
            z = catalog["demp_Z"]
        else:
            raise ValueError("Cannot get photo-z: %s" % method_name)
    else:
        z = _nan_array(len(catalog))
    return z


def get_imag_A10(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "magA10" in catalog.dtype.names:
        return catalog["magA10"]
    elif "i_apertureflux_10_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_apertureflux_10_mag"]
    elif "base_CircularApertureFlux_3_0_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = (
            -2.5 * np.log10(catalog["base_CircularApertureFlux_3_0_instFlux"]) + 27.0
        )
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_imag_A15(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "magA15" in catalog.dtype.names:
        return catalog["magA15"]
    elif "i_apertureflux_15_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_apertureflux_15_mag"]
    elif "base_CircularApertureFlux_4_5_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = (
            -2.5 * np.log10(catalog["base_CircularApertureFlux_4_5_instFlux"]) + 27.0
        )
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_imag_A20(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "magA20" in catalog.dtype.names:
        return catalog["magA20"]
    if "i_apertureflux_20_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_apertureflux_20_mag"]
    elif "base_CircularApertureFlux_6_0_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = (
            -2.5 * np.log10(catalog["base_CircularApertureFlux_6_0_instFlux"]) + 27.0
        )
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_imag(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.

    Args:
        catalog (ndarray):  input catlog
    Returns:
        mag (ndarray):      iband magnitude
    """
    if "mag" in catalog.dtype.names:
        mag = catalog["mag"]
    elif "i_cmodel_mag" in catalog.dtype.names:  # s18 and s19
        mag = catalog["i_cmodel_mag"]
    elif "imag_cmodel" in catalog.dtype.names:  # s15
        mag = catalog["imag_cmodel"]
    elif "modelfit_CModel_instFlux" in catalog.dtype.names:  # pipe 7
        mag = -2.5 * np.log10(catalog["modelfit_CModel_instFlux"]) + 27.0
    else:
        mag = _nan_array(len(catalog))
    return mag


def get_imag_psf(catalog):
    """Returns the i-band magnitude of the objects in the input data or
    simulation catalog. Does not apply any cuts and returns NaNs for invalid
    values.

    Args:
        catalog (ndarray):     input catalog
    Returns:
        magnitude (ndarray):   PSF magnitude
    """
    if "i_psfflux_mag" in catalog.dtype.names:  # s18 and s19
        magnitude = catalog["i_psfflux_mag"]
    elif "imag_psf" in catalog.dtype.names:  # s15
        magnitude = catalog["imag_psf"]
    elif "base_PsfFlux_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = -2.5 * np.log10(catalog["base_PsfFlux_instFlux"]) + 27.0
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_abs_ellip(catalog):
    """Returns the norm of galaxy ellipticities.

    Args:
        catalog (ndarray):  input catlog
    Returns:
        absE (ndarray):     norm of galaxy ellipticities
    """
    if "absE" in catalog.dtype.names:
        absE = catalog["absE"]
    elif "i_hsmshaperegauss_e1" in catalog.dtype.names:  # For S18A
        absE = (
            catalog["i_hsmshaperegauss_e1"] ** 2.0
            + catalog["i_hsmshaperegauss_e2"] ** 2.0
        )
        absE = np.sqrt(absE)
    elif "ishape_hsm_regauss_e1" in catalog.dtype.names:  # For S16A
        absE = (
            catalog["ishape_hsm_regauss_e1"] ** 2.0
            + catalog["ishape_hsm_regauss_e2"] ** 2.0
        )
        absE = np.sqrt(absE)
    elif "ext_shapeHSM_HsmShapeRegauss_e1" in catalog.dtype.names:  # For pipe 7
        absE = (
            catalog["ext_shapeHSM_HsmShapeRegauss_e1"] ** 2.0
            + catalog["ext_shapeHSM_HsmShapeRegauss_e2"] ** 2.0
        )
        absE = np.sqrt(absE)
    else:
        absE = _nan_array(len(catalog))
    return absE


def get_abs_ellip_psf(catalog):
    """Returns the amplitude of ellipticities of PSF

    Args:
        catalog (ndarray):  input catalog
    Returns:
        out (ndarray):      the modulus of galaxy distortions.
    """
    e1, e2 = get_psf_ellip(catalog)
    out = e1**2.0 + e2**2.0
    out = np.sqrt(out)
    return out


def get_radec(catalog):
    """Returns the angular position

    Args:
        catalog (ndarray):  input catalog
    Returns:
        ra (ndarray): ra
        dec (ndarray): dec
    """
    if "ra" in catalog.dtype.names:  # small catalog
        ra = catalog["ra"]
        dec = catalog["dec"]
    elif "i_ra" in catalog.dtype.names:  # s18 & s19
        ra = catalog["i_ra"]
        dec = catalog["i_dec"]
    elif "ira" in catalog.dtype.names:  # s15
        ra = catalog["ira"]
        dec = catalog["idec"]
    elif "coord_ra" in catalog.dtype.names:  # pipe 7
        ra = catalog["coord_ra"]
        dec = catalog["coord_dec"]
    elif "ra_mock" in catalog.dtype.names:  # mock catalog
        ra = catalog["ra_mock"]
        dec = catalog["dec_mock"]
    else:
        ra = _nan_array(len(catalog))
        dec = _nan_array(len(catalog))
    return ra, dec


def get_res(catalog):
    """Returns the resolution

    Args:
        catalog (ndarray):  input catalog
    Returns:
        res (ndarray):      resolution
    """
    if "res" in catalog.dtype.names:
        return catalog["res"]
    elif "i_hsmshaperegauss_resolution" in catalog.dtype.names:  # s18 & s19
        res = catalog["i_hsmshaperegauss_resolution"]
    elif "ishape_hsm_regauss_resolution" in catalog.dtype.names:  # s15
        res = catalog["ishape_hsm_regauss_resolution"]
    elif "ext_shapeHSM_HsmShapeRegauss_resolution" in catalog.dtype.names:  # pipe 7
        res = catalog["ext_shapeHSM_HsmShapeRegauss_resolution"]
    else:
        res = _nan_array(len(catalog))
    return res


def get_sdss_size(catalog, dtype="det"):
    """
    This utility gets the observed galaxy size from a data or sims catalog
    using the specified size definition from the second moments matrix.

    Args:
        catalog (ndarray):  Simulation or data catalog
        dtype (str):        Type of psf size measurement in ['trace', 'determin']
    Returns:
        size (ndarray):     galaxy size
    """
    if "base_SdssShape_xx" in catalog.dtype.names:  # pipe 7
        gal_mxx = catalog["base_SdssShape_xx"] * 0.168**2.0
        gal_myy = catalog["base_SdssShape_yy"] * 0.168**2.0
        gal_mxy = catalog["base_SdssShape_xy"] * 0.168**2.0
    elif "i_sdssshape_shape11" in catalog.dtype.names:  # s18 & s19
        gal_mxx = catalog["i_sdssshape_shape11"]
        gal_myy = catalog["i_sdssshape_shape22"]
        gal_mxy = catalog["i_sdssshape_shape12"]
    elif "ishape_sdss_ixx" in catalog.dtype.names:  # s15
        gal_mxx = catalog["ishape_sdss_ixx"]
        gal_myy = catalog["ishape_sdss_iyy"]
        gal_mxy = catalog["ishape_sdss_ixy"]
    else:
        gal_mxx = _nan_array(len(catalog))
        gal_myy = _nan_array(len(catalog))
        gal_mxy = _nan_array(len(catalog))

    if dtype == "trace":
        size = np.sqrt(gal_mxx + gal_myy)
    elif dtype == "det":
        size = (gal_mxx * gal_myy - gal_mxy**2) ** (0.25)
    else:
        raise ValueError("Unknown PSF size type: %s" % dtype)
    return size


def get_logb(catalog):
    """
    This function gets the log of blendedness
    """
    if "logb" in catalog.dtype.names:
        logb = catalog["logb"]
    elif "base_Blendedness_abs" in catalog.dtype.names:  # pipe 7
        logb = np.log10(np.maximum(catalog["base_Blendedness_abs"], 1.0e-6))
    elif "i_blendedness_abs_flux" in catalog.dtype.names:  # s18
        logb = np.log10(np.maximum(catalog["i_blendedness_abs_flux"], 1.0e-6))
    elif "i_blendedness_abs" in catalog.dtype.names:  # s19
        logb = np.log10(np.maximum(catalog["i_blendedness_abs"], 1.0e-6))
    elif "iblendedness_abs_flux" in catalog.dtype.names:  # s15
        logb = np.log10(np.maximum(catalog["iblendedness_abs_flux"], 1.0e-6))
    else:
        logb = _nan_array(len(catalog))
    return logb


def get_sigma_e(catalog):
    """
    This utility returns the hsm_regauss_sigma values for the catalog, without
    imposing any additional flag cuts.
    In the case of GREAT3-like simulations, the noise rescaling factor is
    applied to match the data.
    """
    if "sigma_e" in catalog.dtype.names:
        return catalog["sigma_e"]
    elif "i_hsmshaperegauss_sigma" in catalog.dtype.names:
        sigma_e = catalog["i_hsmshaperegauss_sigma"]
    elif "ishape_hsm_regauss_sigma" in catalog.dtype.names:
        sigma_e = catalog["ishape_hsm_regauss_sigma"]
    elif "ext_shapeHSM_HsmShapeRegauss_sigma" in catalog.dtype.names:
        sigma_e = catalog["ext_shapeHSM_HsmShapeRegauss_sigma"]
    else:
        sigma_e = _nan_array(len(catalog))
    return sigma_e


def get_psf_size(catalog, dtype="fwhm"):
    """This utility gets the PSF size from a data or sims catalog using the
    specified size definition from the second moments matrix.

    Args:
        catalog (ndarray):  Simulation or data catalog
        dtype (str):        Type of psf size measurement in ['trace', 'det',
                            'fwhm'] (default: 'fwhm')
    Returns:
        size (ndarray):     PSF size
    """
    if "base_SdssShape_psf_xx" in catalog.dtype.names:
        psf_mxx = catalog["base_SdssShape_psf_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_psf_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_psf_xy"] * 0.168**2.0
    elif "i_sdssshape_psf_shape11" in catalog.dtype.names:
        psf_mxx = catalog["i_sdssshape_psf_shape11"]
        psf_myy = catalog["i_sdssshape_psf_shape22"]
        psf_mxy = catalog["i_sdssshape_psf_shape12"]
    elif "ishape_sdss_psf_ixx" in catalog.dtype.names:
        psf_mxx = catalog["ishape_sdss_psf_ixx"]
        psf_myy = catalog["ishape_sdss_psf_iyy"]
        psf_mxy = catalog["ishape_sdss_psf_ixy"]
    else:
        psf_mxx = _nan_array(len(catalog))
        psf_myy = _nan_array(len(catalog))
        psf_mxy = _nan_array(len(catalog))

    if dtype == "trace":
        if "traceR" in catalog.dtype.names:
            size = catalog["traceR"]
        else:
            size = np.sqrt(psf_mxx + psf_myy)
    elif dtype == "det":
        if "detR" in catalog.dtype.names:
            size = catalog["detR"]
        else:
            size = (psf_mxx * psf_myy - psf_mxy**2) ** (0.25)
    elif dtype == "fwhm":
        if "fwhm" in catalog.dtype.names:
            size = catalog["fwhm"]
        else:
            size = 2.355 * (psf_mxx * psf_myy - psf_mxy**2) ** (0.25)
    else:
        raise ValueError("Unknown PSF size type: %s" % dtype)
    return size


def get_gal_ellip(catalog):
    if "e1_regaus" in catalog.dtype.names:  # small catalog
        return catalog["e1_regaus"], catalog["e2_regaus"]
    elif "i_hsmshaperegauss_e1" in catalog.dtype.names:  # catalog
        return catalog["i_hsmshaperegauss_e1"], catalog["i_hsmshaperegauss_e2"]
    elif "ishape_hsm_regauss_e1" in catalog.dtype.names:
        return catalog["ishape_hsm_regauss_e1"], catalog["ishape_hsm_regauss_e2"]
    elif "ext_shapeHSM_HsmShapeRegauss_e1" in catalog.dtype.names:  # S16A
        return (
            catalog["ext_shapeHSM_HsmShapeRegauss_e1"],
            catalog["ext_shapeHSM_HsmShapeRegauss_e2"],
        )
    elif "i_sdssshape_shape11" in catalog.dtype.names:
        mxx = catalog["i_sdssshape_shape11"]
        myy = catalog["i_sdssshape_shape22"]
        mxy = catalog["i_sdssshape_shape12"]
    elif "ishape_sdss_ixx" in catalog.dtype.names:
        mxx = catalog["ishape_sdss_ixx"]
        myy = catalog["ishape_sdss_iyy"]
        mxy = catalog["ishape_sdss_ixy"]
    else:
        raise ValueError("Input catalog does not have required coulmn name")
    return (mxx - myy) / (mxx + myy), 2.0 * mxy / (mxx + myy)


def get_psf_ellip(catalog, return_shear=False):
    """This utility gets the PSF ellipticity (uncalibrated shear) from a data
    or sims catalog.
    """
    if "e1_psf" in catalog.dtype.names:
        return catalog["e1_psf"], catalog["e2_psf"]
    elif "base_SdssShape_psf_xx" in catalog.dtype.names:
        psf_mxx = catalog["base_SdssShape_psf_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_psf_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_psf_xy"] * 0.168**2.0
    elif "i_sdssshape_psf_shape11" in catalog.dtype.names:
        psf_mxx = catalog["i_sdssshape_psf_shape11"]
        psf_myy = catalog["i_sdssshape_psf_shape22"]
        psf_mxy = catalog["i_sdssshape_psf_shape12"]
    elif "ishape_sdss_psf_ixx" in catalog.dtype.names:
        psf_mxx = catalog["ishape_sdss_psf_ixx"]
        psf_myy = catalog["ishape_sdss_psf_iyy"]
        psf_mxy = catalog["ishape_sdss_psf_ixy"]
    else:
        raise ValueError("Input catalog does not have required coulmn name")

    if return_shear:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0, psf_mxy / (
            psf_mxx + psf_myy
        )
    else:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy), 2.0 * psf_mxy / (
            psf_mxx + psf_myy
        )


def get_sigma_e_model(catalog, data_dir="data"):
    """This utility returns a model for the shape measurement uncertainty as a
    function of SNR and resolution.  It uses the catalog directly to get the
    SNR and resolution values.

    The file storing the data used to build the approximate correction is
    expected to be found in sigmae_ratio.dat
    """
    # Get the necessary quantities.
    snr = np.array(get_snr(catalog))
    log_snr = np.log10(snr)
    res = np.array(get_res(catalog))
    # Fix interpolation for NaN/inf
    m = np.isnan(log_snr) | np.isinf(log_snr)
    log_snr[m] = 1.0  # set to minimum value that passes nominal cut
    snr[m] = 10.0

    # Build the baseline model for sigma_e.
    par = np.load(os.path.join(data_dir, "sigma_e_model_par.npy"), allow_pickle=True)[0]
    sigma_e = np.exp(par[2]) * ((snr / 20.0) ** par[0]) * ((res / 0.5) ** par[1])

    # Get the corrections from interpolation amongst saved values.
    dat = np.loadtxt(os.path.join(data_dir, "sigmae_ratio.dat")).transpose()
    saved_snr = dat[0, :]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat[1, :]
    saved_corr = dat[2, :]

    # Interpolate the corrections (which multiply the power-law results).
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_corr, log_snr, res)
    return result * sigma_e


def get_erms_model(catalog, data_dir="data"):
    """This utility returns a model for the RMS ellipticity as a function of
    SNR and resolution.  It uses the catalog directly to get the SNR and
    resolution values.

    The file storing the data used to build the model is expected to be found
    in intrinsicshape_2d.dat

    """
    # Get the necessary quantities.
    snr = np.array(get_snr(catalog))
    log_snr = np.log10(snr)
    res = np.array(get_res(catalog))
    # Fix interpolation for NaN/inf
    m = np.isnan(log_snr) | np.isinf(log_snr)
    log_snr[m] = 1.0  # set to minimum value that passes nominal cut
    snr[m] = 10.0

    # Get saved model values.
    dat = np.loadtxt(os.path.join(data_dir, "intrinsicshape_2d.dat")).transpose()
    saved_snr = dat[0, :]
    log_saved_snr = np.log10(saved_snr)

    saved_res = dat[1, :]
    saved_model = dat[2, :]

    # Interpolate the e_rms values and return them.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_model, log_snr, res)
    return result


def get_weight_model(catalog, data_dir="data"):
    """
    This utility returns a model for the shape measurement weight as a
    function of SNR and resolution.  It relies on two other routines
    to get models for the intrinsic shape RMS and measurement error.
    """
    sigmae_meas = get_sigma_e_model(catalog, data_dir)
    erms = get_erms_model(catalog, data_dir)
    return 1.0 / (sigmae_meas**2 + erms**2)


def get_m_model(
    catalog, weight_bias=True, z_depend=True, data_dir="data"
):
    """
    Routine to get a model for calibration bias m given some input snr
    and resolution values or arrays.
    """
    # Get the necessary quantities.
    snr = np.array(get_snr(catalog))
    log_snr = np.log10(snr)
    res = np.array(get_res(catalog))
    # Fix interpolation for NaN/inf
    m = np.isnan(log_snr) | np.isinf(log_snr)
    log_snr[m] = 1.0  # set to minimum value that passes nominal cut
    snr[m] = 10.0

    # m = -0.1408*((snr/20.)**-1.23)*((res/0.5)**1.76) - 0.0214
    maFname = os.path.join(data_dir, "shear_m_a_model_par.npy")
    m_opt = np.load(maFname, allow_pickle=True).item()["m_opt"]
    fake_x = np.vstack((snr, res))
    model_m = m_func(fake_x, *m_opt)

    data_file = os.path.join(data_dir, "shear_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_m = dat["dm"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_m, log_snr, res)

    if weight_bias:
        result += get_mwt_model(catalog, data_dir=data_dir)

    if z_depend:
        z_file = os.path.join(data_dir, "dnn_z_bin_dm_da_2d.csv")
        zat = Table.read(z_file)
        saved_z = zat["z"]
        saved_m = zat["dm"]
        result += grid_interpolate_1d(
            saved_z, saved_m, np.array(get_photo_z(catalog, "dnn"))
        )

    model_m = model_m + result
    return model_m


def get_mwt_model(catalog, data_dir="data"):
    """
    This function gets a model for calibration bias m due to weight bias, given
    some input snr and resolution values or arrays.
    """
    # Get the necessary quantities.
    snr = np.array(get_snr(catalog))
    log_snr = np.log10(snr)
    res = np.array(get_res(catalog))
    # Fix interpolation for NaN/inf
    m = np.isnan(log_snr) | np.isinf(log_snr)
    log_snr[m] = 1.0  # set to minimum value that passes nominal cut
    snr[m] = 10.0

    # m = -1.31 + (27.26 + (snr/20.)**-1.22) / (res + 20.8)
    maFname = os.path.join(data_dir, "weightBias_m_a_model_par.npy")
    m_opt = np.load(maFname, allow_pickle=True).item()["m_opt"]
    fake_x = np.vstack((snr, res))
    model_m = mwt_func(fake_x, *m_opt)

    data_file = os.path.join(data_dir, "weightBias_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_m = dat["dm"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_m, log_snr, res)
    return result + model_m


def get_c_model(
    catalog, weight_bias=True, z_depend=True, data_dir="data"
):
    """
    This function gets a model for additive bias coefficient a given some input
    snr and resolution values or arrays.
    """
    # Get the necessary quantities.
    snr = np.array(get_snr(catalog))
    log_snr = np.log10(snr)
    res = np.array(get_res(catalog))
    # Fix interpolation for NaN/inf
    m = np.isnan(log_snr) | np.isinf(log_snr)

    log_snr[m] = 1.0  # set to minimum value that passes nominal cut
    snr[m] = 10.0

    # a = 0.175 * ((snr/20.)**-1.07) * (res - 0.508)

    maFname = os.path.join(data_dir, "shear_m_a_model_par.npy")
    a_opt = np.load(maFname, allow_pickle=True).item()["a_opt"]
    fake_x = np.vstack((snr, res))
    model_a = a_func(fake_x, *a_opt)

    data_file = os.path.join(data_dir, "shear_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_a = dat["da"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_a, log_snr, res)
    if weight_bias:
        result += get_awt_model(catalog, data_dir=data_dir)

    if z_depend:
        z_file = os.path.join(data_dir, "dnn_z_bin_dm_da_2d.csv")
        zat = Table.read(z_file)
        saved_z = zat["z"]
        saved_a = zat["da"]
        result += grid_interpolate_1d(
            saved_z, saved_a, np.array(get_photo_z(catalog, "dnn"))
        )

    model_a = model_a + result
    psf_e1, psf_e2 = get_psf_ellip(catalog)
    model_c1 = model_a * psf_e1
    model_c2 = model_a * psf_e2
    return model_c1, model_c2


def get_awt_model(catalog, data_dir="data"):
    """
    This function gets a model for additive bias coefficient a due to weight
    bias given some input snr and resolution values or arrays.
    """
    # Get the necessary quantities.
    snr = np.array(get_snr(catalog))
    log_snr = np.log10(snr)
    res = np.array(get_res(catalog))
    # Fix interpolation for NaN/inf
    m = np.isnan(log_snr) | np.isinf(log_snr)
    log_snr[m] = 1.0  # set to minimum value that passes nominal cut
    snr[m] = 10.0

    # a = -0.089 * (res-0.71) * ((snr/20.)**-2.2)
    maFname = os.path.join(data_dir, "weightBias_m_a_model_par.npy")
    a_opt = np.load(maFname, allow_pickle=True).item()["a_opt"]
    fake_x = np.vstack((snr, res))
    model_a = a_func(fake_x, *a_opt)

    data_file = os.path.join(data_dir, "weightBias_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_a = dat["da"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_a, log_snr, res)
    return result + model_a


def get_wl_cuts(catalog):
    """Returns the weak-lensing cuts"""
    sig_e = get_sigma_e(catalog)
    absE = get_abs_ellip(catalog)
    fwhm = get_psf_size(catalog, "fwhm")
    wlflag = (
        ((get_imag(catalog) - catalog["a_i"]) < 24.5)
        & (absE <= 2.0)
        & (get_res(catalog) >= 0.3)
        & (get_snr(catalog) >= 10.0)
        & (sig_e > 0.0)
        & (sig_e < 0.4)
        & (get_logb(catalog) <= -0.38)
        & (get_imag_A10(catalog) < 25.5)
        & (~np.isnan(fwhm))
    )
    return wlflag


def make_reGauss_calibration_table(catalog):
    out = Table()
    """Updates the columns derived from calibration"""
    sigmae = get_sigma_e_model(catalog)
    out["i_hsmshaperegauss_derived_sigma_e"] = sigmae
    erms = get_erms_model(catalog)
    out["i_hsmshaperegauss_derived_rms_e"] = erms
    out["i_hsmshaperegauss_derived_weight"] = 1.0 / (sigmae**2 + erms**2)
    model_m = get_m_model(catalog, weight_bias=True, z_depend=True)
    out["i_hsmshaperegauss_derived_shear_bias_m"] = model_m
    model_c1, model_c2 = get_c_model(catalog, weight_bias=True, z_depend=True)
    out["i_hsmshaperegauss_derived_shear_bias_c1"] = model_c1
    out["i_hsmshaperegauss_derived_shear_bias_c2"] = model_c2
    return out


def get_sel_bias(weight, magA10, res):
    """This utility gets the selection bias (multiplicative and additive)

    Args:
        weight (ndarray):   weight for dataset.  E.g., lensing shape weight,
                            Sigma_c^-2 weight
        magA10 (ndarray):   aperture magnitude (1 arcsec) for dataset
        res (ndarray):      resolution factor for dataset
    Returns:
        m_sel (float):      multiplicative edge-selection bias
        a_sel (float):      additive edge-selection bias (c1)
        m_sel_err (float):  1-sigma uncertainty in m_sel
        a_sel_err (float):  1-sigma uncertainty in a_sel
    """

    if not (np.all(np.isfinite(weight))):
        raise ValueError("Non-finite weight")
    if not (np.all(weight) >= 0.0):
        raise ValueError("Negative weight")
    wSum = np.sum(weight)

    bin_magA = 0.025
    pedgeM = np.sum(weight[(magA10 >= 25.5 - bin_magA)]) / wSum / bin_magA

    bin_res = 0.01
    pedgeR = np.sum(weight[(res <= 0.3 + bin_res)]) / wSum / bin_res

    m_sel = -0.059 * pedgeM + 0.019 * pedgeR
    a_sel = 0.0064 * pedgeM + 0.0063 * pedgeR

    # assume the errors for 2 cuts are independent.
    m_err = np.sqrt((0.0089 * pedgeM) ** 2.0 + (0.0013 * pedgeR) ** 2.0)
    a_err = np.sqrt((0.0034 * pedgeM) ** 2.0 + (0.0009 * pedgeR) ** 2.0)
    return m_sel, a_sel, m_err, a_err

