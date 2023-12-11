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
import warnings
import numpy as np
import healpy as hp
import scipy.optimize

import astropy.io.fits as pyfits
from astropy.table import Table, join
from scipy.interpolate import griddata

# Field keys used in the noise variance file for simulations
field_names = {
    "XMM": "W02",
    "GAMA09H": "W03",
    "WIDE12H": "W04_WIDE12H",
    "GAMA15H": "W04_GAMA15H",
    "VVDS": "W05",
    "HECTOMAP": "W06",
}


def fix_nan(catalog, key):
    """Fixes NaN entries."""
    x = catalog[key]
    mask = np.isnan(x) | np.isinf(x)
    n_fix = mask.astype(int).sum()
    if n_fix > 0:
        catalog[key][mask] = 0.0
    return


def _nan_array(n):
    """Creates an NaN array."""
    out = np.empty(n)
    out.fill(np.nan)
    return out


def chunkNList(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg
    return out


def chunkPList(seq, per):
    out = []
    last = 0
    while last < len(seq):
        out.append(seq[last : min(last + per, len(seq))])
        last += per
    return out


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


def prepare_field_catalog(filename, fieldname, isim=-1, ngalR=100, ngrid=64):
    """From LSST pipeline data to HSC catalog data

    Args:
        filename (str):     fits filename to read
        fieldname (str):    HSC field information
        isim (int):         simulation subfield information
        ngalR (int):        number of stamps in each row
        ngrid (int):        number of grids in each stamp
    Returns:
        catalog (ndarray): the prepared catalog for a subfield
    """

    # Read the catalog data
    catalog = Table.read(filename)
    colNames = catalog.colnames

    # Load the header to get proper name of flags
    header = pyfits.getheader(filename, 1)
    n_flag = catalog["flags"].shape[1]
    for i in range(n_flag):
        catalog[header["TFLAG%s" % (i + 1)]] = catalog["flags"][:, i]

    # Then, apply mask for permissive cuts
    mask = (
        (~(catalog["base_SdssCentroid_flag"]))
        & (~catalog["ext_shapeHSM_HsmShapeRegauss_flag"])
        & (catalog["base_ClassificationExtendedness_value"] > 0)
        & (~np.isnan(catalog["modelfit_CModel_instFlux"]))
        & (~np.isnan(catalog["modelfit_CModel_instFluxErr"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmShapeRegauss_resolution"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmPsfMoments_xx"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmPsfMoments_xy"]))
        & (~np.isnan(catalog["base_Variance_value"]))
        & (~np.isnan(catalog["modelfit_CModel_instFlux"]))
        & (~np.isnan(catalog["modelfit_CModel_instFluxErr"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmShapeRegauss_resolution"]))
        & (catalog["deblend_nChild"] == 0)
    )
    catalog = catalog[mask]
    if len(catalog) == 0:
        return None
    catalog = catalog[colNames]

    # Add columns with names matching the real data
    data_fields = {
        "ext_shapeHSM_HsmShapeRegauss_e1": "i_hsmshaperegauss_e1",
        "ext_shapeHSM_HsmShapeRegauss_e2": "i_hsmshaperegauss_e2",
        "ext_shapeHSM_HsmShapeRegauss_sigma": "i_hsmshaperegauss_sigma",
        "ext_shapeHSM_HsmShapeRegauss_resolution": "i_hsmshaperegauss_resolution",
    }
    for i in data_fields:
        catalog[data_fields[i]] = catalog[i]

    # Add field name infomation
    catalog["field_name"] = np.string_(fieldname)

    # For simulation
    catalog["isim"] = isim
    catalog["paired"] = False
    # Reweight factor to be determined
    catalog["reweight_s16a"] = 0.0

    catalog["centDist"] = (
        catalog["base_SdssCentroid_y"] % ngrid - ngrid / 2
    ) ** 2.0 + (catalog["base_SdssCentroid_x"] % ngrid - ngrid / 2) ** 2.0
    catalog["centDist"] = np.sqrt(catalog["centDist"])
    catalog = catalog[(catalog["centDist"] < 5.0)]

    # First, keep only detections that are the closest to the grid point
    # Get sorted index by grid index and grid distance
    # Add a gridIndex column instead of using the index column, for compatibility
    catalog["ipos"] = (catalog["base_SdssCentroid_y"] // ngrid) * ngalR + (
        catalog["base_SdssCentroid_x"] // ngrid
    )
    catalog["ipos"] = catalog["ipos"].astype(int)
    inds = np.lexsort([catalog["centDist"], catalog["ipos"]])
    catalog = catalog[inds]
    inds_unique = np.unique(catalog["ipos"], return_index=True)[1]

    catU = catalog[inds_unique]
    """
    # Identify pairs based on their input ipos values
    """
    n = ngalR * ngalR  # Number of grid points
    mask_even = np.in1d(
        catU["ipos"] % n, 2 * np.arange(n / 2).astype("int"), assume_unique=True
    )
    mask_odd = np.in1d(
        catU["ipos"] % n, 2 * np.arange(n / 2).astype("int") + 1, assume_unique=True
    )

    # Find index of broken pairs
    broken_pairs = np.setxor1d(
        catU["ipos"][mask_even] % n, catU["ipos"][mask_odd] % n - 1
    )
    mask_pairs = ~np.in1d(
        catU["ipos"] % n,
        np.concatenate([broken_pairs, broken_pairs + 1]),
        assume_unique=True,
    )

    catalog["paired"][inds_unique] = mask_pairs
    catalog.sort(["paired", "ipos"])

    """
    # Load information from psfPre
    """
    psfFname = "psfPre/psf-%05d.fits" % isim
    psfInfo = pyfits.getheader(psfFname)
    catalog["g1"] = psfInfo["g1"]
    catalog["g2"] = psfInfo["g2"]

    """
    # Load information from catPre
    """
    catFname = "catPre/catalog-%05d.fits" % isim
    catInfo = Table.read(catFname)
    # Join catalogs on grid index
    # Extinction information is contained in catInfo
    catalog = join(catalog, catInfo, "ipos")

    # Add column to make a WL flag for the simulation data:
    catalog["weak_lensing_flag"] = get_wl_cuts(catalog)

    # add a bunch of columns that get used for WL analysis - fake it for now!
    catalog["i_hsmshaperegauss_derived_sigma_e"] = catalog["i_hsmshaperegauss_sigma"]
    catalog["i_hsmshaperegauss_derived_rms_e"] = 0.365

    wt = 1.0 / (
        catalog["i_hsmshaperegauss_derived_rms_e"] ** 2
        + catalog["i_hsmshaperegauss_derived_sigma_e"] ** 2
    )
    catalog["i_hsmshaperegauss_derived_weight"] = wt
    # Unpleasant discovery: the lines below used to say "= 0" and it turns out that meant those
    # columns were ints, which led to all kinds of issues later on.  This fix is for the next time a
    # sims catalog gets turned into HDF5.
    catalog["i_hsmshaperegauss_derived_shear_bias_m"] = 0.0
    catalog["i_hsmshaperegauss_derived_shear_bias_c1"] = 0.0
    catalog["i_hsmshaperegauss_derived_shear_bias_c2"] = 0.0
    return catalog


def galaxy_selector(
    catalog,
    min_snr=None,
    max_snr=None,
    min_res=None,
    max_res=None,
    max_magA10=None,
    max_sigma_e=None,
    max_e=None,
    max_mag=None,
    min_fpfs_flux=None,
    max_fpfs_flux=None,
    lower_psf_size=0.0,
    upper_psf_size=100.0,
    force_90_pair=True,
    max_stampCent_distance=None,
    max_logb=None,
    applyboth="",
    minNpass=2,
    doBOmsk=False,
):
    """Takes a catalog and returns a boolean ndaray with a mask that says which
    galaxies to use for analysis.
    Warning: Cuts are applied in extinction corrected magnitude. For
    simulations, extinction values are extracted from the COSMOS field and
    correpond to the extinction at the position of the COSMOS galaxies used as
    the input of the simulation.

    Args:
        catalog (ndarray):
            catalog on which the cuts apply.
        min_snr (float):
            minimum i-band unforced cmodel SNR cut
        max_snr (float): float
            maximum i-band unforced cmodel SNR cut
        min_res (float):
            minimum i-band resolution cut
        max_res (float):
            maximum i-band resolution cut
        max_mag (float):
            i-band magnitude cut.
        max_magA10 (float):
            maximum i-band aperture magnitude cut
        max_sigma_e (float):
            maximum sigma_e cut. High values can result from various
            measurement failures.
        max_e (float):
            maximum ellipticity cut [2 is a good number to use]
        max_logb (float):
            maximum value of iblendedness_abs_flux
        force_90_pair (float):
            if set to true, impose selection in a way that preserves 90 degree
            rotated pais. Only applies to the simulated data loaded with
            `load_sims_catalog`.
        max_stampCent_distance (float):
            require simulation matches to lie within this many pixels of the
            expected location.  Only really relevant for cases where there are
            off-centered detections and the HSC pipeline is run with detection
            and deblending on [as in e.g. v4]
        applyboth (str):
            which cuts should be applied to both galaxies instead of just 1?
            To be used for tests of selection bias.  If "" then this is not
            done.

    """
    if force_90_pair and "paired" not in catalog.dtype.names:
        raise Exception("cannot cancel shape noise since 'paired' column not found")

    mask = np.ones(len(catalog)).astype(bool)
    if ("npass" in catalog.dtype.names) and minNpass > 0:
        mask &= get_npass(catalog) >= minNpass

    if doBOmsk:
        mask &= ~get_briObj_cuts_s18(catalog)

    if max_stampCent_distance is not None:
        dx = abs(catalog["base_SdssCentroid_x"] % 64 - 32)
        dy = abs(catalog["base_SdssCentroid_y"] % 64 - 32)
        mask &= dx**2.0 + dy**2.0 <= max_stampCent_distance**2.0

    if force_90_pair:
        ind_pair = np.where(catalog["paired"])[0]
        ind_pair1 = ind_pair[0::2]
        ind_pair2 = ind_pair[1::2]
        # Masking elements not in a pair to begin with
        mask &= catalog["paired"]
    else:
        ind_pair1 = np.empty(0)
        ind_pair2 = np.empty(0)

    # Basic sanity checks: none of the quantities that we want to use should
    # be NaN. We also impose the sigma_e and e cuts here, because even just
    # a few bad values can totally mess stuff up.
    e1_psf, e2_psf = get_psf_ellip(catalog)
    e1_regaus, e2_regaus = get_gal_ellip(catalog)
    if (min_res is not None) or (max_sigma_e is not None) or (max_e is not None):
        mask &= (
            (~np.isnan(e1_regaus))
            & (~np.isnan(e2_regaus))
            & (~np.isnan(e1_psf))
            & (~np.isnan(e2_psf))
            & (~np.isnan(get_res(catalog)))
            & (~np.isnan(get_sigma_e(catalog)))
        )
    if (min_snr is not None) or (max_snr is not None):
        mask &= ~np.isnan(get_snr(catalog))
    if max_magA10 is not None:
        mask &= ~np.isnan(get_imag_A10(catalog))
    if max_mag is not None:
        mask &= ~np.isnan(get_imag(catalog))
    if (min_fpfs_flux is not None) or (max_fpfs_flux is not None):
        mask &= ~np.isnan(get_FPFS1_obs(catalog)[4])

    if max_sigma_e is not None:
        sigma_e = get_sigma_e(catalog[mask])
        mask[mask] &= sigma_e <= max_sigma_e
    if max_e is not None:
        absE = get_abs_ellip(catalog[mask])
        mask[mask] &= absE <= max_e

    # The sanity checks are systematically applied to both members of the pair
    # Since NaN's make the data unusable, we have to self-consistently impose
    # this across both 90-degree-rotated pairs (if we're using shape noise
    # cancellation):
    if force_90_pair:
        mask[ind_pair1] = mask[ind_pair1] & mask[ind_pair2]
        mask[ind_pair2] = mask[ind_pair1]

    # Now we move on to the cuts on resolution, S/N and magnitude.
    # We should only try to make the cuts on the galaxies that passed the flag
    # cuts, otherwise we get lots of warnings about invalid values etc.
    if min_res is not None:
        mask[mask] &= get_res(catalog[mask]) >= min_res
    if max_res is not None:
        mask[mask] &= get_res(catalog[mask]) <= max_res

    if min_snr is not None:
        snr = get_snr(catalog[mask])
        mask[mask] &= snr >= min_snr
    if max_snr is not None:
        snr = get_snr(catalog[mask])
        mask[mask] &= snr <= max_snr

    if max_magA10 is not None:
        magA10 = get_imag_A10(catalog[mask])
        mask[mask] &= magA10 <= max_magA10
    if max_mag is not None:
        if "a_i" in catalog.dtype.names:
            mag = get_imag(catalog[mask]) - catalog["a_i"][mask]
        else:
            warnings.warn(
                "Extinction not revised: Do not have Extinction observables in catalog"
            )
            mag = get_imag(catalog[mask])
        mask[mask] &= mag <= max_mag

    if max_logb is not None:
        mask[mask] &= get_logb(catalog[mask]) <= max_logb

    # For PSF size
    psf_size = get_psf_size(catalog[mask])
    if lower_psf_size > 0.0:
        cut_val = np.percentile(psf_size, lower_psf_size)
        mask[mask] &= psf_size >= cut_val
    if upper_psf_size < 100.0:
        cut_val = np.percentile(psf_size, upper_psf_size)
        mask[mask] &= psf_size < cut_val

    if min_fpfs_flux is not None:
        mask[mask] &= get_FPFS1_obs(catalog[mask])[4] >= min_fpfs_flux
    if max_fpfs_flux is not None:
        mask[mask] &= get_FPFS1_obs(catalog[mask])[4] <= max_fpfs_flux

    # For noise cancellation, apply the cut only to the first element of the
    # pair to avoid funny selection biases.
    # But enable introduction of selection bias due to single cuts through applyboth.
    if force_90_pair:
        mask[ind_pair2] = mask[ind_pair1]
        if applyboth != "":
            if "resolution" in applyboth:
                if min_res is not None:
                    print("Applying resolution cut to both objects in pair!")
                    mask[mask] &= get_res(catalog[mask]) >= min_res
            if "snr" in applyboth:
                if min_snr is not None:
                    print("Applying SNR cut to both objects in pair!")
                    snr = get_snr(catalog[mask])
                    mask[mask] &= snr >= min_snr
            if "mag" in applyboth:
                if max_mag is not None:
                    print("Applying magnitude cut to both objects in pair!")
                    mag = get_imag(catalog[mask]) - catalog["a_i"][mask]
                    mask[mask] &= mag < max_mag
    return mask


def weighted_percentile(data, percents, weights=None):
    """Returns the percentile in units of 1% weights specifies the frequency of
    data
    """
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1.0 * w.cumsum() / w.sum() * 100
    y = np.interp(percents, p, d)
    return y


def sliding_window_def(snr, res, weights=None, n_gal=100, n_bin_1d=20, snr_overlap=1.4):
    """sim_analysis_utilities.sliding_window_def

    A routine to define 2d bins in a sample based on its (snr, res)
    using a sliding window.
    It returns an array of lower and upper bin
    edges for the overlapping SNR bins, each of length n_bin_1d.  It
    also returns resolution bin edges within each SNR bin, i.e., the
    lower and upper resolution bin edges have shape (n_bin_1d,
    n_bin_1d).

    Args:
        n_gal (int):            the minimum number of galaxies in a 2d bin, which
                                determines the overlap in res binning
        n_bin_1d (int):         gives the number of bins to use in 1 dimension.
        snr_overlap (float):    determines the overlap in snr binning. Set
                                snr_overlap=1. does not overlap snr bins
    """
    lower_percentiles = np.linspace(0.0, 100.0, n_bin_1d + 1)[:-1]
    delta_perc = lower_percentiles[1] - lower_percentiles[0]
    lower_percentiles -= (0.5 * (snr_overlap - 1.0)) * delta_perc
    lower_percentiles[0] = 0
    upper_percentiles = np.linspace(0.0, 100.0, n_bin_1d + 1)[1:]
    upper_percentiles += (0.5 * (snr_overlap - 1.0)) * delta_perc
    upper_percentiles[-1] = 100.0
    snr_lower = weighted_percentile(snr, lower_percentiles, weights)
    snr_upper = weighted_percentile(snr, upper_percentiles, weights)
    res_lower = np.zeros((n_bin_1d, n_bin_1d))
    res_upper = np.zeros((n_bin_1d, n_bin_1d))
    # Now loop over the bins in SNR.
    for sbin in range(n_bin_1d):
        sbin_mask = (snr >= snr_lower[sbin]) & (snr < snr_upper[sbin])
        tmp_res = res[sbin_mask]
        if weights is not None:
            tmp_wt = weights[sbin_mask]
        else:
            tmp_wt = None
        if len(tmp_res) < 1.1 * n_gal:
            raise RuntimeError("Not enough objects to make " + "sliding window")
        # width of bin in percentiles
        delta_perc = 100.0 * max(float(n_gal) / len(tmp_res), 1.0 / n_bin_1d)
        bounds_percentiles = np.linspace(0.0, 100.0, n_bin_1d + 1)
        cent_percentiles = (bounds_percentiles[:-1] + bounds_percentiles[1:]) / 2.0
        lower_percentiles = cent_percentiles - delta_perc / 2.0
        upper_percentiles = lower_percentiles + delta_perc
        lower_percentiles[0] = 0.0
        upper_percentiles[-1] = 100.0
        res_lower[sbin, :] = weighted_percentile(tmp_res, lower_percentiles, tmp_wt)
        res_upper[sbin, :] = weighted_percentile(tmp_res, upper_percentiles, tmp_wt)
    return snr_lower, snr_upper, res_lower, res_upper


def estimate_subfield_shear(
    catalog,
    use_model=False,
    verbose=False,
    report_resp=True,
    reweight=True,
    roundtrip_data=None,
    force_weight=True,
):
    """The purpose of this routine is to estimate the shear for a given subfield.
    This function expects the true shear to be the same for all entries. If it
    finds >1 subfield index it will raise a RuntimeError.

    Args:
        catalog (ndarray):
            Catalog to use for shear estimation.
        use_model (bool):
            Use get_X_model routines for weights, RMS ellipticity, and sigma_e?  If
            False, then will use catalog entries. (default: False)
        verbose (bool):
            Emit diagnostic statements? (default: False)
        report_resp (bool):
            Return the value of responsivity along with the shears? (default:
            False)
        reweight (bool):
            Use the S16A reweighting?  This will be the same value for each entry
            since the routine works at the subfield level. (default: False)
        roundtrip_data (tuple/None):
            Data to be used for a round-trip analysis given some set of
            calibration corrections that differs from the ones in the catalogs.
            This should be a tuple of (m, c1, c2) values with the same length
            as `catalog`. If it is passed in, then the calibration values in
            the catalog are completely ignored.  If None, then the catalog
            calibration values are used. (default: None)
    """

    if verbose:
        print("Using %d objects in input catalog" % (len(catalog)))

    # The calculation below follows the agreed-upon procedure from
    # http://hscsurvey.pbworks.com/w/page/113068663/notes%20on%20lensing%20signal%20calculation%20with%20S16A%20catalog
    if not use_model:
        weights = catalog["i_hsmshaperegauss_derived_weight"]
        e_rms = catalog["i_hsmshaperegauss_derived_rms_e"]
        if "i_hsmshaperegauss_derived_sigma_e" in catalog.dtype.names:
            sigma_e = catalog["i_hsmshaperegauss_derived_sigma_e"]
        else:
            sigma_e = get_sigma_e(catalog)
    else:
        weights = get_weight_model(catalog)
        e_rms = get_erms_model(catalog)
        sigma_e = get_sigma_e_model(catalog)
    if reweight:
        weights *= catalog["weight"]
    # Enforce equality of weights etc. for galaxies in a pair if shape noise cancellation is requested.
    if force_weight:
        weights[1::2] = weights[0::2]
        e_rms[1::2] = e_rms[0::2]
        sigma_e[1::2] = sigma_e[0::2]

    # Deal gracefully with the case of zero weights (generally due to reweight_option).
    if np.sum(weights) == 0:
        if not report_resp:
            return 0, 0
        else:
            return 0, 0, 0

    e1, e2 = get_gal_ellip(catalog)

    shear_num_1 = np.sum(weights * e1)
    shear_num_2 = np.sum(weights * e2)
    sum_weight = np.sum(weights)
    if roundtrip_data is None:
        bias_num = 0.0
        add_num_1 = 0.0
        add_num_2 = 0.0
    else:
        model_m, model_c1, model_c2 = roundtrip_data
        bias_num = np.sum(weights * model_m)
        add_num_1 = np.sum(weights * model_c1)
        add_num_2 = np.sum(weights * model_c2)
    bias = bias_num / sum_weight

    resp_num = np.sum(weights * e_rms**2)
    responsivity = 1.0 - resp_num / sum_weight

    if responsivity > 2.0 or responsivity < 0.0:
        raise RuntimeError(
            "Error: responsivity is %f with sum_weight %f" % (responsivity, sum_weight)
        )
    if verbose:
        print("Shear responsivity: %f" % responsivity)
    if verbose:
        print("Bias: %f, %f, %f" % (bias, bias_num, sum_weight))
    g1 = shear_num_1 / (2.0 * responsivity * sum_weight * (1.0 + bias)) - add_num_1 / (
        (1.0 + bias) * sum_weight
    )
    g2 = shear_num_2 / (2.0 * responsivity * sum_weight * (1.0 + bias)) - add_num_2 / (
        (1.0 + bias) * sum_weight
    )
    if not report_resp:
        return g1, g2
    else:
        return g1, g2, responsivity


def fitline(xarr, yarr):
    """
    Fit a line y = a + b * x to input x and y arrays by least squares.

    Returns the tuple (a, b, Var(a), Cov(a, b), Var(b)), after performing an
    internal estimate of measurement errors from the best-fitting model
    residuals.

    See Numerical Recipes (Press et al 1992; Section 15.2) for a clear
    description of the details of this simple regression.
    """
    # Get the S values (use default sigma2, best fit a and b still valid for
    # stationary data)
    S, Sx, Sy, Sxx, Sxy = _calculateSvalues(xarr, yarr)
    # Get the best fit a and b
    Del = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / Del
    b = (S * Sxy - Sx * Sy) / Del
    # Use these to estimate the sigma^2 by residuals from the best-fitting
    # model
    ymodel = a + b * xarr
    sigma2 = np.mean((yarr - ymodel) ** 2)
    # And use this to get model parameter error estimates
    var_a = sigma2 * Sxx / Del
    cov_ab = -sigma2 * Sx / Del
    var_b = sigma2 * S / Del
    return (a, b, var_a, cov_ab, var_b)


def _calculateSvalues(xarr, yarr, sigma2=1.0):
    """Calculates the intermediate S values required for basic linear regression.
    See, e.g., Numerical Recipes (Press et al 1992) Section 15.2.
    """
    if len(xarr) != len(yarr):
        raise ValueError("Input xarr and yarr differ in length!")
    if len(xarr) <= 1:
        raise ValueError("Input arrays must have 2 or more values elements.")

    S = len(xarr) / sigma2
    Sx = np.sum(xarr / sigma2)
    Sy = np.sum(yarr / sigma2)
    Sxx = np.sum(xarr * xarr / sigma2)
    Sxy = np.sum(xarr * yarr / sigma2)
    return (S, Sx, Sy, Sxx, Sxy)


def systematics_func(x, m, a):
    return m * x[0, :] + a * x[1, :]


def get_ma_sim(
    g1, g2, g1_true, g2_true, psf_e1, psf_e2, sigma_vals, separate_components=True
):
    """
    This routine takes a list/tuple/NumPy array of values for estimated shear
    (`g1`, `g2`), true shear (`g1_true`, `g2_true`), and PSF ellipticity
    (`psf_e1`, `psf_e2`).  Then it does a GREAT3-like analysis for calibration
    bias and additive bias.

    Depending on the value of `separate_components`, it does this either
    separately for each component, or for both combined.
    """
    # First, just make sure the inputs are NumPy arrays to enable the math that has to happen later.
    g1 = np.array(g1)
    g2 = np.array(g2)
    g1_true = np.array(g1_true)
    g2_true = np.array(g2_true)
    psf_e1 = np.array(psf_e1)
    psf_e2 = np.array(psf_e2)
    sigma_vals = np.array(sigma_vals)

    # Our default algorithm is going to be to fit for
    # g-g_true = m g_true + a e_PSF
    # for each component, using scipy.optimize.curve_fit.
    # This can eventually accomodate weights as needed.
    if separate_components:
        popt1, pcov1 = scipy.optimize.curve_fit(
            systematics_func,
            np.vstack((g1_true, psf_e1)),
            g1 - g1_true,
            sigma=sigma_vals,
        )
        popt2, pcov2 = scipy.optimize.curve_fit(
            systematics_func,
            np.vstack((g2_true, psf_e2)),
            g2 - g2_true,
            sigma=sigma_vals,
        )
        # Return everything: (m, a, sigma_m, sigma_a) first for component 1 then for
        # component 2.
        return (
            popt1[0],
            popt1[1],
            np.sqrt(pcov1[0][0]),
            np.sqrt(pcov1[1][1]),
            popt2[0],
            popt2[1],
            np.sqrt(pcov2[0][0]),
            np.sqrt(pcov2[1][1]),
        )
    else:
        g = np.concatenate([g1, g2])
        g_true = np.concatenate([g1_true, g2_true])
        psf_e = np.concatenate([psf_e1, psf_e2])
        sigma_vals = np.concatenate([sigma_vals, sigma_vals])
        popt, pcov = scipy.optimize.curve_fit(
            systematics_func, np.vstack((g_true, psf_e)), g - g_true, sigma=sigma_vals
        )
        # Return everything: (m, a, sigma_m, sigma_a).
        return popt[0], popt[1], np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])


def get_shear_regauss(catalog, mbias, msel=0.0, asel=0.0):
    """Returns the regauss shear in data *on single galaxy level*.
    Note: shape weight should be added when caluculating ensemble average.

    Args:
        catalog (ndarray):    input hsc catalog
        mbias (float):      average multiplicative bias [m+dm2]
        msel (float):       selection multiplicative bias for the sample
        asel (float):       selection additive bias
        Returns:
        g1 (ndarray):       the first component of shear
        g2 (ndarray):       the second component of shear
    """
    if "i_hsmshaperegauss_derived_weight" in catalog.dtype.names:
        wname = "i_hsmshaperegauss_derived_weight"
        erms_name = "i_hsmshaperegauss_derived_rms_e"
        e1name = "i_hsmshaperegauss_e1"
        e2name = "i_hsmshaperegauss_e2"
        c1name = "i_hsmshaperegauss_derived_shear_bias_c1"
        c2name = "i_hsmshaperegauss_derived_shear_bias_c2"
    elif "ishape_hsm_regauss_derived_shape_weight" in catalog.dtype.names:
        wname = "ishape_hsm_regauss_derived_shape_weight"
        erms_name = "ishape_hsm_regauss_derived_rms_e"
        e1name = "ishape_hsm_regauss_e1"
        e2name = "ishape_hsm_regauss_e2"
        c1name = "ishape_hsm_regauss_derived_shear_bias_c1"
        c2name = "ishape_hsm_regauss_derived_shear_bias_c2"
    else:
        raise ValueError("Cannot process the catalog")

    wsum = np.sum(catalog[wname])
    eres = 1.0 - np.sum(catalog[erms_name] ** 2.0 * catalog[wname]) / wsum
    g1 = (catalog[e1name] / 2.0 / eres - catalog[c1name]) / (1.0 + mbias)
    g2 = (catalog[e2name] / 2.0 / eres - catalog[c2name]) / (1.0 + mbias)
    e1pg, e2pg = get_psf_ellip(catalog)  # PSF shape
    # correcting for selection bias
    g1 = (g1 - e1pg * asel) / (1.0 + msel)
    g2 = (g2 - e2pg * asel) / (1.0 + msel)
    return g1, g2


def get_shear_regauss_mock(datIn, mbias, msel=0.0, version="all"):
    """Returns the regauss shear in mocks *on single galaxy level*.
    Note: shape weight should be added when caluculating ensemble average.

    Args:
        datIn (ndarray):    input mock catalog
        mbias (float):      average multiplicative bias [m+dm2]
        msel (float):       selection multiplicative bias for the sample
        Returns:
        g1 (ndarray):       the first component of shear
        g2 (ndarray):       the second component of shear
    """
    if version == "all":
        erms = (datIn["noise1_int"] ** 2.0 + datIn["noise2_int"] ** 2.0) / 2.0
        eres = 1.0 - np.sum(datIn["weight"] * erms) / np.sum(datIn["weight"])
        # Note: here we assume addtive bias is zero
        g1 = datIn["e1_mock"] / 2.0 / eres / (1.0 + mbias) / (1.0 + msel)
        g2 = datIn["e2_mock"] / 2.0 / eres / (1.0 + mbias) / (1.0 + msel)
    elif version == "shape":
        erms = (datIn["noise1_int"] ** 2.0 + datIn["noise2_int"] ** 2.0) / 2.0
        eres = 1.0 - np.sum(datIn["weight"] * erms) / np.sum(datIn["weight"])
        # Note: here we assume addtive bias is zero
        g1 = (
            (datIn["noise1_int"] + datIn["noise1_mea"])
            / 2.0
            / eres
            / (1.0 + mbias)
            / (1.0 + msel)
        )
        g2 = (
            (datIn["noise2_int"] + datIn["noise2_mea"])
            / 2.0
            / eres
            / (1.0 + mbias)
            / (1.0 + msel)
        )
    elif version == "shear":
        g1 = datIn["shear1_sim"] / (1 - datIn["kappa"])
        g2 = datIn["shear2_sim"] / (1 - datIn["kappa"])
    else:
        raise ValueError("version can only be all, shape or shear")
    return g1, g2


def make_mock_catalog(datIn, mbias, msel=0.0, corr=1.0):
    """Rescales the shear by (1 + mbias) following section 5.6 and calculate
    the mock ellipticities according to eq. (24) and (25) of
    https://arxiv.org/pdf/1901.09488.pdf

    Args:
        datIn (ndaray): Original HSC S19A mock catalog (it should haave m=0)
        mbias (float):  The multiplicative bias
        msel (float):   Selection bias [default=0.]
        corr (float):   Correction term for shell thickness, finite resolution and missmatch
                        between n(z_data) and n(z_mock) due to a limited number source planes
    Returns:
        out (ndarray):  Updated S19A mock catalog (with m=mbias)
    """
    if not isinstance(mbias, (float, int)):
        raise TypeError("multiplicative shear estimation bias should be a float.")
    if not isinstance(msel, (float, int)):
        raise TypeError("multiplicative selection bias should be a float.")

    bratio = (1 + mbias) * (1 + msel) * corr
    out = datIn.copy()
    # Rescaled gamma by (1+m) and then calculate the distortion delta
    gamma_sq = (out["shear1_sim"] ** 2.0 + out["shear2_sim"] ** 2.0) * bratio**2.0
    dis1 = (
        2.0
        * (1 - out["kappa"])
        * out["shear1_sim"]
        * bratio
        / ((1 - out["kappa"]) ** 2 + gamma_sq)
    )
    dis2 = (
        2.0
        * (1 - out["kappa"])
        * out["shear2_sim"]
        * bratio
        / ((1 - out["kappa"]) ** 2 + gamma_sq)
    )
    # Calculate the mock ellitpicities
    de = dis1 * out["noise1_int"] + dis2 * out["noise2_int"]  # for denominators
    dd = dis1**2 + dis2**2.0
    # avoid dividing by zero (this term is 0 under the limit dd->0)
    tmp1 = np.divide(dis1, dd, out=np.zeros_like(dd), where=dd != 0)
    tmp2 = np.divide(dis2, dd, out=np.zeros_like(dd), where=dd != 0)
    # the nominator for e1
    e1_mock = (
        out["noise1_int"]
        + dis1
        + tmp2
        * (1 - (1 - dd) ** 0.5)
        * (dis1 * out["noise2_int"] - dis2 * out["noise1_int"])
    )
    # the nominator for e2
    e2_mock = (
        out["noise2_int"]
        + dis2
        + tmp1
        * (1 - (1 - dd) ** 0.5)
        * (dis2 * out["noise1_int"] - dis1 * out["noise2_int"])
    )
    # update e1_mock and e2_mock
    out["e1_mock"] = e1_mock / (1.0 + de) + out["noise1_mea"]
    out["e2_mock"] = e2_mock / (1.0 + de) + out["noise2_mea"]
    return out


def get_TPid(catalog):
    return catalog["tract"] * 1000 + catalog["patch"]


def get_isIso(catalog):
    """Returns the flag showing whether the galaxy is isolated"""
    if "parent_id" in catalog.dtype.names:
        isIso = catalog["parent_id"] == 0
    elif "parent" in catalog.dtype.names:
        isIso = catalog["parent"] == 0
    else:
        isIso = _nan_array(len(catalog))
    return isIso


def get_cmodel_obj(catalog):
    """Returns the cmodel objective"""
    if "cmodel_obj" in catalog.dtype.names:
        obj = catalog["cmodel_obj"]
    elif "i_cmodel_objective" in catalog.dtype.names:
        obj = catalog["i_cmodel_objective"]
    else:
        obj = _nan_array(len(catalog))
    return obj


def get_briObj_cuts_s18(catalog):
    """Returns the bright object cut (for S18A)"""
    if "brimsk18" in catalog.dtype.names:
        bmsk = catalog["brimsk18"]
    elif "i_mask_s18a_bright_objectcenter" in catalog.dtype.names:
        bmsk = catalog["i_mask_s18a_bright_objectcenter"]
    else:
        bmsk = np.zeros(len(catalog), dtype=bool)
        warnings.warn("cannot find column for s18A Mask")
    return bmsk


def get_briObj_cuts_v1(catalog):
    """The bright object cut (more conservative)"""
    if "brimsk19I" in catalog.dtype.names:
        bmsk = catalog["brimsk19I"]
    elif "i_mask_brightstar_any" in catalog.dtype.names:
        bmsk = catalog["i_mask_brightstar_any"]
    else:
        bmsk = np.zeros(len(catalog), dtype=bool)
        warnings.warn("cannot find column for s19A Mask I")
    return bmsk


def get_mask_briObj_cuts_v2(catalog):
    """The bright object cut applied"""
    if "brimsk19II" in catalog.dtype.names:
        bmsk = catalog["brimsk19II"]
    elif "i_mask_brightstar_halo" in catalog.dtype.names:
        bmsk = (
            catalog["i_mask_brightstar_halo"]
            | catalog["i_mask_brightstar_ghost"]
            | catalog["i_mask_brightstar_blooming"]
        )
    else:
        bmsk = np.zeros(len(catalog), dtype=bool)
        warnings.warn("cannot find column for s19A Mask II")
    return bmsk


def get_pixel_cuts(catalog):
    """Returns pixel cuts"""
    mask = (
        (~catalog["i_deblend_skipped"])
        & (~catalog["i_cmodel_flag_badcentroid"])
        & (~catalog["i_sdsscentroid_flag"])
        & (
            ~catalog["i_detect_isprimary"]
            & (~catalog["i_pixelflags_edge"])
            & (~catalog["i_pixelflags_interpolatedcenter"])
            & (~catalog["i_pixelflags_saturatedcenter"])
            & (~catalog["i_pixelflags_crcenter"])
            & (~catalog["i_pixelflags_bad"])
            & (~catalog["i_pixelflags_suspectcenter"])
            & (~catalog["i_pixelflags_clipped"])
        )
    )
    return mask


def get_FPFS1_obs(data, Delta=2077.966, cRatio=4.0):
    if "fps_momentsG" in data.dtype.names:
        moments = data["fps_momentsG"]
    elif "fpfs_momentsG" in data.dtype.names:
        moments = data["fpfs_momentsG"]
    elif "fpfs_moments" in data.dtype.names:
        moments = data["fpfs_moments"]
    else:
        moments = np.empty((len(data), 4))
        moments.fill(np.nan)
        warnings.warn("cannot find columns for FPFS1 moments")
    # get weight
    const = Delta * cRatio
    weight = moments[:, 0] + const
    # get FPFS flux ratio
    flux = moments[:, 0] / weight
    e1 = moments[:, 1] / (moments[:, 0] + const)
    e2 = moments[:, 2] / (moments[:, 0] + const)
    num1 = -e1
    num2 = -e2
    denom1 = (
        1.0 / np.sqrt(2.0) * (moments[:, 0] - moments[:, 3]) / weight
        + np.sqrt(2) * (moments[:, 1] / weight) ** 2.0
    )
    denom2 = (
        1.0 / np.sqrt(2.0) * (moments[:, 0] - moments[:, 3]) / weight
        + np.sqrt(2) * (moments[:, 2] / weight) ** 2.0
    )
    return num1, num2, denom1, denom2, flux


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


def get_snr_apertures(catalog):
    """This utility computes the S/N for each object in the catalog, based on
    aperture_fluxes. It does not impose any cuts and returns NaNs for invalid
    S/N values.
    """
    if "i_apertureflux_10_fluxsigma" in catalog.dtype.names:  # s18
        snr10 = (
            catalog["i_apertureflux_10_flux"] / catalog["i_apertureflux_10_fluxsigma"]
        )
        snr15 = (
            catalog["i_apertureflux_15_flux"] / catalog["i_apertureflux_15_fluxsigma"]
        )
        snr20 = (
            catalog["i_apertureflux_20_flux"] / catalog["i_apertureflux_20_fluxsigma"]
        )
    elif "i_apertureflux_10_fluxerr" in catalog.dtype.names:  # s19
        snr10 = catalog["i_apertureflux_10_flux"] / catalog["i_apertureflux_10_fluxerr"]
        snr15 = catalog["i_apertureflux_15_flux"] / catalog["i_apertureflux_15_fluxerr"]
        snr20 = catalog["i_apertureflux_20_flux"] / catalog["i_apertureflux_20_fluxerr"]
    elif "base_CircularApertureFlux_3_0_instFlux" in catalog.dtype.names:  # pipe 7
        snr10 = (
            catalog["base_CircularApertureFlux_3_0_instFlux"]
            / catalog["base_CircularApertureFlux_3_0_instFluxErr"]
        )
        snr15 = (
            catalog["base_CircularApertureFlux_4_5_instFlux"]
            / catalog["base_CircularApertureFlux_4_5_instFluxErr"]
        )
        snr20 = (
            catalog["base_CircularApertureFlux_6_0_instFlux"]
            / catalog["base_CircularApertureFlux_6_0_instFluxErr"]
        )
    else:
        snr10 = _nan_array(len(catalog))
        snr15 = _nan_array(len(catalog))
        snr20 = _nan_array(len(catalog))
    return snr10, snr15, snr20


def get_snr_localBG(catalog):
    """This utility computes the S/N for each object in the catalog,
    based on local background flux. It does not impose any cuts
    and returns NaNs for invalid S/N values.
    """
    if "i_localbackground_fluxsigma" in catalog.dtype.names:  # s18
        snrloc = (
            catalog["i_localbackground_flux"] / catalog["i_localbackground_fluxsigma"]
        )
    elif "i_localbackground_fluxerr" in catalog.dtype.names:  # s19
        snrloc = (
            catalog["i_localbackground_flux"] / catalog["i_localbackground_fluxerr"]
        )
    elif "base_LocalBackground_instFlux" in catalog.dtype.names:  # pipe 7
        snrloc = (
            catalog["base_LocalBackground_instFlux"]
            / catalog["base_LocalBackground_instFluxErr"]
        )
    else:
        snrloc = _nan_array(len(catalog))
    return snrloc


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


def get_imag_lb(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "i_localbackground_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_localbackground_mag"]
    elif "base_LocalBackground_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = -2.5 * np.log10(catalog["base_LocalBackground_instFlux"]) + 27.0
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_bs_factor(catalog):
    ratio = np.pi * 9
    if "i_localbackground_flux" in catalog.dtype.names:  # s18
        sb = (
            catalog["i_apertureflux_10_flux"]
            / ratio
            / catalog["i_localbackground_flux"]
        )
        sb = 1.0 / sb
    elif "base_LocalBackground_instFlux" in catalog.dtype.names:  # pipe 7
        sb = (
            catalog["base_CircularApertureFlux_3_0_instFlux"]
            / ratio
            / catalog["base_LocalBackground_instFlux"]
        )
        sb = 1.0 / sb
    else:
        sb = _nan_array((len(catalog)))
    return sb


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


def get_npass(catalog, meas="cmodel"):
    """Returns npass values

    Args:
        catalog (ndarray):  input catalog
        meas (str):         SNR definition used to get npass [default: 'cmodel']
    Returns:
        npass (ndarray):    npass array
    """
    if "npass" in catalog.dtype.names:
        return catalog["npass"]
    elif "g_inputcount_value" in catalog.dtype.names:
        gcountinputs = catalog["g_inputcount_value"]
        rcountinputs = catalog["r_inputcount_value"]
        zcountinputs = catalog["z_inputcount_value"]
        ycountinputs = catalog["y_inputcount_value"]
        if meas == "cmodel":
            pendN = "cmodel_mag"
        elif meas == "aperture":
            pendN = "apertureflux_10_mag"
        else:
            return _nan_array(len(catalog))
        if "forced_g_%ssigma" % pendN in catalog.dtype.names:  # For S18A
            pendN += "sigma"
        if "forced_g_%serr" % pendN in catalog.dtype.names:  # For S19A
            pendN += "err"
        # multi-band detection to remove junk
        g_snr = (2.5 / np.log(10.0)) / catalog["forced_g_%s" % pendN]
        r_snr = (2.5 / np.log(10.0)) / catalog["forced_r_%s" % pendN]
        z_snr = (2.5 / np.log(10.0)) / catalog["forced_z_%s" % pendN]
        y_snr = (2.5 / np.log(10.0)) / catalog["forced_y_%s" % pendN]
    elif "gcountinputs" in catalog.dtype.names:
        gcountinputs = catalog["gcountinputs"]
        rcountinputs = catalog["rcountinputs"]
        zcountinputs = catalog["zcountinputs"]
        ycountinputs = catalog["ycountinputs"]
        # multi-band detection to remove junk
        g_snr = (2.5 / np.log(10.0)) / catalog["gmag_forced_cmodel_err"]
        r_snr = (2.5 / np.log(10.0)) / catalog["rmag_forced_cmodel_err"]
        z_snr = (2.5 / np.log(10.0)) / catalog["zmag_forced_cmodel_err"]
        y_snr = (2.5 / np.log(10.0)) / catalog["ymag_forced_cmodel_err"]
    else:
        out = _nan_array(len(catalog))
        return out

    # Calculate npass
    min_multiband_snr_data = 5.0
    g_mask = (g_snr >= min_multiband_snr_data) & (
        ~np.isnan(g_snr) & (gcountinputs >= 2)
    )
    r_mask = (r_snr >= min_multiband_snr_data) & (
        ~np.isnan(r_snr) & (rcountinputs >= 2)
    )
    z_mask = (z_snr >= min_multiband_snr_data) & (
        ~np.isnan(z_snr) & (zcountinputs >= 2)
    )
    y_mask = (y_snr >= min_multiband_snr_data) & (
        ~np.isnan(y_snr) & (ycountinputs >= 2)
    )
    npass = (
        g_mask.astype(int)
        + r_mask.astype(int)
        + z_mask.astype(int)
        + y_mask.astype(int)
    )
    return npass


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


def get_FDFC_flag(data, hpfname):
    """Returns the Full Depth Full Color (FDFC) cut

    Args:
        data (ndarray):     input catalog array
        hpfname (str):      healpix fname (s16a_wide2_fdfc.fits,
                            s18a_fdfc_hp_contarea.fits, or
                            s19a_fdfc_hp_contarea_izy-gt-5_trimmed_fd001.fits)
    Returns:
        mask (ndarray):     mask array for FDFC region
    """
    ra, dec = get_radec(data)
    m = hp.read_map(hpfname, nest=True, dtype=bool)

    # Get flag
    mfactor = np.pi / 180.0
    indices_map = np.where(m)[0]
    nside = hp.get_nside(m)
    phi = ra * mfactor
    theta = np.pi / 2.0 - dec * mfactor
    indices_obj = hp.ang2pix(nside, theta, phi, nest=True)
    return np.in1d(indices_obj, indices_map)


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
    """This utility gets the observed galaxy size from a data or sims catalog
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
    """Returns the logb"""
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


def get_logbAll(catalog):
    """Returns the logb"""
    if "base_Blendedness_abs" in catalog.dtype.names:  # pipe 7
        logbA = np.log10(np.maximum(catalog["base_Blendedness_abs"], 1.0e-6))
        logbR = np.log10(np.maximum(catalog["base_Blendedness_raw"], 1.0e-6))
        logbO = np.log10(np.maximum(catalog["base_Blendedness_old"], 1.0e-6))
    elif "i_blendedness_abs_flux" in catalog.dtype.names:  # s18
        logbA = np.log10(np.maximum(catalog["i_blendedness_abs_flux"], 1.0e-6))
        logbR = np.log10(np.maximum(catalog["i_blendedness_raw_flux"], 1.0e-6))
        logbO = np.log10(np.maximum(catalog["i_blendedness_old"], 1.0e-6))
    elif "i_blendedness_abs" in catalog.dtype.names:  # s19
        logbA = np.log10(np.maximum(catalog["i_blendedness_abs"], 1.0e-6))
        logbR = np.log10(np.maximum(catalog["i_blendedness_raw"], 1.0e-6))
        logbO = np.log10(np.maximum(catalog["i_blendedness_old"], 1.0e-6))
    else:
        logbA = _nan_array(len(catalog))
        logbR = _nan_array(len(catalog))
        logbO = _nan_array(len(catalog))
    return logbA, logbR, logbO


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


def get_true_shear(catalog):
    """
    This routine accesses the truth tables to get the true shear in the
    Simulation.

    Args:
        catalog (ndarray):   input catalog
    Returns:
        g1_true (ndarray):  input shear g1
        g2_true (ndarray):  input shear g2
    """
    if "g1" in catalog.dtype.names:
        return catalog["g1"], catalog["g2"]
    elif "g1_true" in catalog.dtype.names:
        return catalog["g1_true"], catalog["g2_true"]
    else:
        raise NameError("input catalog does not contain g_1/g_2 or g1_true/g2_true")


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


def get_noi_var(catalog):
    if "noivar" in catalog.dtype.names:  # smallcat
        varNois = catalog["noivar"]
    elif "ivariance" in catalog.dtype.names:  # s18&s19
        varNois = catalog["forced_ivariance"]
    elif "i_variance_value" in catalog.dtype.names:  # s18&s19
        varNois = catalog["i_variance_value"]
    elif "base_Variance_value" in catalog.dtype.names:  # sim
        varNois = catalog["base_Variance_value"]
    else:
        varNois = _nan_array(len(catalog))
    return varNois


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


def get_sdss_ellip(catalog, return_shear=False):
    """This utility gets the SDSS ellipticity (uncalibrated shear) from a data
    or sims catalog.
    """
    if "i_sdssshape_shape11" in catalog.dtype.names:  # s19
        psf_mxx = catalog["i_sdssshape_shape11"]
        psf_myy = catalog["i_sdssshape_shape22"]
        psf_mxy = catalog["i_sdssshape_shape12"]
    elif "ishape_sdss_ixx" in catalog.dtype.names:  # s16
        psf_mxx = catalog["ishape_sdss_ixx"]
        psf_myy = catalog["ishape_sdss_iyy"]
        psf_mxy = catalog["ishape_sdss_ixy"]
    elif "base_SdssShape_xx" in catalog.dtype.names:  # pipeline
        psf_mxx = catalog["base_SdssShape_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_xy"] * 0.168**2.0
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


def get_sigma_e_model(catalog, pltDir="./plot/optimize_weight/"):
    """This utility returns a model for the shape measurement uncertainty as a
    function of SNR and resolution.  It uses the catalog directly to get the
    SNR and resolution values.

    The file storing the data used to build the approximate correction is
    expected to be found in plot/sigmae_ratio.dat
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
    par = np.load(os.path.join(pltDir, "sigma_e_model_par.npy"), allow_pickle=True)[0]
    sigma_e = np.exp(par[2]) * ((snr / 20.0) ** par[0]) * ((res / 0.5) ** par[1])

    # Get the corrections from interpolation amongst saved values.
    dat = np.loadtxt(os.path.join(pltDir, "sigmae_ratio.dat")).transpose()
    saved_snr = dat[0, :]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat[1, :]
    saved_corr = dat[2, :]

    # Interpolate the corrections (which multiply the power-law results).
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_corr, log_snr, res)
    return result * sigma_e


def get_erms_model(catalog, pltDir="./plot/optimize_weight/"):
    """This utility returns a model for the RMS ellipticity as a function of
    SNR and resolution.  It uses the catalog directly to get the SNR and
    resolution values.

    The file storing the data used to build the model is expected to be found in
    plot/eRMS/intrinsicshape_2d.dat

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
    dat = np.loadtxt(os.path.join(pltDir, "intrinsicshape_2d.dat")).transpose()
    saved_snr = dat[0, :]
    log_saved_snr = np.log10(saved_snr)

    saved_res = dat[1, :]
    saved_model = dat[2, :]

    # Interpolate the e_rms values and return them.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_model, log_snr, res)
    return result


def get_weight_model(catalog, pltDir="./plot/optimize_weight/"):
    """
    This utility returns a model for the shape measurement weight as a
    function of SNR and resolution.  It relies on two other routines
    to get models for the intrinsic shape RMS and measurement error.
    """
    sigmae_meas = get_sigma_e_model(catalog, pltDir)
    erms = get_erms_model(catalog, pltDir)
    return 1.0 / (sigmae_meas**2 + erms**2)


def get_m_model(
    catalog, weight_bias=True, photo_z_dep=True, pltDir="./plot/reGausCalib/"
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
    maFname = os.path.join(pltDir, "shear_m_a_model_par.npy")
    m_opt = np.load(maFname, allow_pickle=True).item()["m_opt"]
    fake_x = np.vstack((snr, res))
    model_m = m_func(fake_x, *m_opt)

    data_file = os.path.join(pltDir, "shear_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_m = dat["dm"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_m, log_snr, res)

    if weight_bias:
        result += get_mwt_model(catalog, pltDir=pltDir)

    if photo_z_dep:
        pzDir = os.path.join(pltDir[:-12], "sanityTest")
        z_file = os.path.join(pzDir, "dnn_z_bin_dm_da_2d.csv")
        zat = Table.read(z_file)
        saved_z = zat["z"]
        saved_m = zat["dm"]
        result += grid_interpolate_1d(
            saved_z, saved_m, np.array(get_photo_z(catalog, "dnn"))
        )

    model_m = model_m + result
    return model_m


def get_mwt_model(catalog, pltDir="./plot/reGausCalib/"):
    """Routine to get a model for calibration bias m due to weight bias, given
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
    maFname = os.path.join(pltDir, "weightBias_m_a_model_par.npy")
    m_opt = np.load(maFname, allow_pickle=True).item()["m_opt"]
    fake_x = np.vstack((snr, res))
    model_m = mwt_func(fake_x, *m_opt)

    data_file = os.path.join(pltDir, "weightBias_dm_da_2d.csv")
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
    catalog, weight_bias=True, photo_z_dep=True, pltDir="./plot/reGausCalib/"
):
    """Routine to get a model for additive bias coefficient a given some input
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

    maFname = os.path.join(pltDir, "shear_m_a_model_par.npy")
    a_opt = np.load(maFname, allow_pickle=True).item()["a_opt"]
    fake_x = np.vstack((snr, res))
    model_a = a_func(fake_x, *a_opt)

    data_file = os.path.join(pltDir, "shear_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_a = dat["da"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_a, log_snr, res)
    if weight_bias:
        result += get_awt_model(catalog, pltDir=pltDir)

    if photo_z_dep:
        pzDir = os.path.join(pltDir[:-12], "sanityTest")
        z_file = os.path.join(pzDir, "dnn_z_bin_dm_da_2d.csv")
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


def get_awt_model(catalog, pltDir="./plot/reGausCalib/"):
    """Routine to get a model for additive bias coefficient a due to weight
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
    maFname = os.path.join(pltDir, "weightBias_m_a_model_par.npy")
    a_opt = np.load(maFname, allow_pickle=True).item()["a_opt"]
    fake_x = np.vstack((snr, res))
    model_a = a_func(fake_x, *a_opt)

    data_file = os.path.join(pltDir, "weightBias_dm_da_2d.csv")
    dat = Table.read(data_file)
    saved_snr = dat["snr"]
    log_saved_snr = np.log10(saved_snr)
    saved_res = dat["res"]
    saved_a = dat["da"]
    # Interpolate the model residuals and return them.  These are additional
    # biases beyond the power-law model and so should be added to the power-law.
    result = grid_interpolate_2d(log_saved_snr, saved_res, saved_a, log_snr, res)
    return result + model_a


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


def update_wl_cuts(catalog):
    """Update the weak-lensing cuts"""
    catalog["weak_lensing_flag"] = get_wl_cuts(catalog)
    return catalog


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


def update_reGaus_calibration(catalog):
    """Updates the columns derived from calibration"""
    sigmae = get_sigma_e_model(catalog)
    catalog["i_hsmshaperegauss_derived_sigma_e"] = sigmae
    erms = get_erms_model(catalog)
    catalog["i_hsmshaperegauss_derived_rms_e"] = erms
    catalog["i_hsmshaperegauss_derived_weight"] = 1.0 / (sigmae**2 + erms**2)
    model_m = get_m_model(catalog, weight_bias=True, photo_z_dep=True)
    catalog["i_hsmshaperegauss_derived_shear_bias_m"] = model_m
    model_c1, model_c2 = get_c_model(catalog, weight_bias=True, photo_z_dep=True)
    catalog["i_hsmshaperegauss_derived_shear_bias_c1"] = model_c1
    catalog["i_hsmshaperegauss_derived_shear_bias_c2"] = model_c2
    return catalog


def get_mask_visit_104994(data):
    """We found that visit 104994 has tracking errors, but that visit contributes
    to coadds, we remove this region from the catalog level

    Args:
        data (ndarray): input catalog
    Returns:
        mask (ndarray): mask removing the problematic region
    """
    ra, dec = get_radec(data)

    def _calDistanceAngle(a1, d1):
        """Returns the angular distance on sphere
        a1 (ndarray): ra of galaxies
        d1 (ndarray): dec of galaxies
        """
        a2 = 130.43
        d2 = -1.02
        a1_f64 = np.array(a1, dtype=np.float64) * np.pi / 180.0
        d1_f64 = np.array(d1, dtype=np.float64) * np.pi / 180.0
        a2_f64 = np.array(a2, dtype=np.float64) * np.pi / 180.0
        d2_f64 = np.array(d2, dtype=np.float64) * np.pi / 180.0
        return (
            np.arccos(
                np.cos(d1_f64) * np.cos(d2_f64) * np.cos(a1_f64 - a2_f64)
                + np.sin(d1_f64) * np.sin(d2_f64)
            )
            / np.pi
            * 180.0
        )

    d = _calDistanceAngle(ra, dec)
    mask = (ra > 130.5) & (ra < 131.5) & (dec < -1.5)
    return (d > 0.80) & (~mask)


def del_colnull(data):
    """Deletes the '_isnull' column from the catalog downloaded from database

    Args:
        data (ndarray):     catalog downloaded from database
    Returns:
        data (ndarray):     catalog after removing '_isnull' column
    """
    colns = data.dtype.names
    colns2 = [cn for cn in colns if "_isnull" not in cn]
    data = data[colns2]
    return data


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


def get_binarystar_flags(data):
    """Returns the flags for binary stars (|e|>0.8 & logR<1.8-0.1r)

    Args:
        data (ndarray): an hsc-like catalog

    Returns:
        mask (ndarray):  a boolean (True for binary stars)
    """
    absE = get_abs_ellip(data)
    logR = np.log10(get_sdss_size(data))
    rmag = data["forced_r_cmodel_mag"] - data["a_r"]
    mask = absE > 0.8
    a = 1
    b = 10.0
    c = -18.0
    mask = mask & ((a * rmag + b * logR + c) < 0.0)
    return mask


def get_flag_infield_s19a(ra, dec, fieldname):
    """Returns the flags for each field

    Args:
        ra (ndarray):   ra
        dec (ndarrau):  dec
        fieldname (str):field name
    Returns:
        mask(ndarray):  a boolean mask
    """
    if fieldname == "XMM":
        ramin = 29.0
        ramax = 39.5
        decmin = -6.40
        decmax = -1.19
    elif fieldname == "HECTOMAP":
        ramin = 212.0
        ramax = 250.5
        decmin = 42.21
        decmax = 44.40
    elif fieldname == "GAMA09H":
        ramin = 128
        ramax = 153.50
        decmin = -1.90
        decmax = 4.71
    elif fieldname == "WIDE12H":
        ramin = 153.50
        ramax = 200.0
        decmin = -1.68
        decmax = 4.71
    elif fieldname == "GAMA15H":
        ramin = 206
        ramax = 226.0
        decmin = -1.72
        decmax = 1.31
    elif fieldname == "VVDS":
        ramin = 330.0
        ramax = 363.5
        decmin = -1.01
        decmax = 5.82
    else:
        raise ValueError("input field name incorrect")
    mask = (ra > ramin) & (ra < ramax) & (dec > decmin) & (dec < decmax)
    return mask


def get_mask_G09_good_seeing(data):
    """Gets the mask for the good-seeing region with large high order PSF shape
    residuals

    Parameters:
        data (ndarray):     HSC shape catalog data
    Returns:
        mm (ndarray):       mask array [if False, in the good-seeing region]
    """
    (ra, dec) = get_radec(data)
    mm = (ra >= 132.5) & (ra <= 140.0) & (dec >= 1.6) & (dec < 5.2)
    mm = ~mm
    return mm


def get_shape_weight_regauss(catalog):
    """This utility returns the i-band reGauss shape weight"""
    if "i_hsmshaperegauss_derived_weight" in catalog.dtype.names:  # s19
        weight = catalog["i_hsmshaperegauss_derived_weight"]
    elif "ishape_hsm_regauss_derived_shape_weight" in catalog.dtype.names:  # s16
        weight = catalog["ishape_hsm_regauss_derived_shape_weight"]
    else:
        weight = _nan_array(len(catalog))
    return weight
