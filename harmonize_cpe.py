# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:13:56 2022

@author: Bas van der Velden
"""

import argparse
import numpy as np
import SimpleITK as sitk


def div0(a, b):
    """
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]
    from https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def harmonize_postcontrast(
    s_precontrast_arr: np.ndarray,
    s_postcontrast_arr: np.ndarray,
    flipangle_input: float,
    TR_input: float,
    flipangle_output: float,
    TR_output: float,
    T1_precontrast: float,
    pd: float,
) -> np.ndarray:
    """
    Function to harmonize a postcontrast image from T1-weighted dynamic contrast-
    enhanced MRI with respect to flip angle and repetition time.
    Source: van der Velden et al.
    JMRI 2020: Material and Methods,Harmonization of Parenchymal Enhancement

    Parameters
    ----------
    s_precontrast_arr : np.ndarray
        Numpy array containing the precontrast image of the DCE-MRI
    s_postcontrast_arr : np.ndarray
        Numpy array containing the postcontrast image (at time t)
        of the DCE-MRI to be harmonized
    flipangle_input : float
        The flipangle in degrees with which the DCE-MRI was acquired
    TR_input : float
        The repetition time in ms with which the DCE-MRI was acquired
    flipangle_output : float
        The flipangle in degrees to which s_t should be harmonized
    TR_output : float
        The repetition time in ms to which s_t should be harmonized
    T1_precontrast : float
        T1 value of tissue of interest at precontrast, e.g. 1266.18ms for
        fibroglandular tissue at 1.5T (Rakow-Penner et al., JMRI 2006).
    pd : float
        proton density of the tissue of interest

    Returns
    -------
    np.ndarray
        Harmonized postcontrast numpy array
    """

    fa_input_rad = np.radians(flipangle_input)
    fa_output_rad = np.radians(flipangle_output)

    ## 1: calculate T1_x
    y = div0(s_postcontrast_arr, s_precontrast_arr)
    e_pow = np.exp(-TR_input / T1_precontrast)
    x_num = 1 - (e_pow * np.cos(fa_input_rad))
    x_den = 1 - e_pow
    x = x_num / x_den

    with np.errstate(invalid="ignore"):
        T1_x = div0(-TR_input, np.log(div0((x - y), (x - (y * np.cos(fa_input_rad))))))

    ## 2: calculate s_x at new flipangle and TR
    e_pow2 = np.exp(div0(-TR_output, T1_x))
    s_num = np.sin(fa_output_rad) * (1 - e_pow2)
    s_den = 1 - (e_pow2 * np.cos(fa_output_rad))
    s_postcontrast_harm_arr = pd * div0(s_num, s_den)

    return s_postcontrast_harm_arr


def cpe(
    s_firstpostcontrast_arr: np.ndarray,
    s_lastpostcontrast_arr: np.ndarray,
    fgt_arr: np.ndarray,
    cpe_lowcutoff: float = 0.9,
    cpe_highcutoff: float = 1.0,
) -> np.float64:
    """
    Function to calculate contralateral parenchymal enhancement (CPE) from early and
    late postcontrast scan.
    Source: van der Velden et al. Radiology 2015.
    Optimal cutoffs source: van der Velden et al. Clinical Cancer Research 2017.

    Parameters
    ----------
    s_firstpostcontrast_arr : np.ndarray
        Numpy array containing the first postcontrast image of the DCE-MRI
    s_lastpostcontrast_arr : np.ndarray
        Numpy array containing the last postcontrast image of the DCE-MRI
    fgt_arr : np.ndarray
        Numpy array containing the binary fibroglandular tissue segmentation of
        healthy contralateral breast
    cpe_lowcutoff : float, optional
        Low end of cpe cutoff (standard 0.9 i.e. 90%)
    cpe_highcutoff : float, optional
        High end of cpe cutoff (standard 1.0 i.e. 100%)

    Returns
    -------
    np.float64
        Mean CPE of top x% voxels within mask
    """

    # Calculate CPE of all voxels
    cpe_image = div0(
        s_lastpostcontrast_arr - s_firstpostcontrast_arr, s_firstpostcontrast_arr
    )

    # Calculate mean CPE of top voxels within mask
    cpe_values = np.sort(cpe_image[np.nonzero(fgt_arr)])
    lowcutoff = int(len(cpe_values) * cpe_lowcutoff)
    highcutoff = int(len(cpe_values) * cpe_highcutoff)

    cpe = np.mean(cpe_values[lowcutoff:highcutoff])

    return cpe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to harmonize contralateral parenchymal enhancement (CPE) \
        on DCE-MRI across acquisition parameters"
    )
    parser.add_argument(
        "s_precontrast_im",
        help="filename of the precontrast image",
        type=sitk.ReadImage,
    )
    parser.add_argument(
        "s_firstpostcontrast_im",
        help="filename of the first postcontrast image",
        type=sitk.ReadImage,
    )
    parser.add_argument(
        "s_lastpostcontrast_im",
        help="filename of the last postcontrast image",
        type=sitk.ReadImage,
    )
    parser.add_argument(
        "fgt_im",
        help="filename of the fibroglandular tissue segmentation image",
        type=sitk.ReadImage,
    )
    parser.add_argument(
        "flipangle_input",
        help="flip angle [degrees] at which the DCE-MRI was acquired",
        type=float,
    )
    parser.add_argument(
        "TR_input",
        help="repetition time [ms] at which the DCE-MRI was acquired",
        type=float,
    )
    parser.add_argument(
        "flipangle_output",
        help="flip angle [degrees] to which the postcontrast image should be harmonized",
        type=float,
    )
    parser.add_argument(
        "TR_output",
        help="repetition time [ms] to which the postcontrast image should be harmonized",
        type=float,
    )
    parser.add_argument(
        "T1_precontrast",
        help="T1 value [ms] of tissue of interest at pre-contrast",
        type=float,
    )
    parser.add_argument(
        "pd", help="proton density of the tissue of interest", type=float
    )

    args = parser.parse_args()

    # Load images
    s_precontrast_arr = sitk.GetArrayFromImage(args.s_precontrast_im)
    s_firstpostcontrast_arr = sitk.GetArrayFromImage(args.s_firstpostcontrast_im)
    s_lastpostcontrast_arr = sitk.GetArrayFromImage(args.s_lastpostcontrast_im)
    fgt_arr = sitk.GetArrayFromImage(args.fgt_im)

    # Harmonize postcontrast images
    s_firstpostcontrast_harm_arr = harmonize_postcontrast(
        s_precontrast_arr,
        s_firstpostcontrast_arr,
        args.flipangle_input,
        args.TR_input,
        args.flipangle_output,
        args.TR_output,
        args.T1_precontrast,
        args.pd,
    )

    s_lastpostcontrast_harm_arr = harmonize_postcontrast(
        s_precontrast_arr,
        s_lastpostcontrast_arr,
        args.flipangle_input,
        args.TR_input,
        args.flipangle_output,
        args.TR_output,
        args.T1_precontrast,
        args.pd,
    )

    # Calculate CPE before and after harmonization
    cpe_not_harm = cpe(s_firstpostcontrast_arr, s_lastpostcontrast_arr, fgt_arr)
    cpe_harm = cpe(s_firstpostcontrast_harm_arr, s_lastpostcontrast_harm_arr, fgt_arr)

    print(
        "CPE before harmonization {0:.3f} and after harmonization {1:.3f}".format(
            cpe_not_harm, cpe_harm
        )
    )
