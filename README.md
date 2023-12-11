This repository contains open-source python software that provides the weight
and calibration factors associated with the three-year (S19A) shape catalog
from the HSC SSP survey ([catalog
paper](https://ui.adsabs.harvard.edu/abs/2022PASJ...74..421L/abstract)). The
software is available for use under a BSD 3-clause license contained in the
file [LICENSE.md](LICENSE). See
[here](https://github.com/PrincetonUniversity/hsc-y1-shear-calib/blob/main/gen_hsc_calibrations.py)
for the hsc first-year (S16A) calibration.

Currently the script [gen_hsc_calibrations.py](gen_hsc_calibrations.py) is set
up to read in a FITS catalog containing the columns downloaed from the HSC data
base , including re-Gaussianization shape measurements and CModel magnitudes
and errors, and produce a catalog with the quantities used for ensemble weak
lensing shear estimation. (The contents of [utilities.py](utilities.py) and of
data/ are accessed by [gen_hsc_calibrations.py](gen_hsc_calibrations.py), but
users need not interact with them directly.) For information about
dependencies, command-line arguments, and memory usage, please read the
docstring for [gen_hsc_calibrations.py](gen_hsc_calibrations.py).

## required packages
+ numpy
+ scipy
+ astropy

