#!/usr/bin/env python
import gc
import os
import sys
import numpy as np
from astropy.table import Table
import utils

def main(input_fname, output_fname):
    """
    Main function to output a FITS catalog with calibrations and weights, given
    some input catalog based on the LSST Science Pipelines image processing
    routines.

    Usage: The routine is run as follows:

        python gen_hsc_calibrations.py input_file output_file_name

    It will read in the input file (`input_file`) in FITS format, and produce
    an output FITS file named `output_file_name` with the new information.

    Dependencies: To run this script, you will need to have installed astropy
    and numpy.

    """
    data =  Table.read(input_fname)
    out =  utils.make_reGauss_calibration_table(data)
    out.write(output_fname, overwrite=True)
    return

if __name__=='__main__':
    if len(sys.argv) != 3:
        raise RuntimeError(
            "Wrong number of args; see below for usage info!\n" + str(main.__doc__)
        )

    input_fname = sys.argv[1]
    output_fname = sys.argv[2]
    main(input_fname, output_fname)
