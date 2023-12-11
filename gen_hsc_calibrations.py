#!/usr/bin/env python
import gc
import os
import numpy as np
from astropy.table import Table
import catutil as utilities

del_columns = [
    'i_hsmshaperegauss_derived_bias_m',
    'i_hsmshaperegauss_derived_bias_c1',
    'i_hsmshaperegauss_derived_bias_c2']

anaDir  =   'anaCat_newS19Mask_fdeltacut'
outDir  =   os.path.join(anaDir,'catalog_obs_reGaus_calibrated/fields/')
if not os.path.exists(outDir):
    os.system('mkdir -p %s' %outDir)

obsDir  =   '../../database/s19-Hironao-2/fields/'
if not os.path.isdir(obsDir):
    raise RuntimeError("Data catalog directory %s does not exist "
        "or is not a directory" % (obsDir))

def main():
    for fieldname in utilities.field_names:
        print('processing field: %s' %fieldname)
        obsFname=   os.path.join(obsDir,'%s.fits' %(fieldname))
        zFname  =   os.path.join(obsDir,'%s_pz.fits' %(fieldname))
        outFname=   os.path.join(outDir,'%s_calibrated.fits' %(fieldname))
        data    =   Table.read(obsFname)
        zdat    =   Table.read(zFname)
        data['dnnz_photoz_best']      =   zdat['dnnz_photoz_best']
        data['dnnz_photoz_conf_best'] =   zdat['dnnz_photoz_conf_best']
        data['dnnz_photoz_std_best']  =   zdat['dnnz_photoz_std_best']
        data['dnnz_photoz_err95_min'] =   zdat['dnnz_photoz_err95_min']
        data['dnnz_photoz_err95_max'] =   zdat['dnnz_photoz_err95_max']
        data    =   utilities.update_wl_flags(data)
        mask    =   data['weak_lensing_flag']
        data    =   data[mask]
        data    =   utilities.update_reGaus_calibration(data)
        mask    =   (~np.isnan(data['i_hsmshaperegauss_derived_shear_bias_c1']))
        mask    &=  (~np.isnan(data['i_hsmshaperegauss_derived_shear_bias_c2']))
        mask    &=  (~np.isnan(data['i_hsmshaperegauss_derived_shear_bias_m']))
        mask    &=  utilities.get_FDFC_flag(data,'s19')
        mask    &=  utilities.get_mask_visit_104994(data)
        data    =   data[mask]
        # delete unnecessary columns
        for entry in del_columns:
            del data[entry]
        data.write(outFname,overwrite=True)
        del data
        del zdat
        gc.collect()
    return

if __name__=='__main__':
    main()
