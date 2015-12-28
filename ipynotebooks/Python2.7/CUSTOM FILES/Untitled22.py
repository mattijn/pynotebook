# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
from osgeo import gdal, gdalconst

# <codecell>

def FilesFolder(inGSODFolder, format_end=''):
    st_wmo = [os.path.join(root, name)
               for root, dirs, files in os.walk(inGSODFolder)
                 for name in files                 
                 if name.endswith(format_end)]
    return st_wmo

# <codecell>

import numpy as np

# <codecell>

TRMM_LIST = FilesFolder(r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMClip_Day', format_end='.tif')

# <codecell>

TRMM_LIST[0]

# <codecell>

ds = gdal.Open(TRMM_LIST[0])

# <codecell>

array = ds.ReadAsArray()

# <codecell>

array_msked = np.ma.masked_equal(array,0)
array_msked

# <codecell>


