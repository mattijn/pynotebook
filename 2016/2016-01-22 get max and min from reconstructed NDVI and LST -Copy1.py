
# coding: utf-8

# In[1]:

from osgeo import gdal, ogr, gdalconst
import os
import subprocess as sp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# In[2]:

def saveRaster(path, array, dsSource, datatype=3, formatraster="GTiff", nan=None): 
    """
    Datatypes:
    unknown = 0
    byte = 1
    unsigned int16 = 2
    signed int16 = 3
    unsigned int32 = 4
    signed int32 = 5
    float32 = 6
    float64 = 7
    complex int16 = 8
    complex int32 = 9
    complex float32 = 10
    complex float64 = 11
    float32 = 6, 
    signed int = 3
    
    Formatraster:
    GeoTIFF = GTiff
    Erdas = HFA (output = .img)
    OGC web map service = WMS
    png = PNG
    """
    # Set Driver
    format_ = formatraster #save as format
    driver = gdal.GetDriverByName( format_ )
    driver.Register()
    
    # Set Metadata for Raster output
    cols = dsSource.RasterXSize
    rows = dsSource.RasterYSize
    bands = dsSource.RasterCount
    datatype = datatype#band.DataType
    
    # Set Projection for Raster
    outDataset = driver.Create(path, cols, rows, bands, datatype)
    geoTransform = dsSource.GetGeoTransform()
    outDataset.SetGeoTransform(geoTransform)
    proj = dsSource.GetProjection()
    outDataset.SetProjection(proj)
    
    # Write output to band 1 of new Raster and write NaN value
    outBand = outDataset.GetRasterBand(1)
    if nan != None:
        outBand.SetNoDataValue(nan)
    outBand.WriteArray(array) #save input array
    #outBand.WriteArray(dem)
    
    # Close and finalise newly created Raster
    #F_M01 = None
    outBand = None
    proj = None
    geoTransform = None
    outDataset = None
    driver = None
    datatype = None
    bands = None
    rows = None
    cols = None
    driver = None
    array = None


# In[3]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[4]:

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[ ]:

# get index from tif files
files = listall(r'J:\LST_recon', extension='.tif')


# In[ ]:

base_max = gdal.Open(files[0], gdalconst.GA_ReadOnly).ReadAsArray()
base_min = gdal.Open(files[0], gdalconst.GA_ReadOnly).ReadAsArray()
for idx, file_ in enumerate(files):
    print idx
    ds = gdal.Open(file_, gdalconst.GA_ReadOnly).ReadAsArray()
    base_max = np.maximum.reduce([ds, base_max])
    base_min = np.minimum.reduce([ds, base_min])


# In[ ]:

ds_base = gdal.Open(file_, gdalconst.GA_ReadOnly)


# In[ ]:

saveRaster(r'J:\MAX_MIN_NDVI_recon_LST_recon//LST_recon_min_int.tif', base_min, ds_base, datatype = 3)
saveRaster(r'J:\MAX_MIN_NDVI_recon_LST_recon//LST_recon_max_int.tif', base_max, ds_base, datatype = 3)


# In[ ]:

im = plt.imshow(base_max)
plt.colorbar(im)


# In[ ]:



