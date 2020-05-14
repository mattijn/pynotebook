
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


# In[5]:

# get index from tif files
files_LST  = listall(r'J:\LST_recon', extension='.tif')
files_NDVI = listall(r'J:\NDVI_recon', extension='.tif')
#max_LST = r'J:\MAX_MIN_NDVI_recon_LST_recon//LST_recon_max_int.tif'
#min_LST = r'J:\MAX_MIN_NDVI_recon_LST_recon//LST_recon_min_int.tif'
max_NDVI = r'J:\MAX_MIN_NDVI_recon_LST_recon//NDVI_recon_max_int.tif'
min_NDVI = r'J:\MAX_MIN_NDVI_recon_LST_recon//NDVI_recon_min_int.tif'
alpha = 0.5


# In[6]:

LST_dates = []
LST_doy_list = []
for LST in files_LST:
    #print LST
    LST_year = int(LST[-22:-18])
    LST_doy = int(LST[-7:-4])
    LST_date = datetime(LST_year, 1, 1) + timedelta(LST_doy - 1)
    LST_string = str(LST_date.year).zfill(2)+'-'+str(LST_date.month).zfill(2)+'-'+str(LST_date.day).zfill(2)
    #LST_date = np.datetime64(LST_date)
    #LST_date = pd.Timestamp(np.datetime_as_string(LST_date))    
    LST_dates.append(LST_date)
    LST_yday = LST_date.timetuple().tm_yday
    LST_doy_list.append(LST_yday)    

NDVI_dates = []
NDVI_doy_list = []
for NDVI in files_NDVI:
    #print NDVI
    NDVI_year = int(NDVI[-23:-19])
    NDVI_doy = int(NDVI[-7:-4])
    NDVI_date = datetime(NDVI_year, 1, 1) + timedelta(NDVI_doy - 1)
    NDVI_string = str(NDVI_date.year).zfill(2)+'-'+str(NDVI_date.month).zfill(2)+'-'+str(NDVI_date.day).zfill(2)    
    #NDVI_date = np.datetime64(NDVI_date)
    #NDVI_date = pd.Timestamp(np.datetime_as_string(NDVI_date))
    NDVI_dates.append(NDVI_date)
    NDVI_yday = NDVI_date.timetuple().tm_yday
    NDVI_doy_list.append(NDVI_yday)


# In[ ]:

for idx, date in enumerate(LST_dates[2:365]):
    print date
    print 'LST'
    LST_in = LST_dates[idx]
    LST_single_doy_all_year = np.array(LST_dates)[np.array(LST_doy_list) == date.timetuple().tm_yday]
    
    # calculate maximum and minimum based on selection
    base_file = files_LST[LST_dates.index(LST_single_doy_all_year[0])]
    base_mean = gdal.Open(base_file, gdalconst.GA_ReadOnly).ReadAsArray()
    base_mean -= base_mean
    for jdx, doy in enumerate(LST_single_doy_all_year):
        #print files_LST[LST_dates.index(doy)]
        ds = gdal.Open(files_LST[LST_dates.index(doy)], gdalconst.GA_ReadOnly).ReadAsArray()
        base_mean += ds
        #base_max = np.maximum.reduce([ds, base_max])
        #base_min = np.minimum.reduce([ds, base_min])        
    # save max and min LST
    base_mean /= len(LST_single_doy_all_year)
    ds_base = gdal.Open(files_LST[LST_dates.index(doy)], gdalconst.GA_ReadOnly)
    date_of_year = str(date.timetuple().tm_yday).zfill(3)
    saveRaster(r'J:\MAX_MIN_NDVI_recon_LST_recon\doy_LST//LST_MEAN_'+date_of_year+'.tif', 
               base_mean, ds_base)
    #saveRaster(r'J:\MAX_MIN_NDVI_recon_LST_recon\doy_LST//LST_MIN_'+date_of_year+'.tif', 
    #           base_min, ds_base)    
    
    print 'NDVI'    
    NDVI_in = NDVI_dates.index(date)
    NDVI_single_doy_all_year = np.array(NDVI_dates)[np.array(NDVI_doy_list) == date.timetuple().tm_yday]
    
    # calculate maximum and minimum based on selection
    base_file = files_NDVI[NDVI_dates.index(NDVI_single_doy_all_year[0])]
    base_mean = gdal.Open(base_file, gdalconst.GA_ReadOnly).ReadAsArray()
    base_mean -= base_mean   
    for jdx, doy in enumerate(NDVI_single_doy_all_year):
        #print files_NDVI[NDVI_dates.index(doy)]
        ds = gdal.Open(files_NDVI[NDVI_dates.index(doy)], gdalconst.GA_ReadOnly).ReadAsArray()
        base_mean += ds
        #base_max = np.maximum.reduce([ds, base_max])
        #base_min = np.minimum.reduce([ds, base_min])         
    # save max and min LST
    ds_base = gdal.Open(files_NDVI[NDVI_dates.index(doy)], gdalconst.GA_ReadOnly)
    base_mean /= len(LST_single_doy_all_year)
    date_of_year = str(date.timetuple().tm_yday).zfill(3)
    saveRaster(r'J:\MAX_MIN_NDVI_recon_LST_recon\doy_NDVI//NDVI_MEAN_'+date_of_year+'.tif', 
               base_mean, ds_base)
    #saveRaster(r'J:\MAX_MIN_NDVI_recon_LST_recon\doy_NDVI//NDVI_MIN_'+date_of_year+'.tif', 
    #           base_min, ds_base)        


# In[ ]:

plt.imshow(base_min)


# In[ ]:


LST_date = datetime(LST_year, 1, 1) + timedelta(LST_doy - 1)
LST_date = np.datetime64(LST_date)
LST_date = pd.Timestamp(np.datetime_as_string(LST_date))
LST_date
NDVI_date = datetime(NDVI_year, 1, 1) + timedelta(NDVI_doy - 1)
NDVI_date = np.datetime64(NDVI_date)
NDVI_date = pd.Timestamp(np.datetime_as_string(NDVI_date))
NDVI_date


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



