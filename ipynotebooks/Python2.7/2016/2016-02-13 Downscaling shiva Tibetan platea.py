
# coding: utf-8

# In[1]:

import numpy as np
from osgeo import gdal
import os
import subprocess as sp
import matplotlib.pyplot as plt
import datetime
get_ipython().magic(u'matplotlib inline')


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
    
    if bands > 1:
        for i in range(bands):
            # Write output to band 1 of new Raster and write NaN value
            outBand = outDataset.GetRasterBand(i+1)
            if nan != None:
                outBand.SetNoDataValue(nan)
            outBand.WriteArray(array[i]) #save input array
            #outBand.WriteArray(dem)
    else:
        outBand = outDataset.GetRasterBand(bands)        
        if nan != None:
            outBand.SetNoDataValue(nan)        
        outBand.WriteArray(array) #save input array            
    
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

def listall(RootFolder, varname='',extension='.tif'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[4]:

def nearestDate(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))


# In[ ]:

NDVIClip_Day = r'D:\Downloads\Mattijn@Shiva\2015-10-09 NDVI'
srcfolder = listall(NDVIClip_Day)
dstfolder = r'D:\Downloads\Mattijn@Shiva\NDVI025//'


# In[ ]:

for srcfile in srcfolder:
    dstfile = dstfolder + srcfile[-17::]
    command = ["gdalwarp","-tr","0.25","0.25",srcfile,dstfile,"-overwrite"]
    
    print (sp.list2cmdline(command))
    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)  
    norm.communicate()      

    srcfile = dstfile
    dstfile = dstfolder + "_25_" +srcfile[-17::]
    command = ["gdalwarp","-tr","0.05","0.05",srcfile,dstfile,"-overwrite"]
    
    print (sp.list2cmdline(command))
    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)  
    norm.communicate()


# In[5]:

NDVI = r'D:\Downloads\Mattijn@Shiva\2015-10-09 NDVI'
NDVI_srcfolder = listall(NDVI)

NDVI_dates = []
for NDVI_srcfile in NDVI_srcfolder:
    NDVI_year = int(NDVI_srcfile[-12:-8])
    NDVI_month = int(NDVI_srcfile[-8:-6])
    NDVI_day = int(NDVI_srcfile[-6:-4])
    NDVI_date = datetime.datetime(NDVI_year,NDVI_month,NDVI_day)
    NDVI_dates.append(NDVI_date)
    #print NDVI_date

NDVI_jan = []
NDVI_feb = []
NDVI_mar = []
NDVI_apr = []
NDVI_may = []
NDVI_jun = []
NDVI_jul = []
NDVI_aug = []
NDVI_sep = []
NDVI_oct = []
NDVI_nov = []
NDVI_dec = []

for NDVI_date in NDVI_dates:
    if NDVI_date.month == 1:
        NDVI_jan.append(NDVI_date)
    if NDVI_date.month == 2:
        NDVI_feb.append(NDVI_date)
    if NDVI_date.month == 3:
        NDVI_mar.append(NDVI_date)
    if NDVI_date.month == 4:
        NDVI_apr.append(NDVI_date)
    if NDVI_date.month == 5:
        NDVI_may.append(NDVI_date)
    if NDVI_date.month == 6:
        NDVI_jun.append(NDVI_date)        
    if NDVI_date.month == 7:
        NDVI_jul.append(NDVI_date)
    if NDVI_date.month == 8:
        NDVI_aug.append(NDVI_date)
    if NDVI_date.month == 9:
        NDVI_sep.append(NDVI_date)
    if NDVI_date.month == 10:
        NDVI_oct.append(NDVI_date)        
    if NDVI_date.month == 11:
        NDVI_nov.append(NDVI_date)
    if NDVI_date.month == 12:
        NDVI_dec.append(NDVI_date)            


# In[6]:

NDVI_25 = r'D:\Downloads\Mattijn@Shiva\NDVI025'
NDVI_25_srcfolder = listall(NDVI_25)

NDVI_25_dates = []
for NDVI_25_srcfile in NDVI_25_srcfolder:
    NDVI_25_year = int(NDVI_25_srcfile[-12:-8])
    NDVI_25_month = int(NDVI_25_srcfile[-8:-6])
    NDVI_25_day = int(NDVI_25_srcfile[-6:-4])
    NDVI_25_date = datetime.datetime(NDVI_25_year,NDVI_25_month,NDVI_25_day)
    NDVI_25_dates.append(NDVI_25_date)
    #print NDVI_25_date

NDVI_25_jan = []
NDVI_25_feb = []
NDVI_25_mar = []
NDVI_25_apr = []
NDVI_25_may = []
NDVI_25_jun = []
NDVI_25_jul = []
NDVI_25_aug = []
NDVI_25_sep = []
NDVI_25_oct = []
NDVI_25_nov = []
NDVI_25_dec = []

for NDVI_25_date in NDVI_25_dates:
    if NDVI_25_date.month == 1:
        NDVI_25_jan.append(NDVI_25_date)
    if NDVI_25_date.month == 2:
        NDVI_25_feb.append(NDVI_25_date)
    if NDVI_25_date.month == 3:
        NDVI_25_mar.append(NDVI_25_date)
    if NDVI_25_date.month == 4:
        NDVI_25_apr.append(NDVI_25_date)
    if NDVI_25_date.month == 5:
        NDVI_25_may.append(NDVI_25_date)
    if NDVI_25_date.month == 6:
        NDVI_25_jun.append(NDVI_25_date)        
    if NDVI_25_date.month == 7:
        NDVI_25_jul.append(NDVI_25_date)
    if NDVI_25_date.month == 8:
        NDVI_25_aug.append(NDVI_25_date)
    if NDVI_25_date.month == 9:
        NDVI_25_sep.append(NDVI_25_date)
    if NDVI_25_date.month == 10:
        NDVI_25_oct.append(NDVI_25_date)        
    if NDVI_25_date.month == 11:
        NDVI_25_nov.append(NDVI_25_date)
    if NDVI_25_date.month == 12:
        NDVI_25_dec.append(NDVI_25_date)            


# In[7]:

TRMM = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMM_clip_day_05resolution//'
TRMM_srcfolder = listall(TRMM)

TRMM_dates = []
for TRMM_srcfile in TRMM_srcfolder:
    TRMM_year = int(TRMM_srcfile[-16:-12])
    TRMM_month = int(TRMM_srcfile[-11:-9])
    TRMM_day = int(TRMM_srcfile[-8:-6])
    TRMM_date = datetime.datetime(TRMM_year,TRMM_month,TRMM_day)
    TRMM_dates.append(TRMM_date)
    #print TRMM_date
    
TRMM_jan = []
TRMM_feb = []
TRMM_mar = []
TRMM_apr = []
TRMM_may = []
TRMM_jun = []
TRMM_jul = []
TRMM_aug = []
TRMM_sep = []
TRMM_oct = []
TRMM_nov = []
TRMM_dec = []

for TRMM_date in TRMM_dates:
    if TRMM_date.month == 1:
        TRMM_jan.append(TRMM_date)
    if TRMM_date.month == 2:
        TRMM_feb.append(TRMM_date)
    if TRMM_date.month == 3:
        TRMM_mar.append(TRMM_date)
    if TRMM_date.month == 4:
        TRMM_apr.append(TRMM_date)
    if TRMM_date.month == 5:
        TRMM_may.append(TRMM_date)
    if TRMM_date.month == 6:
        TRMM_jun.append(TRMM_date)        
    if TRMM_date.month == 7:
        TRMM_jul.append(TRMM_date)
    if TRMM_date.month == 8:
        TRMM_aug.append(TRMM_date)
    if TRMM_date.month == 9:
        TRMM_sep.append(TRMM_date)
    if TRMM_date.month == 10:
        TRMM_oct.append(TRMM_date)        
    if TRMM_date.month == 11:
        TRMM_nov.append(TRMM_date)
    if TRMM_date.month == 12:
        TRMM_dec.append(TRMM_date)            


# In[8]:

# get longitude latitude arrays
LAT_005_srcfile = r'D:\Downloads\Mattijn@Shiva\lonlat//LAT_005.tif'
ds_LAT_005 = gdal.Open(LAT_005_srcfile).ReadAsArray()

LAT_025_srcfile = r'D:\Downloads\Mattijn@Shiva\lonlat//LAT.tif'
ds_LAT_025 = gdal.Open(LAT_025_srcfile).ReadAsArray()

LON_005_srcfile = r'D:\Downloads\Mattijn@Shiva\lonlat//LON_005.tif'
ds_LON_005 = gdal.Open(LON_005_srcfile).ReadAsArray()

LON_025_srcfile = r'D:\Downloads\Mattijn@Shiva\lonlat//LON.tif'
ds_LON_025 = gdal.Open(LON_025_srcfile).ReadAsArray()
ds_base = gdal.Open(LON_025_srcfile)


# In[9]:

DEM_005_srcfile = r'D:\Downloads\Mattijn@Shiva\srtm_dem//SRTM_DEM_Clip_005res.tif'
ds_DEM_005 = gdal.Open(DEM_005_srcfile).ReadAsArray()

DEM_025_srcfile = r'D:\Downloads\Mattijn@Shiva\srtm_dem//SRTM_DEM_Clip_005ares.tif'
ds_DEM_025 = gdal.Open(DEM_025_srcfile).ReadAsArray()


# In[ ]:

#jan
for TRMM_date in NDVI_jan:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_jan,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_jan,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      17.343 * (ds_NDVI_025 - ds_NDVI) + 
                      0.328 * (ds_LAT_025 - ds_LAT_005) +
                      -0.439 * (ds_LON_025 - ds_LON_005) +
                      0 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols < 0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM == 0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_jan,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[ ]:

#feb
for TRMM_date in NDVI_feb:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_feb,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_feb,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      8.054 * (ds_NDVI_025 - ds_NDVI) + 
                      0 * (ds_LAT_025 - ds_LAT_005) + 
                      -0.261 * (ds_LON_025 - ds_LON_005) +
                      0 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_feb,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[11]:

#mar
for TRMM_date in NDVI_mar:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_mar,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_mar,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      11.376 * (ds_NDVI_025 - ds_NDVI) + 
                      0 * (ds_LAT_025 - ds_LAT_005) + 
                      0 * (ds_LON_025 - ds_LON_005) +
                      0 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_mar,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[12]:

#apr
for TRMM_date in NDVI_apr:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_apr,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_apr,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      23.243 * (ds_NDVI_025 - ds_NDVI) + 
                      0 * (ds_LAT_025 - ds_LAT_005) + 
                      -0.449 * (ds_LON_025 - ds_LON_005) +
                      -0.003 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_apr,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[13]:

#may
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_may,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_may,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      66.687 * (ds_NDVI_025 - ds_NDVI) + 
                      0 * (ds_LAT_025 - ds_LAT_005) + 
                      -0.826 * (ds_LON_025 - ds_LON_005) +
                      -0.005 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_may,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[14]:

#jun
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_jun,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_jun,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      0 * (ds_NDVI_025 - ds_NDVI) +
                      -2.225 * (ds_LAT_025 - ds_LAT_005) + 
                      0 * (ds_LON_025 - ds_LON_005) +
                      -0.010 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_jun,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[15]:

#jul
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_jul,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_jul,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      0 * (ds_NDVI_025 - ds_NDVI) +
                      -3.44 * (ds_LAT_025 - ds_LAT_005) + 
                      0 * (ds_LON_025 - ds_LON_005) +
                      -0.011 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_jul,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[16]:

#aug
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_aug,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_aug,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      26.455 * (ds_NDVI_025 - ds_NDVI) +
                      -2.495 * (ds_LAT_025 - ds_LAT_005) + 
                      0 * (ds_LON_025 - ds_LON_005) +
                      -0.005 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_aug,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[17]:

#sep
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_sep,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_sep,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      22.639 * (ds_NDVI_025 - ds_NDVI) +
                      -0.006 * (ds_LAT_025 - ds_LAT_005) + 
                      0 * (ds_LON_025 - ds_LON_005) +
                      -1.662 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_sep,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[18]:

#oct
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_oct,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_oct,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      42.422 * (ds_NDVI_025 - ds_NDVI) +
                      0 * (ds_LAT_025 - ds_LAT_005) + 
                      -0.808 * (ds_LON_025 - ds_LON_005) +
                      -0.003 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_oct,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[19]:

#nov
for TRMM_date in NDVI_may:
    try:
        ds_TRMM = gdal.Open(TRMM_srcfolder[TRMM_dates.index(TRMM_date)]).ReadAsArray()
        ds_NDVI = gdal.Open(NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_nov,TRMM_date))]).ReadAsArray()[1::,0:-1]
        ds_NDVI_025 = gdal.Open(NDVI_25_srcfolder[NDVI_25_dates.index(nearestDate(NDVI_25_nov,TRMM_date))]).ReadAsArray()

        ds_TRMM_ols = (
                      6.204 * (ds_NDVI_025 - ds_NDVI) +
                      0 * (ds_LAT_025 - ds_LAT_005) + 
                      0 * (ds_LON_025 - ds_LON_005) +
                      0 * (ds_DEM_025 - ds_DEM_005) 
                      )
        ds_TRMM_ols[np.where(ds_TRMM_ols<0)] = 0
        ds_TRMM_ols[np.where(ds_TRMM==0)] = 0

        folder_out = r'D:\Downloads\Mattijn@Shiva\TRMM_005_35stations//'
        file_out = 'TRMM'+NDVI_srcfolder[NDVI_dates.index(nearestDate(NDVI_nov,TRMM_date))][-13::]
        out_url = folder_out + file_out
        saveRaster(out_url, ds_TRMM_ols, ds_base, 6)
    except:
        print TRMM_date
        continue


# In[ ]:




# In[ ]:

0 * 2 + 1 * 2 + 0 * 3 + 1 * 4


# In[ ]:


im = plt.imshow(ds_TRMM_ols)
plt.colorbar(im)


# In[ ]:

im = plt.imshow(ds_TRMM)
plt.colorbar(im)


# In[ ]:

im = plt.imshow(ds_NDVI_025)
plt.colorbar(im)


# In[ ]:

for NDVI_srcfile in NDVI_srcfolder[0:1]:
    print NDVI_srcfile
    ds_NDVI = gdal.Open(NDVI_srcfile).ReadAsArray()[1::,0:-1]


# In[ ]:




# In[ ]:

ds_test_tmpfilename = gdal.Open(tmpfilename).ReadAsArray()
logging.info(ds_test_tmpfilename[0])

ds=gdal.Open(tmpfilename)
clippedfilename='test'+str(j.toordinal())+'clip.tif' 

path_base = "/var/www/html/wps/CHN_adm"
CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")

command = ["/usr/bin/gdalwarp","-cutline",CHN_adm_gpkg,"-csql","SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = '"+NAME_1+"' and NAME_2 = '"+NAME_2+"' and NAME_3 = '"+NAME_3+"'","-crop_to_cutline","-of","GTiff","-dstnodata","-9999",tmpfilename,clippedfilename, "-overwrite"]

logging.info(sp.list2cmdline(command))
#print (sp.list2cmdline(command))

norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)  
norm.communicate()   

ds=gdal.Open(clippedfilename)
ds_clip = ds.ReadAsArray() 

ds=gdal.Open(tmpfilename)
ds_clip = ds.ReadAsArray()     

