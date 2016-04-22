
# coding: utf-8

# In[1]:

from osgeo import gdal, ogr
import os
import subprocess as sp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# In[2]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[3]:

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[4]:

# 10_DAY_COMPOSITE PRECIPITATION ANOMALIY
# get index from tif files
files = listall(r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\TRMM2_2009\10_Day_Period\10_DaySums_StdNormAnomalyRes', extension='.tif')
index = []
for i in files:
    # get date
    year = int(i[-11:-7])
    doy = int(i[-7:-4])
    date = datetime(year, 1, 1) + timedelta(doy - 1)
    date = np.datetime64(date)
    date = pd.Timestamp(np.datetime_as_string(date))
    index.append(date)
index = np.array(index)
print files[0], index[0]


# In[5]:


# get columns from shp file
shp_filename = r'D:\Data\ChinaShapefile//LC_types.shp'
siteID_list = []
ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()
for feat in lyr:
    # get siteID from Field
    siteID = str(feat.GetField('LC_typce'))
    siteID_list.append(siteID)
siteID_array = np.array(siteID_list)
columns = np.unique(siteID_array).astype(str)   

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)
#df_shp = pd.DataFrame(index=index, columns=columns)
df.head()


# In[6]:

for i in files:
    # load raster GeoTransform, RasterBand    
    try:
        src_ds = gdal.Open(i) 
        gt = src_ds.GetGeoTransform()
        rb = src_ds.GetRasterBand(1)

        # get date
        year = int(i[-11:-7])
        doy = int(i[-7:-4])
        date = datetime(year, 1, 1) + timedelta(doy - 1)
        date = np.datetime64(date)
        date = pd.Timestamp(np.datetime_as_string(date))
        print date
    except Exception, e:
        print e, i
        continue
        
    ds = ogr.Open(shp_filename)
    lyr = ds.GetLayer()
    for feat in lyr:
        try:
            # get siteID from Field

            siteID = str(feat.GetField('LC_typce'))
            #if siteID == '50353':

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get single pixel
            pixel = rb.ReadAsArray(px,py,1,1)
            #array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
            #stationID_mean = np.ma.mean(array_ID_nine)
            # stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = pixel[0][0]
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            #print e, i, feat.GetFID()
            continue            


# In[7]:

df.head()


# In[14]:

df['BA'].plot()


# In[ ]:



