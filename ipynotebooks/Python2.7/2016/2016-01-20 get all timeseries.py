
# coding: utf-8

# In[1]:

from osgeo import gdal, ogr
import os
import subprocess as sp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# In[4]:

get_ipython().magic(u'matplotlib inline')


# In[ ]:

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[ ]:

# get index from tif files
files = listall(r'J:\NDAI_2003-2014', extension='.tif')
index = []
for i in files:
    # get date
    year = int(i[-12:-8])
    doy = int(i[-7:-4])
    date = datetime(year, 1, 1) + timedelta(doy - 1)
    date = np.datetime64(date)
    date = pd.Timestamp(np.datetime_as_string(date))
    index.append(date)
index = np.array(index)


# In[ ]:

# get columns from shp file
shp_filename = r'D:\Data\NDAI_VHI_GROUNDTRUTH\groundtruth_2003_2013.shp'
siteID_list = []
ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()
for feat in lyr:
    # get siteID from Field
    siteID = int(feat.GetField('Site_ID'))
    siteID_list.append(siteID)
siteID_array = np.array(siteID_list)
columns = np.unique(siteID_array).astype(str)    


# In[ ]:

# create empty DataFrame
#df = pd.DataFrame(index=index, columns=columns)
df_shp = pd.DataFrame(index=index, columns=columns)


# In[ ]:

for i in files:
    # load raster GeoTransform, RasterBand    
    try:
        src_ds = gdal.Open(i) 
        gt = src_ds.GetGeoTransform()
        rb = src_ds.GetRasterBand(1)

        # get date
        year = int(i[-12:-8])
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

            siteID = str(int(feat.GetField('Site_ID')))

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get mean of nine pixels surround station ID
            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)
            stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = stationID_mean
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            print e, i, feat.GetFID()
            continue            


# In[ ]:

# save dataframe to pick so it can be loaded if necessary
#df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')
df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl') 


# In[ ]:

df['50353'].plot()


# In[ ]:

# Following bit is to convert CMA database into similar pandas dataframe scheme as RS data


# In[ ]:

import calendar

def get_month_day_range(date):
    """
    For a date 'date' returns the start and end date for the month of 'date'.

    Month with 31 days:
    >>> date = datetime.date(2011, 7, 27)
    >>> get_month_day_range(date)
    (datetime.date(2011, 7, 1), datetime.date(2011, 7, 31))

    Month with 28 days:
    >>> date = datetime.date(2011, 2, 15)
    >>> get_month_day_range(date)
    (datetime.date(2011, 2, 1), datetime.date(2011, 2, 28))
    """
    first_day = date.replace(day = 1)
    last_day = date.replace(day = calendar.monthrange(date.year, date.month)[1])
    return first_day, last_day


# In[ ]:

ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()
for feat in lyr:
    try:
        # get siteID from Field
        siteID = str(int(feat.GetField('Site_ID')))
        
        # get decad
        Year = str(int(feat.GetField('Year')))
        Month = str(int(feat.GetField('Month')))
        Decad = str(int(feat.GetField('Decad')))
        Severity = feat.GetField('Severity')
        if Severity == 'Light':
            severity = 1
        elif Severity == 'Medium':
            severity = 2
        elif Severity == 'Heavy':
            severity = 3     
        
        # print siteID, Year, Month, Decad, Severity
        # get first & last day of month
        first, last = get_month_day_range(pd.Timestamp(Year+'-'+Month+'-'+Decad))
        if Decad == str(1):
            # decad 1
            for i in xrange(first.day, first.day + 10):
                date = pd.Timestamp(Year+'-'+Month+'-'+str(i).zfill(2)).tz_localize('UTC')
                df_shp.ix[date][siteID] = severity
        elif Decad == str(2):
            # decad 2    
            for i in xrange(first.day + 10, first.day + 20):
                date = pd.Timestamp(Year+'-'+Month+'-'+str(i).zfill(2)).tz_localize('UTC')
                df_shp.ix[date][siteID] = severity

        elif Decad == str(3):
            # decad 3    
            for i in xrange(first.day + 20, last.day + 1):
                date = pd.Timestamp(Year+'-'+Month+'-'+str(i).zfill(2)).tz_localize('UTC')
                df_shp.ix[date][siteID] = severity

    except Exception, e:
        print e, feat.GetFID()
        continue


# In[2]:

# save dataframe to pick so it can be loaded if necessary
#df_shp.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//groundtruth_2003_2013.pkl')
df_shp = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//groundtruth_2003_2013.pkl') 


# In[6]:

df_shp['50353'].plot()


# In[ ]:



