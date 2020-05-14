
# coding: utf-8

# In[2]:

from osgeo import gdal, ogr
import os
import subprocess as sp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# In[ ]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[ ]:

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[ ]:

# VHI VHI START
# get index from tif files
files = listall(r'J:\VHI_2003_2013', extension='.tif')
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

print files[0], index[0]
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

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)
df_shp = pd.DataFrame(index=index, columns=columns)

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
            #if siteID == '50353':

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get mean of nine pixels surround station ID
            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)
            array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
            stationID_mean = np.ma.mean(array_ID_nine)
            # stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = stationID_mean
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            print e, i, feat.GetFID()
            continue            
            
# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_VHI_2003_2013.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')             


# In[ ]:

# NVAI NVAI START
# get index from tif files
files = listall(r'D:\Data\NVAI_2003_2013', extension='.tif')
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

print files[0], index[0]
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

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)
df_shp = pd.DataFrame(index=index, columns=columns)

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
            #if siteID == '50353':

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get mean of nine pixels surround station ID
            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)
            array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
            stationID_mean = np.ma.mean(array_ID_nine)
            # stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = stationID_mean
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            print e, i, feat.GetFID()
            continue            
            
# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_NVAI_2003_2013.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')             


# In[ ]:

# NTAI NTAI START
# get index from tif files
files = listall(r'D:\Data\NTAI_2003_2013', extension='.tif')
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

print files[0], index[0]
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

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)
df_shp = pd.DataFrame(index=index, columns=columns)

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
            #if siteID == '50353':

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get mean of nine pixels surround station ID
            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)
            array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
            stationID_mean = np.ma.mean(array_ID_nine)
            # stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = stationID_mean
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            print e, i, feat.GetFID()
            continue            
            
# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_NTAI_2003_2013.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')             


# In[ ]:

# VCI VCI START
# get index from tif files
files = listall(r'D:\Data\VCI_2003_2013', extension='.tif')
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

print files[0], index[0]
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

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)
df_shp = pd.DataFrame(index=index, columns=columns)

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
            #if siteID == '50353':

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get mean of nine pixels surround station ID
            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)
            array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
            stationID_mean = np.ma.mean(array_ID_nine)
            # stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = stationID_mean
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            #print e, i, feat.GetFID()
            continue            
            
# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_VCI_2003_2013.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')             


# In[ ]:

# TCI TCI START
# get index from tif files
files = listall(r'D:\Data\TCI_2003_2013', extension='.tif')
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

print files[0], index[0]
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

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)
df_shp = pd.DataFrame(index=index, columns=columns)

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
            #if siteID == '50353':

            # get lon/lat from GeometryRef
            geom = feat.GetGeometryRef()
            mx,my=geom.GetX(), geom.GetY()  #coord in map units

            # convert from map to pixel coordinates.    
            px = int((mx - gt[0]) / gt[1]) #x pixel
            py = int((my - gt[3]) / gt[5]) #y pixel

            # get mean of nine pixels surround station ID
            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)
            array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
            stationID_mean = np.ma.mean(array_ID_nine)
            # stationID_mean = np.nanmean(array_ID_nine)            
            # set pandas dataframe value
            df.ix[date][siteID] = stationID_mean
            #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
        except Exception, e:
            #print e, i, feat.GetFID()
            continue            
            
# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_TCI_2003_2013.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')             


# In[ ]:




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
                date = pd.Timestamp(Year+'-'+Month+'-'+str(i).zfill(2))#.tz_localize('UTC')
                df_shp.ix[date][siteID] = severity
        elif Decad == str(2):
            # decad 2    
            for i in xrange(first.day + 10, first.day + 20):
                date = pd.Timestamp(Year+'-'+Month+'-'+str(i).zfill(2))#.tz_localize('UTC')
                df_shp.ix[date][siteID] = severity

        elif Decad == str(3):
            # decad 3    
            for i in xrange(first.day + 20, last.day + 1):
                date = pd.Timestamp(Year+'-'+Month+'-'+str(i).zfill(2))#.tz_localize('UTC')
                df_shp.ix[date][siteID] = severity

    except Exception, e:
        print e, feat.GetFID()
        continue


# In[ ]:

# save dataframe to pick so it can be loaded if necessary
#df_shp.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//groundtruth_2003_2013.pkl')
df_shp = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//groundtruth_2003_2013.pkl') 
df_shp.head()


# In[ ]:

df_shp['50353'].plot()


# In[ ]:

# Get landuse classes 9 by 9 pixel surround each station
# read globcover map
glb = r'D:\Data\ChinaWorld_GlobCover\Globcover2009_V2.3_Global_//test2.tif'
src_ds = gdal.Open(glb) 
gt = src_ds.GetGeoTransform()
rb = src_ds.GetRasterBand(1)

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

# create empty DataFrame
df = pd.DataFrame(index=[0], columns=columns)
#df_shp = pd.DataFrame(index=index, columns=columns)

ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()
for feat in lyr:
    try:
        # get siteID from Field

        siteID = str(int(feat.GetField('Site_ID')))
        #if siteID == '50353':

        # get lon/lat from GeometryRef
        geom = feat.GetGeometryRef()
        mx,my=geom.GetX(), geom.GetY()  #coord in map units

        # convert from map to pixel coordinates.    
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel

        # get mean of nine pixels surround station ID
        array_ID_nine = rb.ReadAsArray(px-4,py-4,9,9)
        #array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
        #stationID_mean = np.ma.mean(array_ID_nine)
        stationID_mode = np.bincount(array_ID_nine.flatten()).argmax()
        # stationID_mean = np.nanmean(array_ID_nine)            
        # set pandas dataframe value
        df.ix[0][siteID] = stationID_mode
        #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
    except Exception, e:
        #print e, i, feat.GetFID()
        continue            

# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//RS_LANDUSE.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')   


# In[21]:

# get columns from shp file
shp_filename = r'D:\Data\NDAI_VHI_GROUNDTRUTH\groundtruth2_2003_2013.shp'
siteID_list = []
ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()
for feat in lyr:
    # get siteID from Field
    siteID = int(feat.GetField('Site_ID'))
    siteID_list.append(siteID)
siteID_array = np.array(siteID_list)
columns = np.unique(siteID_array).astype(str)   

# create empty DataFrame
df = pd.DataFrame(index=['province'], columns=columns)
#df_shp = pd.DataFrame(index=index, columns=columns)

ds = ogr.Open(shp_filename)
lyr = ds.GetLayer()
for feat in lyr:
    try:
        # get siteID from Field

        siteID = str(int(feat.GetField('Site_ID')))
        #if siteID == '50353':
        province = feat.GetField('Prov_long')
        province = province.replace('\t','')
        # get lon/lat from GeometryRef
        #geom = feat.GetGeometryRef()
        #mx,my=geom.GetX(), geom.GetY()  #coord in map units

        # convert from map to pixel coordinates.    
        #px = int((mx - gt[0]) / gt[1]) #x pixel
        #py = int((my - gt[3]) / gt[5]) #y pixel

        # get mean of nine pixels surround station ID
        #array_ID_nine = rb.ReadAsArray(px-4,py-4,9,9)
        #array_ID_nine = np.ma.masked_equal(array_ID_nine, 0)
        #stationID_mean = np.ma.mean(array_ID_nine)
        #stationID_mode = np.bincount(array_ID_nine.flatten()).argmax()
        # stationID_mean = np.nanmean(array_ID_nine)            
        # set pandas dataframe value
        df.ix['province'][siteID] = province
        #print siteID#, px, py, stationID_mean, df.ix[date][siteID]
    except Exception, e:
        #print e, i, feat.GetFID()
        continue            

# save dataframe to pick so it can be loaded if necessary
df.to_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//GT_Province.pkl')
#df = pd.read_pickle(r'D:\Data\NDAI_VHI_GROUNDTRUTH//remote_sensing_2003_2013.pkl')   


# In[19]:

province.replace('\t','')


# In[20]:

province


# In[ ]:

gt_list = []
rs_list = []
for i in df.columns:
    #print i
    ID = i
    df_concat = pd.concat([df[ID], df_shp[ID]], axis=1)
    df_concat.dropna(inplace = True)
    gt_list = gt_list + df_concat.ix[:,1].tolist()
    rs_list = rs_list + df_concat.ix[:,0].tolist()
    #plt.scatter(df_concat.ix[:,1],df_concat.ix[:,0])
    #plt.xlim(0,4)
    #plt.xticks([1,2,3],['Light','Medium','Heavy'])
#plt.show()
gt_array = np.array(gt_list)
rs_array = np.array(rs_list)


# In[ ]:

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns  


# In[ ]:

gt_Series = pd.Series(gt_array)
rs_Series = pd.Series(rs_array)


# In[ ]:

df_new = pd.concat([gt_Series, rs_Series], axis=1)
df_new.columns = ['ground-truth','remote-sensing']
df_new.sort('ground-truth', inplace = True)


# In[ ]:

ax = sns.violinplot(df_new['remote-sensing'], groupby=df_new['ground-truth'])
ax.set_xticklabels(['light', 'medium','heavy'])
ax.set_title('NDAI vs CMA')


# In[ ]:

df_new.shape[0]/3


# In[ ]:

df_new['remote-sensing'].describe()


# In[ ]:

from osgeo import gdal

driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(r'J:\NDVI_recon\2006//CN_2006_NDVI_recon.001.tif')
band = file.GetRasterBand(1)
lista = band.ReadAsArray()


# In[ ]:

lista = np.ma.masked_equal(lista, -3000)


# In[ ]:

lista /= 10.


# In[ ]:




# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'list_zeros = np.zeros_like(lista)\nlist_zeros[np.where( lista < 200 )] = 1\nlist_zeros[np.where((200 < lista) & (lista < 400)) ] = 2\nlist_zeros[np.where((400 < lista) & (lista < 600)) ] = 3\nlist_zeros[np.where((600 < lista) & (lista < 800)) ] = 4\nlist_zeros[np.where( lista > 800 )] = 5')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'# reclassification\nlist_zeros = np.zeros_like(lista)\nfor j in  range(file.RasterXSize):\n    for i in  range(file.RasterYSize):\n        if lista[i,j] < 200:\n            list_zeros[i,j] = 1\n        elif 200 < lista[i,j] < 400:\n            list_zeros[i,j] = 2\n        elif 400 < lista[i,j] < 600:\n            list_zeros[i,j] = 3\n        elif 600 < lista[i,j] < 800:\n            list_zeros[i,j] = 4\n        else:\n            list_zeros[i,j] = 5')


# In[ ]:

im = plt.imshow(lista)
plt.colorbar(im)


# In[ ]:

x[np.where( x > 3.0 )]


# In[ ]:

np.where((200 < lista) & (lista < 400)) 


# In[ ]:

# reclassification
lista
for j in  range(file.RasterXSize):
    for i in  range(file.RasterYSize):
        if lista[i,j] < 200:
            lista[i,j] = 1
        elif 200 < lista[i,j] < 400:
            lista[i,j] = 2
        elif 400 < lista[i,j] < 600:
            lista[i,j] = 3
        elif 600 < lista[i,j] < 800:
            lista[i,j] = 4
        else:
            lista[i,j] = 5


