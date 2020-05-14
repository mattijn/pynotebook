
# coding: utf-8

# In[1]:

from osgeo import gdal, ogr
import os
import subprocess as sp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# In[2]:

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[3]:

# get index from tif files
files = listall(r'J:\NDAI_2003-2014', extension='.tif')
index = []
for i in files:
    # get date
    year = int(i[-12:-8])
    doy = int(i[-7:-4])
    date = datetime(year, 1, 1) + timedelta(doy - 1)
    date = np.datetime64(date)
    index.append(date)
index = np.array(index)


# In[4]:

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


# In[5]:

# create empty DataFrame
df = pd.DataFrame(index=index, columns=columns)


# In[36]:

get_ipython().run_cell_magic(u'time', u'', u"for i in files[0:1]:\n    # load raster GeoTransform, RasterBand    \n    try:\n        src_ds = gdal.Open(i) \n        gt = src_ds.GetGeoTransform()\n        rb = src_ds.GetRasterBand(1)\n\n        # get date\n        year = int(i[-12:-8])\n        doy = int(i[-7:-4])\n        date = datetime(year, 1, 1) + timedelta(doy - 1)\n        date = np.datetime64(date)\n        print date\n    except Exception, e:\n        print e, i\n        continue\n        \n    ds = ogr.Open(shp_filename)\n    lyr = ds.GetLayer()\n    for feat in lyr:\n        try:\n            # get siteID from Field\n\n            siteID = str(int(feat.GetField('Site_ID')))\n\n            # get lon/lat from GeometryRef\n            geom = feat.GetGeometryRef()\n            mx,my=geom.GetX(), geom.GetY()  #coord in map units\n\n            # convert from map to pixel coordinates.    \n            px = int((mx - gt[0]) / gt[1]) #x pixel\n            py = int((my - gt[3]) / gt[5]) #y pixel\n\n            # get mean of nine pixels surround station ID\n            array_ID_nine = rb.ReadAsArray(px-1,py-1,3,3)\n            stationID_mean = np.nanmean(array_ID_nine)            \n            # set pandas dataframe value\n            df.ix[date][siteID] = stationID_mean\n            print siteID#, px, py, stationID_mean, df.ix[date][siteID]\n        except Exception, e:\n            print e, i\n            continue            ")


# In[37]:

len(files)


# In[23]:

test = rb.ReadAsArray(px-1,py-1,3,3)
test[0][0] = np.nan


# In[28]:

np.nanmean(test)


# In[ ]:

x = {1: False, 2: True} # no 3

for v in [1,2,3]:
    try:
        print x[v]
    
    except Exception, e:
        print e
        continue


# In[ ]:

df['50353']


# In[ ]:




# In[ ]:




# In[ ]:

int(feat.GetField('Site_ID'))


# In[ ]:

files = listall(r'J:\NDAI_2003-2014', extension='.tif')


# In[ ]:

np.array(index)


# In[ ]:

for i in files:
    # get date
    year = int(i[-12:-8])
    doy = int(i[-7:-4])
    date = datetime(year, 1, 1) + timedelta(doy - 1)
    print date.strftime("%Y%m%d"), i
    
    command = [r'D:\Python27x64//python.exe',r'C:\Program Files\GDAL//gdal_edit.py', '-mo','TIFFTAG_DATETIME='+str(date.strftime("%Y%m%d")), files[0]]

    # log the command 
    print sp.list2cmdline(command)

    norm = sp.Popen(sp.list2cmdline(command), shell=True)  
    norm.communicate()       


# In[ ]:

command = [r'D:\Python27x64//python.exe',r'C:\Program Files\GDAL//gdal_edit.py', '-mo','TIFFTAG_DATETIME='+str(date.strftime("%Y%m%d")), files[0]]

# log the command 
print sp.list2cmdline(command)

norm = sp.Popen(sp.list2cmdline(command), shell=True)  
norm.communicate()    


# In[ ]:



