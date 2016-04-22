
# coding: utf-8

# In[1]:

import subprocess as sp
import os
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

files = listall(r'J:\NDVI_recon', extension='.tif')
index = []
for i in files:
    # get date
    year = int(i[-23:-19])
    doy = int(i[-7:-4])
    date = datetime(year, 1, 1) + timedelta(doy - 1)
    date = np.datetime64(date)
    date = pd.Timestamp(np.datetime_as_string(date))
    index.append(date)
index = np.array(index)
df = pd.DataFrame(index=index)


# In[4]:

month = df.index.map(lambda x: x.month)
year = df.index.map(lambda x: x.year)
day = df.index.map(lambda x: x.day)
doy =  df.index.map(lambda x: x.dayofyear)
# select only dates between certain month
df_sel = (df[(month >= 1) & (month <=6)] + 
          df[(month >= 10) & (month <=12)]).sort_index()
index_sel = pd.Series(np.in1d(df.index,df_sel.index))


# In[20]:

# prulletaria
gdal_translate = r'C:\Program Files\GDAL//gdal_translate.exe'
out_base = r'J:\NDVI_recon_Yongqiao//'


117.42693268
extent = [116.84693268,117.44,33.26,34.10324112]

for idx, val in enumerate(index_sel):    
    if val == True:
        out_file = 'NDVI_Yongqiao_'+str(index[idx].year)+'-'+str(index[idx].month).zfill(2)+'-'+str(index[idx].day).zfill(2)
        command = [gdal_translate, '-projwin', str(extent[0]),str(extent[3]),
                   str(extent[1]),str(extent[2]),'-of','GTiff', files[idx],
                   out_base+out_file+'.tif']
        print (sp.list2cmdline(command))
        norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
        norm.communicate()


# In[ ]:




# In[ ]:



