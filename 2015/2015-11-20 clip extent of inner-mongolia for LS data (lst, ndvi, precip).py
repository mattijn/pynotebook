
# coding: utf-8

# In[1]:

import numpy as np
from osgeo import gdal
import subprocess as sp
import os
get_ipython().magic(u'matplotlib inline')


# In[4]:

def listall(RootFolder, wildcard='', extension='.tif'):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith(extension)]
    return lists


# In[3]:

shapefile_bbox = r'D:\Data\ChinaShapefile//IM_bbox_wrap.shp'
print os.path.splitext(os.path.basename(shapefile_bbox))[0]


# In[5]:

gdalwarp = r'C:\Program Files\GDAL//gdalwarp.exe'
shapefile_bbox = r'D:\Data\ChinaShapefile//IM_bbox_wrap.shp'
folder_in = r'J:\NDAI_2003-2014'
folder_out = r'D:\Data\LS_DATA\NDAI-1day_IM_bbox_warp//'


# In[ ]:

list_files = listall(folder_in)

for in_file_name in list_files:
    
    out_file_name = folder_out + os.path.splitext(os.path.basename(in_file_name))[0] + '_IM_bbox_wrap.tif'
    command = [gdalwarp, '-cutline', shapefile_bbox, '-crop_to_cutline', '-of','GTiff','-dstnodata', '-9999', 
               in_file_name, out_file_name,'-overwrite']
    
    print (sp.list2cmdline(command))

    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
    norm.communicate()


# In[ ]:




# In[ ]:



