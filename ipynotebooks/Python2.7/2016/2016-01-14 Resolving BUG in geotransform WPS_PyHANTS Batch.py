
# coding: utf-8

# In[1]:

from osgeo import gdal
import numpy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[9]:

path = r'C:\Users\lenovo\Downloads//map-70df6198-ba6e-11e5-a13d-4437e647de9f.tif'
ds = gdal.Open(path)
ds.GetGeoTransform()


# In[12]:

path = r'C:\Users\lenovo\Downloads//nonp.2010-03-22.recon.tiff'
ds = gdal.Open(path)
ds.GetGeoTransform()


# In[11]:

float(0.05 * -1)


# In[ ]:



