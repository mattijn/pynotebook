
# coding: utf-8

# In[2]:

import cartopy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import geopandas


# In[4]:

main()


# In[23]:

y = geopandas.read_file(fname)['geometry']
y[0:1]


# In[24]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader

fname = 'D:\Data\ChinaShapefile//PilotAreas.shp'
x = Reader(fname).geometries()
#y = geopandas.read_file(fname)['geometry']

ax = plt.axes(projection=ccrs.Robinson())
ax.add_geometries(y[0:1], ccrs.PlateCarree(), facecolor='white', hatch='xxxx')
plt.show()


# In[10]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

fname = 'D:\Data\ChinaShapefile//PilotAreas.shp'

ax = plt.axes(projection=ccrs.Robinson())
shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                ccrs.PlateCarree(), facecolor='none')
ax.add_feature(shape_feature)
plt.show()


# In[33]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.figure(figsize=(13.84561259054, 6))
ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine(-100))
#ax.background_patch.set_facecolor('none')
ax.outline_patch.set_edgecolor('none')
ax.coastlines(resolution='110m')
#ax.gridlines()


# In[34]:

import requests


# In[ ]:



