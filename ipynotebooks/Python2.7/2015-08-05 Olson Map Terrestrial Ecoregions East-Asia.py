# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
%matplotlib inline 

# <codecell>

extent = [111.91693268, 123.85693268, 49.43324112, 40.67324112]
extent = [72.6,141.5,-12.5,54]

# <codecell>

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader

plt.figure(figsize=(12,12))
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_longitude=120, central_latitude=15))
ax.set_extent(extent)
ax.coastlines(resolution='110m')
ax.gridlines()
bound = r'D:\MicrosoftEdgeDownloads\Ecoregions_EastAsia//ea_clip.shp'
shape_bound = cfeature.ShapelyFeature(Reader(bound).geometries(), ccrs.PlateCarree(), facecolor='b')
ax.add_feature(shape_bound, linewidth='1.0', alpha='1.0')

# <codecell>

from osgeo import gdal, ogr
import sys
import subprocess as sp
import os

# <codecell>

file_in = r'D:\Data\ChinaWorld_GlobCover\Globcover2009_V2.3_Global_//GLOBCOVER_L4_200901_200912_V2.3.tif'

# <codecell>

ds = gdal.Open(file_in)

# <codecell>

band = ds.GetRasterBand(1)

# <codecell>

array = band.ReadAsArray()

# <codecell>

shapefile = r'D:\MicrosoftEdgeDownloads\Ecoregions_EastAsia//ea_clip.shp'
daShapefile = shapefile
driver = ogr.GetDriverByName('ESRI Shapefile')
dataSource = driver.Open(daShapefile, 0)
layer = dataSource.GetLayer()
ex1,ex2,ex3,ex4 = layer.GetExtent()
gdal_translate = 'D:\Python34\Lib\site-packages\osgeo\\gdal_translate.exe'

# <codecell>

file_in = r'D:\Data\ChinaWorld_GlobCover\Globcover2009_V2.3_Global_//GLOBCOVER_L4_200901_200912_V2.3.tif'

# <codecell>

out = r'D:\MicrosoftEdgeDownloads\Ecoregions_EastAsia//eastAsia_GlobCover.tif'
#path = folderout+out
#j2 = "NETCDF:"+r'D:\Downloads\Mattijn@Shakir\CL\CL.nc'+":CL"
paramsnorm = [gdal_translate, "-projwin", str(ex1), str(ex4), str(ex2), str(ex3), file_in, out]
print (sp.list2cmdline(paramsnorm))
#print (j2)
norm = sp.Popen(sp.list2cmdline(paramsnorm), shell=True)     
norm.communicate() 

# <codecell>


