
# coding: utf-8

# In[ ]:

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import urllib
get_ipython().magic(u'matplotlib inline')


# In[ ]:

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28147192:147558%29,Lat%2850%29,Long%2860:80%29],{ansi%28147192:147558%29,Long%280:20%29}%29,%22netcdf%22%29'
path = 'test.nc'
f,h = urllib.urlretrieve(url, path)


# In[ ]:




# In[ ]:

#file = r'C:\Users\lenovo\Downloads//ows (3)'
ds = gdal.Open('NETCDF:'+path+':Band1')


# In[ ]:

array = ds.ReadAsArray()


# In[ ]:

print array.shape
im = plt.imshow(np.ma.masked_equal(array, -3000))
plt.colorbar(im)


# In[ ]:

full_url_wcs = 'http://192.168.1.104:8080/rasdaman/ows?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=NDVI_MOD13C1005_uptodate&FORMAT=application/netcdf&SUBSET=Lat(50.71)&SUBSET=Long(3.04,7.34)&SUBSET=ansi(147192,147558)&SCALEFACTOR=2'
path_wcs = 'test_wcs.nc'
f,h = urllib.urlretrieve(full_url_wcs, path_wcs)


# In[ ]:

ds = gdal.Open('NETCDF:'+path+':Band1')
array = ds.ReadAsArray()
print array.shape
im = plt.imshow(np.ma.masked_equal(array, -3000)/10000., interpolation = 'nearest')
plt.colorbar(im)


# In[ ]:

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28147192:147558%29,Lat%2850%29,Long%2860:80%29],{ansi%28147192:147558%29,Long%280:10%29}%29,%22netcdf%22%29'
path_down = 'test.nc'
f,h = urllib.urlretrieve(url, path_down)
ds_down = gdal.Open('NETCDF:'+path_down+':Band1')
array_down = ds_down.ReadAsArray()

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28147192:147558%29,Lat%2850%29,Long%2860:80%29],{ansi%28147192:147558%29,Long%280:20%29}%29,%22netcdf%22%29'
path_org = 'test.nc'
f,h = urllib.urlretrieve(url, path_org)
ds_org = gdal.Open('NETCDF:'+path_org+':Band1')
array_org = ds_org.ReadAsArray()

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28147192:147558%29,Lat%2850%29,Long%2860:80%29],{ansi%28147192:147558%29,Long%280:40%29}%29,%22netcdf%22%29'
path_up = 'test.nc'
f,h = urllib.urlretrieve(url, path_up)
ds_up = gdal.Open('NETCDF:'+path_up+':Band1')
array_up = ds_up.ReadAsArray()


# In[ ]:

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(311)
img1 = ax1.imshow(array_down, cmap='viridis',interpolation = 'nearest')
ax1.set_xlabel('Long')
ax1.set_ylabel('ansi')
ax1.set_title('DOWNSCALING array shape: '+str(array_down.shape))

ax2 = fig.add_subplot(312)
img2 = ax2.imshow(array_org, cmap='viridis',interpolation = 'nearest')
ax2.set_xlabel('Long')
ax2.set_ylabel('ansi')
ax2.set_title('ORIGINAL array shape: '+str(array_org.shape))

ax3 = fig.add_subplot(313)
img3 = ax3.imshow(array_up, cmap='viridis',interpolation = 'nearest')
ax3.set_xlabel('Long')
ax3.set_ylabel('ansi')
ax3.set_title('UPSCALING array shape: '+str(array_up.shape))

plt.colorbar(img3, orientation='horizontal', ax=ax3)
plt.tight_layout()
#plt.savefig(r'D:\tmp\HANTS_OUT//slice_0_40.png', dpi=200)
plt.show()


# In[ ]:

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28145780%29,Lat%2850:70%29,Long%2860:80%29],{Lat%280:10%29,Long%280:10%29}%29,%22netcdf%22%29'
path_down = 'test.nc'
f,h = urllib.urlretrieve(url, path_down)
ds_down = gdal.Open('NETCDF:'+path_down+':Band1')
array_down = ds_down.ReadAsArray()

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28145780%29,Lat%2850:70%29,Long%2860:80%29],{Lat%280:20%29,Long%280:20%29}%29,%22netcdf%22%29'
path_org = 'test.nc'
f,h = urllib.urlretrieve(url, path_org)
ds_org = gdal.Open('NETCDF:'+path_org+':Band1')
array_org = ds_org.ReadAsArray()

url = 'http://192.168.1.104:8080/rasdaman/ows/wcs?query=for%20c%20in%20%28NDVI_MOD13C1005_uptodate%29%20return%20encode%28%20scale%28%20c[ansi%28145780%29,Lat%2850:70%29,Long%2860:80%29],{Lat%280:40%29,Long%280:40%29}%29,%22netcdf%22%29'
path_up = 'test.nc'
f,h = urllib.urlretrieve(url, path_up)
ds_org = gdal.Open('NETCDF:'+path_org+':Band1')
array_up = ds_org.ReadAsArray()


# In[ ]:

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(131)
img1 = ax1.imshow(np.ma.masked_equal(array_down,0), cmap='viridis',interpolation = 'nearest')
ax1.set_xlabel('Long')
ax1.set_ylabel('Lat')
ax1.set_title('DOWNSCALING array shape: '+str(array_down.shape))

ax2 = fig.add_subplot(132)
img2 = ax2.imshow(array_org, cmap='viridis',interpolation = 'nearest')
ax2.set_xlabel('Long')
ax2.set_ylabel('Lat')
ax2.set_title('ORIGINAL array shape: '+str(array_org.shape))

ax3 = fig.add_subplot(133)
img3 = ax3.imshow(array_up, cmap='viridis',interpolation = 'nearest')
ax3.set_xlabel('Long')
ax3.set_ylabel('Lat')
ax3.set_title('UPSCALING array shape: '+str(array_up.shape))

plt.colorbar(img3, orientation='horizontal')
plt.tight_layout()
plt.savefig(r'D:\tmp\HANTS_OUT//slice_lat_long.png', dpi=200)
plt.show()


# In[ ]:



