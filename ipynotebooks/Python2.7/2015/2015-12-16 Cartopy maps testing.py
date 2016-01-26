
# coding: utf-8

# In[1]:

import cartopy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import geopandas


# In[3]:

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


# In[4]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cfeature


#plt.figure(figsize=(41.53683777162, 18))
plt.figure(figsize=(13.84561259054,6))
ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())
#ax.background_patch.set_facecolor('none')
ax.set_extent([-179,179,-60,80])
coastline = cfeature.COASTLINE.scale='50m'
ax.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')


# In[8]:

import numpy as np
import matplotlib.colors as mcolors
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


# In[9]:

cb_NDAI = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                         c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                         c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                         c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                         c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])


# In[10]:

# ticks of classes
#bounds = [0.,82.875,95.625,108.375,127.5,146.625,159.375,172.125,255.]
bounds = [-1,-0.35,-0.25,-0.15,0,0.15,0.25,0.35,1]
# ticklabels plus colorbar
ticks = ['-1','-0.35','-0.25','-0.15','+0','+.15','+.25','+.35','+1']
cmap = cmap_discretize(cb_NDAI,8)
norm = mcolors.BoundaryNorm(bounds, cmap.N)


# In[11]:

from osgeo import gdal
import numpy as np
in_tif = r'D:\Data\WorldShapefile//MEAN20041015.tif'
ds = gdal.Open(in_tif)
print 'geotransform', ds.GetGeoTransform()
print 'raster X size', ds.RasterXSize
print 'raster Y size', ds.RasterYSize

data = ds.ReadAsArray()
data_ma = np.ma.masked_equal(data,7)
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

xres = gt[1]
yres = gt[5]

# get the edge coordinates and add half the resolution 
# to go to center coordinates
xmin = gt[0] + xres * 0.5
xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
ymax = gt[3] - yres * 0.5

#ds = None
gridlons = np.mgrid[xmin:xmax+xres:xres]
gridlats = np.mgrid[ymax+yres:ymin:yres]


# In[16]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cfeature


#plt.figure(figsize=(41.53683777162, 18))
plt.figure(figsize=(13.84561259054,6))
ax = plt.axes(projection=ccrs.InterruptedGoodeHomolosine())
#ax.background_patch.set_facecolor('none')
ax.set_extent([-179,179,-60,80])
ax.outline_patch.set_edgecolor('gray')
ax.outline_patch.set_linewidth(1)
ax.outline_patch.set_linestyle(':')
coastline = cfeature.COASTLINE.scale='50m'
borders = cfeature.BORDERS.scale='50m'
#land = cfeature.LAND.scale='50m'
#ax.add_feature(cfeature.LAND, facecolor='gainsboro')  
ax.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  

#ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor='blue') 
gl = ax.gridlines(linewidth=1, color='gray', linestyle=':')
gl.xlocator = mticker.FixedLocator(range(-180,190,20))
gl.ylocator = mticker.FixedLocator(range(-60,90,10))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
im = ax.pcolormesh(gridlons, gridlats, data_ma, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap, vmin=-1, vmax=1)
cb = plt.colorbar(im, fraction=0.0476, pad=0.04, ticks=bounds,norm=norm, orientation='horizontal')
cb.set_label('Normalized Drought Anomaly Index')
cb.set_ticklabels(ticks)
#plt.savefig(r'D:\Downloads\Mattijn@Zhou\GlobalDroughtProvince//TEST_1.png', dpi=400)
#plt.clf()
#plt.close()


# In[ ]:



