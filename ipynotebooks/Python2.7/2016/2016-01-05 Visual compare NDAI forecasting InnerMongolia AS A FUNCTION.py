
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
get_ipython().magic(u'matplotlib inline')
import os
from osgeo import gdal
import numpy as np
fname = 'D:\Data\ChinaShapefile//PilotAreas.shp'


# In[2]:

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


# In[3]:

cb_NDAI = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                         c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                         c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                         c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                         c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])


# In[4]:

# ticks of classes
#bounds = [0.,82.875,95.625,108.375,127.5,146.625,159.375,172.125,255.]
bounds = [-1,-0.35,-0.25,-0.15,0,0.15,0.25,0.35,1]
# ticklabels plus colorbar
ticks = ['-1','-0.35','-0.25','-0.15','+0','+.15','+.25','+.35','+1']
cmap = cmap_discretize(cb_NDAI,8)
norm = mcolors.BoundaryNorm(bounds, cmap.N)


# In[5]:

def listall(RootFolder, wildcard='', extension='.tif'):
    lists = [os.path.join(root, name)    
                 for root, dirs, files in os.walk(RootFolder)
                   for name in files
                   if wildcard in name
                     if name.endswith(extension)]
    return lists


# In[6]:

def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1


# In[7]:

ndai_w7h1 = listall(r'D:\tmp\ndai_out\NDAI\window7horizon1')
ndai_w14h7 = listall(r'D:\tmp\ndai_out\NDAI\window14horizon7')
ndai_w28h14 = listall(r'D:\tmp\ndai_out\NDAI\window28horizon14')
ndai_obs = listall(r'D:\Data\LS_DATA\NDAI-1day_IM_bbox_warp')

print ndai_obs[0][-25:-17]


# In[8]:

ndai_obs[2699]


# In[10]:

for ndai in ndai_obs[2699:]:
    try:
        frc_w7h1 = ndai_w7h1[[i for i, s in enumerate(ndai_w7h1) if ndai[-25:-17] in s][0]]
        frc_w14h7 = ndai_w14h7[[i for i, s in enumerate(ndai_w14h7) if ndai[-25:-17] in s][0]]
        frc_w28h14 = ndai_w28h14[[i for i, s in enumerate(ndai_w28h14) if ndai[-25:-17] in s][0]]
        #print frc_w7h1, frc_w14h7, frc_w28h14
    except:
        #print ndai[-25:-17]
        continue
        
    # files are same doy and year DO WORK
    print ndai[-25:-17]
    year = str(ndai[-25:-21])
    doy = str(ndai[-20:-17])

    ds_obs = gdal.Open(ndai)
    print 'geotransform', ds_obs.GetGeoTransform()
    print 'raster X size', ds_obs.RasterXSize
    print 'raster Y size', ds_obs.RasterYSize

    data_obs = ds_obs.ReadAsArray()
    data_ma_obs = np.ma.masked_equal(data_obs,-9999)
    data_ma_obs = np.ma.masked_invalid(data_ma_obs)
    gt = ds_obs.GetGeoTransform()
    proj = ds_obs.GetProjection()

    xres = gt[1]
    yres = gt[5]

    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds_obs.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds_obs.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5

    #ds = None
    gridlons = np.mgrid[xmin:xmax+xres:xres]
    gridlats = np.mgrid[ymax+yres:ymin:yres]            

    data_w7h1 = gdal.Open(frc_w7h1).ReadAsArray()
    data_ma_w7h1 = np.ma.masked_equal(data_w7h1,-9999)
    data_ma_w7h1 = np.ma.masked_invalid(data_ma_w7h1)

    data_w14h7 = gdal.Open(frc_w14h7).ReadAsArray()
    data_ma_w14h7 = np.ma.masked_equal(data_w14h7,-9999)
    data_ma_w14h7 = np.ma.masked_invalid(data_ma_w14h7)

    data_w28h14 = gdal.Open(frc_w28h14).ReadAsArray()
    data_ma_w28h14 = np.ma.masked_equal(data_w28h14,-9999)
    data_ma_w28h14 = np.ma.masked_invalid(data_ma_w28h14)


    print 'data loaded start preparing plot'
    ## PLOT @@@@@@@@@@@@@@@ 1 @@@@@@@@@@@@@
    #plt.figure(figsize=(41.53683777162, 18))
    plt.figure(figsize=(13.84561259054,12))
    ax = plt.subplot(221,projection=ccrs.PlateCarree())
    #ax.background_patch.set_facecolor('none')
    ax.set_extent([111.5,124.3,40.3,49.8])
    ax.outline_patch.set_edgecolor('gray')
    ax.outline_patch.set_linewidth(1)
    ax.outline_patch.set_linestyle(':')

    coastline = cfeature.COASTLINE.scale='10m'
    borders = cfeature.BORDERS.scale='10m'
    land = cfeature.LAND.scale='10m'
    ocean = cfeature.OCEAN.scale='10m'

    ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue') 
    ax.add_feature(cfeature.LAND, facecolor='gainsboro')  
    ax.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  

    #ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor='blue') 
    gl = ax.gridlines(linewidth=1, color='gray', linestyle=':')
    #gl.xlocator = mticker.FixedLocator(range(-180,190,20))
    #gl.ylocator = mticker.FixedLocator(range(-60,90,10))
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER

    im = ax.pcolormesh(gridlons, gridlats, data_ma_obs, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap, vmin=-1, vmax=1, zorder=2)
    #cb = plt.colorbar(im, fraction=0.0476, pad=0.04, ticks=bounds,norm=norm, orientation='horizontal')
    #cb.set_label('Normalized Drought Anomaly Index')
    #cb.set_ticklabels(ticks)

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), facecolor='none')
    ax.add_feature(shape_feature, zorder=3, linewidth=0.5)
    ax.set_title('OBSERVED '+'year: '+year+' doy: '+doy)


    ## !@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+
    ## PLOT @@@@@@@@@@@@@@@ 2 @@@@@@@@@@@@@
    ## !@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+


    ax = plt.subplot(222,projection=ccrs.PlateCarree())
    #ax.background_patch.set_facecolor('none')
    ax.set_extent([111.5,124.3,40.3,49.8])
    ax.outline_patch.set_edgecolor('gray')
    ax.outline_patch.set_linewidth(1)
    ax.outline_patch.set_linestyle(':')

    coastline = cfeature.COASTLINE.scale='10m'
    borders = cfeature.BORDERS.scale='10m'
    ocean = cfeature.OCEAN.scale='10m'
    land = cfeature.LAND.scale='10m'

    ax.add_feature(cfeature.LAND, facecolor='gainsboro')  
    ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')  
    ax.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  

    #ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor='blue') 
    gl = ax.gridlines(linewidth=1, color='gray', linestyle=':')
    #gl.xlocator = mticker.FixedLocator(range(-180,190,20))
    #gl.ylocator = mticker.FixedLocator(range(-60,90,10))
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    im = ax.pcolormesh(gridlons, gridlats, data_ma_w7h1, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap, vmin=-1, vmax=1, zorder=2)
    #cb = plt.colorbar(im, fraction=0.0476, pad=0.04, ticks=bounds,norm=norm, orientation='horizontal')
    #cb.set_label('Normalized Drought Anomaly Index')
    #cb.set_ticklabels(ticks)

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), facecolor='none')
    ax.add_feature(shape_feature, zorder=3, linewidth=0.5)
    ax.set_title('FORECAST WINDOW 7 HORIZON 1: '+'year: '+year+' doy: '+doy)


    ## !@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+
    ## PLOT @@@@@@@@@@@@@@@ 2 @@@@@@@@@@@@@
    ## !@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+


    ax = plt.subplot(223,projection=ccrs.PlateCarree())
    #ax.background_patch.set_facecolor('none')
    ax.set_extent([111.5,124.3,40.3,49.8])
    ax.outline_patch.set_edgecolor('gray')
    ax.outline_patch.set_linewidth(1)
    ax.outline_patch.set_linestyle(':')

    coastline = cfeature.COASTLINE.scale='10m'
    borders = cfeature.BORDERS.scale='10m'
    ocean = cfeature.OCEAN.scale='10m'
    land = cfeature.LAND.scale='10m'

    ax.add_feature(cfeature.LAND, facecolor='gainsboro')  
    ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')  
    ax.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  

    #ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor='blue') 
    gl = ax.gridlines(linewidth=1, color='gray', linestyle=':')
    #gl.xlocator = mticker.FixedLocator(range(-180,190,20))
    #gl.ylocator = mticker.FixedLocator(range(-60,90,10))
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    im = ax.pcolormesh(gridlons, gridlats, data_ma_w14h7, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap, vmin=-1, vmax=1, zorder=2)
    cb = plt.colorbar(im, fraction=0.0476, pad=0.04, ticks=bounds,norm=norm, orientation='horizontal')
    cb.set_label('Normalized Drought Anomaly Index')
    cb.set_ticklabels(ticks)

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), facecolor='none')
    ax.add_feature(shape_feature, zorder=3, linewidth=0.5)
    ax.set_title('FORECAST WINDOW 14 HORIZON 7: '+'year: '+year+' doy: '+doy)

    ## !@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+
    ## PLOT @@@@@@@@@@@@@@@ 4 @@@@@@@@@@@@@
    ## !@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+!@#$%^&*()-+


    ax = plt.subplot(224,projection=ccrs.PlateCarree())
    #ax.background_patch.set_facecolor('none')
    ax.set_extent([111.5,124.3,40.3,49.8])
    ax.outline_patch.set_edgecolor('gray')
    ax.outline_patch.set_linewidth(1)
    ax.outline_patch.set_linestyle(':')

    coastline = cfeature.COASTLINE.scale='10m'
    borders = cfeature.BORDERS.scale='10m'
    ocean = cfeature.OCEAN.scale='10m'
    land = cfeature.LAND.scale='10m'

    ax.add_feature(cfeature.LAND, facecolor='gainsboro')  
    ax.add_feature(cfeature.OCEAN, facecolor='lightsteelblue')  
    ax.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')  

    #ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor='blue') 
    gl = ax.gridlines(linewidth=1, color='gray', linestyle=':')
    #gl.xlocator = mticker.FixedLocator(range(-180,190,20))
    #gl.ylocator = mticker.FixedLocator(range(-60,90,10))
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    
    im = ax.pcolormesh(gridlons, gridlats, data_ma_w28h14, transform=ccrs.PlateCarree(), norm=norm, cmap=cmap, vmin=-1, vmax=1, zorder=2)
    cb = plt.colorbar(im, fraction=0.0476, pad=0.04, ticks=bounds,norm=norm, orientation='horizontal')
    cb.set_label('Normalized Drought Anomaly Index')
    cb.set_ticklabels(ticks)

    shape_feature = ShapelyFeature(Reader(fname).geometries(),
                                    ccrs.PlateCarree(), facecolor='none')
    ax.add_feature(shape_feature, zorder=3, linewidth=0.5)
    ax.set_title('FORECAST WINDOW 28 HORIZON 14: '+'year: '+year+' doy: '+doy)                    
    plt.tight_layout()
    plt.savefig(r'D:\tmp\ndai_out\PNG//OBS_FRC_'+year+'_'+doy+'.png', dpi=100,bbox_inches='tight')
    plt.clf()
    plt.close()            


# In[ ]:

1


# In[ ]:

from osgeo import gdal
import numpy as np
tif_forecast = r'D:\tmp\ndai_out\NDAI\window14horizon7//NDAI_W14H07_2009_021.tif'
ds_frc = gdal.Open(tif_forecast)
print 'geotransform', ds_frc.GetGeoTransform()
print 'raster X size', ds_frc.RasterXSize
print 'raster Y size', ds_frc.RasterYSize

data_frc = ds_frc.ReadAsArray()
data_ma_frc = np.ma.masked_equal(data_frc,-9999)
data_ma_frc = np.ma.masked_invalid(data_ma_frc)
gt = ds_frc.GetGeoTransform()
proj = ds_frc.GetProjection()

xres = gt[1]
yres = gt[5]

# get the edge coordinates and add half the resolution 
# to go to center coordinates
xmin = gt[0] + xres * 0.5
xmax = gt[0] + (xres * ds_frc.RasterXSize) - xres * 0.5
ymin = gt[3] + (yres * ds_frc.RasterYSize) + yres * 0.5
ymax = gt[3] - yres * 0.5

#ds = None
gridlons = np.mgrid[xmin:xmax+xres:xres]
gridlats = np.mgrid[ymax+yres:ymin:yres]

# plt.imshow(data_ma, cmap='viridis')
# plt.show()


# In[ ]:

tif_observed = r'D:\Data\LS_DATA\NDAI-1day_IM_bbox_warp//NDAI_2009_021_IM_bbox_wrap.tif'
ds_obs = gdal.Open(tif_observed)
print 'geotransform', ds_obs.GetGeoTransform()
print 'raster X size', ds_obs.RasterXSize
print 'raster Y size', ds_obs.RasterYSize

data_obs = ds_obs.ReadAsArray()
data_ma_obs = np.ma.masked_equal(data_obs,-9999)
data_ma_obs = np.ma.masked_invalid(data_ma_obs)
plt.imshow(data_ma_obs, cmap='viridis')
plt.show()


# In[ ]:



