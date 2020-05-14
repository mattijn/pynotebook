
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcb
import cartopy.feature as cfeature
from matplotlib import gridspec
from datetime import datetime
import warnings
from osgeo import gdal
import numpy as np


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

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    Return a reverse colormap from a LinearSegmentedColormaps
    
    cmap: LinearSegmentedColormap instance
    name: name of new cmap (default 'my_cmap_r')

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   
    
    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
        
    LinearL = dict(zip(k,reverse))
    my_cmap_r = mcolors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r


# In[3]:

date = datetime(2004,10,15)
extent = [-179,179,-60,80]


# In[4]:

date_str = '20041015'
prefix = ['P0','P1','P2','P3','MEAN','DC']
folder = r'D:\Downloads\Mattijn@Zhou\GlobalDroughtProvince\tif//'
in_rasters = []
for pre in prefix:    
    out_raster = folder + pre + date_str + '.tif'
    print out_raster
    in_rasters.append(out_raster)

data1 = np.ma.masked_equal(gdal.Open(in_rasters[0]).ReadAsArray(),7)
data2 = np.ma.masked_equal(gdal.Open(in_rasters[1]).ReadAsArray(),7)
data3 = np.ma.masked_equal(gdal.Open(in_rasters[2]).ReadAsArray(),7)
data4 = np.ma.masked_equal(gdal.Open(in_rasters[3]).ReadAsArray(),7)
data5 = np.ma.masked_equal(gdal.Open(in_rasters[4]).ReadAsArray(),7)
data6 = np.ma.masked_equal(gdal.Open(in_rasters[5]).ReadAsArray(),7)


# In[5]:

data6.max()


# In[6]:

in_tif = in_rasters[0]
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


# In[7]:

drought_cat_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                      c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#C4DC73')])

drought_per_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                      c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#FEFFD3')])

drought_avg_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                                      c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                                      c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                                      c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                                      c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])

drought_per_tci_cmap_r = reverse_colourmap(drought_per_tci_cmap, name = 'drought_per_tci_cmap_r')
drought_cat_tci_cmap_r = reverse_colourmap(drought_cat_tci_cmap, name = 'drought_cat_tci_cmap_r')


# In[8]:


#extent = [111.91693268, 123.85693268, 49.43324112, 40.67324112]
#extent = [73.5,140,14,53.6]    


fig = plt.figure(figsize=(27.69123,12))
gs = gridspec.GridSpec(3, 3)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax1
# PLOT TOP LEFT
ax1 = fig.add_subplot(gs[0,0], projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0))
ax1.set_extent(extent)
ax1.outline_patch.set_edgecolor('none')
# gridlines
gl1 = ax1.gridlines()
# pcolormesh
bounds1 = [0.25,0.5,0.75,1]
cmap1 = cmap_discretize(drought_per_tci_cmap_r,6)   
norm1 = mcolors.BoundaryNorm(bounds1, cmap1.N)
im1 = ax1.pcolormesh(gridlons, gridlats, data1, transform=ccrs.PlateCarree(), norm=norm1, cmap=cmap1, vmin=0, vmax=1)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax2
# PLOT MIDDLE LEFT
ax2 = fig.add_subplot(gs[1,0], projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0))
ax2.set_extent(extent)
ax2.outline_patch.set_edgecolor('none')
# gridlines
gl2 = ax2.gridlines()
# pcolormesh
bounds2 = [0.25,0.5,0.75,1]
cmap2 = cmap_discretize(drought_per_tci_cmap_r,6)
norm2 = mcolors.BoundaryNorm(bounds2, cmap2.N)
im2 = ax2.pcolormesh(gridlons, gridlats, data2, transform=ccrs.PlateCarree(), norm=norm2, cmap=cmap2, vmin=0, vmax=1)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax3
# PLOT BOTTOM LEFT
ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0))
ax3.set_extent(extent)
ax3.outline_patch.set_edgecolor('none')
# gridlines
gl3 = ax3.gridlines()
# pcolormesh
bounds3 = [0.25,0.5,0.75,1]
cmap3 = cmap_discretize(drought_per_tci_cmap_r,6)
norm3 = mcolors.BoundaryNorm(bounds3, cmap3.N)
im3 = ax3.pcolormesh(gridlons, gridlats, data3, transform=ccrs.PlateCarree(), norm=norm3, cmap=cmap3, vmin=0, vmax=1)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax4
# PLOT BOTTOM MIDDLE
ax4 = fig.add_subplot(gs[2,1], projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0))
ax4.set_extent(extent)
ax4.outline_patch.set_edgecolor('none')
# gridlines
gl4 = ax4.gridlines()
# pcolormesh
bounds4 = [0.25,0.5,0.75,1]
cmap4 = cmap_discretize(drought_per_tci_cmap_r,6)
norm4 = mcolors.BoundaryNorm(bounds4, cmap4.N)
im4 = ax4.pcolormesh(gridlons, gridlats, data4, transform=ccrs.PlateCarree(), norm=norm4, cmap=cmap4, vmin=0, vmax=1)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax5
# PLOT BOTTOM RIGHT
ax5 = fig.add_subplot(gs[2,2], projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0))
ax5.set_extent(extent)
ax5.outline_patch.set_edgecolor('none')
# gridlines
gl5 = ax5.gridlines(linewidth=1, color='gray', linestyle=':') 
# pcolormesh
bounds5 = [-1,-0.35,-0.25,-0.15,0,0.15,0.25,0.35,1]
cmap5 = cmap_discretize(drought_avg_tci_cmap,8)
norm5 = mcolors.BoundaryNorm(bounds5, cmap5.N)
im5 = ax5.pcolormesh(gridlons, gridlats, data5, transform=ccrs.PlateCarree(), norm=norm5, cmap=cmap5, vmin=-1, vmax=1)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax6
# PLOT CENTER
ax6 = fig.add_subplot(gs[0:2,1:3], projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0))
ax6.set_extent(extent)
ax6.outline_patch.set_edgecolor('gray')
ax6.outline_patch.set_linewidth(1)
ax6.outline_patch.set_linestyle(':')
# features
coastline = cfeature.COASTLINE.scale='50m'
borders = cfeature.BORDERS.scale='50m'
ax6.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
ax6.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
# gridlines
gl6 = ax6.gridlines(linewidth=1, color='gray', linestyle=':')
gl6.xlocator = mticker.FixedLocator(range(-180,190,20))
gl6.ylocator = mticker.FixedLocator(range(-60,90,10))
gl6.xformatter = LONGITUDE_FORMATTER
gl6.yformatter = LATITUDE_FORMATTER
# pcolormesh
bounds6 = [0, 1, 2, 3, 4, 5]
cmap6 = cmap_discretize(drought_cat_tci_cmap_r,5)
norm6 = mcolors.BoundaryNorm(bounds6, cmap6.N)
im6 = ax6.pcolormesh(gridlons, gridlats, data6, transform=ccrs.PlateCarree(), norm=norm6, cmap=cmap6, vmin=0, vmax=4)

#date = i[-7:]
#year = date[-4::]
#doy = date[-7:-4]
#date_out = datetime.datetime.strptime(str(year)+'-'+str(doy),'%Y-%j')
date_label = 'Date: '+str(date.year) +'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
# ADD LABELS FOR EACH PLOT
ax1.set_title('Percentage of Slight Drought', weight='semibold', fontsize=12)
ax2.set_title('Percentage of Moderate Drought', weight='semibold', fontsize=12)
ax3.set_title('Percentage of Severe Drought', weight='semibold', fontsize=12)
ax4.set_title('Percentage of Extreme Drought', weight='semibold', fontsize=12)
ax5.set_title('Average of NDAI', weight='semibold', fontsize=12)
ax6.set_title('Drought Alert at Province Level. '+date_label, fontsize=20, weight='semibold', color='k')

# ADD LEGEND IN ALL PLOTS
# -------------------------Ax 1
#cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
cbax1 = fig.add_axes([0.03, 0.7, 0.011, 0.10]) # including tight_layout()

#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_per_tci_cmap,6)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5]
bounds_ticks = [1.5, 2.5, 3.5, 4.5]
bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cb2 = mcb.ColorbarBase(cbax1, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')
#cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)



# -------------------------Ax 2
#cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
cbax2 = fig.add_axes([0.03, 0.37, 0.011, 0.10]) # including tight_layout()

#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_per_tci_cmap,6)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5]
bounds_ticks = [1.5, 2.5, 3.5, 4.5]
bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cb2 = mcb.ColorbarBase(cbax2, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')
#cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)   


# -------------------------Ax 3
#cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
cbax3 = fig.add_axes([0.03, 0.04, 0.011, 0.10]) # including tight_layout()

#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_per_tci_cmap,6)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5]
bounds_ticks = [1.5, 2.5, 3.5, 4.5]
bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cb2 = mcb.ColorbarBase(cbax3, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')
#cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)    


# -------------------------Ax 4
#cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
cbax4 = fig.add_axes([0.36, 0.04, 0.011, 0.10]) # including tight_layout()

#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_per_tci_cmap,6)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5]
bounds_ticks = [1.5, 2.5, 3.5, 4.5]
bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cb2 = mcb.ColorbarBase(cbax4, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')
#cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)        


# -------------------------Ax 5
#cbax5 = fig.add_axes([0.85, 0.15, 0.011, 0.16]) # without tight_layout()
cbax5 = fig.add_axes([0.6922, 0.04, 0.011, 0.16]) # including tight_layout()    

#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_avg_tci_cmap,8)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5,6,7,8,9]
bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5,6.6,7.5,8.5]
bounds_ticks_name = [' ', '-0.35', ' ', '-0.15','0','0.15',' ','0.35',' ']
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cb2 = mcb.ColorbarBase(cbax5, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')        
cb2.set_ticklabels(bounds_ticks_name)     

# ------------------------Ax 6
#cbax6 = fig.add_axes([0.79, 0.48, 0.020, 0.30]) # without tight_layout()
cbax6 = fig.add_axes([0.37, 0.4, 0.020, 0.20]) # without tight_layout()    

#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_cat_tci_cmap,5)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5,6]
bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5]
bounds_ticks_name = ['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Slight Drought', 'No Drought']
norm = mcolors.BoundaryNorm(bounds, cmap.N)
cb2 = mcb.ColorbarBase(cbax6, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')
#cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)
cb2.ax.tick_params(labelsize=12)
#         # ADD LAKES AND RIVERS 
#         #FOR PLOT 1
#         lakes = cfeature.LAKES.scale='110m'
#         rivers = cfeature.RIVERS.scale='110m'        
#         ax1.add_feature(cfeature.LAKES)
#         ax1.add_feature(cfeature.RIVERS)         

#         #FOR PLOT 2        
#         ax2.add_feature(cfeature.LAKES)
#         ax2.add_feature(cfeature.RIVERS)         

#         #FOR PLOT 3        
#         ax3.add_feature(cfeature.LAKES)
#         ax3.add_feature(cfeature.RIVERS)                 

#         #FOR PLOT 4        
#         ax4.add_feature(cfeature.LAKES)
#         ax4.add_feature(cfeature.RIVERS)         

#         #FOR PLOT 5
#         ax5.add_feature(cfeature.LAKES)
#         ax5.add_feature(cfeature.RIVERS)                 

#FOR PLOT 6        
#lakes = cfeature.LAKES.scale='50m'
#rivers = cfeature.RIVERS.scale='50m'        
#ax6.add_feature(cfeature.LAKES)
#ax6.add_feature(cfeature.RIVERS)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax1.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
ax2.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax2.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
ax3.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax3.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
ax4.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax4.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')                
ax5.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax5.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
ax6.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax6.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')                

with warnings.catch_warnings():
    # This raises warnings since tight layout cannot
    # handle gridspec automatically. We are going to
    # do that manually so we can filter the warning.
    warnings.simplefilter("ignore", UserWarning)
    gs.tight_layout(fig, rect=[None,None,None,None])

#gs.update(wspace=0.03, hspace=0.03)
path_out = r'D:\Downloads\Mattijn@Zhou\GlobalDroughtProvince\png//Global_'
file_out = 'DroughtAlert_'+str(date.timetuple().tm_yday).zfill(3)+str(date.year).zfill(4)+'.png'
filepath = path_out+file_out 

fig.savefig(filepath, dpi=200, bbox_inches='tight')
print filepath
#plt.show()        
fig.clf()        
plt.close()
#del record#,county
ram = None    


# In[ ]:

get_ipython().magic(u'matplotlib inline')


# In[ ]:

my_cmap = drought_per_tci_cmap


# In[ ]:

fig = plt.figure(figsize=(12, 32))
ax1 = fig.add_axes([0.03, 0.7, 0.011, 0.10]) # including tight_layout()
#cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
my_cmap = cmap_discretize(drought_per_tci_cmap,6)
#for key in cmap._segmentdata:
#    cmap._segmentdata[key] = list(reversed(cmap._segmentdata[key]))  

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5]
bounds_ticks = [1.5, 2.5, 3.5, 4.5]
bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
norm = mcolors.BoundaryNorm(bounds, my_cmap.N)
cb2 = mcb.ColorbarBase(ax1, cmap=my_cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     #boundaries=[0]+bounds+[13],
                                     #extend='both',
                                     extendfrac='auto',
                                     ticklocation='right',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='vertical')
#cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)
plt.show()


# In[ ]:

my_cmap


# In[ ]:

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap
    name (default 'my_cmap_r')
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   
    
    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
        
    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r


# In[ ]:

my_cmap_r = reverse_colourmap(my_cmap)


# In[ ]:

#my_cmap_r._segmentdata
arg = cmap_discretize(my_cmap_r,6)


# In[ ]:

fig = plt.figure(figsize=(8, 2))
ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap = my_cmap, norm=norm,orientation='horizontal')
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap = my_cmap_r, norm=norm, orientation='horizontal')


# In[ ]:

import matplotlib as mpl


# In[ ]:



