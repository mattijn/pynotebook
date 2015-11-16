# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
%matplotlib inline 
import matplotlib as mpl

# <codecell>

import matplotlib.colors as mcolors
import matplotlib
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
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

# <codecell>

#tci_cmap = make_colormap([c('#F29813'), c('#D8DC44'),0.2, c('#D8DC44'), c('#7EC5AD'),0.4, c('#7EC5AD'), c('#5786BE'),0.6, 
#                          c('#5786BE'), c('#41438D'),0.8, c('#41438D')])
drought_cat_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                      c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#C4DC73')])

# <codecell>

drought_per_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                      c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#FEFFD3')])

# <codecell>

drought_avg_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                                      c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                                      c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                                      c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                                      c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])

# <codecell>



extent = [111.91693268, 123.85693268, 49.43324112, 40.67324112]'#C4DC73'
extent = [74,130,11,53.6]

# <codecell>

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader

fig = plt.figure(figsize=(12,12))
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_longitude=100))
ax.set_extent(extent)
#ax.stock_img()
# Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
#states_provinces = cfeature.NaturalEarthFeature(
#    category='cultural',
#    name='admin_1_states_provinces_lines',
#    scale='50m',
#    facecolor='none')
ax.add_feature(cfeature.LAND, facecolor='0.85')
ax.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=0.05, edgecolor='black')
#ax.add_feature(states_provinces, edgecolor='gray')
#ax.coastlines(resolution='110m')
#ax.border(resolution='110m')
ax.gridlines()
#bound = r'D:\MicrosoftEdgeDownloads\Ecoregions_EastAsia//ea_clip.shp'
#shape_bound = cfeature.ShapelyFeature(Reader(bound).geometries(), ccrs.PlateCarree(), facecolor='b')
#ax.add_feature(shape_bound, linewidth='1.0', alpha='1.0')
# mark a known place to help us geo-locate ourselves
ax.plot(116.4, 39.3, 'ks', markersize=7, transform=ccrs.Geodetic())
ax.text(117, 40., 'Beijing', weight='semibold', transform=ccrs.Geodetic())

cbax1 = fig.add_axes([0.80, 0.25, 0.05, 0.5])

#cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(drought_avg_tci_cmap,10)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5,6,7,8,9,10,11]
bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5,6.6,7.5,8.5,9.5]
bounds_ticks_name = ['-1', ' ', 'c', 'd','e','f','g','h','i','j','k']
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(cbax1, cmap=cmap,
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
cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)


# <codecell>


datetime.datetime.strptime('1936-077','%Y-%j')

# <codecell>

date = '1612014'

# <codecell>

import datetime
year = date[-4::]
doy = date[-7:-4]
date_out = datetime.datetime.strptime(str(year)+'-'+str(doy),'%Y-%j')
'Date: '+str(date_out.year) +'-'+str(date_out.month)+'-'+str(date_out.day)

# <codecell>


# <codecell>

'''
Make a colorbar as a separate figure.
'''

from matplotlib import pyplot
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig = pyplot.figure(figsize=(8,3))
ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])

# The third example illustrates the use of custom length colorbar
# extensions, used on a colorbar with discrete intervals.
cmap = mpl.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],
    [1., .8, 0.], [1., .4, 0.]])
cmap.set_over((1., 0., 0.))
cmap.set_under((0., 0., 1.))

bounds = [-1., -.5, 0., .5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb3 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                     norm=norm,
                                     boundaries=[-1.5]+bounds+[1.5],
                                     extend='both',
                                     # Make the length of each extension
                                     # the same as the length of the
                                     # interior colors:
                                     extendfrac='auto',
                                     ticks=bounds,
                                     #spacing='uniform',
                                     orientation='horizontal')
cb3.set_label('Custom extension lengths, some other units')

# <codecell>

# By Jake VanderPlas
# License: BSD-style

import matplotlib.pyplot as plt
import numpy as np


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


if __name__ == '__main__':
    N = 5

    x = np.random.randn(40)
    y = np.random.randn(40)
    c = np.random.randint(N, size=40)

    # Edit: don't use the default ('jet') because it makes @mwaskom mad...
    plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, 'cubehelix'))
    plt.colorbar(ticks=range(N))
    plt.clim(-0.5, N - 0.5)
    plt.show()

# <codecell>

# Make a figure and axes with dimensions as desired.
fig = pyplot.figure(figsize=(8,3))
ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])

#cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
cmap = cmap_discretize(tci_cmap,10)
cmap.set_over('0.25')
cmap.set_under('0.75')

# If a ListedColormap is used, the length of the bounds array must be
# one greater than the length of the color list.  The bounds must be
# monotonically increasing.
bounds = [1, 2, 3, 4, 5, 6]
bounds_ticks = [1.5, 2.5, 3.5, 4.5, 5.5]
bounds_ticks_name = ['a', 'b', 'c', 'd', 'e']
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                     norm=norm,
                                     # to use 'extend', you must
                                     # specify two extra boundaries:
                                     boundaries=[0]+bounds+[13],
                                     extend='both',
                                     extendfrac='auto',
                                     ticklocation='bottom',
                                     ticks=bounds_ticks,#_name, # optional
                                     spacing='proportional',
                                     orientation='horizontal')
cb2.set_label('Discrete intervals, some other units')
cb2.set_ticklabels(bounds_ticks_name)

# <codecell>

np.linspace(3,6)

# <codecell>


