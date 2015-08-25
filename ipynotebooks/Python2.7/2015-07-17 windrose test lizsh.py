# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from windrose import WindroseAxes 
from matplotlib import mpl 
from matplotlib import pyplot as plt 
import matplotlib.colorbar as cbar 
import matplotlib.cm as cm 
from numpy.random import random 
from numpy import arange 
%matplotlib inline

# <codecell>

#Create wind speed and direction variables 
ws = random(500)*6 
wd = random(500)*360 
 
def new_axes(): 
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w') 
    rect = [0.1, 0.1, 0.8, 0.8] 
    ax = WindroseAxes(fig, rect, axisbg='w') 
    fig.add_axes(ax) 
    return ax 
 
def set_legend(ax): 
    l = ax.legend(axespad=-0.10) 
    plt.setp(l.get_texts(), fontsize=8) 

#windrose like a stacked histogram with normed (displayed in percent) results 
ax = new_axes() 
r = ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white') 
#set_legend(ax) 
#fig = plt.gcf() 
cax, _ = cbar.make_axes(ax) 
 
bounds = ax._info["bins"] 
bounds[-1] = bounds[-2] 
colors = [o.get_facecolor() for o in ax.patches_list] 
cmap = mpl.colors.ListedColormap(colors) 
norm = mpl.colors.BoundaryNorm(bounds, cmap.N) 
cb2 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, 
                                     norm=norm, 
                                     # to use 'extend', you must 
                                     # specify two extra boundaries: 
                                     boundaries=[0]+bounds+[13], 
                                     ticks=bounds, # optional 
                                     extend='max', 
                                     spacing='proportional', 
                                     ) 
                                      
cax.set_yticklabels(["%4.3f" % v for v in bounds]) 
plt.show() 

# <codecell>

"""
Reprojecting images from a Geostationary projection
---------------------------------------------------

This example demonstrates Cartopy's ability to project images into the desired
projection on-the-fly. The image itself is retrieved from a URL and is loaded
directly into memory without storing it intermediately into a file. It
represents pre-processed data from Moderate-Resolution Imaging
Spectroradiometer (MODIS) which has been put into an image in the data's
native Geostationary coordinate system - it is then projected by cartopy
into a global Miller map.

"""
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
from io import BytesIO

import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def geos_image():
    """
    Return a specific MODIS image by retrieving it from a github gist URL.

    Returns
    -------
    img : numpy array
        The pixels of the image in a numpy array.
    img_proj : cartopy CRS
        The rectangular coordinate system of the image.
    img_extent : tuple of floats
        The extent of the image ``(x0, y0, x1, y1)`` referenced in
        the ``img_proj`` coordinate system.
    origin : str
        The origin of the image to be passed through to matplotlib's imshow.

    """
    url = ('https://gist.github.com/pelson/5871263/raw/a568da18e578ef3286a6a0779ee7985fa8ac683f/EIDA50_201211061300_clip2.png')
    img_handle = BytesIO(urlopen(url).read())
    img = plt.imread(img_handle)
    img_proj = ccrs.Geostationary(satellite_height=35786000)
    img_extent = (-5500000, 5500000, -5500000, 5500000)
    return img, img_proj, img_extent, 'upper'


def main():
    ax = plt.axes(projection=ccrs.Miller())
    ax.coastlines()
    ax.set_global()
    img, crs, extent, origin = geos_image()
    plt.imshow(img, transform=crs, extent=extent, origin=origin, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()

# <codecell>

ax = plt.axes(projection=ccrs.Miller())
ax.coastlines()
ax.set_global()
img, crs, extent, origin = geos_image()
plt.imshow(img, transform=crs, extent=extent, origin=origin, cmap='gray')
plt.show()

# <codecell>


