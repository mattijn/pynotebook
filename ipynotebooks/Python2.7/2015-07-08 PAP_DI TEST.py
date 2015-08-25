# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pywps.Process import WPSProcess 
import pydap.client
from pydap.client import open_url
import numpy as np
import datetime as dt
import os
import sys
import gdal
import shutil
import logging
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_cl
from mpl_toolkits.basemap import Basemap
from osgeo import osr, gdal
%matplotlib inline

# <codecell>

def convertXY(xy_source, inproj, outproj):
    # function to convert coordinates

    shape = xy_source[0,:,:].shape
    size = xy_source[0,:,:].size

    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(inproj, outproj)
    xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))

    xx = xy_target[:,0].reshape(shape)
    yy = xy_target[:,1].reshape(shape)

    return xx, yy

# <codecell>

def PRECIP_DI_CAL(date='2014-06-06',bbox=[-87.5, -31.1, -29.3, 0.1]):
    opendap_url_mon='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc'
    opendap_url_ltm='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.ltm.nc'

    # what is the input of the module
    logging.info(date) 
    logging.info(bbox) 

    # convert iso-date to gregorian calendar and get the month
    dta=(dt.datetime.strptime(date,'%Y-%m-%d').date()-dt.datetime.strptime('1800-01-01','%Y-%m-%d').date()).days
    mon=(dt.datetime.strptime(date,'%Y-%m-%d').date()).month

    # open opendap connection and request the avaialable time + lon/lat
    dataset_mon = open_url(opendap_url_mon)
    time=dataset_mon.time[:]
    lat=dataset_mon.lat[:]
    lon=dataset_mon.lon[:]
    dt_ind=next((index for index,value in enumerate(time) if value > dta),0)-1


    # convert bbox into coordinates and convert OL lon to GPCP lon where needed
    minlon = bbox[0]
    if minlon < 0: minlon += 360 #+ 180 # GPCP is from 0-360, OL is from -180-180
    maxlon = bbox[2]
    if maxlon < 0: maxlon += 360 #+ 180 # GPCP is from 0-360, OL is from -180-180
    minlat = bbox[1]
    maxlat = bbox[3]

    lat_sel = (lat>minlat)&(lat<maxlat)
    lat_sel[np.nonzero(lat_sel)[0]-1] = True

    # ugly method to decide if there are two areas to select
    # prepare lon/lat subset arrays
    check_if = 0 # If this one is 1, than there are two areas to check
    if minlon >= maxlon:
        check_if = 1
        lon_sel = np.invert((lon<minlon)&(lon>maxlon))
    else:
        lon_sel = (lon>minlon)&(lon<maxlon)    

    # request the subset from opendap
    dataset_mon=dataset_mon['precip'][dt_ind,lat_sel,lon_sel]
    dataset_ltm = open_url(opendap_url_ltm)
    dataset_ltm=dataset_ltm['precip'][mon-1,lat_sel,lon_sel]
    
    mon = np.ma.masked_less((dataset_mon['precip'][:]).squeeze(),0)
    ltm = np.ma.masked_less((dataset_ltm['precip'][:]).squeeze(),0)

    # if two areas make sure the subset is applied appropriate 
    if check_if == 1:
        subset = np.ones((mon.shape[0]), dtype=bool)[None].T * lon_sel
        mon = np.roll(mon, 
                      len(lon)/2, 
                      axis=1)[np.roll(subset, 
                                      len(lon)/2, 
                                      axis=1)].reshape(mon.shape[0],
                                                       subset.sum()/mon.shape[0])    
        ltm = np.roll(ltm, 
                      len(lon)/2, 
                      axis=1)[np.roll(subset, 
                                      len(lon)/2, 
                                      axis=1)].reshape(ltm.shape[0],
                                                       subset.sum()/ltm.shape[0])    


    # calculate PAP
    PAP=(mon-ltm)/(ltm+1)*100

    # prepare output for GDAL
    papFileName = 'PRECIP_DI'+date +'.tif'        
    driver = gdal.GetDriverByName( "GTiff" )
    ds = driver.Create(papFileName, mon.shape[1], mon.shape[0], 1,gdal.GDT_Int16)
    
    # set projection information
    projWKT='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
    ds.SetProjection(projWKT) 

    # set geotransform information
    geotransform_input = dataset_mon.lon[:]
    if check_if == 1:
        dataset_mon.lon[:][np.where(dataset_mon.lon[:] > 180)] -= 360
        geotransform_input = np.roll(dataset_mon.lon[:], len(lon)/2, axis=0)[np.roll(subset, len(lon)/2, axis=1)[0]]
    geotransform = (min(geotransform_input),2.5, 0,max(dataset_mon.lat[:]),0,-2.5) 
    ds.SetGeoTransform(geotransform) 

    # write the data
    ds.GetRasterBand(1).WriteArray(PAP)
    #ds=None
    return ds

# <codecell>

def PLOT_MAP(ds):
    data = ds.ReadAsArray()
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

    ds = None

    # create a grid of xy coordinates in the original projection
    xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]

    # Create the figure and basemap object
    fig = plt.figure(figsize=(12, 6))
    m = Basemap(projection='robin', lon_0=0, resolution='c')

    # Create the projection objects for the convertion
    # original (Albers)
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)

    # Get the target projection from the basemap object
    outproj = osr.SpatialReference()
    outproj.ImportFromProj4(m.proj4string)

    # Convert from source projection to basemap projection
    xx, yy = convertXY(xy_source, inproj, outproj)

    # plot the data
    my_list = ['#C80000','#FF6400','#FFAA00','#FFFF00','#FFE100','#C8E100','#64E100','#FFFFFF','#FFFFFF','#00C8F0','#0064C8','#3200E1','#7D00E1','#C800C8','#960096','#5A005A']    
    my_cmap = mpl_cl.ListedColormap(my_list, name='my_name')
    im1 = m.pcolormesh(xx, yy, data[:,:].T, cmap=my_cmap, vmin=-200, vmax=200)
    #im1.cmap.set_under()
    #im1.cmap.set_over()
    bounds = [-175,-125, -25, 25, 75, 125, 175]
    cbar = plt.colorbar(im1, fraction=0.024, pad=0.04, ticks=bounds, format='%1i')
    plt.title('Preciptation Anomaly Percentage', size=14)

    # annotate
    m.drawcountries()
    m.drawcoastlines(linewidth=.5)

    plt.show()    

# <markdowncell>

# Test the method for:
# 1. South-America (both lonmin and lonmax are below 0)
# 2. Africa (lonmin is below zero and lonmax above, this the most crucial one)
# 3. East-Asia (both lonmin and lonmax are above 0)

# <codecell>

tibet = PRECIP_DI_CAL(date='2015-07-17',bbox=[72.6,17,104.3,38.4])
PLOT_MAP(tibet)

# <codecell>

SOUTHAMERICA = PRECIP_DI_CAL(date='2014-06-06',bbox=[-87.5, -56.6, -32.7, 13.7])
PLOT_MAP(SOUTHAMERICA)

# <codecell>

AFRICA       = PRECIP_DI_CAL(date='2015-06-06',bbox=[-19.1, -36.1,  59.1, 37.9])
PLOT_MAP(AFRICA)

# <codecell>

EASTASIA     = PRECIP_DI_CAL(date='2014-06-06',bbox=[   73, -11.3, 141.2,   54])
PLOT_MAP(EASTASIA)

# <markdowncell>

# 
# 
# FROM HERE TEST CARTOPY

# <codecell>

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import matplotlib.colors as mcolors
%matplotlib inline

# <codecell>

def setMap(rasterBase):

    # Read the data and metadata
    ds = rasterBase#gdal.Open(rasterBase)
    #band = ds.GetRasterBand(20)
    
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    #proj = ds.GetProjection()
    
    nan = ds.GetRasterBand(1).GetNoDataValue()
    if nan != None:
        data = np.ma.masked_equal(data,value=nan)
    
    xres = gt[1]
    yres = gt[5]
    
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    
    x = ds.RasterXSize 
    y = ds.RasterYSize  
    extent = [ gt[0],gt[0]+x*gt[1], gt[3],gt[3]+y*gt[5]]
    #ds = None
    img_extent = (extent[0], extent[1], extent[2], extent[3])
    
    # create a grid of xy coordinates in the original projection
    #xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    
    return extent, img_extent#, xy_source, proj

# <codecell>

#tif_test = r'D:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\NDVI\DaySums_Anomaly//NDVI_IM_2003001.tif'
extent, img_extent = setMap(tibet)

ds = tibet#gdal.Open(tif_test)
array = ds.ReadAsArray()
array = ma.masked_equal(array, array[0][0])
array = np.flipud(array)

# <codecell>

proj = ccrs.Robinson()
fig = plt.figure(figsize=[16,10]) 

extents = proj.transform_points(ccrs.Geodetic(),
                                np.array([extent[0], extent[1]]),
                                np.array([extent[2], extent[3]]))
img_extents = img_extent# (extents[0][0], extents[1][0], extents[0][1], extents[1][1] ) 

ax = plt.axes(projection=proj)

my_list = ['#C80000','#FF6400','#FFAA00','#FFFF00','#FFE100','#C8E100','#64E100','#FFFFFF','#FFFFFF','#00C8F0','#0064C8','#3200E1','#7D00E1','#C800C8','#960096','#5A005A']    
my_cmap = mpl_cl.ListedColormap(my_list, name='my_name')

im=ax.imshow(array, origin='upper', extent=img_extents, transform=ccrs.PlateCarree(), cmap=my_cmap, vmin=-200, vmax=200,  interpolation='nearest')
ax.set_global()
ax.set_xmargin(0.05)
ax.set_ymargin(0.10)
ax.coastlines()
ax.gridlines()

ax.plot(116.38833, 39.92889, 'ro', markersize=7, transform=ccrs.Geodetic())
ax.text(98, 37, 'Beijing', transform=ccrs.Geodetic())

title = 'NDVI Standardized Anomaly 2009 DOY'#+raster[j][-7:-4]
plt.suptitle(title, y=0.86, fontsize=22)    

cax = fig.add_axes([0.182, 0.14, 0.661, 0.03]) # location of the legend
bounds = [-175,-125, -25, 25, 75, 125, 175]
cbar = plt.colorbar(im, cax=cax, ticks=bounds, format='%1i', orientation='horizontal')
plt.title('Preciptation Anomaly Percentage', size=14)
#outPath = r'C:\out.png'

#plt.savefig(outPath, dpi=400)
#print outPath
plt.show()
plt.close(fig)
fig.clf() 

# <codecell>

img_extent

# <codecell>


# <codecell>

def display(image, display_min, display_max):
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    image //= (display_max - display_min + 1) / 256.
    return image.astype(np.uint8)

def lut_display(image, display_min, display_max):
    lut = np.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image) 

# <codecell>

ds, pap, mon, ltm = PRECIP_DI_CAL(date='2014-06-06',bbox=[-19.1, -36.1,  59.1, 37.9])

# <codecell>

PLOT_MAP(ds)

# <codecell>

im2 = plt.imshow(lut_display(pap.astype('int32'), pap.min(), pap.max()))
plt.colorbar(im2)

# <codecell>

im3 = plt.imshow(pap)
plt.colorbar(im3)

# <codecell>

im4 = plt.imshow(pap.astype(np.uint8))
plt.colorbar(im4)

# <codecell>

tes2 = ((mon-ltm)/(ltm+1))*100

# <codecell>

im = plt.imshow(test)
plt.colorbar(im)

# <codecell>

im = plt.imshow(tes2)
plt.colorbar(im)

# <codecell>


