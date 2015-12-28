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
    cbar = plt.colorbar(im1, fraction=0.024, pad=0.04, 
                        ticks=[-175, -125,-75, -25, 25, 75, 125, 175 ])
    plt.title('Preciptation Anomaly Percentage', size=14)

    # annotate
    m.drawcountries()
    m.drawcoastlines(linewidth=.5)

    plt.show()    

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

# <markdowncell>

# Test the method for:
# 1. South-America (both lonmin and lonmax are below 0)
# 2. Africa (lonmin is below zero and lonmax above, this the most crucial one)
# 3. East-Asia (both lonmin and lonmax are above 0)

# <codecell>

SOUTHAMERICA = PRECIP_DI_CAL(date='2014-06-06',bbox=[-87.5, -56.6, -32.7, 13.7])
PLOT_MAP(SOUTHAMERICA)

# <codecell>

AFRICA       = PRECIP_DI_CAL(date='2014-06-06',bbox=[-19.1, -36.1,  59.1, 37.9])
PLOT_MAP(AFRICA)

# <codecell>

EASTASIA     = PRECIP_DI_CAL(date='2014-06-06',bbox=[   73, -11.3, 141.2,   54])
PLOT_MAP(EASTASIA)

# <codecell>

def PRECIP_test(date='2014-06-06',bbox=[-87.5, -31.1, -29.3, 0.1]):
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
    
    return PAP, ds

# <codecell>

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
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

# <codecell>

# plot the data
my_list = ['#C80000','#FF6400','#FFAA00','#FFFF00','#FFE100','#C8E100','#64E100','#B4FFBE','#BEFFFF','#00C8F0','#0064C8','#3200E1','#7D00E1','#C800C8','#960096','#5A005A']
my_cmap = mpl_cl.ListedColormap(my_list, name='my_name')

# <codecell>

PAP, ds = PRECIP_test(date='2014-06-06',bbox=[-87.5, -31.1, -29.3, 0.1])

# <codecell>

##write the result VCI to disk
# get parameters
geotransform = ds.GetGeoTransform()
spatialreference = ds.GetProjection()
ncol = ds.RasterXSize
nrow = ds.RasterYSize
nband = 1

trans = ds.GetGeoTransform()
extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
  trans[3] + ds.RasterYSize*trans[5], trans[3])

# Create figure
fig = plt.imshow(PAP, cmap=my_cmap, vmin=-200, vmax=200, extent=extent, interpolation='nearest')#vmin=-0.4, vmax=0.4
plt.colorbar(fig, ticks=[-175, -125,-75, -25, 25, 75, 125, 175 ])
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

(200+200)*1000

# <codecell>

PAP[np.where(PAP>200)]=200
PAP[np.where(PAP<-200)]=-200
PAP += 200
PAP //= (400 - 0 + 1) / 255.
PAP = PAP.astype(np.uint8) 
PAP += 1

# <codecell>

##write the result VCI to disk
# get parameters
geotransform = ds.GetGeoTransform()
spatialreference = ds.GetProjection()
ncol = ds.RasterXSize
nrow = ds.RasterYSize
nband = 1

trans = ds.GetGeoTransform()
extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
  trans[3] + ds.RasterYSize*trans[5], trans[3])

# Create figure
fig = plt.imshow(PAP, cmap=my_cmap, vmin=0, vmax=255, extent=extent, interpolation='nearest')#vmin=-0.4, vmax=0.4
plt.colorbar(fig, ticks=[ 15,  47,  79, 111, 143, 175, 207, 239])
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

bounds = np.array([-175, -150,-125,-100,-75,-50, -25, 0,25,50, 75,100, 125,150, 175 ])
((bounds+200) * 255 / 400 ) 

# <codecell>


