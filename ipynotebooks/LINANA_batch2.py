# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

"""
oXo-oXo-oXo-oXo-oXo-oXo-oXo
"""

RasterPath = r'H:\PHD_Thesis\Thesis code\all methods at present\soil heat flux\Ma_SEBAL\output\tiff//G0_2009154_albers_5km_heihesub.tif'
ShapefilePath = r'C:\Python33\boundary//heihe_upmiddownstream'
OutFolder = r'C:\Python33\output//'
Prefix = 'G0_Ma_SEBAL'
TitlePlot = '2009-06-03'
LegendDescription = 'G0_Ma_SEBAL (W/m2)'

"""
oXo-oXo-oXo-oXo-oXo-oXo-oXo
"""

from mpl_toolkits.basemap import Basemap
from osgeo import osr, gdal
import matplotlib.pyplot as plt
import numpy as np
print ('modules loaded')



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
    



# Read the data and metadata
ds = gdal.Open(RasterPath)
#band = ds.GetRasterBand(20)

data = ds.ReadAsArray()
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

nan=ds.GetRasterBand(1).GetNoDataValue()
if nan!=None:
    data=np.ma.masked_equal(data,value=nan)

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

# create a grid of xy coordinates in the original projection
xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]



print ('boundary coordinates map\n',gt)



# Create the figure and basemap object
fig = plt.figure(figsize=(10, 10))
#ax = plt.subplot(111)

#m = Basemap(projection='robin', lon_0=0, resolution='c')
width_x = (extent[1]-extent[0])*1.3
height_y = (extent[2]-extent[3])*1.1

m = Basemap(projection='aea', resolution='c', width = width_x , height = height_y, lat_0=40.2,lon_0=99.6,)

# Create the projection objects for the conversion
# original (Albers)
inproj = osr.SpatialReference()
inproj.ImportFromWkt(proj)

# Get the target projection from the basemap object
outproj = osr.SpatialReference()
outproj.ImportFromProj4(m.proj4string)

# Convert from source projection to basemap projection
xx, yy = convertXY(xy_source, inproj, outproj)


###
# plot the data (first layer) 24bands
for i in range(0,24):
#for i in range(ds.RasterCount):   
    v = np.linspace(data[i].min(), data[i].max(), 5, endpoint=True)
    im1 = m.pcolormesh(xx, yy, data[i,:,:].T, cmap=plt.cm.jet) # data[0,:,:] select band here
    m.colorbar(im1, location='bottom',ticks=v, pad='6%').set_label(LegendDescription)
    plt.title(TitlePlot+' hour '+str(i)+'_UTC', fontsize=20)

    # annotate
    m.readshapefile(ShapefilePath, 'shp', drawbounds=True, )
    m.drawmeridians(np.arange(98,103,1), linewidth=.2, labels=[1,0,0,1], labelstyle='+/-', color='grey' ) 
    m.drawparallels(np.arange(37,43,1), linewidth=.2, labels=[1,0,0,1], labelstyle='+/-', color='grey')
    m.drawmapboundary(linewidth=0.5, color='grey')
    ToPath = OutFolder+Prefix+'_band'+str(i)+'.png'
    plt.savefig(ToPath,dpi=200)
    fig.clf()
    print (ToPath+ ' - OK')




