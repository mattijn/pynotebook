# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import sys
import urllib
from osgeo import gdal
import numpy
import numpy as np
import numpy.ma as ma
from lxml import etree
from datetime import datetime, timedelta
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# <codecell>

startt=145792
start=datetime.fromtimestamp((startt-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
start

# <codecell>

endt=150728
end=datetime.fromtimestamp((endt-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
end

# <codecell>

dates = end-start
dates

# <codecell>

dates.days / 8

# <codecell>

start + timedelta(8)

# <codecell>


# <codecell>

dates_iso = []
dates_ansi = []
for i in range(0,618):
    iso = start+timedelta(i * 8)
    dates_iso.append(iso)
    
    ansi = startt+(i * 8)
    dates_ansi.append(ansi)
dates_iso = dates_iso[308::]

# <codecell>

dates_iso

# <codecell>

dates_iso[308::]

# <codecell>

spl_arr=[118.9,45.295,118.96,45.355]
cube_arr=[]
endpoint='http://159.226.117.95:58080/rasdaman/ows'
for k,d in enumerate(dates_ansi[308::]):
    print d
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='GetCoverage'
    field['COVERAGEID']='LST_MOD11C2005'#'trmm_3b42_coverage_1'
    field['SUBSET']=['ansi('+str(d)+')',
                     'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                    'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
    field['FORMAT']='image/tiff'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    print full_url
    tmpfilename='test'+str(d)+'.tif'
    f,h = urllib.urlretrieve(full_url,tmpfilename)
    print h
    
    ds=gdal.Open(tmpfilename)
    ds_array = ds.ReadAsArray()
    ds_array_ma = ma.masked_equal(ds_array,-3000)    
    ds_array_ma = ds_array_ma / 10000.

    #write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    directory = r'D:\Downloads\Mattijn@Jia\LST_rasdaman//'
    nvaiFileName = 'LST_'+dates_iso[k].strftime("%Y%m%d")+'.tif'
    file_out = directory+nvaiFileName
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(file_out, ncol, nrow, nband, gdal.GDT_Float32)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(ds_array_ma)
    dst_dataset = None        
    
    #print d

# <codecell>

from osgeo import gdal,ogr
import struct

shp_filename = r'D:\Downloads\Mattijn@Shiva//point_location.shp'

ds=ogr.Open(shp_filename)
lyr=ds.GetLayer()

#first get codes from the 
codes_all = []
for feat in lyr:
    code = feat.GetField(0)
    codes_all.append(codes_all)

# <codecell>

directory = r'D:\Downloads\Mattijn@Shiva\2015-10-09 NDVI//'
files = os.listdir(directory)

# <codecell>

files

# <codecell>

#src_filename = r'D:\Downloads\Mattijn@Shiva\2015-10-09 NDVI//NDVI_20000307.tif'
shp_filename = r'D:\Downloads\Mattijn@Shiva//point_location.shp'


ds=ogr.Open(shp_filename)
lyr=ds.GetLayer()

#first get codes from the 
codes_all = []
for feat in lyr:
    code = feat.GetField(0)
    codes_all.append(code)
c_all = numpy.asarray(codes_all)
    
for i in files:
    
    src_ds=gdal.Open(directory+i) 
    gt=src_ds.GetGeoTransform()
    rb=src_ds.GetRasterBand(1)

    ds=ogr.Open(shp_filename)
    lyr=ds.GetLayer()
    values_all = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        mx,my=geom.GetX(), geom.GetY()  #coord in map units

        #Convert from map to pixel coordinates.
        #Only works for geotransforms with no rotation.
        #If raster is rotated, see http://code.google.com/p/metageta/source/browse/trunk/metageta/geometry.py#493
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel

        intval=rb.ReadAsArray(px,py,1,1)
        values_all.append(intval[0][0])
        #print intval[0] #intval is a numpy array, length=1 as we only asked for 1 pixel value

    v_all = numpy.asarray(values_all)
    c_all = numpy.vstack((c_all,v_all))

code_values = c_all.T

# <codecell>

numpy.savetxt(r'D:\Downloads\Mattijn@Shiva//code_values.csv', code_values, delimiter = ',')

# <codecell>

def excel_date(date1):
    temp = datetime(1899, 12, 30)
    delta = date1 - temp
    return float(delta.days) + (float(delta.seconds) / 86400)

# <codecell>

excel_dates = []
for i in files:
    python_date =  datetime(int(i[5:9]),int(i[9:11]),int(i[11:13]))
    excel_dat = excel_date(python_date)
    excel_dates.append(excel_dat)

# <codecell>

numpy.savetxt(r'D:\Downloads\Mattijn@Shiva//date_values.csv',numpy.asarray(excel_dates),delimiter=',')

# <codecell>


# <codecell>

#also get DEM values
directory = r'D:\Downloads\Mattijn@Shiva\srtm_dem//'
files = os.listdir(directory)
#src_filename = r'D:\Downloads\Mattijn@Shiva\2015-10-09 NDVI//NDVI_20000307.tif'
shp_filename = r'D:\Downloads\Mattijn@Shiva//point_location.shp'


ds=ogr.Open(shp_filename)
lyr=ds.GetLayer()

#first get codes from the 
codes_all = []
for feat in lyr:
    code = feat.GetField(0)
    codes_all.append(code)
c_all = numpy.asarray(codes_all)
    
for i in files[0:1]:
    
    src_ds=gdal.Open(directory+files[1]) 
    gt=src_ds.GetGeoTransform()
    rb=src_ds.GetRasterBand(1)

    ds=ogr.Open(shp_filename)
    lyr=ds.GetLayer()
    values_all = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        mx,my=geom.GetX(), geom.GetY()  #coord in map units

        #Convert from map to pixel coordinates.
        #Only works for geotransforms with no rotation.
        #If raster is rotated, see http://code.google.com/p/metageta/source/browse/trunk/metageta/geometry.py#493
        px = int((mx - gt[0]) / gt[1]) #x pixel
        py = int((my - gt[3]) / gt[5]) #y pixel

        intval=rb.ReadAsArray(px,py,1,1)
        values_all.append(intval[0][0])
        #print intval[0] #intval is a numpy array, length=1 as we only asked for 1 pixel value

    v_all = numpy.asarray(values_all)
    c_all = numpy.vstack((c_all,v_all))

code_values = c_all.T      

# <codecell>

numpy.savetxt(r'D:\Downloads\Mattijn@Shiva//dem_values.csv', code_values, delimiter = ',')

# <codecell>

cube_arr_ma.shape
len(dates_ansi[65:247])

# <codecell>

for i in range(len(dates_ansi[65:247])):
    print i
    print cube_arr_ma[i]

# <codecell>

cube_arr_ma[181].shape

# <codecell>

%matplotlib inline

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
fig = plt.imshow(cube_arr_ma[181]/10000., extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

#write the result VCI to disk
# get parameters
geotransform = ds.GetGeoTransform()
spatialreference = ds.GetProjection()
ncol = ds.RasterXSize
nrow = ds.RasterYSize
nband = 1

# create dataset for output
fmt = 'GTiff'
nvaiFileName = 'NVAI'+cur_date.strftime("%Y%m%d")+'.tif'
driver = gdal.GetDriverByName(fmt)
dst_dataset = driver.Create(nvaiFileName, ncol, nrow, nband, gdal.GDT_Byte)
dst_dataset.SetGeoTransform(geotransform)
dst_dataset.SetProjection(spatialreference)
dst_dataset.GetRasterBand(1).WriteArray(NVAI)
dst_dataset = None

