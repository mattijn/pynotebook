
# coding: utf-8

# In[7]:

get_ipython().magic(u'matplotlib inline')
from osgeo import gdal
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt


# In[13]:

driver = gdal.GetDriverByName('NetCDF')


# In[14]:

driver.Register()


# In[20]:

def modelout_to_geotif(netcdfile, outdir, VarName):
    """ Given the path to the netcdf output,
    produces geotiffs for each band of the given variable name.

    Input::
        netcdfile: string path to netcdf file
        VarName: string variable name in netcdf file (e.g. 'GridPrec')

    Output::
        creates geotiffs for each 'band' (e.g. time step). Output is written
        to current working directory.


    """
    try:
        nci = gdal.Open('NETCDF:{0}:{1}'.format(netcdfile, VarName))
        ncd = Dataset(netcdfile)
    except:
        print("Could not open input file: {0}".format(netcdfile))
        return

    try:
        geotransform = nci.GetGeoTransform()
    except:
        print("Could not get geotransform.\n")
        return
    try:
        projection = nci.GetProjection()
    except:
        print("Could not get projection.\n")
        return

    numbands = nci.RasterCount
    times = [o2d(t) for t in ncd.variables['time']]
    print("Found {0} bands to process for {1}".format(numbands, VarName))
    for i in range(numbands):
        time = times[i]
        band = nci.GetRasterBand(i+1)
        raster = band.ReadAsArray()
        y,x = raster.shape

        output_file = "{0}_{1}.tif".format(VarName, time.strftime('%Y%m%d_%H'))
        subdir = os.path.join(outdir, VarName)
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        output_file = os.path.join(subdir, output_file)
        if os.path.exists(output_file):
            continue
        # Create gtif
        driver = gdal.GetDriverByName("GTiff")
        #TODO: Make this more generic
        dst_ds = driver.Create(output_file, x, y, 1, gdal.GDT_CFloat32 )

        # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
        dst_ds.SetGeoTransform( geotransform )

        # set the reference info 
        dst_ds.SetProjection( projection )

        # write the band
        dst_ds.GetRasterBand(1).WriteArray(raster)


# In[24]:

in_ = r'D:\Downloads//CL.nc'
out_ =  r'D:\Downloads\12//'
modelout_to_geotif(in_,out_,'lat')


# In[15]:

from osgeo import gdal, ogr
import sys
import subprocess as sp
import os

def clipraster(folderin, shapefile, folderout, format_end=''):

    files = [os.path.join(root, name)
               for root, dirs, files in os.walk(folderin)
                 for name in files                 
                 if name.endswith(format_end)]
    
    daShapefile = shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(daShapefile, 0)
    layer = dataSource.GetLayer()
    ex1,ex2,ex3,ex4 = layer.GetExtent()
    gdal_translate = 'C:\Python34\Lib\site-packages\osgeo\\gdal_translate.exe'

    for j in files[0:1]:
        out = j[-2:-1]+'xinjiang.tif'
        path = folderout+out
        j2 = "NETCDF:"+r'D:\Downloads\Mattijn@Shakir\CL\CL.nc'+":CL"
        paramsnorm = [gdal_translate, "-projwin", str(ex1), str(ex4), str(ex2), str(ex3), 
                      j2, path]
        print (sp.list2cmdline(paramsnorm))
        print (j2)
        norm = sp.Popen(sp.list2cmdline(paramsnorm), shell=True)     
        norm.communicate() 


# In[18]:

layer = gdal.Open('NETCDF:"'+r'D:\Downloads\Mattijn@Shakir\CL\CL.nc'+'":lat')


# In[ ]:




# Open and Clip NETCDF file field capacity and permanent wilting point

# In[ ]:

folderout = r'D:\Downloads\Mattijn@Shakir\CL_Xinjiang//'
folderin = r'D:\Downloads\Mattijn@Shakir\CL'
#folderin = 'C:\Downloads\TH33//'
#folderin = 'C:\Users\Matt\Downloads\Globcover2009_V2.3_Global_//'
shapefile = r'D:\Downloads\Mattijn@Shakir\xijiang border//xinjiang.shp'
format_end = 'nc'
clipraster(folderin,shapefile,folderout,format_end)


# In[ ]:

def OpenRaster(filepath, band=1):
    """
    In:
    filepath = path to file
    band     = 1

    Return:
    raster   = raster info
    array    = numpy array
    extent   = extent info
    """
    raster = gdal.Open(filepath, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(band)
    array = band.ReadAsArray()
    extent = raster.GetGeoTransform()
    return raster, array, extent

def saveRaster(path, array, dsSource, datatype=3, formatraster="GTiff", nan=None): 
    """
    Datatypes:
    unknown = 0
    byte = 1
    unsigned int16 = 2
    signed int16 = 3
    unsigned int32 = 4
    signed int32 = 5
    float32 = 6
    float64 = 7
    complex int16 = 8
    complex int32 = 9
    complex float32 = 10
    complex float64 = 11
    float32 = 6, 
    signed int = 3
    
    Formatraster:
    GeoTIFF = GTiff
    Erdas = HFA (output = .img)
    OGC web map service = WMS
    png = PNG
    """
    # Set Driver
    format_ = formatraster #save as format
    driver = gdal.GetDriverByName( format_ )
    driver.Register()
    
    # Set Metadata for Raster output
    cols = dsSource.RasterXSize
    rows = dsSource.RasterYSize
    bands = dsSource.RasterCount
    datatype = datatype#band.DataType
    
    # Set Projection for Raster
    outDataset = driver.Create(path, cols, rows, bands, datatype)
    geoTransform = dsSource.GetGeoTransform()
    outDataset.SetGeoTransform(geoTransform)
    proj = dsSource.GetProjection()
    outDataset.SetProjection(proj)
    
    # Write output to band 1 of new Raster and write NaN value
    outBand = outDataset.GetRasterBand(1)
    if nan != None:
        outBand.SetNoDataValue(nan)
    outBand.WriteArray(array) #save input array
    #outBand.WriteArray(dem)
    
    # Close and finalise newly created Raster
    #F_M01 = None
    outBand = None
    proj = None
    geoTransform = None
    outDataset = None
    driver = None
    datatype = None
    bands = None
    rows = None
    cols = None
    driver = None
    array = None


# Remove top and right line of array

# In[ ]:

raster, array, extent = OpenRaster('C:\Users\Matt\Downloads\AWC_CLASS2//MERIS_CN_Reclass_Clip.tif')
array_out = array[1::].T[0:-1].T
ds_ref = gdal.Open(r'G:\Data\0_DAILY_INTERVAL_NDVI_TRMM\InnerMongolia\2009\OUTPUT_SummerOnly\COMB//ph.tif')
path_out = r'C:\Users\Matt\Downloads\AWC_CLASS2//meris_cn.tif'


# In[ ]:

saveRaster(path_out, array_out, ds_ref, datatype=6)

