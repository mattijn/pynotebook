
# coding: utf-8

# In[7]:

from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
gdal.PushErrorHandler('CPLQuietErrorHandler')
get_ipython().magic(u'matplotlib inline')


# In[8]:

def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)


# In[94]:

def zonal_stats(vector_path, raster_path, nodata_value):
    
    # open raster layer
    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()
    
    # set raster nodata value
    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
    
    # open vector layer
    vds = ogr.Open(vector_path, GA_ReadOnly)  
    assert(vds)
    vlyr = vds.GetLayer(0)    
    
    # compare EPSG values of vector and raster and change projection if necessary
    sourceSR = vlyr.GetSpatialRef()
    sourceSR.AutoIdentifyEPSG()
    EPSG_sourceSR = sourceSR.GetAuthorityCode(None)
    
    targetSR = osr.SpatialReference(wkt=rds.GetProjection())
    targetSR.AutoIdentifyEPSG()
    EPSG_targetSR = targetSR.GetAuthorityCode(None)
    
    if EPSG_sourceSR != EPSG_sourceSR:
        # reproject vector geometry to same projection as raster
        print 'unequal projections'    
        sourceSR = vlyr.GetSpatialRef()
        targetSR = osr.SpatialReference()
        targetSR.ImportFromWkt(rds.GetProjectionRef())
        coordTrans = osr.CreateCoordinateTransformation(sourceSR,targetSR)    
        
    """do the work"""
    global_src_extent = None
    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')
    
    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature() 
    
    while feat is not None:
        # print statement after each hunderds features
        fid = int(feat.GetFID())  
        prov = str(feat.GetField('NAME_1'))
        print("start with feature %s " % (fid)) 
        print("province: %s " % (prov)) 
        #if fid % 500 == 0:
        #    print("finished first %s features" % (fid))
    
        if not global_src_extent:
            #print 'bbox county'
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)
        
            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5]
            )
        
        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())
        
        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        try:
            masked = np.ma.MaskedArray(
                src_array,
                mask = np.logical_or(
                    src_array == nodata_value,
                    np.logical_not(rv_array)
                )
            )
            
            #print 'feature ID: ',int(feat.GetFID())
            
            # GET STATISTICS FOR EACH COUNTY
            print 'do something'
            print masked.shape
            #return masked
            values, counts = np.unique(masked, return_counts=True)
            print values, counts
            pixels = counts[0]
            total_pixels_prov = np.ma.count(masked)
            print pixels
            comb = [prov,pixels,total_pixels_prov]
            
            #county_stats = getStatsCounty(cnty_array = masked, feat=feat)            
            stats.append(comb)
            
            rvds = None
            mem_ds = None
            feat = vlyr.GetNextFeature()
            
        except np.ma.MaskError: 
            # catch MaskError, ignore feature containing no valid corresponding raster data set
            # in my case the the most southern county of hainan is not totally within the raster extent            
            print 'feature ID: ',fid, ' maskError, ignore county and lets continue'
            
            rvds = None
            mem_ds = None
            feat = vlyr.GetNextFeature()            
    
    vds = None
    rds = None
    return stats#, src_array, rv_array, masked


# In[99]:

vector_path = r'D:\Data\ChinaShapefile\CHN_adm//CHN_adm1.shp'
raster_path = r'D:\Data\ChinaWorld_GlobCover//CN_VEGETATION_CROP.tif'
nodata_value = np.nan


# In[100]:

stats = zonal_stats(vector_path, raster_path, nodata_value)


# In[101]:

df = pd.DataFrame(stats)
df.columns=['prov','veg','total']
df.set_index(['prov'], inplace=True)
df


# In[98]:

df2['crop'].to_csv(r'D:\Data\NDAI_VHI_GROUNDTRUTH//cropland.csv')


# In[105]:

df3 = pd.concat([df['veg'],df2],axis=1)


# In[112]:

df3


# In[111]:

(df3/30000).to_csv(r'D:\Data\NDAI_VHI_GROUNDTRUTH//crop_veg.csv')


# In[104]:

df3['veg']-df3['crop']


# In[27]:

values, counts = np.unique(masked, return_counts=True)
pixels = counts[0]
total_pixels_prov = np.ma.count(masked)


# In[36]:




# In[38]:

stats.size


# In[ ]:



