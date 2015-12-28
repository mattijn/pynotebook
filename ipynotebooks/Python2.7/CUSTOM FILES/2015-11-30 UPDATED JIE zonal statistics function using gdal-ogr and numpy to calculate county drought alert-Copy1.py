
# coding: utf-8

# In[4]:

from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
gdal.PushErrorHandler('CPLQuietErrorHandler')
get_ipython().magic(u'matplotlib inline')


# In[5]:

def getStatsCounty(cnty_array, feat):
    """
    Core function to calculate statistics to be applied for each county
    
    Input:  
    cnty_array   = Masked raster array of a single county    
    feat         = feature of shapefile to extract ID
    Output: 
    county_stats = Dictionary containing the stats for the county
    """
    
    dc=0
    #percentage of no drought
    p0=(cnty_array[(cnty_array <= 0) ]).size*1.0/cnty_array.size
    if p0>=0.5: dc=1
    #percentage of no drought
    p1=(cnty_array[(cnty_array<=-0.15)]).size*1.0/cnty_array.size
    if p1>0.5: dc=2
    #percentage of no drought
    p2=(cnty_array[(cnty_array<=-0.25) ]).size*1.0/cnty_array.size
    if p2>=0.5: dc=3
    #percentage of no drought
    p3=(cnty_array[cnty_array <=-0.35]).size*1.0/cnty_array.size
    if p3>=0.5: dc=4
    #print cnty_array.count(),np.nanmin(cnty_array)
    ct=cnty_array.count()
    county_stats = {
        'MINIMUM': np.nan if ct<2 else float(np.nanmin(cnty_array)),
        'MEAN': np.nan if ct<2 else float(np.nanmean(cnty_array)),
        'MAX': np.nan if ct<2 else float(np.nanmax(cnty_array)),
        'STD': np.nan if ct<2 else float(np.nanstd(cnty_array)),
        'SUM': np.nan if ct<2 else float(np.nansum(cnty_array)),
        'COUNT': int(cnty_array.count()),
        'FID': int(feat.GetFID()),  
        'P0':p0,
        'P1':p1,
        'P2':p2,
        'P3':p3,
        'DC':dc}
    
    return county_stats


# In[6]:

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


# In[7]:

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
        if fid % 500 == 0:
            print("finished first %s features" % (fid))
    
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
                mask=np.logical_or(
                    src_array == nodata_value,
                    np.logical_not(rv_array)
                )
            )
            
            #print 'feature ID: ',int(feat.GetFID())
            
            # GET STATISTICS FOR EACH COUNTY
            county_stats = getStatsCounty(cnty_array = masked, feat=feat)            
            stats.append(county_stats)
            
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


# In[8]:

vector_path = r'D:\Data\ChinaShapefile\Chinese_version_counties//counties_test.shp'
raster_path = r'D:\Data\NDAI\NDAI_2014//NDAI_2014_008.tif'
nodata_value = np.nan


# In[9]:

stats = zonal_stats(vector_path, raster_path, nodata_value)


# In[31]:

df_stats = pd.DataFrame(stats)
df_stats.set_index('FID', inplace=True)


# In[14]:

# read shapefile and concatate on index using a 'inner' join
# meaning counties without statistics info will be ignored
gdf = gpd.read_file(vector_path)
frames  = [df_stats,gdf]
gdf_df_stats = gpd.pd.concat(frames, axis=1, join='inner')
gdf_df_stats.index.rename('FID', inplace=True)
gdf_df_stats.geometry = gdf_df_stats.geometry.astype(gpd.geoseries.GeoSeries) # overcome bug 


# In[59]:




# In[75]:

gdf_df_stats.to_file(r'D:\Data\ChinaShapefile\Chinese_version_counties//counties_test_concat.shp')


# In[ ]:




# In[12]:

# open vector layer
vds = ogr.Open(vector_path, GA_ReadOnly)  
assert(vds)
vlyr = vds.GetLayer(0)    

# compare EPSG values of vector and raster and change projection if necessary
sourceSR = vlyr.GetSpatialRef()
sourceSR.AutoIdentifyEPSG()
EPSG_sourceSR = sourceSR.GetAuthorityCode(None)


# In[33]:

vlyr_def = vlyr.GetLayerDefn() #get definitions of the layer


# In[35]:

field_names = [vlyr_def.GetFieldDefn(i).GetName() for i in range(vlyr_def.GetFieldCount())] #store the field names as a list of strings
print len(field_names)# so there should be just one at the moment called "FID"
field_names #will show you the current field names


# In[36]:

import geopandas as gpd


# In[10]:

new_field = ogr.FieldDefn('DC', ogr.OFTInteger) #
layer.CreateField(new_field) #self explaining


# In[ ]:



