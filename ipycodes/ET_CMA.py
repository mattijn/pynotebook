# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

filepath = r'E:\Data\WangKun@Mattijn\Atmospheric data_hourly\2011\201101'
filestations = r'E:\Data\WangKun@Mattijn\Radiation_hourly\AWS_stations_Aalbers.csv'
folderDEM = r'E:\Data\WangKun@Mattijn\DEM_TILE5'
prefix = 'P'
outFolder = r'E:\Data\WangKun@Mattijn\outfolder\P//'

# <codecell>

from scipy import stats
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree as KDTree
import statsmodels.api as sm
import numpy as np
import pandas as pd
import gdal
import os
from __future__ import division

# <codecell>

def FILES(inGSODFolder):
    st_wmo = [os.path.join(root, name)
               for root, dirs, files in os.walk(inGSODFolder)
                 for name in files                 
                 if name.endswith('')]
    return st_wmo

# <codecell>

def DEMfiles(inDEMFolder):
    dem_files = [os.path.join(root, name)
               for root, dirs, files in os.walk(inDEMFolder)
                 for name in files
                 if name.endswith('.TIF')]
    return dem_files

# <codecell>

def inRaster(fileDEM):
    raster = gdal.Open(fileDEM, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(1)
    dem = band.ReadAsArray()
    extent = raster.GetGeoTransform()
    return raster, dem, extent

# <codecell>

def saveRaster(path, array, datatype=6, formatraster="GTiff"):
    # Set Driver
    format_ = formatraster #save as format
    driver = gdal.GetDriverByName( format_ )
    driver.Register()
    
    # Set Metadata for Raster output
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    bands = raster.RasterCount
    datatype = 6#band.DataType
    
    # Set Projection for Raster
    outDataset = driver.Create(outFilename, cols, rows, bands, datatype)
    geoTransform = raster.GetGeoTransform()
    outDataset.SetGeoTransform(geoTransform)
    proj = raster.GetProjection()
    outDataset.SetProjection(proj)
    
    # Write output to band 1 of new Raster
    outBand = outDataset.GetRasterBand(1)
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

# <codecell>

def GIDS(x,y):    
    x_y = [(x,y)] # for kd_tree that starts counting at bottom left with 0,0
    #print x_y
    x_ = (x*extent[1]+extent[0]+extent[1]/2) # longitude aalbers projection (meters)
    y_ = (y*extent[5]+extent[3]+extent[5]/2) # latitude aalbers projection (meters)
    long_lat = np.array([[x_,y_]])
    
    #print long_lat
    
    dem_1 = dem[y,x] # elevation x_y coordinate
    dist_tree, ix_tree = tree.query(x_y, k=8, eps=0, p=1) # returns distance and index
    df_selection = df.ix[ix_tree.ravel()]
    
    #print 'elevation from x_y =', dem_1
    #print '\n8 nearest neighbours\n', df_selection
    
    Longi = df_selection.ix[:,9] # meters - Aalbers projection
    Lati = df_selection.ix[:,10] # meters - Aalbers projection
    hi = df_selection.ix[:,6]
    
    ##TEMP
    ti = df_selection.ix[:,2]
    #print 'ti\n',ti    
    
    pr_var = zip(Longi,Lati,hi) # combines predictor variables as tuples
    y = zip(ti) # dependent variable
    X = sm.add_constant(pr_var, prepend=True) # multiple linear regression
    
    #fit the model
    mlr = sm.OLS(y,X).fit()
    b0,b1,b2,b3 = mlr.params
    #print '\nMultiLineair regression\n', mlr.summary()
    
    long_lat_stations = df_selection.as_matrix(columns=['POINT_X','POINT_Y']) 
                                 
    di = cdist(long_lat_stations, long_lat, 'euclidean') # Returns Eucleadian distance in meters between grid cell and selected weather-stations
    #print '\nDistances\n',di
    
    # prepare datasets as flattened numpy array or as single values
    Hi = df_selection.as_matrix(columns=['Elev']).flatten()
    Ti = df_selection.as_matrix(columns=['TEMP']).flatten()
    #Ti = (Ti-32)*(5/9)
    
    di_ = di.flatten()
    long_lat_ = long_lat.flatten()
    Longi_ = df_selection.as_matrix(columns=['POINT_X',]).flatten()
    Lati_ = df_selection.as_matrix(columns=['POINT_Y',]).flatten()
    #print '\nhi\n',Hi
    #print '\nti\n',Ti
    #print '\ndi\n',di
    #print '\nlong_lat\n',long_lat
    #print '\nlongi\n',Longi_
    #print '\nlati\n',Lati_
    
    top =    sum( (1/di_)**2 )**-1
    long_f = b1*(long_lat_[0]-Longi_)
    lat_f =  b2*(long_lat_[1]-Lati_)
    h_f =    b3*(dem_1-Hi)
    middle = Ti + long_f + lat_f + h_f
    end = (1/di_)**2 
    comb = top * sum( middle * end )
    #print comb
    return comb

# <codecell>

# select meteorological data file
file_ = FILES(filepath)
f = file_[0]
print f

# create range of dates
rng = pd.date_range('1/1/2011', '30/1/2011', freq='D')
print rng

# <codecell>

# read file and parse the dates correctly
df = pd.read_csv(f, sep='\t', header=None,  
                 parse_dates={'datetime': [1,2,3,4]}, 
                 date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d %H'))

# set name of columns
columns_df = ['datetime','Station','STP','a','TEMP','b','c','d','REHU','e','f','g','PRCP','h','i','WDSP','j','k','l','m','n','o', \
              'p','q','r','s','t']
df.columns = columns_df

# set date-time as index and select the variables needed
df.set_index('datetime', inplace=True)
df_sel = df[['Station', 'STP','TEMP','REHU','PRCP','WDSP']]

# replace missing, blank and values for snow as NaN
df_sel.replace(32766, np.nan, inplace=True)
df_sel.replace(32744, np.nan, inplace=True)
df_sel.replace(32700, np.nan, inplace=True)

# make appropriate concersions to right unit
df_sel.STP *= 0.1
df_sel.TEMP *= 0.1
df_sel.PRCP *= 0.1
df_sel.STP *= 0.1

# group by Station ID and resample to daily values
df_selDay = df_sel.groupby('Station').resample('1D')
df_selDay = df_selDay.drop('Station',1)
df_selDay = df_selDay.reset_index()
df_selDay = df_selDay.set_index('datetime')
print df_selDay.head()

# <codecell>

rsters = DEMfiles(folderDEM)
#st_wmo = GSODfiles(folderGSOD)
for rster in rsters:    
    raster, dem, extent = inRaster(rster)
    #for date in date_all:
    for i in rng[0:1]:
        # read station file
        df_stations = pd.read_csv(filestations)        
        
        df_stations['Y_LAT'] = (df_stations.POINT_Y-extent[3])/extent[5]
        df_stations['X_LON'] = (df_stations.POINT_X-extent[0])/extent[1]
        
        #print '\ndate of range\n', i
        data_left = df_selDay.ix[i]    
        sel_merge = pd.merge(data_left, df_stations, on='Station', how='inner', )
        #print '\nInner merge on station and data frame\n', sel_merge.head()
        
        ##TEMP
        df = sel_merge[pd.notnull(sel_merge['TEMP'])]
        df.reset_index(drop=True, inplace=True)
        #print '\nDrop not null and reset index\n', df.head()

        Longscaled = df.ix[:,12]
        #print '\nlongscaled\n', Longscaled
        Latscaled = df.ix[:,11]
        #print '\nlatscaled\n', Latscaled
        tree = KDTree(zip(Longscaled,Latscaled), leafsize=11)
    
        tp = np.zeros([dem.shape[1],dem.shape[0]])
        
        #x = 0
        #y = 0
        #print GIDS(x,y)
        for x in range(0,dem.shape[1],1):
            for y in range(0,dem.shape[0],1):
                tp[x][y] = GIDS(x,y)
        tp = tp.T
        
        #save output as raster
        if len(raster.GetDescription()) == 43: # fixed code to check the length of the path
            add = raster.GetDescription()[-6:-4]
        else:
            add = raster.GetDescription()[-5:-4]
            
        date = str(i.year)+str(i.month).zfill(2)+str(i.day).zfill(2)
        outFilename = outFolder+prefix+str(date)+'_'+str(add.zfill(2))+'.tif'
        print outFilename
        
        saveRaster(outFilename, tp)
        outFilename = None
        tp = None