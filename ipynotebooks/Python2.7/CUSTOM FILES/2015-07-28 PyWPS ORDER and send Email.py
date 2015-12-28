# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib2
import urllib
from osgeo import gdal
import numpy
import numpy as np
import numpy.ma as ma
from lxml import etree
#import datetime
from datetime import datetime
#import matplotlib
#import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from cStringIO import StringIO
#import cStringIO
import jdcal
import json
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
import base64
import uuid
import re
from pydap.client import open_url
import datetime as dt
import os
import sys
import gdal
import shutil
import logging
#%matplotlib inline
#plt.style.use('ggplot')

# <codecell>

gdal.GDT_Float64

# <codecell>

def PRECIP_DI_CAL(coverageID, request_name, from_date_order,bbox_order, directory):
    opendap_url_mon='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc'
    opendap_url_ltm='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.ltm.nc'

    # what is the input of the module
    logging.info(from_date_order) 
    logging.info(bbox_order) 

    # convert iso-date to gregorian calendar and get the month
    dta=(dt.datetime.strptime(from_date_order,'%Y-%m-%d').date()-dt.datetime.strptime('1800-01-01','%Y-%m-%d').date()).days
    mon=(dt.datetime.strptime(from_date_order,'%Y-%m-%d').date()).month

    # open opendap connection and request the avaialable time + lon/lat
    dataset_mon = open_url(opendap_url_mon)
    time=dataset_mon.time[:]
    lat=dataset_mon.lat[:]
    lon=dataset_mon.lon[:]
    dt_ind=next((index for index,value in enumerate(time) if value > dta),0)-1


    # convert bbox into coordinates and convert OL lon to GPCP lon where needed
    minlon = bbox_order[0]
    if minlon < 0: minlon += 360 #+ 180 # GPCP is from 0-360, OL is from -180-180
    maxlon = bbox_order[2]
    if maxlon < 0: maxlon += 360 #+ 180 # GPCP is from 0-360, OL is from -180-180
    minlat = bbox_order[1]
    maxlat = bbox_order[3]

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
    
#     PAP[np.where(PAP>200)]=200
#     PAP[np.where(PAP<-200)]=-200
#     PAP += 200
#     PAP //= (400 - 0 + 1) / 256.
#     PAP = PAP.astype(np.uint8)    

    # prepare output for GDAL
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm
    
    #papFileName = 'PRECIP_DI'+date +'.tif'        
    driver = gdal.GetDriverByName( "GTiff" )
    ds = driver.Create(directory, mon.shape[1], mon.shape[0], 1)
    
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
    ds=None
    
    print directory

# <codecell>

def _NTAI_CAL(coverageID, request_name, from_date_order,bbox_order, directory):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_11c2_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    #print [start_date, end_date]

    #generate the dates list 
    cur_date=datetime.strptime(from_date_order,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='modis_11c2_cov'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(bbox_order[1])+','+str(bbox_order[3])+')',
                        'Long('+str(bbox_order[0])+','+str(bbox_order[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    NTAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
#     NTAI += 1
#     NTAI *= 1000
#     NTAI //= (2000 - 0 + 1) / 255. # instead of 256 to make space for zero values
#     NTAI = NTAI.astype(numpy.uint8)
#     NTAI += 1 # So 0 values are reserved for mask

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    #NTAIFileName = 'NTAI'+cur_date.strftime("%Y%m%d")+'.tif'
    
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm
    
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(directory, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(NTAI)
    dst_dataset = None
    print directory

# <codecell>

def _NVAI_CAL(coverageID, request_name, from_date_order,bbox_order, directory):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    #print [start_date, end_date]

    #generate the dates list 
    cur_date=datetime.strptime(from_date_order,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(bbox_order[1])+','+str(bbox_order[3])+')',
                        'Long('+str(bbox_order[0])+','+str(bbox_order[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    NVAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
#     NVAI += 1
#     NVAI *= 1000
#     NVAI //= (2000 - 0 + 1) / 255. # instead of 256 to make space for zero values
#     NVAI = NVAI.astype(numpy.uint8)
#     NVAI += 1 # So 0 values are reserved for mask

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    #nvaiFileName = 'NVAI'+cur_date.strftime("%Y%m%d")+'.tif'
    
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm
    
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(directory, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(NVAI)
    dst_dataset = None
    print directory

# <codecell>

def _VCI_CAL(coverageID, request_name, from_date_order,bbox_order, directory):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    #print [start_date, end_date]

    #generate the dates list 
    cur_date=dt.datetime.strptime(from_date_order,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(bbox_order[1])+','+str(bbox_order[3])+')',
                        'Long('+str(bbox_order[0])+','+str(bbox_order[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
#     VCI *= 1000
#     VCI //= (1000 - 0 + 1) / 255. # instead of 256 to make space for zero values
#     VCI = VCI.astype(numpy.uint8)
#     VCI += 1 # So 0 values are reserved for mask

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    #vciFileName = 'VCI'+cur_date.strftime("%Y%m%d")+'.tif'
    
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm    
    
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(directory, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(VCI)
    dst_dataset = None
    print directory

# <codecell>

def _TCI_CAL(coverageID, request_name, from_date_order,bbox_order, directory):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_11c2_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    #print [start_date, end_date]

    #generate the dates list 
    cur_date=datetime.strptime(from_date_order,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='modis_11c2_cov'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(bbox_order[1])+','+str(bbox_order[3])+')',
                        'Long('+str(bbox_order[0])+','+str(bbox_order[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    ##VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    TCI=(numpy.amax(cube_arr_ma,0)-cube_arr_ma[cur_pos,:,:])*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
#     TCI *= 1000
#     TCI //= (1000 - 0 + 1) / 255. #instead of 256
#     TCI = TCI.astype(numpy.uint8)
#     TCI += 1 #so mask values are reserverd for 0 

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    #tciFileName = 'TCI'+cur_date.strftime("%Y%m%d")+'.tif'
    
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm     
    
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(directory, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(TCI)
    dst_dataset = None
    print directory

# <codecell>

def _VCIvhi_CAL(date,spl_arr):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    #print [start_date, end_date]

    #generate the dates list 
    cur_date=datetime.strptime(date,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                        'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    #VCI *= 1000
    #VCI //= (1000 - 0 + 1) / 256.
    #VCI = VCI.astype(numpy.uint8) 
    return VCI,ds

def _TCIvhi_CAL(date,spl_arr):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_11c2_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    #print [start_date, end_date]

    #generate the dates list 
    cur_date=datetime.strptime(date,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='modis_11c2_cov'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                        'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    ##VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    TCI=(numpy.amax(cube_arr_ma,0)-cube_arr_ma[cur_pos,:,:])*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    return TCI, cur_date

def _VHI_CAL(coverageID, request_name, from_date_order,bbox_order, directory,alpha = 0.5):
    
    TCI, cur_date = _TCIvhi_CAL(from_date_order,bbox_order)
    VCI, ds = _VCIvhi_CAL(from_date_order,bbox_order)
    VHI = (alpha * VCI ) + ((1-alpha)* TCI)
    
#     VHI *= 1000
#     VHI //= (1000 - 0 + 1) / 255. #instead of 256
#     VHI = VHI.astype(numpy.uint8)
#     VHI += 1 #so mask values are reserverd for 0 

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    #vhiFileName = 'VHI'+cur_date.strftime("%Y%m%d")+'.tif'
    
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm  
    
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(directory, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(VHI)
    dst_dataset = None
    print directory 

# <codecell>

def email(info_order, url_link):
    msg = MIMEMultipart()
    msg['From'] = 'm.vanhoek@radi.ac.cn'
    msg['To'] = info_order[-1]
    msg['Subject'] = 'Data request East-Asian Drought monitoring system'
    message =  """
    Dear %(name)s,

    Thanks for your interesting in the East-Asian Drought monitoring system.

    Your request has been automatically generated and the files are ready
    to be downloaded.

    Please follow the following link:

    %(link_to_outfolder)s

    We hope the request serves your need.

    Best of luck,

    Team East-Asian Drought monitoring system
    RADI-CAS


    p.s Please cite the corresponding paper to support continous future support
    of the website

    """ % {"name":info_order[2].capitalize(),"link_to_outfolder":url_link}
    msg.attach(MIMEText(message))

    mailserver = smtplib.SMTP('smtp.radi.ac.cn',25)
    # identify ourselves to smtp gmail client
    mailserver.ehlo()
    # secure our email with tls encryption
    mailserver.starttls()
    # re-identify ourselves as an encrypted connection
    mailserver.ehlo()
    mailserver.login('m.vanhoek@radi.ac.cn', 'Radi2014')
    mailserver.sendmail('m.vanhoek@radi.ac.cn',info_order[-1],msg.as_string())
    mailserver.quit()

# <codecell>

# get a UUID - URL safe, Base64
def get_a_uuid():
    r_uuid = base64.urlsafe_b64encode(uuid.uuid4().bytes)
    return r_uuid.replace('=', '')

# <codecell>

def getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):    

    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='GetCoverage'
    field['COVERAGEID']=coverageID#'trmm_3b42_coverage_1'
    field['SUBSET']=['ansi("'+str(from_date_order)+'")',
                     'Lat('+str(bbox_order[1])+','+str(bbox_order[3])+')',
                    'Long('+str(bbox_order[0])+','+str(bbox_order[2])+')']
    field['FORMAT']='image/tiff'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values    
    print full_url    
    
    flnm=request_name+'_'+from_date_order+'_'+str(bbox_order[0])+'_'+str(bbox_order[1])+'_'+str(bbox_order[2])+'_'+str(bbox_order[3])
    flnm=re.sub('[^0-9a-zA-Z]+', '_',flnm)
    flnm+='.tif'
    directory+=flnm
    f,h = urllib.urlretrieve(full_url,directory)
    
    print directory
    #return directory,f

# <codecell>

def collect_data(array_order, from_date_order,bbox_order, endpoint, directory):
    for i in array_order:
        if i == 'p_gpcp':            
            coverageID = 'None'
            #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):
        if i == 'p_trmm':
            coverageID = 'trmm_3b42_coverage_1'
            #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):
        if i == 't_lst':
            coverageID = 'modis_11c2_cov'
            request_name = i
            getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory)
        if i == 'et_radi':
            coverageID = 'radi_et_v1'
            #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory):
        if i == 'ndvi_modis':
            coverageID = 'modis_13c1_cov'
            request_name = i        
            getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory)
        if i == 'di_pap':
            coverageID = 'gpcp'
            request_name = i        
            PRECIP_DI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        if i == 'di_vci':
            coverageID = 'modis_13c1_cov'
            request_name = i        
            _VCI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        if i == 'di_tci':
            coverageID = 'modis_11c2_cov'
            request_name = i        
            _TCI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        if i == 'di_vhi':
            coverageID = 'modis_11c2_cov'
            request_name = i        
            _VHI_CAL(coverageID, request_name, from_date_order,bbox_order, directory,alpha = 0.5)
        if i == 'di_nvai':
            coverageID = 'modis_11c2_cov'
            request_name = i        
            _NVAI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        if i == 'di_ntai':
            coverageID = 'modis_11c2_cov'
            request_name = i        
            _NTAI_CAL(coverageID, request_name, from_date_order,bbox_order, directory)
        if i == 'di_netai':
            coverageID = 'modis_11c2_cov'
            #request_name = i        
            #getData(coverageID, request_name, from_date_order, bbox_order, endpoint, directory): 

# <codecell>

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_ORDER",
            title="Compute the order",
            abstract="""Module to order data""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs

        self.fromDateIn = self.addLiteralInput(identifier="from_date_order",
                    title = "The start date to be calcualted",
                                          type=type(''))

        #self.toDateIn = self.addLiteralInput(identifier="to_date",
        #            title = "The final date to be calcualted",
        #                                  type=type(''))   
        
        self.bboxIn = self.addLiteralInput(identifier="bbox_order",
                    title="spatial area",
                    type=type(''))

        self.arrayIn = self.addLiteralInput(identifier="array_order",
                    title="client selection",
                    type=type(''))

        self.infoIn = self.addLiteralInput(identifier="info_order",
                    title = "client info",
                    type=type(''))       
    ##
    # Execution part of the process
    def execute(self):
        # 1. Load the data
        from_date_order = self.fromDateIn.getValue()
        bbox_order = self.bboxIn.getValue()
        array_order = self.arrayIn.getValue() 
        info_order = self.infoIn.getValue() 

        logging.info(from_date_order)
        logging.info(bbox_order)
        logging.info(array_order)
        logging.info(info_order)          
        
        # 2. Do the Work
        # set endpoint
        endpoint='http://159.226.117.95:8080/rasdaman/ows'

        # get unique user ID and create path
        uuid_in = get_a_uuid()
        server_path = r'D:\GoogleChromeDownloads\MyWebSites'+'\\'+uuid_in+'\\'
        client_path = 'http://159.226.117.95/wpsoutputs/'+uuid_in
        if not os.path.exists(server_path):
            os.makedirs(server_path)

        # request and prepare all data
        collect_data(array_order, from_date_order,bbox_order, endpoint, server_path)
        
        # 3. Output 
        # send the goddamn email
        email(info_order,client_path)        
        
        return

# <codecell>

# from_date_order = '2014-01-01'
# bbox_order = [108.1, 32.5, 126.5, 43.8]
# array_order = ["p_gpcp","p_trmm","t_lst","et_modis","et_radi","ndvi_modis","di_pap","di_vci","di_tci","di_vhi","di_nvai","di_ntai","di_netai"]
# info_order = ["0.15", "ascii", "mattijn", "radi", "mattijn@gmail.com"]

# <codecell>

urlIn = '107.5,27,133.2,45.2'

# <codecell>

map(float, urlIn.split(','))

# <codecell>


