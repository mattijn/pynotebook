# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pywps.Process import WPSProcess 
import types
import os
import geojson
import subprocess as sp
import json
import logging
import sys
import urllib
from osgeo import gdal
import numpy as np
from lxml import etree
from datetime import datetime, timedelta
import pandas as pd
from cStringIO import StringIO
from datetime import datetime, timedelta

# <codecell>

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()

# <codecell>

def datelist_irregular_coverage(root, date, no_observations):
    #get ANSI start date of coverage
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    start_date=int((lc.split(' '))[2])
    #print [start_date, end_date]

    #convert user date to ISO date
    cur_date = datetime.strptime(date,"%Y-%m-%d")
    #convert start date coverage to ISO date
    start = datetime.fromtimestamp((start_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)
    #print start    

    # get sample size coefficients from XML root
    sample_size = root[0][3][0][5][0][1].text #sample size
    #print root[0][3][0][5][0][1].text #sample size
    
    # use coverage start_date and sample_size array to create all dates in ANSI
    array_stepsize = np.fromstring(sample_size, dtype=int, sep=' ')
    #print np.fromstring(sample_size, dtype=int, sep=' ')
    array_all_ansi = array_stepsize + start_date   
    
    # create array of all dates in ISO
    list_all_dates = []
    for stepsize in array_stepsize:
        date_and_stepsize = start + timedelta(stepsize - 1)
        list_all_dates.append(date_and_stepsize)
        #print date_and_stepsize
    array_all_dates = np.array(list_all_dates)      

    # subtract user date of all dates in ISO 
    # and find the nearest available coverage date
    array_diff_dates = array_all_dates - cur_date
    idx_nearest_date = find_nearest(array_diff_dates, timedelta(0))
    date_list = array_all_dates[idx_nearest_date-no_observations:idx_nearest_date+1]
    
    # return datelist in ANSI and the index of the nearest date
    return date_list[::-1]

# <codecell>

def getDateList(date, no_observations):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://192.168.1.104:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#NDVI_MOD13C1005_uptodate
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    print full_url
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)

    try:
        # regular coverage: should contain a fixed temporal resolution 
        # get it and compute date_list    
        temp_res = int((root[0][3][0][5].text).split(' ')[2])
        print 'regular coverage'
        print temp_res

        # convert all required dates in ISO date format
        date_start = datetime(int(date[0:4]),int(date[5:7]),int(date[8:10]))
        date_list = []
        date_list.append(date_start)
        for i in range(1,no_observations+1):
            #print i
            date_list.append(date_start - (i *timedelta(days=temp_res)))
    except:
        # irregular coverage: get date_list according to the offset
        print 'irregular coverage'
        date_list = datelist_irregular_coverage(root, date, no_observations)    
    return date_list

# <codecell>

extent=[108.8,38.2,121.1,43.5]
date = "2014-04-01"
no_observations = 4
coverageID = 'NDVI_MOD13C1005_uptodate'

# <codecell>

date_list = getDateList(date, no_observations, coverageID)
# request data use WCS service baed on extend and clip based on sql query
array_NDAI = []
endpoint='http://192.168.1.104:8080/rasdaman/ows'
for j in date_list:
    #logging.info(j)
    #d = 150842
    date_in_string = '"'+str(j.year)+'-'+str(j.month).zfill(2)+'-'+str(j.day).zfill(2)+'"'

    #logging.info(date_in_string)
    #logging.info(str(extent[1]))
    #logging.info(str(extent[3]))
    #logging.info(str(extent[0]))
    #logging.info(str(extent[2]))

    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='GetCoverage'
    field['COVERAGEID']=coverageID#'trmm_3b42_coverage_1'#NDVI_MOD13C1005_uptodate
    field['SUBSET']=['ansi('+date_in_string+')',#['ansi('+str(d)+')',
                     'Lat('+str(extent[1])+','+str(extent[3])+')',
                    'Long('+str(extent[0])+','+str(extent[2])+')']
    field['FORMAT']='image/tiff'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values

    logging.info(full_url)
    print full_url
    tmpfilename='test'+str(j.toordinal())+'.tif'

    #logging.info(tmpfilename)
    f,h = urllib.urlretrieve(full_url,tmpfilename)
    #logging.info(h)
    #print h
    #ds_test_tmpfilename = gdal.Open(tmpfilename).ReadAsArray()
    #logging.info(ds_test_tmpfilename[0])

    #ds=gdal.Open(tmpfilename)
#         clippedfilename='test'+str(j.toordinal())+'clip.tif' 

#         path_base = "/var/www/html/wps/CHN_adm"
#         CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")

#         command = ["/usr/bin/gdalwarp","-cutline",CHN_adm_gpkg,"-csql","SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = '"+NAME_1+"' and NAME_2 = '"+NAME_2+"' and NAME_3 = '"+NAME_3+"'","-crop_to_cutline","-of","GTiff","-dstnodata","-9999",tmpfilename,clippedfilename, "-overwrite"]

#         logging.info(sp.list2cmdline(command))
#         #print (sp.list2cmdline(command))

#         norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)  
#         norm.communicate()   

#         ds=gdal.Open(clippedfilename)
#         ds_clip = ds.ReadAsArray() 

    ds=gdal.Open(tmpfilename)
    ds_clip = ds.ReadAsArray()         

    #logging.info(ds_clip[0])

    array_NDAI.append(ds_clip)

# <codecell>

array_NDAI

# <codecell>


