
# coding: utf-8

# In[1]:

#!/usr/bin/env python


# from pywps.Process import WPSProcess 
import logging

import matplotlib 
# logging.info('get backend before set')
# print('get backend before set')
# logging.info(matplotlib.matplotlib_fname())
# print(matplotlib.matplotlib_fname())
# logging.info(matplotlib.get_backend())
# print(matplotlib.get_backend())
# matplotlib.rcParams['backend'] = 'AGG'
# matplotlib.use('AGG')

# logging.info('get backend after set')
# logging.info(matplotlib.get_backend())

import sys
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcb
import cartopy.feature as cfeature
from matplotlib import gridspec
from datetime import datetime
import warnings
from osgeo import gdal
import numpy as np
from osgeo import gdal, ogr, osr
import sys
import pandas as pd
import geopandas as gpd
from matplotlib import gridspec
from cartopy.io import shapereader
import shapely.geometry as sgeom
import numpy as np
import matplotlib as mpl
import urllib
import numpy
import numpy as np
import numpy.ma as ma
from lxml import etree
from datetime import datetime, timedelta
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
# sys.path.insert(0, r'/var/www/html/wps/pywps/processes/NDAI_PROCESSING/CDMA')
# import cdma


# In[16]:

def datelist_irregular_coverage(root, start_date, start, cur_date):
    """
    retrieve irregular datelist and requested current position in regards to total no. of observations
    """
    
    #root[0]                - wcs:CoverageDescription
    #root[0][0]             - boundedBy 
    #root[0][0][0]          - Envelope
    #root[0][0][0][0]       - lowerCorner
    # --- 
    #root[0]                - wcs:CoverageDescription
    #root[0][3]             - domainSet
    #root[0][3][0]          - gmlrgrid:ReferenceableGridByVectors
    #root[0][3][0][5]       - gmlrgrid:generalGridAxis
    #root[0][3][0][5][0]    - gmlrgrid:GeneralGridAxis
    #root[0][3][0][5][0][1] - gmlrgrid:coefficients

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
    
    # create array of all dates as DOY
    list_all_yday = []
    for j in array_all_dates:
        yday = j.timetuple().tm_yday
        list_all_yday.append(yday)
        #print yday
    array_all_yday = np.array(list_all_yday)    
    
    # subtract user date of all dates in ISO 
    # to find the nearest available coverage date
    array_diff_dates = array_all_dates - cur_date
    idx_nearest_date = find_nearest(array_diff_dates, timedelta(0))
    nearest_date = array_all_dates[idx_nearest_date]    
    
    # select all coresponding DOY of all years for ANSI and ISO dates
    array_selected_ansi = array_all_ansi[array_all_yday == nearest_date.timetuple().tm_yday]
    array_selected_dates = array_all_dates[array_all_yday == nearest_date.timetuple().tm_yday]
    print array_selected_ansi
    
    # get index of nearest date in selection array
    idx_nearest_date_selected = numpy.where(array_selected_dates==nearest_date)[0][0]  
    print idx_nearest_date_selected
    
    # return datelist in ANSI and the index of the nearest date
    return array_selected_ansi, idx_nearest_date_selected

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()


# In[2]:

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists


# In[115]:

logging.info('test')
#date='2015-06-30'
endpoint='http://192.168.1.104:8080/rasdaman/ows'
field={}
field['SERVICE']='WCS'
field['VERSION']='2.0.1'
field['REQUEST']='DescribeCoverage'
field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
url_values = urllib.urlencode(field,doseq=True)
full_url = endpoint + '?' + url_values
data = urllib.urlopen(full_url).read()
root = etree.fromstring(data)
lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
start_date = int((lc.split(' '))[2])
end_date = int((uc.split(' '))[2])
#print [start_date, end_date]

#generate the dates list 
start = datetime.fromtimestamp((start_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)
#print start

try:    
    # get sample size coefficients from XML root
    sample_size = root[0][3][0][5][0][1].text #sample size
    #print root[0][3][0][5][0][1].text #sample size

    # use coverage start_date and sample_size array to create all dates in ANSI
    array_stepsize = np.fromstring(sample_size, dtype=int, sep=' ')
    #print np.fromstring(sample_size, dtype=int, sep=' ')
    array_all_ansi = array_stepsize + start_date  
    #print 'irregular'
    print 'irregular'
    #print array_all_ansi
except IndexError:
    datelist, cur_pos = datelist_regular_coverage(root, start_date, start, cur_date)
    #print 'regular'
    print 'regular'

# create array of all dates in ISO
list_all_dates = []
for stepsize in array_stepsize:
    date_and_stepsize = start + timedelta(stepsize - 1)
    list_all_dates.append(date_and_stepsize)
    #print date_and_stepsize
array_all_dates = np.array(list_all_dates)
#print array_all_dates

# create array of all dates in string
array_all_date_string = []
for i in array_all_dates:
    date_string = str(i.year).zfill(2)+'-'+str(i.month).zfill(2)+'-'+str(i.day).zfill(2)
    array_all_date_string.append(date_string)
#print array_all_date_string
    
dates_wcs = []
for i in array_all_date_string:
    year = int(i[-10:-6])
    month = int(i[-5:-3])
    day = int(i[-2::])
    last_date_wcs = datetime(year,month,day)
    dates_wcs.append(last_date_wcs)
#print dates_wcs
dates_unique = dates_wcs
array_sel_date_string = []
for i in dates_unique:
    date_string = str(i.year).zfill(2)+'-'+str(i.month).zfill(2)+'-'+str(i.day).zfill(2)
    array_sel_date_string.append(date_string)

array_sel_date_string.sort()
#print 'all missing dates: ', array_sel_date_string
#array_sel_even_date_string = np.array_split(array_sel_date_string,4)[0]
print 'all missing dates: ', array_sel_date_string


# In[118]:

print datetime.fromtimestamp((end_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)


# In[108]:

date = array_sel_date_string[-2]
print date


# In[109]:

##request image cube for the specified date and area by WCS.
#firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
endpoint='http://192.168.1.104:8080/rasdaman/ows'
field={}
field['SERVICE']='WCS'
field['VERSION']='2.0.1'
field['REQUEST']='DescribeCoverage'
field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'LST_MOD11C2005_uptodate'
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
cur_date = datetime.strptime(date,"%Y-%m-%d")
#startt=145775
start = datetime.fromtimestamp((start_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)


# In[111]:

print start_date, start, cur_date
datelist, cur_pos = datelist_irregular_coverage(root, start_date, start, cur_date)


# In[112]:

datelist[cur_pos]


# In[114]:

for d in datelist:
    print datetime.fromtimestamp((d - (datetime(1970,1,1) - datetime(1601,1,1)).days)*24*60*60)


# In[ ]:



