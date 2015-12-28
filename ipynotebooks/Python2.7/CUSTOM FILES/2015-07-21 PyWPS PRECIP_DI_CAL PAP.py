# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#%matplotlib inline
from pywps.Process import WPSProcess 
import pydap
from pydap.client import open_url
import matplotlib,pylab
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
import sys
import gdal
import shutil
import logging

# <codecell>




# In[29]:

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
    
    PAP[np.where(PAP>200)]=200
    PAP[np.where(PAP<-200)]=-200
    PAP += 200
    PAP //= (400 - 0 + 1) / 256.
    PAP = PAP.astype(np.uint8) 

