# coding: utf-8

# # The procedule for calculate precipitation based drougt index
# This procedule aims to calculate different precipitation based drought index based on grided long term precipitaion data.
# The data source: http://www.esrl.noaa.gov/psd/data/gridded/data.gpcp.html
# The data is fetched on requests by opendap.
# 
# The dataset contains both long term average and monthly mean preciptation, thus we can calculate different precipitation based anomally indicators. 
# Precipitation Anomaly Percentage (PAP): the ratio between precipitation anomally and long term average. This index can be calculated on different temporal scales like monthly, seasonal, annual.
# 

# In[25]:

#%matplotlib inline
from pywps.Process import WPSProcess 
import pydap
from pydap.client import open_url
#import matplotlib,pylab
#from matplotlib.pyplot import figure, show, savefig
#import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
import sys
import gdal
import shutil
import logging


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

    # prepare output for GDAL
    papFileName = 'PRECIP_DI'+date +'.tif'        
    driver = gdal.GetDriverByName( "GTiff" )
    ds = driver.Create(papFileName, mon.shape[1], mon.shape[0], 1)
    
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
    
    return papFileName
    ## fetch data

if __name__== '__main__':
    #starttime=datetime.now()
    res=PRECIP_DI_CAL()
    #timedelta=(datetime.now()-starttime)
    #print 'Total running time: %s' % timedelta.seconds
    # plot grid data
    


# In[ ]:

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_PRECIP_DI_CAL",
            title="Precipitation based Drought Index",
            abstract="""This process intend to Calculate precipitation based drought index.""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs
        
        
        self.boxIn = self.addBBoxInput(identifier="bbox",
                    title="Spatial region")

        self.dateIn = self.addLiteralInput(identifier="date",
                    title = "The date to be calcualted",
                                          type=type(''))

        ##
        # Adding process outputs

        self.dataOut = self.addComplexOutput(identifier="map",
                title="Output PAP image",
                useMapscript = True,
                formats =  [
                           {'mimeType':'image/tiff'},
                           {'mimeType':'image/png'} # ADDED: 07-08-2015 CHECK IF IT RESOLVE ISSUE
                           ])

        #self.textOut = self.addLiteralOutput(identifier = "text",
         #       title="Output literal data")
    ##
    # Execution part of the process
    def execute(self):
        
        #Get the xml setting file string
        # Get the box value
        BBOXObject = self.boxIn.getValue()
        CoordTuple = BBOXObject.coords
        
        #Get the date string
        date = self.dateIn.getValue()
        #logging.info(CoordTuple)
        #logging.info(date)


        #date='2013-06-30'
        #spl_arr=[70,30,80,50]
        spl_arr=[CoordTuple[0][0],CoordTuple[0][1],CoordTuple[1][0],CoordTuple[1][1]]
        #logging.info(date)
        #logging.info(spl_arr)
        papfn=PRECIP_DI_CAL(date,spl_arr)
        self.dataOut.setValue( papfn )
        return

if __name__== '__main__':
    starttime=datetime.now()
    print RegionHants('http://localhost/html/HANTS_PS.xml')
    timedelta=(datetime.now()-starttime)
    print 'Total running time: %s' % timedelta.seconds
