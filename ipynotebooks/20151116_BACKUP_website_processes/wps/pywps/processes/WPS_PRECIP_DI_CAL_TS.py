from pywps.Process import WPSProcess 
import pydap.client
from pydap.client import open_url
import numpy as np
from datetime import datetime as dt
import os
import sys
import gdal
import shutil
import logging
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_cl
from mpl_toolkits.basemap import Basemap
from osgeo import osr, gdal
import pandas as pd
from datetime import datetime
import cStringIO
import json

def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    logging.info('function `unix_time` complete')
    return delta.total_seconds()    

def unix_time_millis(dt):
    logging.info('function `unix_time_millis` complete')
    return int(unix_time(dt) * 1000)

def PAP_DI_CAL_TIMESERIES(from_date='2010-06-06', to_date='2014-06-06', lon_in = 113.9797, lat_in = 42.7202):

    opendap_url_mon='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc'
    opendap_url_ltm='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.ltm.nc'

    # convert iso-date to gregorian calendar and get the month
    fo_dta=(dt.strptime(from_date,'%Y-%m-%d').date()-dt.strptime('1800-01-01','%Y-%m-%d').date()).days
    to_dta=(dt.strptime(to_date,'%Y-%m-%d').date()-dt.strptime('1800-01-01','%Y-%m-%d').date()).days
    mon=(dt.strptime(from_date,'%Y-%m-%d').date()).month

    # open opendap connection and request the avaialable time + lon/lat
    dataset_mon = open_url(opendap_url_mon)
    time=dataset_mon.time[:]
    lat=dataset_mon.lat[:]
    lon=dataset_mon.lon[:]

    fo_dt_ind=next((index for index,value in enumerate(time) if value > fo_dta),0)-1
    to_dt_ind=next((index for index,value in enumerate(time) if value > to_dta),0)-1

    if lon_in < 0: lon_in += 360 #+ 180 # GPCP is from 0-360, OL is from -180-180

    lon_index = next((index for index,value in enumerate(lon) if value > lon_in),0)-1
    lat_index = next((index for index,value in enumerate(lat) if value < lat_in),0)-1

    time_sel = (time>fo_dta)&(time<to_dta)
    time_sel[np.nonzero(time_sel)[0]-1] = True

    dataset_mon=dataset_mon['precip'][time_sel,lat_index,lon_index]
    mon = np.ma.masked_less((dataset_mon['precip'][:]).squeeze(),0)

    months_index = np.ones(shape=(12),dtype=bool)
    dataset_ltm = open_url(opendap_url_ltm)
    dataset_ltm=dataset_ltm['precip'][months_index,lat_index,lon_index]

    ltm = np.ma.masked_less((dataset_ltm['precip'][:]).squeeze(),0)

    from_date_ordinal = datetime.toordinal(datetime(1800,1,1)) + time[fo_dt_ind]
    from_date_ordinal = datetime.fromordinal(int(from_date_ordinal))

    end_date_ordinal = datetime.toordinal(datetime(1800,1,1)) + time[to_dt_ind]
    end_date_ordinal = datetime.fromordinal(int(end_date_ordinal))

    date_range = pd.date_range(from_date_ordinal, end_date_ordinal, freq='MS')
    
    try:
        ts = pd.Series(mon, index=date_range[:-1])
    except:
        ts = pd.Series(mon, index=date_range)

    new_dates = []
    new_values = []
    for i,j in zip(ts.index, ts):
        #print i.month, j, ltm[i.month-1]
        new_dates.append(i)
        new_values.append((j-ltm[i.month-1])/(ltm[i.month-1]+1)*100)
    PAP = pd.Series(new_values, index=new_dates)

    # data preparation for HighCharts: Output need to be in JSON format with time 
    # in Unix milliseconds
    dthandler = lambda obj: (
    unix_time_millis(obj)
    if isinstance(obj, datetime)
    or isinstance(obj, date)
    else None)

    output1 = cStringIO.StringIO()

    logging.info('ready to dump files to JSON')
    # np.savetxt(output, pyhants, delimiter=',')
    out1 = json.dump(PAP.reset_index().as_matrix().tolist(), output1, default=dthandler)

    logging.info('dates converted from ISO 8601 to UNIX in ms')             
    
    return output1

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_PRECIP_DI_CAL_TS",
            title="Compute PAP TIMESERIES",
            abstract="""Module to compute PAP TimeSeries based on GPCP data""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs
        self.lonIn = self.addLiteralInput(identifier="pap_lon",
                    title="Longitude",type=type(''))

        self.latIn = self.addLiteralInput(identifier="pap_lat",
                    title="Latitude",type=type(''))        

        self.fromDateIn = self.addLiteralInput(identifier="from_date",
                    title = "The start date to be calcualted",
                                          type=type(''))

        self.toDateIn = self.addLiteralInput(identifier="to_date",
                    title = "The final date to be calcualted",
                                          type=type(''))        

        ##
        # Adding process outputs

        self.papOut = self.addComplexOutput(identifier  = "pap_ts", 
                                        title       = "Pap Timeseries",
                                        formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json   


    ##
    # Execution part of the process
    def execute(self):
	logging.info('we are alrady here')
        # Load the data
        From_date = self.fromDateIn.getValue()
        To_date = self.toDateIn.getValue()
        Lon_in = float(self.lonIn.getValue())
        Lat_in = float(self.latIn.getValue())
        logging.info(Lat_in)

        
        # Do the Work
        pap_out = PAP_DI_CAL_TIMESERIES(from_date=From_date, to_date=To_date, lon_in=Lon_in, lat_in=Lat_in)
        
        # Save to out
        self.papOut.setValue( pap_out )
        return


