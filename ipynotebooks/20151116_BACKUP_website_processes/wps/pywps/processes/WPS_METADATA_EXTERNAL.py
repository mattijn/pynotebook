from pywps.Process import WPSProcess 
import urllib
from datetime import datetime
from lxml import etree
import pydap
from pydap.client import open_url
import logging

def metadata_opendap_gpcp():
    opendap_url_ltm='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.ltm.nc'
    opendap_url_mon='http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/gpcp/precip.mon.mean.nc'
    
    # open opendap connection and request the avaialable time + lon/lat
    dataset_mon = open_url(opendap_url_mon)
    time=dataset_mon.time[:]
    lat=dataset_mon.lat[:]
    lon=dataset_mon.lon[:]    
    
    # CONVERT DATES from_date, to_date
    start_date = time[0]
    end_date = time[-1]
    start=datetime.fromtimestamp((start_date-(datetime(1970,1,1)-datetime(1800,1,1)).days)*24*60*60)
    end=datetime.fromtimestamp((end_date-(datetime(1970,1,1)-datetime(1800,1,1)).days)*24*60*60)    
    
    from_date = str(start.year)+"-"+str(start.month).zfill(2)+"-"+str(start.day).zfill(2)
    to_date = str(end.year)+"-"+str(end.month).zfill(2)+"-"+str(end.day).zfill(2)
    
    # GET BBOX lonmin, latmin, lonmax, latmax
    lonmin = str(lon[0])
    latmin = str(lat[0])
    lonmax = str(lon[-1])
    latmax = str(lat[-1])
    
    # GET RESOLUTION temp_resolution, spatial_resolution
    temp_resolution = str(1) + 'm'
    spatial_resolution = str(lon[1]-lon[0]) + 'deg'
    return from_date, to_date, temp_resolution, lonmin, latmin, lonmax, latmax, spatial_resolution    

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_METADATA_EXTERNAL",
            title="Metadata of the available coverage",
            abstract="""Module to get all metadata of the coverages included in the system""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs
        ##

        ##
        # Adding process outputs
        
        self.gpcp_Out = self.addLiteralOutput(identifier="gpcp",
                title="metadata gpcp")        


    ##
    # Execution part of the process
    def execute(self):

        gpcp = metadata_opendap_gpcp()

        self.gpcp_Out.setValue( gpcp )        
        return

