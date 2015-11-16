from pywps.Process import WPSProcess 
import urllib
from datetime import datetime
from lxml import etree
import pydap
from pydap.client import open_url
import logging

def metadata_rsdm(CoverageName):
    """
    Returns the Capabilities of a coverage.
    
    In:  
    CoverageName : Name of the coverage ID in RASDAMAN
    
    Out:
    from_date : First date of Coverage
    end_date  : Last date of Coverage
    temp_resolution : Temporal resolution
    spatial_resolution : Spatial resolution
    lonmin    : Longitude min
    latmin    : Latitude min
    lonmax    : Longitude max
    latmax    : Latitude max
    """
    endpoint='http://localhost:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']=CoverageName#'modis_13c1_cov'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)

    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])

    # CONVERT DATES from_date, to_date
    start=datetime.fromtimestamp((start_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)
    end=datetime.fromtimestamp((end_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)
    
    from_date = str(start.year)+"-"+str(start.month).zfill(2)+"-"+str(start.day).zfill(2)
    to_date = str(end.year)+"-"+str(end.month).zfill(2)+"-"+str(end.day).zfill(2)
    
    # GET BBOX lonmin, latmin, lonmax, latmax
    lonmin = str((lc.split(' '))[1])
    latmin = str((lc.split(' '))[0])
    lonmax = str((uc.split(' '))[1])
    latmax = str((uc.split(' '))[0])
    
    # GET RESOLUTION temp_resolution, spatial_resolution
    temp_resolution = str((root[0][3][0][5].text).split(' ')[2]) + 'd'
    spatial_resolution = str((root[0][3][0][3].text).split(' ')[1]) + 'deg'
    return from_date, to_date, temp_resolution, lonmin, latmin, lonmax, latmax, spatial_resolution

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
            identifier = "WPS_METADATA",
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

        self.modis_13c1_cov_Out = self.addLiteralOutput(identifier="modis_13c1_cov",
                title="metadata modis_13c1_cov")

        self.modis_11c2_cov_Out = self.addLiteralOutput(identifier="modis_11c2_cov",
                title="metadata modis_11c2_cov")
        
        self.trmm_3b42_coverage_1_Out = self.addLiteralOutput(identifier="trmm_3b42_coverage_1",
                title="metadata trmm_3b42_coverage_1")
        
        self.gpcp_Out = self.addLiteralOutput(identifier="gpcp",
                title="metadata gpcp")        


    ##
    # Execution part of the process
    def execute(self):
        
        modis_13c1_cov = metadata_rsdm('modis_13c1_cov')
        modis_11c2_cov = metadata_rsdm('modis_11c2_cov')
        trmm_3b42_coverage_1 = metadata_rsdm('trmm_3b42_coverage_1')
        gpcp = metadata_opendap_gpcp()


        self.modis_13c1_cov_Out.setValue( modis_13c1_cov )
        self.modis_11c2_cov_Out.setValue( modis_11c2_cov )
        self.trmm_3b42_coverage_1_Out.setValue( trmm_3b42_coverage_1 )
        self.gpcp_Out.setValue( gpcp )        
        return

