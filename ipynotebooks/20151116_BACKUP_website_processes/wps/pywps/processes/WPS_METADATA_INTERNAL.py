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

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_METADATA_INTERNAL",
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

        self.LST_MOD11C2005_Out = self.addLiteralOutput(identifier="LST_MOD11C2005",
                title="metadata LST_MOD11C2005")

        self.NDVI_MOD13C1005_Out = self.addLiteralOutput(identifier="NDVI_MOD13C1005",
                title="metadata NDVI_MOD13C1005")
        
        self.NDAI_1km_Out = self.addLiteralOutput(identifier="NDAI_1km",
                title="metadata NDAI_1km")        

    ##
    # Execution part of the process
    def execute(self):
        
        LST_MOD11C2005 = metadata_rsdm('LST_MOD11C2005')
        NDVI_MOD13C1005 = metadata_rsdm('NDVI_MOD13C1005')
        NDAI_1km = metadata_rsdm('NDAI_1km')

        self.LST_MOD11C2005_Out.setValue( LST_MOD11C2005 )
        self.NDVI_MOD13C1005_Out.setValue( NDVI_MOD13C1005 )
        self.NDAI_1km_Out.setValue( NDAI_1km )
        return

