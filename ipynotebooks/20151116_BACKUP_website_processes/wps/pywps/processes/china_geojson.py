# In[ ]:
from __future__ import division
from pywps.Process import WPSProcess
import types
import logging
import os
import geojson
import subprocess as sp
import json

def ExtractGeoJSON(NAME_1in, NAME_2in, NAME_3in):
    """
    Function to query OGC GeoPackage of China to extract county boundary as JSON
    input
    NAME_1in = Province/Shang (e.g: 'Anhui')
    NAME_2in = Regon/Shi (e.g: 'Bengbu')
    NAME_3in = County/Qian (e.g: 'Guzhen')
    
    output
    Boundary of County in GeoJSON format    
    """

    # get path url for OGC GeoPackage
    path_gpkg = "/var/www/html/wps/CHN_adm"
    CHN_adm_gpkg = os.path.join(path_gpkg, "CHN_adm.gpkg")
    
    # set path url for GeoJSON
    path_geojson = "/var/www/html/wps/wpsoutputs/"
    CHN_adm_geojson = os.path.join(path_geojson, "CHN_adm_selection.geojson")
    if os.path.exists(CHN_adm_geojson):
        os.remove(CHN_adm_geojson)
        logging.info('previous geojson file removed')
    
    command = ["/usr/bin/ogr2ogr", "-f", "GeoJSON", CHN_adm_geojson, "-sql", 
               "SELECT NAME_1, NAME_2, NAME_3 FROM CHN_adm3 WHERE NAME_1 = "+NAME_1in+" and NAME_2 = "+NAME_2in+" and NAME_3 = "+NAME_3in+"", 
               CHN_adm_gpkg, "-s_srs", "EPSG:4326","-t_srs","EPSG:900913", "-skipfailures", "-nlt", "LINESTRING"]
    
    # log the command 
    logging.info(sp.list2cmdline(command))
    
    norm = sp.Popen(sp.list2cmdline(command), shell=True)  
    norm.communicate()     

    with open(CHN_adm_geojson) as f:
        geojson2ol = json.load(f)    
    return geojson2ol, CHN_adm_geojson

# In[ ]:

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__( self,
            identifier       = "china_geojson",
            title            = "Extract GeoJSON boundary",
            abstract         = "Extract and transform county data from OGC GeoPackage to GeoJSON for OL",
            version          = "1.0",
            storeSupported   = True,
            statusSupported  = True)
        ##
        # Adding process inputs

        self.NAME_1    = self.addLiteralInput(  identifier  = "Province",
                                                title       = "Province",
                                                type        = types.StringType)        
        
        self.NAME_2    = self.addLiteralInput(  identifier  = "Prefecture",
                                                title       = "Region",
                                                type        = types.StringType)

        self.NAME_3    = self.addLiteralInput(  identifier  = "County",
                                                title       = "County",
                                                type        = types.StringType)
        ##
        # Adding process outputs
        self.GeoJSON = self.addComplexOutput( identifier  = "Bound_GeoJSON",
                                                title       = "Resulting Country Boundary",                                                
                                                formats     = [
                                                              {"mimeType":"application/json"}
                                                              ])
    # In[ ]:

    ##
    # Execution part of the process
    def execute(self):
        #Get input names
        NAME_1in = self.NAME_1.getValue()
        NAME_1in = '"'+NAME_1in+'"'
        NAME_2in = self.NAME_2.getValue()
        NAME_2in = '"'+NAME_2in+'"'
        NAME_3in = self.NAME_3.getValue()
        NAME_3in = '"'+NAME_3in+'"'
        logging.info(NAME_1in)
        logging.info(NAME_2in)
        logging.info(NAME_3in)

        #NAME_3in = "'Guzhen'"

        GeoJSONout, CHN_adm_geojson = ExtractGeoJSON(NAME_1in, NAME_2in, NAME_3in)       

        self.GeoJSON.setValue( CHN_adm_geojson )#GeoJSONout )

        logging.info(os.getcwd())
        return


