# PyWPS::Execute
# http://localhost/cgi-bin/pywps.cgi?request=Execute&identifier=mapserver_test&service=WPS&version=1.0.0&responsedocument=[map=@asReference=true]
# coding: utf-8
# sld: http://159.226.117.95/mapdata/v0.6/xml/sld_raster2.xml
#
# MapServer::GetMap
# http://159.226.117.95/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps25513-tmpA6rFXE.map&LAYERS=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=TRUE&SLD=http%3A%2F%2F159.226.117.95%2Fmapdata%2Fv0.6%2Fxml%2Fsld_raster5.xml&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=EPSG%3A900913&BBOX=10801469.339531,6418264.3901563,10958012.373438,6574807.4240625&WIDTH=256&HEIGHT=256
#
# MapServer::GetLegend
# http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps25513-tmpA6rFXE.map&LAYERS=map&FORMAT=image%2Fpng&VERSION=1.1.1&REQUEST=GetLegendGraphic
# In[ ]:
from __future__ import division
from pywps.Process import WPSProcess 
import types
import logging
import os
import sys
import urllib
from osgeo import gdal
import numpy
import numpy as np
import numpy.ma as ma
import cStringIO

def display(image, display_min, display_max):
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    image //= (display_max - display_min + 1) / 256.
    return image.astype(np.uint8)

def lut_display(image, display_min, display_max):
    lut = np.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image) 

# In[ ]:

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__( self,
            identifier       = "mapserver_test",
            title            = "Test PyWPS Process MapServer",
            abstract         = "Extensive test, whether it is possible to output WMS/WCS URL using MapServer",
            version          = "1.0",
            storeSupported   = True,
            statusSupported  = True)
        ##
        # Adding process inputs
        
        self.boxIn     = self.addBBoxInput(     identifier  = "bbox",
                                                title       = "Spatial region")         

        self.dateIn    = self.addLiteralInput(  identifier  = "date",
                                                title       = "The date to be calcualted",
                                                type        = types.StringType,
                                                default     = '"2015-01-02"')     
        ##
        # Adding process outputs
        self.outputMap = self.addComplexOutput( identifier  = "map",
                                                title       = "Resulting output map",
                                                useMapscript= True,
                                                formats     = [
                                                              #{"mimeType":"image/tiff"},
                                                              {"mimeType":"image/png"}
                                                              ])
    # In[ ]:   

    def load_tif(self,d,spl_arr):    
        endpoint='http://159.226.117.95:8080/rasdaman/ows'
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
        
        logging.info(full_url)
        
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)

        ds=gdal.Open(tmpfilename)
        array = ds.ReadAsArray()
        array[array < 0] = -1
        #array_ma = ma.masked_equal(array,0)

        # do math
        array_ma = lut_display(array, -1, 10000)
        #array_ma[array_ma == 255] = 0
        nan = 0#int(array_ma.min())        

        ##write the result to disk
        # get parameters
        geotransform = ds.GetGeoTransform()
        spatialreference = ds.GetProjection()
        ncol = ds.RasterXSize
        nrow = ds.RasterYSize
        nband = 1

        # create dataset for output
        fmt = 'GTiff'
        FileName = 'TIF_TEST.tif'
        driver = gdal.GetDriverByName(fmt)
        dst_dataset = driver.Create(FileName, ncol, nrow, nband, 1)
        dst_dataset.SetGeoTransform(geotransform)
        dst_dataset.SetProjection(spatialreference)
        if nan != None:
            dst_dataset.GetRasterBand(1).SetNoDataValue(nan) 
        dst_dataset.GetRasterBand(1).WriteArray(array_ma)
        dst_dataset = None
        return FileName

    ##
    # Execution part of the process
    def execute(self):
        d = self.dateIn.getValue() 
        d = '"'+d+'"'
        logging.info(d)

        #d = '"2015-01-02"'

        bbox = self.boxIn.getValue()
        bbox = bbox.coords
        spl_arr = [bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1]]
        logging.info(spl_arr)           
        #spl_arr = [96.288,26.758,120.282,42.491] #lon/south, lat/east, lon/north, lat/west

        #logging.info(d)
        #logging.info(spl_arr)  
        
        loadTif = self.load_tif(d,spl_arr)
        self.outputMap.setValue( loadTif )

        logging.info(os.getcwd())
        return

