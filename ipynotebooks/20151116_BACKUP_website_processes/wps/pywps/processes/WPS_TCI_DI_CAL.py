# coding: utf-8

# # TCI calculation process
# This process intend to calculate the Temperature Condition index (TCI) for a specific area. The fomula of the index is:
# TCI =max(LST)-LST/(max(LST)-min(LST))
# where the LST is Land surface temperature.
# This is a WPS process served by PyWPS. 
# 
# Input:
# bBox:a rectangle box which specifies the processing area.
# date: a date string specifies the date to be calculated. The date format should be "YYYY-MM-DD".
# 
# Output:
# file:
# format:
# 
# The process internally retrieves NDVI data set from a rasdaman database.
# 
# Client side execute script:
# http://localhost/cgi-bin/pywps.cgi?service=wps&version=1.0.0&request=execute&identifier=WPS_TCI_CAL&datainputs=[date=2005-02-06;bbox=50,10,120,60]&responsedocument=image=@asReference=true

# In[1]:

from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib
from osgeo import gdal
import numpy
import numpy.ma as ma
from lxml import etree
from datetime import datetime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def _TCI_CAL(date,spl_arr):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://localhost:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='LST_MOD11C2005'#'trmm_3b42_coverage_1'
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
    cur_date=datetime.strptime(date,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='LST_MOD11C2005'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                        'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        #print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    ##VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    TCI=(numpy.amax(cube_arr_ma,0)-cube_arr_ma[cur_pos,:,:])*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    TCI *= 1000
    TCI //= (1000 - 0 + 1) / 255. #instead of 256
    TCI = TCI.astype(numpy.uint8)
    TCI += 1 #so mask values are reserverd for 0 

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    tciFileName = 'TCI'+cur_date.strftime("%Y%m%d")+'.tif'
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(tciFileName, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(TCI)
    dst_dataset = None
    return tciFileName

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_TCI_DI_CAL",
            title="TCI calculation process",
            abstract="""This process intend to calculate the Temperature Condition index (TCI) for a specific area..""",
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
                title="Output TCI image",
                useMapscript= True,
                formats =  [{'mimeType':'image/tiff'}])

        #self.textOut = self.addLiteralOutput(identifier = "text",
         #       title="Output literal data")

    
    ##
    # Execution part of the process
    def execute(self):

        # Get the box value
        BBOXObject = self.boxIn.getValue()
        CoordTuple = BBOXObject.coords
        
        #Get the date string
        date = self.dateIn.getValue()
        
        logging.info(CoordTuple)
        logging.info(date)        

        #date='2013-06-30'
        #spl_arr=[70,30,80,50]
        spl_arr=[CoordTuple[0][0],CoordTuple[0][1],CoordTuple[1][0],CoordTuple[1][1]]
        
        logging.info(date)
        logging.info(spl_arr)
        
        tcifn=_TCI_CAL(date,spl_arr)
        self.dataOut.setValue( tcifn )
        #self.textOut.setValue( self.textIn.getValue() )
        #os.remove(vcifn)
        logging.info(os.getcwd())
        return


