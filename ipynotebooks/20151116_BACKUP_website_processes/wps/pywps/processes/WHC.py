"""
DummyProcess to check the WPS structure

Author: Jorge de Jesus (jorge.jesus@gmail.com) as suggested by Kor de Jong
"""
from pywps.Process import WPSProcess
import types
import os
import geojson
import subprocess as sp
import json
import logging
import sys
import urllib
from osgeo import gdal
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from cStringIO import StringIO
from datetime import datetime, timedelta

def getHistogramsFromWCS(NAME_1,NAME_2,NAME_3,extent,date,no_observations):
    # convert all required dates in ISO date format
    date_start = datetime(int(date[0:4]),int(date[5:7]),int(date[8:10]))
    date_list = []
    date_list.append(date_start)
    for i in range(1,no_observations+1):
        #print i
        date_list.append(date_start + (i *timedelta(days=8)))

    # request data use WCS service baed on extend and clip based on sql query
    array_NDAI = []
    endpoint='http://192.168.1.104:8080/rasdaman/ows'
    for j in date_list:
        #d = 150842
        date_in_string = '"'+str(j.year)+'-'+str(j.month).zfill(2)+'-'+str(j.day).zfill(2)+'"'
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='NDAI_1km'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+date_in_string+')',#['ansi('+str(d)+')',
                         'Lat('+str(extent[1])+','+str(extent[3])+')',
                        'Long('+str(extent[0])+','+str(extent[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        logging.info(full_url)
        #print full_url
        tmpfilename='test'+str(j.toordinal())+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        logging.info(h)
        #print h

        #ds=gdal.Open(tmpfilename)
        clippedfilename='test'+str(j.toordinal())+'clip.tif' 

        path_base = "/var/www/html/wps/CHN_adm"
        CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")

        command = ["/usr/bin/gdalwarp", "-cutline", CHN_adm_gpkg, "-csql", "SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = '"+NAME_1+"' and NAME_2 = '"+NAME_2+"' and NAME_3 = '"+NAME_3+"'",
                   "-crop_to_cutline", "-of", "GTiff", "-dstnodata","-9999",tmpfilename, clippedfilename, "-overwrite"] # 

        logging.info(sp.list2cmdline(command))
        #print (sp.list2cmdline(command))

        norm = sp.Popen(sp.list2cmdline(command), shell=True)  
        norm.communicate()   

        ds=gdal.Open(clippedfilename)
        ds_clip = ds.ReadAsArray() 

        array_NDAI.append(ds_clip)


    array_NDAI = np.asarray(array_NDAI)
    #array_NDAI_ma = np.ma.masked_equal(array_NDAI, -9999)  
    y_list=[]
    bincenters_list=[]
    for k in range(0,len(date_list)):
        ds_hist_data = array_NDAI[k][array_NDAI[k] != -9999]
        y,binEdges=np.histogram(ds_hist_data,bins=100, range=(-1,1), normed=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        y_list.append(y)
        bincenters_list.append(bincenters)

    date_list_string = []
    for m in date_list:
        logging.info(m)
        #print m
        date_list_string.append(str(m.year)+'-'+str(m.month).zfill(2)+'-'+str(m.day).zfill(2))

    return date_list_string, bincenters_list, y_list

class Process(WPSProcess):
     def __init__(self):

        ##
        # Process initialization
         WPSProcess.__init__( self,
              identifier = "WHC", # must be same, as filename
              title            = "Histogram computation based on County ",
              version = "1.0",
              abstract         = "Module to compute Histograms of numerous NDAI observations",
	      statusSupported=True)

         self.Input1 = self.addLiteralInput(identifier = "input1",
                                            title = "Input1 number",
                                            type=types.IntType,
                                            default="100")
         self.Input2= self.addLiteralInput(identifier="input2",
                                           title="Input2 number",
                                           type=types.IntType,
                                          default="200")
         self.Output1=self.addLiteralOutput(identifier="output1",
                                            title="Output1 add 1 result")
         self.Output2=self.addLiteralOutput(identifier="output2",title="Output2 subtract 1 result" )
	 self.NAME_1          = self.addLiteralInput(  identifier    = "Province",
                                                      title         = "Chinese Province",
                                                      type          = types.StringType)

         self.NAME_2          = self.addLiteralInput(  identifier    = "Prefecture",
                                                      title         = "Chinese Prefecture",
                                                      type          = types.StringType)

         self.NAME_3          = self.addLiteralInput(  identifier    = "County",
                                                      title         = "Chinese County",
                                                      type          = types.StringType)

         self.bboxCounty      = self.addLiteralInput(  identifier    = "ExtentCounty",
                                                      title         = "The Extent of the web-based selected County",
                                                      type          = types.StringType)   
        
         self.date            = self.addLiteralInput(  identifier    = "date",
                                                      title         = "The selected date of interest",
                                                      type          = types.StringType)

         self.no_observations = self.addLiteralInput(  identifier    = "num_observations",
                                                      title         = "The number of succeeding observations",
                                                      type          = types.StringType)  
	 self.label_ts1       = self.addLiteralOutput( identifier    = "label_ts1", 
                                                      title         = "Label of the first observations") 
     def execute(self):

	NAME_1 = str(self.NAME_1.getValue())
        NAME_2 = str(self.NAME_2.getValue())
        NAME_3 = str(self.NAME_3.getValue())                
        

        extent = np.asarray(self.bboxCounty.getValue().strip().split(','))
        logging.info(extent)
        extent = np.fromstring(self.bboxCounty.getValue(), dtype=float, sep=',')
	logging.info(extent)
        date = str(self.date.getValue())        
        no_observations = int(self.no_observations.getValue())
	# Do the Work
        #date_list_string, bincenters_list, y_list = getHistogramsFromWCS(NAME_1,NAME_2,NAME_3,extent,date,no_observations)
	logging.info(date_list_string[0])
        # Save to out 
        self.Output1.setValue(int(self.Input1.getValue())+1)
        self.Output2.setValue(int(self.Input1.getValue())-1)
	self.label_ts1.setValue( "hello")
        return
