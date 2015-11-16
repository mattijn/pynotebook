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

def getFracAreaFromWCS(NAME_1,NAME_2,NAME_3,extent,date,no_observations):

    #logging.info(NAME_1)
    #logging.info(NAME_2) 
    #NAME_3, extent, date, no_observations)

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
        #logging.info(j)
        #d = 150842
        date_in_string = '"'+str(j.year)+'-'+str(j.month).zfill(2)+'-'+str(j.day).zfill(2)+'"'

        #logging.info(date_in_string)
        #logging.info(str(extent[1]))
        #logging.info(str(extent[3]))
        #logging.info(str(extent[0]))
        #logging.info(str(extent[2]))

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

        #logging.info(tmpfilename)
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        #logging.info(h)
        #print h
        #ds_test_tmpfilename = gdal.Open(tmpfilename).ReadAsArray()
        #logging.info(ds_test_tmpfilename[0])

        #ds=gdal.Open(tmpfilename)
        clippedfilename='test'+str(j.toordinal())+'clip.tif' 

        path_base = "/var/www/html/wps/CHN_adm"
        CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")

        command = ["/usr/bin/gdalwarp","-cutline",CHN_adm_gpkg,"-csql","SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = '"+NAME_1+"' and NAME_2 = '"+NAME_2+"' and NAME_3 = '"+NAME_3+"'","-crop_to_cutline","-of","GTiff","-dstnodata","-9999",tmpfilename,clippedfilename, "-overwrite"]

        logging.info(sp.list2cmdline(command))
        #print (sp.list2cmdline(command))

        norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)  
        norm.communicate()   

        ds=gdal.Open(clippedfilename)
        ds_clip = ds.ReadAsArray() 

        #logging.info(ds_clip[0])

        array_NDAI.append(ds_clip)


    array_NDAI = np.asarray(array_NDAI)
    #array_NDAI_ma = np.ma.masked_equal(array_NDAI, -9999)  
    y_list=[]
    bincenters_list=[]
    for k in range(0,len(date_list)):
        ds_hist_data = array_NDAI[k][array_NDAI[k] != -9999]
        sorted_data = np.sort(ds_hist_data)

        bincenters = np.concatenate([sorted_data, sorted_data[[-1]]])[~np.isnan(np.concatenate([sorted_data, sorted_data[[-1]]]))]
        y = np.arange(sorted_data.size+1)/ (np.arange(sorted_data.size+1).max().astype(float) / 100)
        y = y[~np.isnan(np.concatenate([sorted_data, sorted_data[[-1]]]))]        
        #y = np.arange(sorted_data.size+1)/ (np.arange(sorted_data.size+1).max().astype(float) / 100)
        #bincenters = np.concatenate([sorted_data, sorted_data[[-1]]])
        y_list.append(y)
        bincenters_list.append(bincenters)

    date_list_string = []
    for m in date_list:
        #logging.info(m)
        #print m
        date_list_string.append(str(m.year)+'-'+str(m.month).zfill(2)+'-'+str(m.day).zfill(2))

    #logging.info(date_list_string)
    #logging.info(bincenters_list)
    #logging.info(y_list)

    return date_list_string, bincenters_list, y_list


class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__( self,
            identifier       = "WPS_FRACAREA_COMPUTATION",
            title            = "Histogram computation based on County ",
            abstract         = "Module to compute Histograms of numerous NDAI observations",
            version          = "1.0",
            storeSupported   = True,
            statusSupported  = True)

        ##
        # Adding process inputs

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

        ##
        # Adding process outputs

        self.label_ts1       = self.addLiteralOutput( identifier    = "label_ts1", 
                                                      title         = "Label of the first observations")        

        self.frac_ts1        = self.addComplexOutput( identifier    = "frac_ts1", 
                                                      title         = "Histogram of the first observations",
                                                      formats       = [{'mimeType':'text/xml'}]) 
        
        self.label_ts2       = self.addLiteralOutput( identifier    = "label_ts2", 
                                                      title         = "Label of the first observations")

        self.frac_ts2        = self.addComplexOutput( identifier    = "frac_ts2", 
                                                      title         = "Histogram of the first observations",
                                                      formats       = [{'mimeType':'text/xml'}]) 
        
        self.label_ts3       = self.addLiteralOutput( identifier    = "label_ts3", 
                                                      title         = "Label of the first observations")

        self.frac_ts3        = self.addComplexOutput( identifier    = "frac_ts3", 
                                                      title         = "Histogram of the first observations",
                                                      formats       = [{'mimeType':'text/xml'}]) 
        
        self.label_ts4       = self.addLiteralOutput( identifier    = "label_ts4", 
                                                      title         = "Label of the first observations")

        self.frac_ts4        = self.addComplexOutput( identifier    = "frac_ts4", 
                                                      title         = "Histogram of the first observations",
                                                      formats       = [{'mimeType':'text/xml'}]) 
        
        self.label_ts5       = self.addLiteralOutput( identifier    = "label_ts5", 
                                                      title         = "Label of the first observations")

        self.frac_ts5        = self.addComplexOutput( identifier    = "frac_ts5", 
                                                      title         = "Histogram of the first observations",
                                                      formats       = [{'mimeType':'text/xml'}])         


    ##
    # Execution part of the process
    def execute(self):
        # Load the data
        NAME_1 = str(self.NAME_1.getValue())
        NAME_2 = str(self.NAME_2.getValue())
        NAME_3 = str(self.NAME_3.getValue())                
        

        #extent = np.asarray(self.bboxCounty.getValue().strip().split(','))
        #logging.info(extent)
        extent = np.fromstring(self.bboxCounty.getValue(), dtype=float, sep=',')
        logging.info(extent)
        date = str(self.date.getValue())        
        no_observations = int(self.no_observations.getValue())
        
        # Do the Work
        date_list_string, bincenters_list, y_list = getFracAreaFromWCS(NAME_1,NAME_2,NAME_3,extent,date,no_observations)

        #logging.info(date_list_string[0])
        # Save to out        
        try:
            label_ts1 = date_list_string[0]            
            frac_ts1 = StringIO()
            json.dump(pd.Series(y_list[0],bincenters_list[0]).reset_index().as_matrix().tolist(), frac_ts1)
            self.label_ts1.setValue( label_ts1 )
            self.frac_ts1.setValue( frac_ts1 )
            logging.info('ts1')
            logging.info(label_ts1)
            #logging.info(frac_ts1.getvalue())
        except :
            logging.info('ts1 didnt work')
            pass


        try:
            label_ts2 = date_list_string[1]
            frac_ts2 = StringIO()
            json.dump(pd.Series(y_list[1],bincenters_list[1]).reset_index().as_matrix().tolist(), frac_ts2)
            self.label_ts2.setValue( label_ts2 )        
            self.frac_ts2.setValue( frac_ts2 )
            logging.info('ts2')            
            logging.info(label_ts2)
            #logging.info(frac_ts2.getvalue())

        except:
            logging.info('ts2 didnt work')
            pass

        try:
            label_ts3 = date_list_string[2]
            frac_ts3 = StringIO()
            json.dump(pd.Series(y_list[2],bincenters_list[2]).reset_index().as_matrix().tolist(), frac_ts3)
            self.label_ts3.setValue( label_ts3 )        
            self.frac_ts3.setValue( frac_ts3 )
            logging.info('ts3')  
            logging.info(label_ts3)
            #logging.info(frac_ts3.getvalue())
          
        except:
            logging.info('ts3 didnt work')
            pass        

        try:
            label_ts4 = date_list_string[3]
            frac_ts4 = StringIO()
            json.dump(pd.Series(y_list[3],bincenters_list[3]).reset_index().as_matrix().tolist(), frac_ts4)
            self.label_ts4.setValue( label_ts4 )        
            self.frac_ts4.setValue( frac_ts4 )
            logging.info('ts4')  
            logging.info(label_ts4)
            #logging.info(frac_ts4.getvalue())
          
        except :
            logging.info('ts4 didnt work')
            pass     

        try:
            label_ts5 = date_list_string[4]
            frac_ts5 = StringIO()
            json.dump(pd.Series(y_list[4],bincenters_list[4]).reset_index().as_matrix().tolist(), frac_ts5)
            self.label_ts5.setValue( label_ts5 )        
            self.frac_ts5.setValue( frac_ts5 )
            logging.info('ts5')  
            logging.info(label_ts5)
            #logging.info(frac_ts5.getvalue())
          
        except:
            logging.info('ts5 didnt work')
            pass 

        return


# In[ ]:

#NAME_1 = "Anhui"
#NAME_2 = "Bengbu"
#NAME_3 = "Guzhen"
#extent = [117.04640962322863,33.00404358318741,117.59765626636589,33.50222015793983] # left, bottom, right, top
#date = "2014-01-01"
#no_observations = 4


# In[ ]:

#date_list_string, bincenters_list, y_list = getHistogramsFromWCS(NAME_1,NAME_2,NAME_3,extent,date,no_observations)


# In[ ]:

#for l in range(0,len(date_list_string)):
#    print l
#    plt.plot(bincenters_list[l],y_list[l])
#plt.show()


