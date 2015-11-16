# coding: utf-8
from __future__ import division
#from osgeo import gdal
import numpy as np
#import scipy.signal
import os
#import matplotlib.pyplot as plt
#import nitime.algorithms as tsa
import random
import os
#from osgeo import gdal
import scipy.signal
#from mpl_toolkits.axes_grid1 import host_subplot
#import mpl_toolkits.axisartist as AA
from pywps.Process import WPSProcess
from types import FloatType
import xml.etree.cElementTree as etree
import jdcal
import cStringIO
import pandas as pd
import json
import datetime
import logging
#from PIL import Image

# --------*-*-*-*-*-*--------  WPSProcess  --------*-*-*-*-*-*--------  WPSProcess  --------*-*-*-*-*-*-----
# Computing diagonal for each row of a 2d array. See: http://stackoverflow.com/q/27214027/2459096
def makediag3d(M):
    b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
    b[:, ::M.shape[1] + 1] = M
    
    logging.info('function `makediag3d` complete')    
    return b.reshape(M.shape[0], M.shape[1], M.shape[1]) 

def get_starter_matrix(base_period_len, sample_count, frequencies_considered_count):
    nr = min(2 * frequencies_considered_count + 1,
                  sample_count)  # number of 2*+1 frequencies, or number of input images
    mat = np.zeros(shape=(nr, sample_count))
    mat[0, :] = 1
    ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
    cs = np.cos(ang)
    sn = np.sin(ang)
    # create some standard sinus and cosinus functions and put in matrix
    i = np.arange(1, frequencies_considered_count + 1)
    ts = np.arange(sample_count)
    for column in xrange(sample_count):
        index = np.mod(i * ts[column], base_period_len)
        # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
        mat[2 * i - 1, column] = cs.take(index)
        mat[2 * i, column] = sn.take(index)

    logging.info('function `get_starter_matrix` complete')
    return mat

def HANTS(sample_count, inputs,
          frequencies_considered_count=3,
          outliers_to_reject='Lo',
          low=0., high=255,
          fit_error_tolerance=5,
          delta=0.1):
    """
    Function to apply the Harmonic analysis of time series applied to arrays

    sample_count    = nr. of images (total number of actual samples of the time series)
    base_period_len    = length of the base period, measured in virtual samples
            (days, dekads, months, etc.)
    frequencies_considered_count    = number of frequencies to be considered above the zero frequency
    inputs     = array of input sample values (e.g. NDVI values)
    ts    = array of size sample_count of time sample indicators
            (indicates virtual sample number relative to the base period);
            numbers in array ts maybe greater than base_period_len
            If no aux file is used (no time samples), we assume ts(i)= i,
            where i=1, ..., sample_count
    outliers_to_reject  = 2-character string indicating rejection of high or low outliers
            select from 'Hi', 'Lo' or 'None'
    low   = valid range minimum
    high  = valid range maximum (values outside the valid range are rejeced
            right away)
    fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
            fit are rejected)
    dod   = degree of overdeterminedness (iteration stops if number of
            points reaches the minimum required for curve fitting, plus
            dod). This is a safety measure
    delta = small positive number (e.g. 0.1) to suppress high amplitudes
    """

    # define some parameters
    base_period_len = sample_count  #

    # check which setting to set for outlier filtering
    if outliers_to_reject == 'Hi':
        sHiLo = -1
    elif outliers_to_reject == 'Lo':
        sHiLo = 1
    else:
        sHiLo = 0

    nr = min(2 * frequencies_considered_count + 1,
             sample_count)  # number of 2*+1 frequencies, or number of input images

    # create empty arrays to fill
    outputs = np.zeros(shape=(inputs.shape[0], sample_count))

    mat = get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)

    # repeat the mat array over the number of arrays in inputs
    # and create arrays with ones with shape inputs where high and low values are set to 0
    mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
    p = np.ones_like(inputs)
    p[(low >= inputs) | (inputs > high)] = 0
    nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

    # prepare for while loop
    ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false

    dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
    noutmax = sample_count - nr - dod
    # prepare to add delta to suppress high amplitudes but not for [0,0]
    Adelta = np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
    Adelta[:, 0, 0] -= delta
    
    for _ in xrange(sample_count):
        if ready.all():
            break        
        
        # multiply outliers with timeseries
        za = np.einsum('ijk,ik->ij', mat, p * inputs)
        #print za

        # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
        diag = makediag3d(p)
        #print diag
        
        A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
        # add delta to suppress high amplitudes but not for [0,0]
        A += Adelta
        #A[:, 0, 0] = A[:, 0, 0] - delta
        #print A

        # solve linear matrix equation and define reconstructed timeseries
        zr = np.linalg.solve(A, za)
        #print zr
        
        outputs = np.einsum('ijk,kj->ki', mat.T, zr)
        #print outputs

        # calculate error and sort err by index
        err = p * (sHiLo * (outputs - inputs))
        rankVec = np.argsort(err, axis=1, )

        # select maximum error and compute new ready status
        maxerr = np.max(err, axis=-1)
        #maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
        ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)        

        # if ready is still false
        if not ready.all():
            j = rankVec.take(sample_count - 1, axis=-1)

            p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                int)  #*check
            nout += 1

    logging.info('function `HANTS` complete')
    return outputs


# In[2]:

def convert_ansi_date(date, offset=0.5):
    logging.info('function `convert_ansi_date` complete')
    return jdcal.jd2gcal(2305812.5, date + offset) # 0.5 offset is to adjust from night to noon

def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    #logging.info('function `unix_time` complete')
    return delta.total_seconds()    

def unix_time_millis(dt):
    #logging.info('function `unix_time_millis` complete')
    return int(unix_time(dt) * 1000)

def anomaly_computation(ts, norm=1):
    if norm == 2:
        anom_ts = ts.groupby([ts.index.month, ts.index.day]).apply(lambda g: (g - g.mean())/(g.max()-g.min()))
    if norm == 1:
        anom_ts = ts.groupby([ts.index.month, ts.index.day]).apply(lambda g: g - g.mean())
    if norm == 0:
        anom_ts = ts.groupby([ts.index.month, ts.index.day]).apply(lambda g: (g - g.mean())/g.std())
    return anom_ts



# In[ ]:

class Process(WPSProcess):
     def __init__(self):
         
         WPSProcess.__init__(self,
              identifier       = "ewma", 
              title            = "***---*** Anomaly Computation ***---***",
              version          = "1",
              storeSupported   = "true",
              statusSupported  = "false",
              abstract         = "This service provide access to compute the absolute or standardized anomaly including exponentially-weighted moving average")
              
         #self.input  = self.addLiteralInput(identifier  = "input_ts",
         #                                   title       = "Input time series to reconstruct",
         #                                   type        = FloatType)
         self.sign1  = self.addComplexInput(identifier  = "EWMAInputNDVI",
                                            title       = "Input time series NDVI anomaly",
                                            formats     = [{'mimeType':'text/xml'}])
                
         self.wshor  = self.addLiteralInput(identifier  = "EWMAWindowSizeShort",
                                            title       = "Window size for short-term EWMA")

         self.wlong  = self.addLiteralInput(identifier  = "EWMAWindowSizeLong",
                                            title       = "Window size for long-term EWMA")

         self.stand  = self.addLiteralInput(identifier  = "EWMAStandardization",
                                            title       = "Compute absolute or standardized anomaly")

         self.nfreq  = self.addLiteralInput(identifier  = "no_freq",
                                            title       = "Number of frequencies to consider")

         self.hants  = self.addLiteralInput(identifier  = "based_on_hants",
                                            title       = "Based on HANTS yes or no")

         self.outli  = self.addLiteralInput(identifier  = "outlier_type",
                                            title       = "Which type of values are considered as outliers")

                                            
#         self.resu1 = self.addComplexOutput(identifier  = "AnomalyNDVI", 
#                                            title       = "Output NDVI anomaly data",
#                                            formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json

         self.resu2 = self.addComplexOutput(identifier  = "EWMANDVI_ewma", 
                                            title       = "Output NDVI anomaly expx. weighted moving average",
                                            formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json                                            

         self.short = self.addComplexOutput(identifier  = "EWMANDVI_short", 
                                            title       = "Output NDVI anomaly expx. weighted moving average",
                                            formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json                                            

         self.long_ = self.addComplexOutput(identifier  = "EWMANDVI_long", 
                                            title       = "Output NDVI anomaly expx. weighted moving average",
                                            formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json                                            
     def execute(self):  


         # In[3]:

         # parse XML into Python
         parser= etree.XMLParser(encoding="utf-8")

         link1 = str(self.sign1.getValue())#r'C:\Users\lenovo\Desktop//ows.xml'

         window_size_short = int(self.wshor.getValue())
         window_size_long = int(self.wlong.getValue())
         standard = int(self.stand.getValue())

         based_on_hants = int(self.hants.getValue())

         frequencies = int(self.nfreq.getValue())
         variable_out = int(self.outli.getValue())
         if variable_out == 1:
             outliers = "Hi"    
         elif variable_out == 2:
             outliers = "Lo"
         else:
             outliers = "None" 

         #logging.info('window size', window_size)
         logging.info('type anomaly', standard)

         tree = etree.parse(link1, parser = parser)
         root = tree.getroot()

         # read grid envelope of domain set
         xml_low_env = cStringIO.StringIO(root[1][0][0][0][0].text)
         xml_high_env = cStringIO.StringIO(root[1][0][0][0][1].text)
         
         # load grid envelope as numpy array
         low_env = np.loadtxt(xml_low_env, dtype='int', delimiter=' ')
         high_env = np.loadtxt(xml_high_env, dtype='int', delimiter=' ')
         ts_shape = high_env - low_env + 1
         
         easting = ts_shape[0]
         northing = ts_shape[1]
         time = ts_shape[2]+1
         
         ## extract the dates
         #sd = ansi_date_to_greg_date(low_env[2]+140734)
         #ed = ansi_date_to_greg_date(high_env[2]+140734)
         
         # extract the values we need from the parsed XML
         sta_date_ansi = np.loadtxt(cStringIO.StringIO(root[0][0][0].text))[2] # 150116
         end_date_ansi = np.loadtxt(cStringIO.StringIO(root[0][0][1].text))[2] # 150852
         sta_date_rasd = np.loadtxt(cStringIO.StringIO(root[1][0][0][0][0].text))[2] # 9382
         end_date_rasd = np.loadtxt(cStringIO.StringIO(root[1][0][0][0][1].text))[2] # 9427
         timestep_date = np.loadtxt(cStringIO.StringIO(root[1][0][5].text))[2] # 16
         
         # compute the start and end-date
         dif_date_anra = sta_date_ansi - sta_date_rasd 
         dif_date_rasd = end_date_rasd - sta_date_rasd + 1
         end_date_anra = dif_date_rasd * timestep_date + sta_date_rasd + dif_date_anra
         
         sd = convert_ansi_date(sta_date_ansi) # (2012, 1, 2, 0.5)
         ed = convert_ansi_date(end_date_anra) # (2014, 1, 7, 0.5)
         
         # convert dates to pandas date_range
         str_date = str(sd[1])+'.'+str(sd[2])+'.'+str(sd[0])+'.'+str(int(np.round(sd[3]*24)))+':00'
         end_date = str(ed[1])+'.'+str(ed[2])+'.'+str(ed[0])+'.'+str(int(np.round(ed[3]*24)))+':00'
         freq_date = str(int(timestep_date))+'D'
         dates = pd.date_range(str_date,end_date, freq=freq_date)
         dates = dates[:-1]
         
         logging.info('start date in ANSI', sta_date_ansi, 'is equal to ISO 8601', dates[0])
         logging.info('end date in ANSI', end_date_anra, 'is equal to ISO 8601', dates[-1])
         logging.info('number of timesteps', time)
         
         # read data block of range set
         xml_ts = cStringIO.StringIO(root[2][0][1].text)
         
         # load data block as numpy array
         ts = np.loadtxt(xml_ts, dtype='float', delimiter=',')
         #ts = ts[0:-1]
         #ts_reshape = ts.reshape((easting*northing,time)) #Easting = ts_shape[0], Northing = ts_shape[1], time = ts_shape[2]

         # In[8]:
         if based_on_hants == 1:
             ts_reshape_trmm=ts[None] 
             ts_reshape = HANTS(sample_count=ts_reshape_trmm.shape[1], inputs=ts_reshape_trmm/100, frequencies_considered_count=frequencies,  outliers_to_reject=outliers)
             ts_reshape = ts_reshape[0]*100
         if based_on_hants == 0:
             ts_reshape = ts
         
         
         # In[8]:
         
         df = pd.DataFrame(ts_reshape/1000., dates)#ts_reshape.flatten(), dates)
         df_daly = df.asfreq('D', method='pad')
         df_anom = anomaly_computation(df_daly, norm=standard)#standard
         df_ewma_short = pd.ewma(df_anom,span=window_size_short)
         df_ewma_long = pd.ewma(df_anom,span=window_size_long)
	 df_ewma = df_ewma_short - df_ewma_long
         
         # In[9]:
         
         #plt.Figure(figsize(16,4))
         #ax = df_anom.plot(style='b', lw=1, alpha=0.5)
         #df_ewma.plot(style='m', lw=2, ax=ax)
         
         
         # In[ ]:
         
         # data preparation for HighCharts: Output need to be in JSON format with time 
         # in Unix milliseconds
         dthandler = lambda obj: (
          unix_time_millis(obj)
          if isinstance(obj, datetime.datetime)
          or isinstance(obj, datetime.date)
          else None)

         output1 = cStringIO.StringIO()
         output2_short = cStringIO.StringIO()      
         output3_long = cStringIO.StringIO()      
         
         logging.info('ready to dump files to JSON')
         # np.savetxt(output, pyhants, delimiter=',')
         out1 = json.dump(df_ewma.reset_index().as_matrix().tolist(), output1, default=dthandler)
         out2_short = json.dump(df_ewma_short.reset_index().as_matrix().tolist(), output2_short, default=dthandler)
         out3_long = json.dump(df_ewma_long.reset_index().as_matrix().tolist(), output3_long, default=dthandler)
         logging.info('dates converted from ISO 8601 to UNIX in ms')         
         
         # ouput
         self.resu2.setValue(output1)
         self.short.setValue(output2_short)
         self.long_.setValue(output3_long)
         return
