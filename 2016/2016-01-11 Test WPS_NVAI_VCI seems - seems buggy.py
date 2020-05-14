
# coding: utf-8

# In[2]:

#from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib2
import numpy as np
from lxml import etree
import datetime
import pandas as pd
from cStringIO import StringIO
import jdcal
import json


# In[ ]:




# In[3]:




# In[62]:


# In[49]:
def GetCoverageNames():
    file_json = r'/var/www/html/wps/pywps/processes/coverages_names.json'
    with open(file_json) as json_data:
        d = json.load(json_data)
    _CoverageID_NDVI = d['COVG_NAME_NDVI_MOD13C1005']
    _CoverageID_LST  = d['COVG_NAME_LST_MOD11C2005']
    return _CoverageID_NDVI, _CoverageID_LST

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

def convert_ansi_date(date, offset=0.5):
    logging.info('function `convert_ansi_date` complete')
    return jdcal.jd2gcal(2305812.5, date + offset) # 0.5 offset is to adjust from night to noon

def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    logging.info('function `unix_time` complete')
    return delta.total_seconds()    

def unix_time_millis(dt):
    logging.info('function `unix_time_millis` complete')
    return int(unix_time(dt) * 1000)

def region(pixel):
    """
    Extract pixel or regio:
    region1pix = single lat/lon
    region4pix = block of 4 pixels [2x2]
    region9pix = block of 9 pixels [3x3]
    """
    if pixel == 0:
        regionpix = [0,0]
    elif pixel == 4:
        regionpix = [-0.025,0.025]
    elif pixel == 9:
        regionpix = [-0.075,0.075]
    else:
        return
    return regionpix


# In[50]:

def REQUEST_DATA_XML(lon_center, lat_center, pix_offset, coverageID):
    
    regionpix = region(pix_offset)
    
    Long1 = str(lon_center + regionpix[0])
    Long2 = str(lon_center + regionpix[1])
    Lat1 = str(lat_center + regionpix[0])
    Lat2 = str(lat_center + regionpix[1])
    
    full_url = "http://localhost:8080/rasdaman/ows/wcs2?service=WCS&version=2.0.1&request=GetCoverage&coverageId="+coverageID+"&subset=Long("+Long1+","+Long2+")&subset=Lat("+Lat1+","+Lat2+")"
    f = urllib2.urlopen(full_url)     
    root = etree.fromstring(f.read())    
    
    # read grid envelope of domain set
    xml_low_env = StringIO(root[1][0][0][0][0].text)
    xml_high_env = StringIO(root[1][0][0][0][1].text)

    # load grid envelope as numpy array
    low_env = np.loadtxt(xml_low_env, dtype='int', delimiter=' ')
    high_env = np.loadtxt(xml_high_env, dtype='int', delimiter=' ')
    ts_shape = high_env - low_env + 1

    easting = ts_shape[0]
    northing = ts_shape[1]
    time = ts_shape[2]    

    ## extract the dates
    #sd = ansi_date_to_greg_date(low_env[2]+140734)
    #ed = ansi_date_to_greg_date(high_env[2]+140734)

    # extract the values we need from the parsed XML
    sta_date_ansi = np.loadtxt(StringIO(root[0][0][0].text))[2] # 150116
    end_date_ansi = np.loadtxt(StringIO(root[0][0][1].text))[2] # 150852
    sta_date_rasd = np.loadtxt(StringIO(root[1][0][0][0][0].text))[2] # 9382
    end_date_rasd = np.loadtxt(StringIO(root[1][0][0][0][1].text))[2] # 9427
    #timestep_date = np.loadtxt(StringIO(root[1][0][5].text))[2] # 16

    try:
        # check if regular coverage and ignore empty warnings
        with warnings.catch_warnings():        
            warnings.simplefilter("ignore")
            timestep_date = np.loadtxt(StringIO(root[1][0][5].text))[2] # 16    
        cov_reg = 1
        print 'regular coverages'
    except:        
        # check if irregular coverage
        array_stepsize = np.loadtxt(StringIO(root[1][0][5][0][1].text)) #array sample interval 
        cov_reg = 0    
        print 'irregular coverages'    

    # compute the start and end-date
    dif_date_anra = sta_date_ansi - sta_date_rasd
    dif_date_rasd = end_date_rasd - sta_date_rasd + 1


    # convert dates to pandas date_range
    str_date = pd.Timestamp.fromtimestamp((sta_date_ansi.astype('<m8[D]') - 
                                           (np.datetime64('1970-01-01') - np.datetime64('1601-01-01'))
                                           ).astype('<m8[s]').astype(int))
    end_date = pd.Timestamp.fromtimestamp((end_date_ansi.astype('<m8[D]') - 
                                           (np.datetime64('1970-01-01') - np.datetime64('1601-01-01'))
                                           ).astype('<m8[s]').astype(int))
    print cov_reg
    if cov_reg == 1:
        # regular coverage    
        freq_date = str(int(timestep_date))+'D'
        dates = pd.date_range(str_date,end_date, freq=freq_date)
        dates = dates[:-1]
        print 'dates regular'
    elif cov_reg == 0:
        # irregular coverage
        time_delta = pd.TimedeltaIndex(array_stepsize, unit = 'D')
        dates = pd.Series(np.array(str_date).repeat(len(array_stepsize)))
        dates += time_delta
        print 'dates irregular'    

    logging.info('dates converted from ANSI to ISO 8601')    

    # read data block of range set
    xml_ts = StringIO(root[2][0][1].text)

    # load data block as numpy array
    ts = np.loadtxt(xml_ts, dtype='float', delimiter=',')

    try:
        ts_reshape = ts.reshape((easting*northing,time)) #Easting = ts_shape[0], Northing = ts_shape[1], time = ts_shape[2]
    except:
        # sometimes length regular coverages is incorrect
        ts = ts[:-1]
        ts_reshape = ts.reshape((easting*northing,time)) 

    return ts_reshape, dates,str_date,end_date, time


# In[51]:

def COMPUTE_NVAI(ts_reshape, dates, str_date, end_date, time, pyhants=0, ann_freq=6, outliers_rj='None', 
                 from_date='2010-01-01', to_date='2012-01-01'):
    """
    
    pyhants: 0 means will NOT use, 1 will use
    ann_freq: number of frequencies (PER YEAR!)
    outliers_rj = 'None', 'Hi' or 'Lo'
    
    """
    
    # compute mean so regional statistics can be computed as well
    ts_mean = ts_reshape.mean(axis=0)
    ndvi = pd.Series(ts_mean.flatten()/10000.,dates, name='nvdi')

    # Interpolate NDVI 16 day interval to 1 day interval using linear interpolation
    x = pd.date_range(str_date,end_date,freq='D')
    ndvi_int = ndvi.reindex(x)
    #ndvi_int = ndvi_int.fillna(method='pad')
    ndvi_int = ndvi_int.interpolate(method='linear')
    ndvi_int.name = 'ndvi_int'
    
    
    if pyhants == 0:

        #Compute NVAI using 16 day interval NDVI data        
        #nvai = ndvi.groupby([ndvi.index.month]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        #nvai.name = 'nvai'
        
        # Compute NVAI using daily interpolated NDVI data
        nvai = ndvi_int.groupby([ndvi_int.index.month,ndvi_int.index.day]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        nvai.name = 'nvai'  
        nvai_sel = nvai.ix[from_date:to_date]        
    
    elif pyhants == 1:
        # Compute PyHANTS first and then calculate mean
        frequencies = len(ndvi_int) / 365 * ann_freq
        outliers = outliers_rj
        #ts_reshape = ts_reshape.mean(axis=0)
        pyhants = HANTS(sample_count=time, inputs=ts_reshape/100, frequencies_considered_count=frequencies,  outliers_to_reject=outliers)
        pyhants *= 100
        pyhants_mean = pyhants.mean(axis=0)
        ndvi_pyhants = pd.Series(pyhants_mean.flatten()/10000.,dates, name='ndvi_pyhants')        
        
        # interpolate reconstructed NDVI to daily values
        x = pd.date_range(str_date,end_date,freq='D')
        ndvi_pyhants_int = ndvi_pyhants.reindex(x)
        ndvi_pyhants_int = ndvi_pyhants_int.interpolate(method='linear')
        ndvi_pyhants_int.name = 'ndvi_pyhants_int'     
        
        # Compute NVAI using daily interpolated PyHANTS reconstructed NDVI data
        nvai = ndvi_pyhants_int.groupby([ndvi_pyhants_int.index.month,ndvi_pyhants_int.index.day]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        nvai.name = 'nvai'
        nvai_sel = nvai.ix[from_date:to_date]
    
    # data preparation for HighCharts: Output need to be in JSON format with time 
    # in Unix milliseconds
    dthandler = lambda obj: (
    unix_time_millis(obj)
    if isinstance(obj, datetime.datetime)
    or isinstance(obj, datetime.date)
    else None)

    nvai_json = StringIO()

    logging.info('ready to dump files to JSON')
    # np.savetxt(output, pyhants, delimiter=',')
    out1 = json.dump(nvai_sel.reset_index().as_matrix().tolist(), nvai_json, default=dthandler)

    logging.info('dates converted from ISO 8601 to UNIX in ms')             
    
    return nvai_json #nvai_sel


# In[52]:

def COMPUTE_NTAI(ts_reshape, dates, str_date, end_date, time, pyhants=0, ann_freq=6, outliers_rj='None', 
                 from_date='2010-01-01', to_date='2012-01-01'):
    """
    
    pyhants: 0 means will NOT use, 1 will use
    ann_freq: number of frequencies (PER YEAR!)
    outliers_rj = 'None', 'Hi' or 'Lo'
    
    """
    
    # compute mean so regional statistics can be computed as well
    ts_mean = ts_reshape.mean(axis=0)
    lst = pd.Series(ts_mean.flatten()/10000.,dates, name='lst')

    # Interpolate NDVI 16 day interval to 1 day interval using linear interpolation
    x = pd.date_range(str_date,end_date,freq='D')
    lst_int = lst.reindex(x)
    #ndvi_int = ndvi_int.fillna(method='pad')
    lst_int = lst_int.interpolate(method='linear')
    lst_int.name = 'lst_int'
    
    
    if pyhants == 0:

        #Compute NVAI using 16 day interval NDVI data        
        #nvai = ndvi.groupby([ndvi.index.month]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        #nvai.name = 'nvai'
        
        # Compute NVAI using daily interpolated NDVI data
        ntai = lst_int.groupby([lst_int.index.month,lst_int.index.day]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        ntai.name = 'ntai'  
        ntai_sel = ntai.ix[from_date:to_date]        
    
    elif pyhants == 1:
        # Compute PyHANTS first and then calculate mean
        frequencies = len(lst_int) / 365 * ann_freq
        outliers = outliers_rj
        #ts_reshape = ts_reshape.mean(axis=0)
        pyhants = HANTS(sample_count=time, inputs=ts_reshape/100, frequencies_considered_count=frequencies,  outliers_to_reject=outliers)
        pyhants *= 100
        pyhants_mean = pyhants.mean(axis=0)
        lst_pyhants = pd.Series(pyhants_mean.flatten()/10000.,dates, name='lst_pyhants')        
        
        # interpolate reconstructed LST to daily values
        x = pd.date_range(str_date,end_date,freq='D')
        lst_pyhants_int = lst_pyhants.reindex(x)
        lst_pyhants_int = lst_pyhants_int.interpolate(method='linear')
        lst_pyhants_int.name = 'lst_pyhants_int'     
        
        # Compute NTAI using daily interpolated PyHANTS reconstructed NTAI data
        ntai = lst_pyhants_int.groupby([lst_pyhants_int.index.month,lst_pyhants_int.index.day]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        ntai.name = 'ntai'
        ntai_sel = ntai.ix[from_date:to_date]
    
    # data preparation for HighCharts: Output need to be in JSON format with time 
    # in Unix milliseconds
    dthandler = lambda obj: (
    unix_time_millis(obj)
    if isinstance(obj, datetime.datetime)
    or isinstance(obj, datetime.date)
    else None)

    ntai_json = StringIO()

    logging.info('ready to dump files to JSON')
    # np.savetxt(output, pyhants, delimiter=',')
    out1 = json.dump(ntai_sel.reset_index().as_matrix().tolist(), ntai_json, default=dthandler)

    logging.info('dates converted from ISO 8601 to UNIX in ms')             
    
    return ntai_json #nvai_sel


# In[53]:

def COMPUTE_VCI(ts_reshape, dates, str_date, end_date, time, pyhants=0, ann_freq=6, outliers_rj='None', 
                 from_date='2010-01-01', to_date='2012-01-01'):
    """
    
    pyhants: 0 means will NOT use, 1 will use
    ann_freq: number of frequencies (PER YEAR!)
    outliers_rj = 'None', 'Hi' or 'Lo'
    
    """
    
    # compute mean so regional statistics can be computed as well
    ts_mean = ts_reshape.mean(axis=0)
    ndvi = pd.Series(ts_mean.flatten()/10000.,dates, name='nvdi')

    # Interpolate NDVI 16 day interval to 1 day interval using linear interpolation
    x = pd.date_range(str_date,end_date,freq='D')
    ndvi_int = ndvi.reindex(x)
    #ndvi_int = ndvi_int.fillna(method='pad')
    ndvi_int = ndvi_int.interpolate(method='linear')
    ndvi_int.name = 'ndvi_int'
    
    
    if pyhants == 0:

        #Compute NVAI using 16 day interval NDVI data        
        #nvai = ndvi.groupby([ndvi.index.month]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        #nvai.name = 'nvai'
        
        # Compute NVAI using daily interpolated NDVI data
        vci = ndvi_int.groupby([ndvi_int.index.month,ndvi_int.index.day]).apply(lambda g: (g - g.min())/(g.max() - g.min()))
        vci.name = 'vci'  
        vci_sel = vci.ix[from_date:to_date]        
    
    elif pyhants == 1:
        # Compute PyHANTS first and then calculate mean
        frequencies = len(ndvi_int) / 365 * ann_freq
        outliers = outliers_rj
        #ts_reshape = ts_reshape.mean(axis=0)
        pyhants = HANTS(sample_count=time, inputs=ts_reshape/100, frequencies_considered_count=frequencies,  outliers_to_reject=outliers)
        pyhants *= 100
        pyhants_mean = pyhants.mean(axis=0)
        ndvi_pyhants = pd.Series(pyhants_mean.flatten()/10000.,dates, name='ndvi_pyhants')        
        
        # interpolate reconstructed NDVI to daily values
        x = pd.date_range(str_date,end_date,freq='D')
        ndvi_pyhants_int = ndvi_pyhants.reindex(x)
        ndvi_pyhants_int = ndvi_pyhants_int.interpolate(method='linear')
        ndvi_pyhants_int.name = 'ndvi_pyhants_int'     
        
        # Compute NVAI using daily interpolated PyHANTS reconstructed NDVI data
        vci = ndvi_pyhants_int.groupby([ndvi_pyhants_int.index.month,ndvi_pyhants_int.index.day]).apply(lambda g: (g - g.min())/(g.max() - g.min()))
        vci.name = 'vci'
        vci_sel = vci.ix[from_date:to_date]
    
    # data preparation for HighCharts: Output need to be in JSON format with time 
    # in Unix milliseconds
    dthandler = lambda obj: (
    unix_time_millis(obj)
    if isinstance(obj, datetime.datetime)
    or isinstance(obj, datetime.date)
    else None)

    vci_json = StringIO()

    logging.info('ready to dump files to JSON')
    # np.savetxt(output, pyhants, delimiter=',')
    out1 = json.dump(vci_sel.reset_index().as_matrix().tolist(), vci_json, default=dthandler)

    logging.info('dates converted from ISO 8601 to UNIX in ms')             
    
    return vci_json #nvai_sel


# In[54]:

def COMPUTE_TCI(ts_reshape, dates, str_date, end_date, time, pyhants=0, ann_freq=6, outliers_rj='None', 
                 from_date='2010-01-01', to_date='2012-01-01'):
    """
    
    pyhants: 0 means will NOT use, 1 will use
    ann_freq: number of frequencies (PER YEAR!)
    outliers_rj = 'None', 'Hi' or 'Lo'
    
    """
    
    # compute mean so regional statistics can be computed as well
    ts_mean = ts_reshape.mean(axis=0)
    lst = pd.Series(ts_mean.flatten()/10000.,dates, name='lst')

    # Interpolate NDVI 16 day interval to 1 day interval using linear interpolation
    x = pd.date_range(str_date,end_date,freq='D')
    lst_int = lst.reindex(x)
    #ndvi_int = ndvi_int.fillna(method='pad')
    lst_int = lst_int.interpolate(method='linear')
    lst_int.name = 'lst_int'
    
    
    if pyhants == 0:

        #Compute NVAI using 16 day interval NDVI data        
        #nvai = ndvi.groupby([ndvi.index.month]).apply(lambda g: (g - g.mean())/(g.max() - g.min()))
        #nvai.name = 'nvai'
        
        # Compute NVAI using daily interpolated NDVI data
        tci = lst_int.groupby([lst_int.index.month,lst_int.index.day]).apply(lambda g: (g.max() - g)/(g.max() - g.min()))
        tci.name = 'tci'  
        tci_sel = tci.ix[from_date:to_date]        
    
    elif pyhants == 1:
        # Compute PyHANTS first and then calculate mean
        frequencies = len(lst_int) / 365 * ann_freq
        outliers = outliers_rj
        #ts_reshape = ts_reshape.mean(axis=0)
        pyhants = HANTS(sample_count=time, inputs=ts_reshape/100, frequencies_considered_count=frequencies,  outliers_to_reject=outliers)
        pyhants *= 100
        pyhants_mean = pyhants.mean(axis=0)
        lst_pyhants = pd.Series(pyhants_mean.flatten()/10000.,dates, name='lst_pyhants')        
        
        # interpolate reconstructed LST to daily values
        x = pd.date_range(str_date,end_date,freq='D')
        lst_pyhants_int = lst_pyhants.reindex(x)
        lst_pyhants_int = lst_pyhants_int.interpolate(method='linear')
        lst_pyhants_int.name = 'lst_pyhants_int'     
        
        # Compute NTAI using daily interpolated PyHANTS reconstructed NTAI data
        tci = lst_pyhants_int.groupby([lst_pyhants_int.index.month,lst_pyhants_int.index.day]).apply(lambda g: (g.max() - g)/(g.max() - g.min()))
        tci.name = 'tci'
        tci_sel = tci.ix[from_date:to_date]
    
    # data preparation for HighCharts: Output need to be in JSON format with time 
    # in Unix milliseconds
    dthandler = lambda obj: (
    unix_time_millis(obj)
    if isinstance(obj, datetime.datetime)
    or isinstance(obj, datetime.date)
    else None)

    tci_json = StringIO()

    logging.info('ready to dump files to JSON')
    # np.savetxt(output, pyhants, delimiter=',')
    out1 = json.dump(tci_sel.reset_index().as_matrix().tolist(), tci_json, default=dthandler)

    logging.info('dates converted from ISO 8601 to UNIX in ms')             
    
    return tci_json #nvai_sel


# In[80]:

def COMPUTE_VHI(vci_json, tci_json, alpha = 0.5):
    # load VCI data
    vci = pd.read_json(vci_json.getvalue())
    vci.columns = ["date","vci"]
    vci['date'] = pd.to_datetime(vci['date'],unit='ms')
    vci.set_index(['date'],inplace=True)

    # load TCI data
    tci = pd.read_json(tci_json.getvalue())
    tci.columns = ["date","tci"]
    tci['date'] = pd.to_datetime(tci['date'],unit='ms')
    tci.set_index(['date'],inplace=True)
    
    # compute VHI    
    vhi = (alpha * vci['vci']) + ((1-alpha) * tci['tci'])
    
    # data preparation for HighCharts: Output need to be in JSON format with time 
    # in Unix milliseconds
    dthandler = lambda obj: (
    unix_time_millis(obj)
    if isinstance(obj, datetime.datetime)
    or isinstance(obj, datetime.date)
    else None)

    vhi_json = StringIO()

    logging.info('ready to dump files to JSON')
    # np.savetxt(output, pyhants, delimiter=',')
    out1 = json.dump(vhi.reset_index().as_matrix().tolist(), vhi_json, default=dthandler)

    logging.info('dates converted from ISO 8601 to UNIX in ms')             
    
    return vhi_json #nvai_sel    

# In[ ]:



# In[ ]:

lon_center = float(self.lonIn.getValue())
lat_center = float(self.latIn.getValue())
pix_offset = int(self.pixOff.getValue())                

from_date = str(self.fromDateIn.getValue())
to_date = str(self.toDateIn.getValue())

pyhants = int(self.pyHants.getValue())
ann_freq = int(self.freqan.getValue())
outliers_rj = str(self.outlier.getValue())    

CoverageID_NDVI, CoverageID_LST = GetCoverageNames()    

# 2. Do the Work
# Request NDVI data and compute VCI and compute NVAI 
ndvi,dates,str_date,end_date,time=REQUEST_DATA_XML(lon_center,lat_center,pix_offset,coverageID=CoverageID_NDVI)
vci_json = COMPUTE_VCI(ndvi,dates,str_date,end_date,time,pyhants,ann_freq,outliers_rj,from_date,to_date)
nvai_json = COMPUTE_NVAI(ndvi,dates,str_date,end_date,time,pyhants,ann_freq,outliers_rj,from_date,to_date)


# In[ ]:

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_NVAI_VCI_TS",
            title="Compute NVAI_VCI TIMESERIES",
            abstract="""Module to compute NVAI_VCI TimeSeries based on NDVI data""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs
        self.lonIn = self.addLiteralInput(identifier="lon_center",
                    title="Longitude",
                    type=type(''))

        self.latIn = self.addLiteralInput(identifier="lat_center",
                    title="Latitude",
                    type=type(''))

        self.fromDateIn = self.addLiteralInput(identifier="from_date",
                    title = "The start date to be calcualted",
                                          type=type(''))

        self.toDateIn = self.addLiteralInput(identifier="to_date",
                    title = "The final date to be calcualted",
                                          type=type(''))   
        
        self.pixOff = self.addLiteralInput(identifier="pix_offset",
                    title="pixel offset, 0, 4 or 9",
                    type=type(''))

        self.pyHants = self.addLiteralInput(identifier="pyhants",
                    title="exclude pyhants (0) or include pyhants (1)",
                    type=type(''))

        self.freqan = self.addLiteralInput(identifier="ann_freq",
                    title = "number of annual (!) frequencies",
                                          type=type(''))

        self.outlier = self.addLiteralInput(identifier="outliers_rj",
                    title = "what type of outliers to reject, Hi, Lo, or None",
                                          type=type(''))        
        
        ##
        # Adding process outputs
        self.nvaiOut = self.addComplexOutput(identifier  = "nvai_ts", 
                                        title       = "NVAI Timeseries",
                                        formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json   

        self.vciOut = self.addComplexOutput(identifier  = "vci_ts", 
                                        title       = "VCI Timeseries",
                                        formats     = [{'mimeType':'text/xml'}]) #xml/application ||application/json   
        
    ##
    # Execution part of the process
    def execute(self):
        # 1. Load the data
        lon_center = float(self.lonIn.getValue())
        lat_center = float(self.latIn.getValue())
        pix_offset = int(self.pixOff.getValue())                
        
        from_date = str(self.fromDateIn.getValue())
        to_date = str(self.toDateIn.getValue())
        
        pyhants = int(self.pyHants.getValue())
        ann_freq = int(self.freqan.getValue())
        outliers_rj = str(self.outlier.getValue())    

        CoverageID_NDVI, CoverageID_LST = GetCoverageNames()    
        
        # 2. Do the Work
        # Request NDVI data and compute VCI and compute NVAI 
        ndvi,dates,str_date,end_date,time=REQUEST_DATA_XML(lon_center,lat_center,pix_offset,coverageID=CoverageID_NDVI)
        vci_json = COMPUTE_VCI(ndvi,dates,str_date,end_date,time,pyhants,ann_freq,outliers_rj,from_date,to_date)
        nvai_json = COMPUTE_NVAI(ndvi,dates,str_date,end_date,time,pyhants,ann_freq,outliers_rj,from_date,to_date)
        
        # 3. Save to out
        self.vciOut.setValue( vci_json )
        self.nvaiOut.setValue( nvai_json )
        return

