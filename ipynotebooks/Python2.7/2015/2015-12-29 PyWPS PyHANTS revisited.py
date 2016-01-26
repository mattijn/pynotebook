
# coding: utf-8

# In[ ]:



# # WPS process for regional HANTS processing
# This process implements the HANTS reconstruction algorithm, which can be used to process reginal data. In addition, this algorithm has been published by WPS protocal. 
# The input for the wps is a valid link to a xml, where one can set all parameters involed in the processing.
# The wps procedure return a url link where users can donwload the reconstruction result.
# 
# The processing process is implemented under parallel schema, whcih can speed up the processing by adding more cpu resources to the computation server. Specifically, the basic process structure is composed of one reading (data retrieving) thread, one writing thread and several paralel CPU processes. The reason of using just one thead for reading and writing is that the speed of I/O on the server is limited. 
# Currently, on a VM ubuntu 14.4 system (3 cpu cores), to precess a NDVI time series data set (141(column)X100(row)X23(sences)) cost 90 second in total.
# 
# Limitations & further expected improvement
#     *Data source. Now the process can only retrieve data from rasdaman database. It should be improved to support file   system based data. In that way, normal users can upload original data to specific ftp server and then request the reconstruction service.
#     *Data type. Now the process only suport 16-bit integer data for I/O. More data type support need to be added and users can set the data type based on their real data.
#     *Output in random time interval. Now the output file hold the same time stamp as the input file. 
#     *Redesign the parameter setting file by xlsd.
#     *A web form which can be used by users to fill parameter setting file.
#     *Client side visualization of reconstruction result.
# Script to excute the wps:
#     http://localhost/cgi-bin/pywps.cgi?service=wps&version=1.0.0&request=execute&identifier=WPS_HANTS_RECON_BATCH&datainputs=[ps_xml=http://localhost/html/HANTS_PS.xml]
# 
# Author: Jie Zhou, Martijn
# Date: 14/04/2015
# 

#from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib2
import urllib
import gdal
import osr
import numpy as np
import json
from datetime import datetime, timedelta
import multiprocessing as mp
import threading
import shutil
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:

def find_values(id, json_repr):
    results = []
    def _decode_dict(a_dict):
        try: results.append(a_dict[id])
        except KeyError: pass
        return a_dict
    json.loads(json_repr, object_hook=_decode_dict)  # return value ignored
    return results


# In[ ]:

# The kernal of HANTS algorithm. Implemented by Mattijn.
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

    logging.info('HANTS: function `get_starter_matrix` complete')
    return mat

def HANTS(sample_count, inputs,
          frequencies_considered_count=3,
          outliers_to_reject='Lo',
          low=0., high=255,
          fit_error_tolerance=5,
          dod = 5,
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
    logger = mp.get_logger()
    logger.info('HANTS: HANTS is active %s', inputs.shape)

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

    #dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
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

    logging.info('HANTS: function `HANTS` complete')
    #print 'function HANTS complete'
    return outputs


# In[ ]:

# Paralel schema for region processing using HANTS
def __ReadLineData(inq, paramsDict, latlist, status, tFlag):
    try:
        logger = mp.get_logger()
        logger.info("__ReadLineData %s start.", 'Reading')  
        iline=0
        
        while True:
            with status.get_lock():
                if status.value == 10 : break
            if inq.qsize() > 100 :
                continue

            #retrieve the data
            inStartDate = (datetime.strptime(paramsDict['inStartDate'], "%Y-%m-%d")-datetime(1601,1,1)).days
            inEndDate = (datetime.strptime(paramsDict['inEndDate'], "%Y-%m-%d")-datetime(1601,1,1)).days
            field = {}
            #field['SERVICE'] = 'WCS'
            #field['VERSION'] = '2.0.1'
            #field['REQUEST'] = 'ProcessCoverages'
            field['query'] = 'for c in ('+paramsDict['coverageID']+') return encode( scale( c[ansi('+str(inStartDate)+':'+str(inEndDate)+'),'+'Lat('+str(latlist[iline])+'),'+'Long('+str(paramsDict['lLon'])+':'+str(paramsDict['rLon'])+')],{ansi('+str(inStartDate)+':'+str(inEndDate)+'),Long(0:'+str(paramsDict['xScale'])+')}),"netcdf")'            
            url_values = urllib.unquote_plus(urllib.urlencode(field, doseq=True))
            full_url = paramsDict['wcsEndpoint'] + '?' + url_values
            logger.info('__ReadLineData full_url to rasdaman is: %s', full_url)
            
            # retrieve the file
            tmpfilename = 'WCS'+str(mp.current_process().name)+'.nc'
            f,h = urllib.urlretrieve(full_url,tmpfilename)
            
            data = gdal.Open('NETCDF:'+tmpfilename+':Band1').ReadAsArray()
            logger.info('__ReadLineData data shape: %s', data.shape)
            
            #fcsv = urllib.urlopen(full_url)
            #str1 = fcsv.read()
            #str1 = (str1[38:len(str1)-19].split('},{'))
            #nsample = len(str1)
            #data = np.array(map(int, ','.join(str1).split(','))).reshape(nsample,-1)
            
            with tFlag.get_lock():
                logger.info('__ReadLineData just before putting')
                inq.put((iline, data.T))                
                logger.info('__ReadLineData just after putting')                
                tFlag[iline] = 1
                iline = iline + 1
                if iline == len(tFlag): break

    except:
        logger.warning("__ReadLineData: Reading Failed: %s", sys.exc_info())
        logging.error("__ReadLineData: Reading Failed: %s", mp.current_process().name, sys.exc_info())
        with status.get_lock():
                status.value = 10


# In[ ]:

def __WriteLineData(outq, paramsDict, status, tFlag):
    
    try:
        logger = mp.get_logger()
        logger.info("__WriteLineData %s start.", 'Writing')  
        print 'Writing...'
        #generate filenames
        startDate = datetime.strptime(paramsDict['outStartDate'], "%Y-%m-%d")
        endDate = datetime.strptime(paramsDict['outEndDate'], "%Y-%m-%d")
        numdays = (endDate - startDate).days
        print 'startDate, endDate, numdays:', startDate, endDate, numdays
        date_list = [(startDate + timedelta(days = x)).strftime('%Y-%m-%d') for x in np.arange(0, numdays, paramsDict['outInterval'])]
        print 'date_list:', date_list
        odir = paramsDict['outDir'] + '/' + 'recon'
        if os.path.isdir(odir):
            shutil.rmtree(odir)
        os.mkdir(odir)
        filenames = [os.path.join(odir, '.'.join([paramsDict['outPrefix'], 
                                                  str(dt), paramsDict['outSuffix'], "tiff"])) for dt in date_list]
        
        #create tiff files
        for ifile in filenames:
            #print 'ifile:',ifile
            driver = gdal.GetDriverByName( "GTiff" )
            ds = driver.Create(ifile, paramsDict['xOSize'], paramsDict['yOSize'], 1, gdal.GDT_Float32)
            
            #set projection information 
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(paramsDict['ProjectionEPSG'])
            ds.SetProjection(srs.ExportToWkt()) 
            geotransform = (float(paramsDict['lLon']), paramsDict['outSpatialResolution'], 0,
                            float(paramsDict['uLat']), 0, float(paramsDict['outSpatialResolution']))  
            ds.SetGeoTransform(geotransform) 
            ds = None
            
        while True:
            #print "outq size:", outq.qsize()
            with status.get_lock():
                if status.value == 10: break
            #sys.stdout.flush()
            with tFlag.get_lock():
                felist = np.where(np.asarray(tFlag)!= 4)[0]
                if len(felist) == 0: break
                #now begin to write 
                if outq.qsize() == 0: continue
                outTruple = outq.get()
                
                data = outTruple[1]
                data = data.T
                iline = outTruple[0]
                tFlag[iline] = 4
            for i in range(len(filenames)):
                ds = gdal.Open(filenames[i], gdal.GA_Update)
                ds.GetRasterBand(1).WriteArray(data[i:i+1,:], 0, paramsDict['yOSize'] - iline - 1)
                ds = None
        logger.info("__WriteLineData: %s end.", 'Writing')  

    except:
        
        logger.warning("__WriteLineData: Writing Failed: %s", sys.exc_info())
        logging.error("__WriteLineData: Writing Failed: %s", sys.exc_info())
        with status.get_lock():
                status.value = 10


# In[ ]:

def __RegionHants_Subprocess(inq, outq, paramsDict, status, tFlag):
    #logger.info("%s start.", "__RegionHants_Subprocess")
    try:
        logger = mp.get_logger()
        logger.info("__RegionHants_Subprocess: %s start.", mp.current_process().name)

        while True:
            with status.get_lock():
                if status.value == 10: break
            with tFlag.get_lock():
                if  len(np.where(np.asarray(tFlag) < 2)[0]) == 0:
                    break
                if inq.qsize() == 0: continue
                    
                inTruple = inq.get()
                logger.info('__RegionHants_Subprocess: inTruple: %s', inTruple[0])
                tFlag[inTruple[0]]=2

            data = inTruple[1]
            logger.info('__RegionHants_Subprocess: shape data: %s', data.shape)
            outData = np.zeros_like(data)
            logger.info('__RegionHants_Subprocess: shape outData: %s', outData.shape)
            logger.info('__RegionHants_Subprocess: data range: %s', data.shape[0]-1)
            
            # slice only rows contianing 90% or more valid values
            percent = np.sum(data == paramsDict['FillValue'], axis=-1) / float(data.shape[1])
            percent = percent >= 0.9
            slice = np.invert(percent).nonzero()[0]
            slice_nan = percent.nonzero()[0]               
            # prepare out data with only nan values
            outData = np.zeros_like(data).astype(float)
            
            # apply HANTS lazily with maximum 100 rows each time
            for k in range(0, (len(slice)), 50):
                if k + 50 < (len(slice)):
                    l = k + 50
                else:
                    l = k + (len(slice)) - k
                print ('start',k,'end',l,data[slice][k:l].shape)
                logger.info('__RegionHants_Subprocess: start %s, end %s, shape %s', k, l, data[slice][k:l].shape)
            
#                 # HANTS COMPUTATION
#                 outData[slice][k:l] = HANTS(data.shape[1], data[slice][k:l]/100,
#                                        frequencies_considered_count = paramsDict['nf'],
#                                        outliers_to_reject = 'Lo',
#                                        low = paramsDict['low'], 
#                                        high = paramsDict['high'],
#                                        fit_error_tolerance = paramsDict['toe'],
#                                        dod = paramsDict['dod'],
#                                        delta = 0.1)

            # HANTS COMPUTATION
            outData[slice] = HANTS(data.shape[1], data[slice]/100,
                                   frequencies_considered_count = paramsDict['nf'],
                                   outliers_to_reject = 'Lo',
                                   low = paramsDict['low'], 
                                   high = paramsDict['high'],
                                   fit_error_tolerance = paramsDict['toe'],
                                   dod = paramsDict['dod'],
                                   delta = 0.1)                
            
            # set nan values to invalid values and correct scaling
            outData[slice_nan] = np.nan
            outData *= 100

            with tFlag.get_lock():
                tFlag[inTruple[0]] = 3
                outq.put((inTruple[0], outData))
        logger.info("__RegionHants_Subprocess: %s end successfully with status %s", mp.current_process().name, status.value) 
    except:
        logger.warning("__RegionHants_Subprocess: %s Failed: %s", mp.current_process().name, sys.exc_info())
        logging.error("__RegionHants_Subprocess: %s Failed: %s", mp.current_process().name, sys.exc_info())
        with status.get_lock():
                status.value = 10


# In[ ]:

def RegionHants(ps_json): #ps_xml
    try:        
        with open(ps_json) as data_file:
            hants_ps = data_file.read()
        
        paramsDict = {}
        #Data source params. Currently only support retrieving data from rasdaman
        paramsDict['wcsEndpoint'] = find_values('WCSEndpoint', hants_ps)[0]
        paramsDict['coverageID'] = find_values('CoverageID', hants_ps)[0]

        paramsDict['inStartDate'] = find_values('StartDate', hants_ps)[0] # DateRange
        paramsDict['inEndDate'] = find_values('EndDate', hants_ps)[0] # DateRange
        paramsDict['inInterval'] = find_values('Interval', hants_ps)[0] # DateRange

        paramsDict['SpatialResolution'] = find_values('SpatialResolution', hants_ps)[0]
        paramsDict['ProjectionEPSG'] = find_values('ProjectionEPSG', hants_ps)[0]
        paramsDict['FillValue'] = find_values('FillValue', hants_ps)[0]

        paramsDict['lLon'] = find_values('LeftLon', hants_ps)[0]
        paramsDict['rLon'] = find_values('RightLon', hants_ps)[0]
        paramsDict['uLat'] = find_values('UpperLat', hants_ps)[0]
        paramsDict['bLat'] = find_values('BottomLat', hants_ps)[0]

        #Algorithm parameters
        paramsDict['bPer'] = find_values('BasePeriod', hants_ps)[0] 
        paramsDict['nf'] = find_values('NumberOfFrequencies', hants_ps)[0]
        paramsDict['per'] = map(int,find_values('Periods', hants_ps)[0].split(',')) 
        paramsDict['toe'] = find_values('ToleranceOfError', hants_ps)[0]
        paramsDict['low'] = find_values('LowValue', hants_ps)[0]
        paramsDict['high'] = find_values('HighValue', hants_ps)[0]
        paramsDict['dod'] = find_values('DegreeOfOverdetermination', hants_ps)[0]

        #output settings
        paramsDict['outDir'] = find_values('OutDataDir', hants_ps)[0]
        paramsDict['outPrefix'] = find_values('OutPrefix', hants_ps)[0]
        paramsDict['outSuffix'] = find_values('OutSuffix', hants_ps)[0]
        paramsDict['outSpatialResolution'] = find_values('OutSpatialResolution', hants_ps)[0]

        paramsDict['outStartDate'] = find_values('StartDate', hants_ps)[1] # RegularOut
        paramsDict['outEndDate'] = find_values('EndDate', hants_ps)[1] # RegularOut 
        paramsDict['outInterval'] = find_values('Interval', hants_ps)[1] # RegularOut

        baseRes = paramsDict['SpatialResolution']
        outRes = paramsDict['outSpatialResolution'] #basic resolution of output data
        paramsDict['xBSize'] = int((int(paramsDict['rLon']) - int(paramsDict['lLon'])) / baseRes)
        paramsDict['yBSize'] = int((int(paramsDict['uLat']) - int(paramsDict['bLat'])) / baseRes)
        paramsDict['xOSize'] = int(paramsDict['xBSize'] * baseRes / outRes) + 1
        paramsDict['yOSize'] = int(paramsDict['yBSize'] * baseRes / outRes) + 1
        paramsDict['xScale'] = 20#(paramsDict['xOSize']-1) * baseRes
        paramsDict['yScale'] = (paramsDict['yOSize']-1) * baseRes        
        
        ##Now begin to processing the regional data. 
        latlist = np.arange(float(paramsDict['bLat']),
                            float(paramsDict['uLat']) + 0.1 * outRes,
                            outRes)

        # Define an input data queue
        inq = mp.Queue()
        # Define an output data queue
        outq = mp.Queue()
        #Define a status indicator array
        status = mp.Value('b', 0)

        latlist = latlist[:]
        tFlag = mp.Array('i', np.zeros(len(latlist), np.int))

        #config a filehandles for mp logger
        logger = mp.get_logger()
        fhandler = logging.FileHandler(filename=paramsDict['outDir'] + '/RegionalHants.log', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.INFO)

        logger.info("RegionHants: %s start.",'Main process') 

        rt = threading.Thread(target = __ReadLineData, 
                              name = 'Reading data',
                              args = (inq, paramsDict, latlist, status, tFlag,))
        rt.start()


        procs=[]
        for i in range(mp.cpu_count() + 2):
            p = mp.Process(target = __RegionHants_Subprocess, 
                           args = (inq, outq, paramsDict, status, tFlag,),
                           name = "reconstruction process " + str(i))
            p.start()
            logger.info('RegionHants: %s is alive: %s.', p.name, p.is_alive())
            procs.append(p)

        wt = threading.Thread(target = __WriteLineData,
                              name = 'Writing data',
                              args = (outq,paramsDict,status,tFlag))
        wt.start()
        
        rt.join()

        wt.join()

        for p in procs: p.terminate()

        inq = None
        outq = None
        status = None
        logger.info("RegionHants: %s end.", 'Main process') 

        return paramsDict['outDir']
    except:
        logger = mp.get_logger()
        logger.warning("RegionHants: %s Failed: %s", mp.current_process().name, sys.exc_info())
        logging.error("RegionHants: %s Failed: %s", mp.current_process().name, sys.exc_info())
        return 'Processing Failed with following error information:' + str(sys.exc_info())


# In[ ]:

HANTS_PS_JSON = u'D:\\tmp\\HANTS_PS.json'
RegionHants(HANTS_PS_JSON)


# In[ ]:

with open(HANTS_PS_JSON) as data_file:
    hants_ps = data_file.read()

paramsDict = {}
#Data source params. Currently only support retrieving data from rasdaman
paramsDict['wcsEndpoint'] = find_values('WCSEndpoint', hants_ps)[0]
paramsDict['coverageID'] = find_values('CoverageID', hants_ps)[0]

paramsDict['inStartDate'] = find_values('StartDate', hants_ps)[0] # DateRange
paramsDict['inEndDate'] = find_values('EndDate', hants_ps)[0] # DateRange
paramsDict['inInterval'] = find_values('Interval', hants_ps)[0] # DateRange

paramsDict['SpatialResolution'] = find_values('SpatialResolution', hants_ps)[0]
paramsDict['ProjectionEPSG'] = find_values('ProjectionEPSG', hants_ps)[0]
paramsDict['FillValue'] = find_values('FillValue', hants_ps)[0]

paramsDict['lLon'] = find_values('LeftLon', hants_ps)[0]
paramsDict['rLon'] = find_values('RightLon', hants_ps)[0]
paramsDict['uLat'] = find_values('UpperLat', hants_ps)[0]
paramsDict['bLat'] = find_values('BottomLat', hants_ps)[0]

#Algorithm parameters
paramsDict['bPer'] = find_values('BasePeriod', hants_ps)[0] 
paramsDict['nf'] = find_values('NumberOfFrequencies', hants_ps)[0]
paramsDict['per'] = map(int,find_values('Periods', hants_ps)[0].split(',')) 
paramsDict['toe'] = find_values('ToleranceOfError', hants_ps)[0]
paramsDict['low'] = find_values('LowValue', hants_ps)[0]
paramsDict['high'] = find_values('HighValue', hants_ps)[0]
paramsDict['dod'] = find_values('DegreeOfOverdetermination', hants_ps)[0]

#output settings
paramsDict['outDir'] = find_values('OutDataDir', hants_ps)[0]
paramsDict['outPrefix'] = find_values('OutPrefix', hants_ps)[0]
paramsDict['outSuffix'] = find_values('OutSuffix', hants_ps)[0]
paramsDict['outSpatialResolution'] = find_values('OutSpatialResolution', hants_ps)[0]

paramsDict['outStartDate'] = find_values('StartDate', hants_ps)[1] # RegularOut
paramsDict['outEndDate'] = find_values('EndDate', hants_ps)[1] # RegularOut 
paramsDict['outInterval'] = find_values('Interval', hants_ps)[1] # RegularOut

baseRes = paramsDict['SpatialResolution']
outRes = paramsDict['outSpatialResolution'] #basic resolution of output data
paramsDict['xBSize'] = int((int(paramsDict['rLon']) - int(paramsDict['lLon'])) / baseRes)
paramsDict['yBSize'] = int((int(paramsDict['uLat']) - int(paramsDict['bLat'])) / baseRes)
paramsDict['xOSize'] = int(paramsDict['xBSize'] * baseRes / outRes) + 1
paramsDict['yOSize'] = int(paramsDict['yBSize'] * baseRes / outRes) + 1
paramsDict['xScale'] = 20#(paramsDict['xOSize']-1) * baseRes
paramsDict['yScale'] = (paramsDict['yOSize']-1) * baseRes        

##Now begin to processing the regional data. 
latlist = np.arange(float(paramsDict['bLat']),
                    float(paramsDict['uLat']) + 0.1 * outRes,
                    outRes)


# In[ ]:

test.shape


# In[ ]:

full_url = 'http://159.226.117.95:58080/rasdaman/ows/wcs?query=for c in (NDVI_MOD13C1005_uptodate) return encode( scale( c[ansi(147192:147558),Lat(15.0),Long(90:110)],{ansi(147192:147558),Long(0:20.0)}),"csv")'
fcsv = urllib.urlopen(full_url)


# In[ ]:

str1 = fcsv.read()


# In[ ]:

str1


# In[ ]:

str2 = str1[1:-1].split('},{')
nsample = len(str2)
str3 = map(int, ','.join(str2).split(','))
data = np.array(str3).reshape(nsample,-1)


# In[ ]:

print data.shape
plt.imshow(np.ma.masked_equal(data,-3000), interpolation='nearest')


# In[ ]:

full_url = 'http://159.226.117.95:58080/rasdaman/ows/wcs?query=for c in (NDVI_MOD13C1005_uptodate) return encode( scale( c[ansi(147192:147558),Lat(15.0),Long(90:110)],{ansi(147192:147558),Long(0:20.0)}),"netcdf")'
# retrieve the file
tmpfilename = 'test2.nc'
f,h = urllib.urlretrieve(full_url,tmpfilename)

data = gdal.Open('NETCDF:'+tmpfilename+':Band1').ReadAsArray()
print('data first line: %s', data.shape)


# In[ ]:

plt.imshow(np.ma.masked_equal(data.T,-3000), interpolation='nearest')


# In[ ]:

str1 = fcsv.read()
#str1 = (str1[38:len(str1)-19].split('},{'))
#nsample = len(str1)
#data = np.array(map(int, ','.join(str1).split(','))).reshape(nsample,-1)


# In[ ]:




# In[ ]:

driver = gdal.GetDriverByName( "GTiff" )
ds = driver.Create(r'D:\tmp\HANTS_OUT\recon\HANTS.2004-01-01.recon.tiff', 
                   paramsDict['xOSize'], paramsDict['yOSize'], 1, gdal.GDT_Int16)

#set projection information 

srs = osr.SpatialReference()
srs.ImportFromEPSG(paramsDict['ProjectionEPSG'])
ds.SetProjection(srs.ExportToWkt())  
geotransform = (float(paramsDict['lLon']), paramsDict['outSpatialResolution'], 0,
                float(paramsDict['uLat']), 0, float(paramsDict['outSpatialResolution']))  
ds.SetGeoTransform(geotransform) 
ds = None


# In[ ]:

with open(HANTS_PS_JSON) as data_file:
    hants_ps = data_file.read()


paramsDict = {}
#Data source params. Currently only support retrieving data from rasdaman
paramsDict['wcsEndpoint'] = find_values('WCSEndpoint', hants_ps)[0]
paramsDict['coverageID'] = find_values('CoverageID', hants_ps)[0]

paramsDict['inStartDate'] = find_values('StartDate', hants_ps)[0] # DateRange
paramsDict['inEndDate'] = find_values('EndDate', hants_ps)[0] # DateRange
paramsDict['inInterval'] = find_values('Interval', hants_ps)[0] # DateRange

paramsDict['SpatialResolution'] = find_values('SpatialResolution', hants_ps)[0]
paramsDict['ProjectionEPSG'] = find_values('ProjectionEPSG', hants_ps)[0]
paramsDict['FillValue'] = find_values('FillValue', hants_ps)[0]

paramsDict['lLon'] = find_values('LeftLon', hants_ps)[0]
paramsDict['rLon'] = find_values('RightLon', hants_ps)[0]
paramsDict['uLat'] = find_values('UpperLat', hants_ps)[0]
paramsDict['bLat'] = find_values('BottomLat', hants_ps)[0]

#Algorithm parameters
paramsDict['bPer'] = find_values('BasePeriod', hants_ps)[0] 
paramsDict['nf'] = find_values('NumberOfFrequencies', hants_ps)[0]
paramsDict['per'] = map(int,find_values('Periods', hants_ps)[0].split(',')) 
paramsDict['toe'] = find_values('ToleranceOfError', hants_ps)[0]
paramsDict['low'] = find_values('LowValue', hants_ps)[0]
paramsDict['high'] = find_values('HighValue', hants_ps)[0]
paramsDict['dod'] = find_values('DegreeOfOverdetermination', hants_ps)[0]

#output settings
paramsDict['outDir'] = find_values('OutDataDir', hants_ps)[0]
paramsDict['outPrefix'] = find_values('OutPrefix', hants_ps)[0]
paramsDict['outSuffix'] = find_values('OutSuffix', hants_ps)[0]
paramsDict['outSpatialResolution'] = find_values('OutSpatialResolution', hants_ps)[0]

paramsDict['outStartDate'] = find_values('StartDate', hants_ps)[1] # RegularOut
paramsDict['outEndDate'] = find_values('EndDate', hants_ps)[1] # RegularOut 
paramsDict['outInterval'] = find_values('Interval', hants_ps)[1] # RegularOut

baseRes = paramsDict['SpatialResolution']
outRes = paramsDict['outSpatialResolution'] #basic resolution of output data
paramsDict['xBSize'] = int((int(paramsDict['rLon']) - int(paramsDict['lLon'])) / baseRes)
paramsDict['yBSize'] = int((int(paramsDict['uLat']) - int(paramsDict['bLat'])) / baseRes)
paramsDict['xOSize'] = int(paramsDict['xBSize'] * baseRes / outRes) + 1
paramsDict['yOSize'] = int(paramsDict['yBSize'] * baseRes / outRes) + 1
paramsDict['xScale'] = (paramsDict['xOSize']-1) * baseRes
paramsDict['yScale'] = (paramsDict['yOSize']-1) * baseRes   


# In[ ]:

int((110 - 70) / 0.05)


# In[ ]:

(800 * 0.05 / 0.5) + 1


# In[ ]:

(81-1) * baseRes


# In[ ]:

paramsDict["xScale"]


# In[ ]:

# Enclosing the regional processing as a WPS service.
class Process(WPSProcess):

    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_HANTS_RECON_BATCH",
            title="HANTS regional processing",
            abstract="""This process intend to reconstruct regional EO data-set using HANTS.""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs
        
        self.settingIn = self.addLiteralInput(identifier = "ps_xml",
                title = "Parameter settings xml file",
                type = type(''))

        ##
        # Adding process outputs

        self.dataLinkOut = self.addLiteralOutput(identifier = "result_dirlink",
                title = "Result files location",
                type = type(''))

        #self.textOut = self.addLiteralOutput(identifier = "text",
         #       title="Output literal data")
    ##
    # Execution part of the process
    def execute(self):
        
        #Get the xml setting file string
        psfile = self.settingIn.getValue()
        logging.info(psfile)
        
        outDir = RegionHants(psfile)
        self.dataLinkOut.setValue( outDir )
        return

if __name__ == '__main__':
    starttime = datetime.now()
    HANTS_PS_JSON = r'D:\tmp\\HANTS_PS.json'
    RegionHants(HANTS_PS_JSON)
    #print RegionHants('http://localhost/html/HANTS_PS.xml')
    timedelta = (datetime.now()-starttime)
    print 'Total running time: %s' % timedelta.seconds

