
# coding: utf-8

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

# In[ ]:

from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib2
import urllib
import gdal
import numpy as np
from lxml import etree
from datetime import datetime, timedelta
import multiprocessing as mp
import threading
import shutil


# The kernal of HANTS algorithm. Implemented by Marttijn.

# In[6]:

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

    logging.info('function `HANTS` complete')
    return outputs


# Paralel schema for region processing using HANTS

# In[ ]:

def __ReadLineData(inq,paramsDict,latlist,status,tFlag):
    try:
        logger=mp.get_logger()
        logger.info("%s start.",'Reading')  
        iline=0
  
        while True:
            with status.get_lock():
                if status.value==10 : break
            if inq.qsize() > 100 :
                continue


            #retrieve the data
            inStartDate=(datetime.strptime(paramsDict['inStartDate'],"%Y-%m-%d")-datetime(1601,01,01)).days
            inEndDate=(datetime.strptime(paramsDict['inEndDate'],"%Y-%m-%d")-datetime(1601,01,01)).days
            field={}
            field['SERVICE']='WCS'
            field['VERSION']='2.0.1'
            field['REQUEST']='ProcessCoverages'
            field['query']='for c in ('+paramsDict['coverageID'] +             ') return encode( scale( c[ansi('+str(inStartDate)+':'+str(inEndDate)+'),'+             'Lat('+str(latlist[iline])+'),' +             'Long('+str(paramsDict['lLon'])+':'+str(paramsDict['rLon'])+')],{ansi('+            str(inStartDate)+':'+str(inEndDate)+'),Long(0:'+str(paramsDict['xScale'])+')}),"csv")'

            url_values = urllib.unquote_plus(urllib.urlencode(field,doseq=True))
            full_url = paramsDict['wcsEndpoint'] + '?' + url_values
            fcsv= urllib.urlopen(full_url)
            str1=fcsv.read()
            str1=(str1[38:len(str1)-19].split('},{'))
            nsample=len(str1)
            data = np.array(map(int,','.join(str1).split(','))).reshape(nsample,-1)
            
            with tFlag.get_lock():
                inq.put((iline,data))
                tFlag[iline]=1
                iline = iline +1
                if iline ==len(tFlag): break

    except:
        logger.warning("Reading Failed: %s",sys.exc_info())
        logging.error("reading Failed: %s",mp.current_process().name,sys.exc_info())
        with status.get_lock():
                status.value=10

     
    
def __WriteLineData(outq,paramsDict,status,tFlag):
    
    try:
        logger=mp.get_logger()
        logger.info("%s start.",'Writing')  
        #generate filenames
        startDate=datetime.strptime(paramsDict['outStartDate'],"%Y-%m-%d")
        endDate=datetime.strptime(paramsDict['outEndDate'],"%Y-%m-%d")
        numdays=(endDate-startDate).days
        date_list = [(startDate + timedelta(days=x)).strftime('%Y-%m-%d') for x in np.arange(0, numdays,paramsDict['outInterval'])]
        odir=paramsDict['outDir']+'/'+'recon'
        if os.path.isdir(odir) :
            shutil.rmtree(odir)
        os.mkdir(odir)
        filenames=[os.path.join(odir,'.'.join([paramsDict['outPrefix'],dt,paramsDict['outSuffix'],"tiff"])) for                    dt in date_list]
        #create tiff files
        for ifile in filenames:
            driver = gdal.GetDriverByName( "GTiff" );
            ds = driver.Create( ifile, paramsDict['xOSize'], paramsDict['yOSize'], 1,gdal.GDT_Int16)
            #set projection information
 
            ds.SetProjection(paramsDict['ProjectionWKT'])  
            geotransform = (float(paramsDict['lLon']),paramsDict['outSpatialResolution'],
                            0,float(paramsDict['uLat']),0,float(paramsDict['outSpatialResolution']))  
            ds.SetGeoTransform(geotransform) 
            ds=None
 
        while True:
            #print "outq size:", outq.qsize()
            with status.get_lock():
                if status.value==10 : break
            #sys.stdout.flush()
            with tFlag.get_lock():
                felist=np.where(np.asarray(tFlag)!=4)[0]
                if len(felist) == 0:break
                #now begin to write 
                if outq.qsize()==0 :continue
                outTruple=outq.get()
                
                data=outTruple[1]
                data=data.transpose()
                iline=outTruple[0]
                tFlag[iline]=4
            for i in range(len(filenames)):
                ds= gdal.Open(filenames[i],gdal.GA_Update)
                ds.GetRasterBand(1).WriteArray(data[i:i+1,:],0,paramsDict['yOSize']-iline-1)
                ds=None
        logger.info("%s end.",'Writing')  

    except:
        
        logger.warning("Writing Failed: %s",sys.exc_info())
        logging.error("writing Failed: %s",sys.exc_info())
        with status.get_lock():
                status.value=10

   
    
def __RegionHants_Subprocess(inq,outq,paramsDict,status,tFlag):
    
    try:

        logger=mp.get_logger()
        logger.info("%s start.",mp.current_process().name)

        while True:
            with status.get_lock():
                if status.value==10 : break
            with tFlag.get_lock():
                if  len(np.where(np.asarray(tFlag)<2 )[0]) ==0:
                    break
                if inq.qsize()==0 :continue
       
                inTruple = inq.get()
                tFlag[inTruple[0]]=2
             
            data=inTruple[1]
            outData=np.zeros_like(data)

            for i in range(data.shape[0]-1):
                   outData[i:i+1,:]=HANTS(data.shape[1], data[i:i+1,:],
                                   frequencies_considered_count=paramsDict['nf'],
                                   outliers_to_reject='Lo',
                                   low=paramsDict['low'], 
                                   high=paramsDict['high'],
                                   fit_error_tolerance=paramsDict['toe'],
                                   dod=paramsDict['dod'],
                                   delta=0.1)

            with tFlag.get_lock():
                tFlag[inTruple[0]]=3
                outq.put((inTruple[0],outData))
        logger.info("%s end successfully with status %s",mp.current_process().name,status.value) 
    except:
        logger.warning("%s Failed: %s",mp.current_process().name,sys.exc_info())
        logging.error("%s Failed: %s",mp.current_process().name,sys.exc_info())
        with status.get_lock():
                status.value=10
   
        
        
            
    
def RegionHants(ps_xml):
    try:
        xml=urllib2.urlopen(ps_xml)
        tree = etree.parse(xml)
        paramsDict={}
        #Data source params. Currently only support retrieving data from rasdaman
        paramsDict['wcsEndpoint']=tree.find('.//WCSEndpoint').text
        paramsDict['coverageID'] = tree.find('.//CoverageID').text

        paramsDict['inStartDate'] =tree.find('.//DateRange/StartDate').text
        paramsDict['inEndDate'] =tree.find('.//DateRange/EndDate').text
        paramsDict['inInterval'] =int(tree.find('.//DateRange/Interval').text)
        
        paramsDict['SpatialResolution'] =float(tree.find('.//SpatialResolution').text)
        paramsDict['ProjectionWKT'] =tree.find('.//ProjectionWKT').text

        paramsDict['lLon'] =tree.find('.//LeftLon').text
        paramsDict['rLon'] =tree.find('.//RightLon').text
        paramsDict['uLat'] =tree.find('.//UpperLat').text
        paramsDict['bLat'] =tree.find('.//BottomLat').text

        #Algorithm parameters
        paramsDict['bPer'] =int(tree.find('.//BasePeriod').text)
        paramsDict['nf'] =int(tree.find('.//NumberOfFrequencies').text)
        paramsDict['per'] =map(int,((tree.find('.//Periods').text).split(',')))
        paramsDict['toe'] = float(tree.find('.//ToleranceOfError').text)
        paramsDict['low'] =float(tree.find('.//LowValue').text)
        paramsDict['high'] =float(tree.find('.//HighValue').text)
        paramsDict['dod'] = int(tree.find('.//DegreeOfOverdetermination').text)

        #output settings
        paramsDict['outDir'] =tree.find('.//OutDataDir').text
        paramsDict['outPrefix'] =tree.find('.//OutPrefix').text
        paramsDict['outSuffix'] =tree.find('.//OutSuffix').text
        paramsDict['outSpatialResolution'] =float(tree.find('.//OutSpatialResolution').text)

        paramsDict['outStartDate'] =tree.find('.//RegularOut/StartDate').text
        paramsDict['outEndDate'] =tree.find('.//RegularOut/EndDate').text
        paramsDict['outInterval'] =int(tree.find('.//RegularOut/Interval').text)


        baseRes=paramsDict['SpatialResolution']
        outRes=paramsDict['outSpatialResolution'] #basic resolution of output data
        paramsDict['xBSize']=int((int(paramsDict['rLon'])-int(paramsDict['lLon']))/baseRes)
        paramsDict['yBSize']=int((int(paramsDict['uLat'])-int(paramsDict['bLat']))/baseRes)
        paramsDict['xOSize']=int(paramsDict['xBSize']*baseRes/outRes)+1
        paramsDict['yOSize']=int(paramsDict['yBSize']*baseRes/outRes)+1
        paramsDict['xScale']=(paramsDict['xOSize']-1)*baseRes
        paramsDict['yScale']=(paramsDict['yOSize']-1)*baseRes
        
        
        ##Now begin to processing the regional data. 
        latlist=np.arange(float(paramsDict['bLat']),float(paramsDict['uLat'])+0.1*outRes,outRes)

        # Define an input data queue
        inq = mp.Queue()
        # Define an output data queue
        outq = mp.Queue()
        #Define a status indicator array
        status = mp.Value('b', 0)

        latlist=latlist[:]
        tFlag=mp.Array('i',np.zeros(len(latlist),np.int))

        #config a filehandles for mp logger
        logger=mp.get_logger()
        fhandler = logging.FileHandler(filename=paramsDict['outDir']+'/RegionalHants.log', mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.INFO)

        logger.info("%s start.",'Main process') 

        rt=threading.Thread(target=__ReadLineData,name = 'reading data',
                            args=(inq,paramsDict,latlist,status,tFlag,))
        rt.start()


        procs=[]
        for i in range(mp.cpu_count()+2):
            p=mp.Process(target=__RegionHants_Subprocess, 
                            args=(inq,outq,paramsDict,status,tFlag,),
                            name="reconstruction process"+str(i))
            p.start()
            logger.info('%s is alive: %s.',p.name,p.is_alive())
            procs.append(p)

        wt=threading.Thread(target=__WriteLineData,name = 'Writing data',args=(outq,paramsDict,status,tFlag))
        wt.start()
        
        rt.join()

        wt.join()

        for p in procs :p.terminate()
 
        inq=None
        outq=None
        status=None
        logger.info("%s end.",'Main process') 

        return paramsDict['outDir']
    except:
        logger.warning("%s Failed: %s",mp.current_process().name,sys.exc_info())
        logging.error("%s Failed: %s",mp.current_process().name,sys.exc_info())
        return 'Processing Failed with following error information:'+sys.exc_info()


# Enclosing the regional processing as a WPS service.

# In[13]:

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
        
        self.settingIn = self.addLiteralInput(identifier="ps_xml",
                title="Parameter settings xml file",
                type=type(''))

        ##
        # Adding process outputs

        self.dataLinkOut = self.addLiteralOutput(identifier="result_dirlink",
                title="Result files location",
                type=type(''))

        #self.textOut = self.addLiteralOutput(identifier = "text",
         #       title="Output literal data")
    ##
    # Execution part of the process
    def execute(self):
        
        #Get the xml setting file string
        psfile = self.settingIn.getValue()
        logging.info(psfile)
        
        outDir=RegionHants(psfile)
        self.dataLinkOut.setValue( outDir )
        return

if __name__== '__main__':
    starttime=datetime.now()
    print RegionHants('http://localhost/html/HANTS_PS.xml')
    timedelta=(datetime.now()-starttime)
    print 'Total running time: %s' % timedelta.seconds

