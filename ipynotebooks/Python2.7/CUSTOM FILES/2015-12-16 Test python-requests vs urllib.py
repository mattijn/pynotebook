
# coding: utf-8

# In[8]:

import numpy as np
import sys
import requests
import urllib
from datetime import datetime, timedelta
import numpy
from lxml import etree
from osgeo import gdal


# In[9]:

def datelist_regular_coverage(root, start_date, start, cur_date):
    """
    retrieve regular datelist and requested current position in regards to total no. of observations
    """

    #print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,1,1)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,1,1)).days
    datelist=range(start+1,end_date-1,365)
    print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,1,1)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    print ('Current position:',cur_pos)    
    
    return datelist, cur_pos


# In[10]:

def datelist_irregular_coverage(root, start_date, start, cur_date):
    """
    retrieve irregular datelist and requested current position in regards to total no. of observations
    """
    
    #root[0]                - wcs:CoverageDescription
    #root[0][0]             - boundedBy 
    #root[0][0][0]          - Envelope
    #root[0][0][0][0]       - lowerCorner
    # --- 
    #root[0]                - wcs:CoverageDescription
    #root[0][3]             - domainSet
    #root[0][3][0]          - gmlrgrid:ReferenceableGridByVectors
    #root[0][3][0][5]       - gmlrgrid:generalGridAxis
    #root[0][3][0][5][0]    - gmlrgrid:GeneralGridAxis
    #root[0][3][0][5][0][1] - gmlrgrid:coefficients

    # get sample size coefficients from XML root
    sample_size = root[0][3][0][5][0][1].text #sample size
    #print root[0][3][0][5][0][1].text #sample size
    
    # use coverage start_date and sample_size array to create all dates in ANSI
    array_stepsize = np.fromstring(sample_size, dtype=int, sep=' ')
    #print np.fromstring(sample_size, dtype=int, sep=' ')
    array_all_ansi = array_stepsize + start_date   
    
    # create array of all dates in ISO
    list_all_dates = []
    for stepsize in array_stepsize:
        date_and_stepsize = start + timedelta(stepsize - 1)
        list_all_dates.append(date_and_stepsize)
        #print date_and_stepsize
    array_all_dates = np.array(list_all_dates)  
    
    # create array of all dates as DOY
    list_all_yday = []
    for j in array_all_dates:
        yday = j.timetuple().tm_yday
        list_all_yday.append(yday)
        #print yday
    array_all_yday = np.array(list_all_yday)    
    
    # subtract user date of all dates in ISO 
    # to find the nearest available coverage date
    array_diff_dates = array_all_dates - cur_date
    idx_nearest_date = find_nearest(array_diff_dates, timedelta(0))
    nearest_date = array_all_dates[idx_nearest_date]    
    
    # select all coresponding DOY of all years for ANSI and ISO dates
    array_selected_ansi = array_all_ansi[array_all_yday == nearest_date.timetuple().tm_yday]
    array_selected_dates = array_all_dates[array_all_yday == nearest_date.timetuple().tm_yday]
    print array_selected_ansi
    
    # get index of nearest date in selection array
    idx_nearest_date_selected = numpy.where(array_selected_dates==nearest_date)[0][0]  
    print idx_nearest_date_selected
    
    # return datelist in ANSI and the index of the nearest date
    return array_selected_ansi, idx_nearest_date_selected


# In[11]:

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()


# In[12]:

date = '2014-06-25'
##request image cube for the specified date and area by WCS.
#firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
endpoint='http://192.168.1.104:8080/rasdaman/ows'
field={}
field['SERVICE']='WCS'
field['VERSION']='2.0.1'
field['REQUEST']='DescribeCoverage'
field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
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
#startt=145775
start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)


# In[13]:

try:    
    datelist, cur_pos = datelist_irregular_coverage(root, start_date, start, cur_date)
    print 'irregular'
except IndexError:
    datelist, cur_pos = datelist_regular_coverage(root, start_date, start, cur_date)
    print 'regular'


# In[14]:

spl_arr = [-179,-60,179,90]


# In[ ]:

def urlretrieve(datelist,spl_arr):    
    #retrieve the data cube
    cube_arr=[]
    for d in datelist[0:4]:
        print 'NDVI: ', d        
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
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
    return 


# In[ ]:

def requests_session(datelist,spl_arr):    
    #retrieve the data cube
    s = requests.Session()
    cube_arr=[]
    for d in datelist[0:4]:
        print 'NDVI: ', d        
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                        'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        #f,h = urllib.urlretrieve(full_url,tmpfilename)
        
        r = s.get(full_url, stream=True)
        chunk_size = 16 * 1024
        with open(tmpfilename, 'wb') as fd:
            for chunk in r.iter_content(chunk_size):
                fd.write(chunk)        
        
        #print h
        ds=gdal.Open(tmpfilename)
        cube_arr.append(ds.ReadAsArray())
        #print d  
    return cube_arr


# In[ ]:

def urllib2_chunk(datelist,spl_arr):    
    #retrieve the data cube
    #r = requests.Session()
    cube_arr = []
    for d in datelist[0:4]:
        print 'NDVI: ', d        
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
        field['SUBSET']=['ansi('+str(d)+')',
                         'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                        'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
        field['FORMAT']='image/tiff'
        url_values = urllib.urlencode(field,doseq=True)
        full_url = endpoint + '?' + url_values
        urls.append(full_url)
        #print full_url
        tmpfilename='test'+str(d)+'.tif'
        #f,h = urllib.urlretrieve(full_url,tmpfilename)
        
        response = urllib2.urlopen(full_url)
        CHUNK = 256 * 1024
        with open(tmpfilename, 'wb') as f:
            while True:
                chunk = response.read(CHUNK)
                if not chunk: break
                f.write(chunk)        
        
        #print h
        ds=gdal.Open(tmpfilename)
        cube_arr.append(ds.ReadAsArray())
        #print d  
    return cube_arr


# In[ ]:

get_ipython().magic(u'timeit cube_arr =  urlretrieve(datelist,spl_arr)')


# In[ ]:

get_ipython().magic(u'timeit cube_arr = requests_session(datelist,spl_arr)')


# In[ ]:

get_ipython().magic(u'timeit cube_arr = urllib2_chunk(datelist,spl_arr)')


# In[15]:

cube_arr=[]
urls = []
names = []
for d in datelist[0:4]:
    print 'NDVI: ', d        
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='GetCoverage'
    field['COVERAGEID']='NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
    field['SUBSET']=['ansi('+str(d)+')',
                     'Lat('+str(spl_arr[1])+','+str(spl_arr[3])+')',
                    'Long('+str(spl_arr[0])+','+str(spl_arr[2])+')']
    field['FORMAT']='image/tiff'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    urls.append(full_url)
    #print full_url
    tmpfilename='test'+str(d)+'.tif'
    names.append(tmpfilename)
    #f,h = urllib.urlretrieve(full_url,tmpfilename)
    #print h
    #ds=gdal.Open(tmpfilename)
    #cube_arr.append(ds.ReadAsArray())
    #print d  


# In[ ]:

def get(ix,u):
    r = grequests.get(u, stream=True)
    chunk_size = 16 * 1024
    tmpfilename = names[ix]
    with open(tmpfilename, 'wb') as fd:
        for chunk in r.iter_content(chunk_size):
            fd.write(chunk)


# In[ ]:

rs = (get(ix,u) for ix, u in enumerate(urls))


# In[ ]:

grequests.map(rs)


# In[ ]:

grequests.


# In[ ]:

from urlparse import urlparse
from threading import Thread
import httplib, sys
from Queue import Queue

concurrent = 200
cube_arr=[]
def doWork():
    
    s = requests.Session()    
    while True:
        url = q.get()
        status, url = getStatus(url)
        doSomethingWithResult(status, url,s, cube_arr)        
        q.task_done()
    return cube_arr

def getStatus(ourl):
    try:
        url = urlparse(ourl)
        conn = httplib.HTTPConnection(url.netloc)   
        conn.request("HEAD", url.path)
        res = conn.getresponse()
        return res.status, ourl
    except:
        return "error", ourl

def doSomethingWithResult(status, url,s,cube_arr):
    urls = ['http://192.168.1.104:8080/rasdaman/ows?SUBSET=ansi%28145908%29&SUBSET=Lat%28-60%2C90%29&SUBSET=Long%28-179%2C179%29&SERVICE=WCS&FORMAT=image%2Ftiff&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID=NDVI_MOD13C1005_uptodate',
 'http://192.168.1.104:8080/rasdaman/ows?SUBSET=ansi%28146274%29&SUBSET=Lat%28-60%2C90%29&SUBSET=Long%28-179%2C179%29&SERVICE=WCS&FORMAT=image%2Ftiff&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID=NDVI_MOD13C1005_uptodate',
 'http://192.168.1.104:8080/rasdaman/ows?SUBSET=ansi%28146639%29&SUBSET=Lat%28-60%2C90%29&SUBSET=Long%28-179%2C179%29&SERVICE=WCS&FORMAT=image%2Ftiff&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID=NDVI_MOD13C1005_uptodate',
 'http://192.168.1.104:8080/rasdaman/ows?SUBSET=ansi%28147004%29&SUBSET=Lat%28-60%2C90%29&SUBSET=Long%28-179%2C179%29&SERVICE=WCS&FORMAT=image%2Ftiff&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID=NDVI_MOD13C1005_uptodate']
    tmpfilename = names[urls.index(url)]
    print status, url
    r = s.get(full_url, stream=True)
    chunk_size = 16 * 1024
    print tmpfilename, '\n'#tmpfilename = names[ix]
    with open(tmpfilename, 'wb') as fd:
        for chunk in r.iter_content(chunk_size):
            fd.write(chunk)
    cube_arr.append(gdal.Open(tmpfilename).ReadAsArray())
    

q = Queue(concurrent * 2)
for i in range(concurrent):
    t = Thread(target=doWork)
    t.daemon = True
    t.start()
try:
    for url in urls:
        q.put(url.strip())
    q.join()
except KeyboardInterrupt:
    sys.exit(1)


# In[ ]:

cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)


# In[ ]:

plt.imshow(cube_arr[1] - cube_arr[0])
plt.show()
# plt.imshow(cube_arr[1])
# plt.show()
# plt.imshow(cube_arr[2])
# plt.show()
# plt.imshow(cube_arr[3])
# plt.show()


# In[ ]:

np.count_nonzero(cube_arr[1] == cube_arr[0])


# In[ ]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[ ]:

import numpy.ma as ma


# In[ ]:

urls.index('http://192.168.1.104:8080/rasdaman/ows?SUBSET=ansi%28147004%29&SUBSET=Lat%28-60%2C90%29&SUBSET=Long%28-179%2C179%29&SERVICE=WCS&FORMAT=image%2Ftiff&REQUEST=GetCoverage&VERSION=2.0.1&COVERAGEID=NDVI_MOD13C1005_uptodate')


# In[35]:

import eventlet
from eventlet.green import urllib2


# In[33]:

ixs = range(len(urls))
cube_arr = []

def fetch(url, ix):
    print "opening url: ", ix
    
    response = urllib2.urlopen(url)
    # Create in-memory file and initialize it with the content
    gdal.FileFromMemBuffer('/vsimem/tiffinmem', response.read())
    # Open the in-memory file
    ds = gdal.Open('/vsimem/tiffinmem')
    assert ds    
    cube_arr.append(ds.ReadAsArray())
    out = url     
    print "done with url: ", ix
    return ix

# create pool of threads
pool = eventlet.GreenPool(200)

# start farming jobs
for ix in pool.imap(fetch, urls, ixs):
    print "finished url", ix


# In[27]:

import requests


# In[28]:

cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)


# In[32]:

cube_arr_ma[0]


# In[1]:

#import urllib
import urllib2
import gdal


# In[2]:

url = 'https://raw.githubusercontent.com/mattijn/pynotebook/master/idata/ano_DOY2002170.tif'
tmp_filename1 = 'tmp1.tif'
tmp_filename2 = 'tmp2.tif'
tmp_filename3 = 'tmp3.tif'


# In[3]:

content = urllib2.urlopen(url)
chunk_size = 16 * 1024
with open(tmp_filename1, 'wb') as f:
    while True:
        chunk = content.read(chunk_size)
        if not chunk: break
        f.write(chunk) 


# In[4]:

ds = gdal.Open(tmp_filename1)
assert ds


# In[ ]:

import urllib2
content = urllib2.urlopen(url)
output = open(tmp_filename,'wb')
output.write(content.read())
output.close()

#urllib.urlretrieve(url,filename_tmp)
#ds = gdal.Open(filename_tmp)

#assert ds


# In[5]:

content = open(tmp_filename1, mode='rb').read()
# Create in-memory file and initialize it with the content
gdal.FileFromMemBuffer('/vsimem/tiffinmem', content)
# Open the in-memory file
ds = gdal.Open('/vsimem/tiffinmem')
assert ds


# In[6]:

content = urllib2.urlopen(url)
gdal.FileFromMemBuffer('/vsimem/tiffinmem', content.read())
# Open the in-memory file
ds = gdal.Open('/vsimem/tiffinmem')
assert ds


# In[7]:

ds.ReadAsArray()[0][0]


# In[18]:

import urllib3
import requests


# In[16]:

content = urllib2.urlopen(urls[0])
gdal.FileFromMemBuffer('/vsimem/tiffinmem', content.read())
# Open the in-memory file
ds = gdal.Open('/vsimem/tiffinmem')
assert ds


# In[ ]:




# In[ ]:

gdal.FileFromMemBuffer('tif_in_memory', response.read())
ds = gdal.Open('tif_in_memory')


# In[ ]:

assert ds


# In[ ]:

test_file = r'D:\Data\NDAI\NDAI_2014//NDAI_2014_008.tif'
content = open(test_file, mode='rb').read()


# In[ ]:

assert content


# In[ ]:

conn = urllib3.connection_from_url('http://192.168.1.104:8080/rasdaman/ows?')


# In[ ]:

http = urllib3.PoolManager()
r = http.request("GET", 'https://raw.githubusercontent.com/mattijn/pynotebook/master/idata/ano_DOY2002170.tif')
r.getheader("transfer-encoding")


# In[ ]:

gdal.FileFromMemBuffer('tiffinmem2', r.data)
ds = gdal.Open('tiffinmem2')


# In[ ]:




# In[ ]:

for chunk in r.stream():
    print chunk


# In[ ]:

with open('arh.tif', 'wb') as fp:


# In[ ]:

gdal.FileFromMemBuffer('tiffinmem2', r1.data)
ds = gdal.Open('tiffinmem2')


# In[ ]:

ds


# In[ ]:

new_arr = gdal.Open(names[1]).ReadAsArray() - gdal.Open(names[0]).ReadAsArray()
plt.imshow(new_arr)


# In[ ]:

import os
import urllib2
import math

def downloadChunks(url):
    """Helper to download large files
        the only arg is a url
       this file will go to a temp directory
       the file will also be downloaded
       in chunks and print out how much remains
    """

    baseFile = os.path.basename(url)

    #move the file to a more uniq path
    os.umask(0002)
    temp_path = "/tmp/"
    try:
        file = os.path.join(temp_path,baseFile)

        req = urllib2.urlopen(url)
        total_size = int(req.info().getheader('Content-Length').strip())
        downloaded = 0
        CHUNK = 256 * 10240
        with open(file, 'wb') as fp:
            while True:
                chunk = req.read(CHUNK)
                downloaded += len(chunk)
                print math.floor( (downloaded / total_size) * 100 )
                if not chunk: break
                fp.write(chunk)
    except urllib2.HTTPError, e:
        print "HTTP Error:",e.code , url
        return False
    except urllib2.URLError, e:
        print "URL Error:",e.reason , url
        return False

    return file
    
#use it like this
#downloadChunks("http://localhost/a.zip")


# In[ ]:

fileout = downloadChunks(full_url)


# In[ ]:

import urllib2
to_mem = urllib2.urlopen(full_url)
to_mem.read()[3::]


# In[ ]:

tifinxml


# In[ ]:

r = requests.get(full_url)
r.content


# In[ ]:

gdal.FileFromMemBuffer('tiffinmem2', r.content)
#ds = gdal.Open('tiffinmem2')


# In[ ]:

req = urllib2.urlopen(full_url)


# In[ ]:

file_in = tmpfilename
content = open(file_in, mode='rb').read()
gdal.FileFromMemBuffer('/vsimem/tiffinmem', content)
ds = gdal.Open('/vsimem/tiffinmem')


# In[ ]:

content = None
ds = None


# In[ ]:

content == r.content


# In[ ]:

open('tiffinmem2', mode='rb').read()


# In[ ]:

for i in req.info():
    print i


# In[ ]:

gdal.FileFromMemBuffer('tifinmem', output.getvalue())


# In[ ]:

ds = gdal.Open('tiffinmem')
ds


# In[ ]:

import logging
import requests
from cStringIO import StringIO
#logging.basicConfig(level=logging.DEBUG)
#s = requests.Session()
#s.get(full_url)
r = requests.get(full_url)


# In[ ]:

ds = gdal.Open(r.content)
ds


# In[ ]:

r = requests.get(full_url, stream=True)
chunk_size = 16 * 1024
with open(tmpfilename, 'wb') as fd:
    for chunk in r.iter_content(chunk_size):
        fd.write(chunk)


# In[ ]:

ds = gdal.Open(tmpfilename)
ds.ReadAsArray()


# In[ ]:

import grequests


# In[ ]:



