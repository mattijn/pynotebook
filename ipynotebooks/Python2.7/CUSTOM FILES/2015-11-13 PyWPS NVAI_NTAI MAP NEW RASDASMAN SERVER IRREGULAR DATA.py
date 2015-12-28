# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib
from osgeo import gdal
import numpy
import numpy as np
import numpy.ma as ma
from lxml import etree
from datetime import datetime
from datetime import timedelta
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# <codecell>


# <codecell>

def _NTAI_CAL(date,spl_arr):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:58080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='LST_MOD11C2005'#'trmm_3b42_coverage_1'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    print full_url
    data = urllib.urlopen(full_url).read()
    root = etree.fromstring(data)
    lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
    uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
    start_date=int((lc.split(' '))[2])
    end_date=int((uc.split(' '))[2])
    print [start_date, end_date]

    #generate the dates list 
    cur_date=datetime.strptime(date,"%Y-%m-%d")
    startt=145775
    start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
    print start
    tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    if tmp_date > start :
        start=(tmp_date-datetime(1601,01,01)).days
    else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    datelist=range(start+1,end_date-1,365)
    print datelist

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,01,01)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    print ('Current position:',cur_pos)
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
        print full_url
        tmpfilename='test'+str(d)+'.tif'
        f,h = urllib.urlretrieve(full_url,tmpfilename)
        print h
        ds=gdal.Open(tmpfilename)

        cube_arr.append(ds.ReadAsArray())
        print d

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    NTAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    NTAI += 1
    NTAI *= 1000
    NTAI //= (2000 - 0 + 1) / 255.
    NTAI = NTAI.astype(numpy.uint8)
    NTAI += 1
    return NTAI,ds

# <codecell>


# <codecell>

date='2011-01-01'
spl_arr=[108.8,38.2,121.1,43.5]
spl_arr=[75,25.5,103.75,39]
#alpha=0.5
#VHI,ds = _VHI_CAL(date,spl_arr)
#TCI, cur_date = _TCI_CAL(date,spl_arr)
NTAI, ds = _NTAI_CAL(date,spl_arr)
#VHI = (alpha * VCI ) + ((1-alpha)* TCI)

# <codecell>

start

# <codecell>


# <codecell>

##request image cube for the specified date and area by WCS.
#firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
endpoint='http://159.226.117.95:58080/rasdaman/ows'
field={}
field['SERVICE']='WCS'
field['VERSION']='2.0.1'
field['REQUEST']='DescribeCoverage'
field['COVERAGEID']='LST_MOD11C2005_uptodate'#'trmm_3b42_coverage_1'
url_values = urllib.urlencode(field,doseq=True)
full_url = endpoint + '?' + url_values
print full_url
data = urllib.urlopen(full_url).read()
root = etree.fromstring(data)
lc = root.find(".//{http://www.opengis.net/gml/3.2}lowerCorner").text
uc = root.find(".//{http://www.opengis.net/gml/3.2}upperCorner").text
start_date=int((lc.split(' '))[2])
end_date=int((uc.split(' '))[2])
print [start_date, end_date]

# <codecell>

#generate the dates list 
cur_date=datetime.strptime(date,"%Y-%m-%d")
#startt=145796
start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)

# <codecell>

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

sample_size = root[0][3][0][5][0][1].text #sample size
root[0][3][0][5][0][1].text #sample size

# <codecell>

array_stepsize = np.fromstring(sample_size, dtype=int, sep=' ')
np.fromstring(sample_size, dtype=int, sep=' ')

# <codecell>

array_all_dates = []
for stepsize in array_stepsize:
    date_and_stepsize = start + timedelta(stepsize - 1)
    array_all_dates.append(date_and_stepsize)
    print date_and_stepsize

# <codecell>

for j in array_all_dates:
    print j.timetuple().tm_yday

# <codecell>

# function for parsing strings using specific format
get_datetime = lambda s: datetime.strptime(s,"%Y-%m-%d")


base_date='2011-01-01'
base = get_datetime(base_date)
later = filter(lambda d: get_datetime(d[0]) > base, array_all_dates)
closest_date = min(later, key = lambda d: get_datetime(d[0]))

# <codecell>

base

# <codecell>

from datetime import timedelta, datetime
base_date = '2011-01-17'
b_d = datetime.strptime(base_date, "%Y-%m-%d")
def func(d):
    #d =  datetime.strptime(x[0], "%Y-%m-%d")
    delta =  d - b_d if d > b_d else timedelta.max
    return delta

lis2 = [('10/30 04:30 PM', '+1.5 -111', '-1.5 +101'), ('10/30 04:24 PM', '+1.5 -110', '-1.5     +100'), ('10/30 04:21 PM', '+1.5 -111', '-1.5 +101'), ('10/30 04:15 PM', '+1.5 -112', '-1.5 +102'), ('10/30 04:14 PM', '+1.5 -110', '-1.5 +100'), ('10/30 03:57 PM', '+1.5 -111', '-1.5 +101'), ('10/30 03:40 PM', '+1.5 -110', '-1.5 +100'), ('10/30 03:31 PM', '+1.5 -111', '-1.5 +101'), ('10/30 03:30 PM', '+1.5 -109', '-1.5 -101'), ('10/30 03:25 PM', '+1.5 -107', '-1.5 -103'), ('10/30 03:24 PM', '+1.5 -110', '-1.5 +100'), ('10/30 03:23 PM', '+1.5 -108', '-1.5 -102'), ('10/30 03:22 PM', '+1.5 -106', '-1.5 -104'), ('10/30 02:14 PM', '+1.5 -104', '-1.5 -106'), ('10/30 01:41 PM', '+1.5 -105', '-1.5 -105'), ('10/30 01:37 PM', '+1.5 -107', '-1.5 -103'), ('10/30 01:36 PM', '+1.5 -105', '-1.5 -105'), ('10/30 01:06 PM', '+1.5 -103', '-1.5 -107'), ('10/30 12:56 PM', '+2 -111', '-2 +101'), ('10/30 12:53 PM', '+2 -110', '-2 +100'), ('10/30 12:50 PM', '+2 -113', '-2 +103'), ('10/30 12:49 PM', '+2 -112', '-2 +102'), ('10/30 12:46 PM', '+2 -113', '-2 +103'), ('10/30 12:45 PM', '+2 -110', '-2 +100'), ('10/30 12:43 PM', '+2 -108', '-2 -102'), ('10/30 12:38 PM', '+2.5 -116', '-2.5 +106'), ('10/30 12:38 PM', '+2.5 -113', '-2.5 +103'), ('10/30 12:37 PM', '+2.5 -110', '-2.5 +100'), ('10/30 10:30 AM', '+2.5 -105', '-2.5 -105'), ('10/30 10:07 AM', '+3 -113', '-3 +103'), ('10/30 09:55 AM', '+3 -112', '-3 +102'), ('10/30 09:51 AM', '+3 -110', '-3 +100'), ('10/30 09:32 AM', '+3 -109', '-3 -101'), ('10/30 06:04 AM', '+3 -110', '-3 +100'), ('10/30 03:16 AM', '+3 -107', '-3 -103'), ('10/30 03:14 AM', '+3.5 -116', '-3.5 +106'), ('10/30 01:03 AM', '+3.5 -115', '-3.5 +105'), ('10/30 12:17 AM', '+3.5 -110', '-3.5 +100'), ('10/29 08:52 PM', '+3.5 -108', '-3.5 -102'), ('10/29 01:31 PM', '+3.5 -105', '-3.5 -105'), ('10/29 06:48 AM', '+3.5 -110', '-3.5 +100'), ('10/29 06:47 AM', '+3.5 -109', '-3.5 -101'), ('10/29 05:39 AM', '+3.5 -113', '-3.5 +103'), ('10/29 03:34 AM', '+3.5 -108', '-3.5 -102'), ('10/29 12:44 AM', '+3.5 -110', '-3.5 +100'), ('10/29 12:41 AM', '+3.5 -107', '-3.5 -103'), ('10/29 12:40 AM', '+3.5 -105', '-3.5 -105'), ('10/28 12:52 PM', '+4 -105', '-4 -105')]
min(array_all_dates, key = func)

# <codecell>


# <codecell>

get_datetime(date)

# <codecell>

root[0][3][1][0].text

# <codecell>

#generate the dates list 
cur_date=datetime.strptime(date,"%Y-%m-%d")
startt=145775
start=datetime.fromtimestamp((start_date-(datetime(1970,01,01)-datetime(1601,01,01)).days)*24*60*60)
print start
tmp_date=datetime(start.year,cur_date.month,cur_date.day)
if tmp_date > start :
    start=(tmp_date-datetime(1601,01,01)).days
else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
datelist=range(start+1,end_date-1,365)
print datelist

# <codecell>


# <codecell>

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

# <codecell>

ja = ((NVAI+1)*1000)
ja.max()

# <codecell>

ja //= (2000 - 0 + 1) / 255.
ja.max()

# <codecell>

NVAI += 1
NVAI *= 1000
NVAI //= (2000 - 0 + 1) / 255.
NVAI = NVAI.astype(numpy.uint8)
NVAI += 1

# <codecell>

%matplotlib inline
nvai_cmap = make_colormap([c('#F29813'), c('#D8DC44'),0.2, c('#D8DC44'), c('#7EC5AD'),0.4, c('#7EC5AD'), c('#5786BE'),0.6, 
                          c('#5786BE'), c('#41438D'),0.8, c('#41438D')])

nvai_cmap = make_colormap([c('#781800'), c('#B34700'),0.1, c('#B34700'), c('#F09400'),0.2, c('#F09400'), c('#FFBE3B'), 0.3, 
                       c('#FFBE3B'), c('#FFD88A'),0.4, c('#FFD88A'), c('#FFD88A'),0.5, c('#FFFFFF'), c('#B6D676'), 0.6,
                       c('#B6D676'), c('#8BBA2D'),0.7, c('#8BBA2D'), c('#60A100'),0.8, c('#60A100'), c('#1B8500'), 0.9,
                       c('#1B8500'), c('#006915')])

# <codecell>

##write the result VCI to disk
# get parameters
geotransform = ds.GetGeoTransform()
spatialreference = ds.GetProjection()
ncol = ds.RasterXSize
nrow = ds.RasterYSize
nband = 1

trans = ds.GetGeoTransform()
extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
  trans[3] + ds.RasterYSize*trans[5], trans[3])

# Create figure
fig = plt.imshow(NVAI, cmap=cmap_discretize(nvai_cmap,10), vmin=-1, vmax=1, extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

##write the result VCI to disk
# get parameters
geotransform = ds.GetGeoTransform()
spatialreference = ds.GetProjection()
ncol = ds.RasterXSize
nrow = ds.RasterYSize
nband = 1

trans = ds.GetGeoTransform()
extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
  trans[3] + ds.RasterYSize*trans[5], trans[3])

# Create figure
fig = plt.imshow(NVAI, cmap=cmap_discretize(nvai_cmap,10), vmin=0, vmax=255, extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

bounds = np.array([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
(bounds * 255 / 2 ) 

# <codecell>

ntai_cmap = make_colormap([c('#7F00FF'), c('#3F94FF'),0.1, c('#3F94FF'), c('#78CBFE'),0.2, c('#78CBFE'), c('#99EDFE'), 0.3, 
                       c('#99EDFE'), c('#D9FED9'),0.4, c('#D9FED9'), c('#FFFFFF'),0.5, c('#FFFFFF'), c('#FEFE4C'), 0.6,
                       c('#FEFE4C'), c('#FFCC00'),0.7, c('#FFCC00'), c('#FF7F00'),0.8, c('#FF7F00'), c('#FF0000'), 0.9,
                       c('#FF0000'), c('#7E0000')])

# <codecell>

##write the result VCI to disk
# get parameters
geotransform = ds.GetGeoTransform()
spatialreference = ds.GetProjection()
ncol = ds.RasterXSize
nrow = ds.RasterYSize
nband = 1

trans = ds.GetGeoTransform()
extent = (trans[0], trans[0] + ds.RasterXSize*trans[1],
  trans[3] + ds.RasterYSize*trans[5], trans[3])

# Create figure
fig = plt.imshow(NTAI, cmap=cmap_discretize(ntai_cmap,10), vmin=0, vmax=255, extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

def _NVAI_CAL(date,spl_arr):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://159.226.117.95:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
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
        field['COVERAGEID']='modis_13c1_cov'#'trmm_3b42_coverage_1'
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
    NVAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    NVAI += 1
    NVAI *= 1000
    NVAI //= (2000 - 0 + 1) / 255. # instead of 256 to make space for zero values
    NVAI = NVAI.astype(numpy.uint8)
    NVAI += 1 # So 0 values are reserved for mask

    ##write the result VCI to disk
    # get parameters
#     geotransform = ds.GetGeoTransform()
#     spatialreference = ds.GetProjection()
#     ncol = ds.RasterXSize
#     nrow = ds.RasterYSize
#     nband = 1

#     # create dataset for output
#     fmt = 'GTiff'
#     nvaiFileName = 'NVAI'+cur_date.strftime("%Y%m%d")+'.tif'
#     driver = gdal.GetDriverByName(fmt)
#     dst_dataset = driver.Create(nvaiFileName, ncol, nrow, nband, gdal.GDT_Byte)
#     dst_dataset.SetGeoTransform(geotransform)
#     dst_dataset.SetProjection(spatialreference)
#     dst_dataset.GetRasterBand(1).WriteArray(NVAI)
#     dst_dataset = None
    return NVAI, ds#nvaiFileName

# <codecell>


