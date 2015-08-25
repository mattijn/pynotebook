# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# coding: utf-8

# # VCI calculation process
# This process intend to calculate the Vegetation Condition index (VCI) for a specific area. The fomula of the index is:
# VCI =NDVI/(max(NDVI)-min(NDVI))
# where the NDVI is Normalized Difference Vegetation Index.
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
# http://localhost/cgi-bin/pywps.cgi?service=wps&version=1.0.0&request=execute&identifier=WPS_VCI_CAL&datainputs=[date=2005-02-06;bbox=50,10,120,60]&responsedocument=image=@asReference=true

# In[1]:

# <codecell>

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
from __future__ import division

# <codecell>

def _VCI_CAL(date,spl_arr):

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
    VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    VCI *= 1000
    VCI //= (1000 - 0 + 1) / 256.
    VCI = VCI.astype(numpy.uint8)  
    
    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    vciFileName = 'VCI'+cur_date.strftime("%Y%m%d")+'.tif'
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(vciFileName, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).WriteArray(VCI*200)
    dst_dataset = None
    return VCI,ds#vciFileName

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

vci_cmap = make_colormap([c('#FFFF00'), c('#A35600'),0.1, c('#A35600'), c('#D89A71'),0.2, c('#D89A71'), c('#ECD9C8'),0.3, 
                          c('#ECD9C8'), c('#DEDEE0'),0.4, c('#DEDEE0'), c('#DEDFE1'),0.5, c('#DEDFE1'), c('#74FE75'),0.6, 
                          c('#74FE75'), c('#87DE00'),0.7, c('#87DE00'), c('#4FB94B'),0.8, c('#4FB94B'), c('#178A49'),0.9, 
                          c('#178A49'), c('#004642')])

# <codecell>

date='2015-01-01'
spl_arr=[73.2,-9.5,140.9,53.9]

VCI,ds = _VCI_CAL(date,spl_arr)

# <codecell>

  

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
fig = plt.imshow(VCI, cmap=cmap_discretize(vci_cmap,10), vmin=0, vmax=255, extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

np.linspace(0.,255.,10)

# <codecell>

VCI

# <codecell>


# <codecell>


# <codecell>

date='2013-06-30'
spl_arr=[70,30,80,50]

# <codecell>

VCI,ds = _VCI_CAL(date,spl_arr)

# <codecell>

%matplotlib inline

# <codecell>


# <codecell>

cmap_discretize(vci_cmap,10)

# <codecell>

import numpy as np

# <codecell>

import matplotlib

# <codecell>

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

# <codecell>

lut_display((VCI*100).astype(np.int32), VCI.min(),VCI.max())

# <codecell>

VCI *= 1000
VCI //= (1000 - 0 + 1) / 256.
VCI = VCI.astype(np.uint8) 

# <codecell>

256./8

# <codecell>

    $('.map_di').on('change', function() {

    // Check lonlat exist
    //if (typeof lonlat == 'undefined') {
    //	alert('Computer says no. First click on land to retrieve a coordinate.');
    //	close_dialog_event();
    //	return;
    //};

    selector = $('input[name="di"]:checked', '#map_di_form').val()
    if (selector == 'pap') {
        //Add Derived Drought Indices OPTIONS
        console.log('di_maps_pap');	
        $('#ts_di_pap_refresh').attr('src', './js/img/ls.gif');

    // Execute function
    //executePrecipitationAnomalyPercentageSLDQuery();
    alert('execute!');		

    // Show right options		
        document.getElementById("map_di_pap_opacity").style.display = 'inline-block';
        document.getElementById("map_di_pap_opacity").style.width = '100%';
        document.getElementById("map_di_vci_opacity").style.display = 'none';

    $('#map_di_pap_refresh').on('click', function () {
    if (selector == 'pap') {
        console.log('map_di_pap_check');
        $('#map_di_pap_refresh').attr('src', './js/img/ls.gif');

    //executePrecipitationAnomalyPercentageSLDQuery();
    alert('execute!');
    } else {
        console.log('map_di_pap_uncheck');
    }
    });


    $("#layer_opacity_map_di_pap").slider({
    value: 70,
    slide: function (e, ui) {
        var map_di_pap = map.getLayersByName("SLD test");
        map_di_pap[0].setOpacity(ui.value / 100);
    }
    });
    };

    //   - - - - - - -- - - - - - - - - -- - - - - - - - -


    if (selector == 'vci') {
        //Add Derived Drought Indices OPTIONS
        console.log('di_maps_vci');	
        $('#ts_di_vci_refresh').attr('src', './js/img/ls.gif');

    // Execute function
    executePrecipitationAnomalyPercentageSLDQuery();		

    // Show right options
        document.getElementById("map_di_vci_opacity").style.display = 'inline-block';
        document.getElementById("map_di_vci_opacity").style.width = '100%';
        document.getElementById("map_di_pap_opacity").style.display = 'none';

    $('#map_di_vci_refresh').on('click', function () {
    if (selector == 'vci') {
        console.log('map_di_vci_check');
        $('#map_di_vci_refresh').attr('src', './js/img/ls.gif');

    //executePrecipitationAnomalyPercentageSLDQuery();
    alert('execute!');
    } else {
        console.log('map_di_vci_uncheck');
    }
    });


    $("#layer_opacity_map_di_vci").slider({
    value: 70,
    slide: function (e, ui) {
        var map_di_vci = map.getLayersByName("SLD test");
        map_di_vci[0].setOpacity(ui.value / 100);
    }
    });

    };
    });

