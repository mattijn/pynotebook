import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcb
import cartopy.feature as cfeature
from matplotlib import gridspec
from datetime import datetime
import warnings
from osgeo import gdal
import numpy as np
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import sys
import pandas as pd
import matplotlib
import geopandas as gpd
from matplotlib import gridspec
from cartopy.io import shapereader
import shapely.geometry as sgeom
import numpy as np
import matplotlib as mpl
import urllib
import numpy
import numpy as np
import numpy.ma as ma
from lxml import etree
from datetime import datetime, timedelta
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import subprocess as sp

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

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()
	
def _NVAI_CAL(date,spl_arr):

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
    #print start

    #tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    #if tmp_date > start :
    #    start=(tmp_date-datetime(1601,01,01)).days
    #else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    #datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    #cur_epoch=(cur_date-datetime(1601,01,01)).days
    #cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)

    try:    
        datelist, cur_pos = datelist_irregular_coverage(root, start_date, start, cur_date)
        print 'irregular'
    except IndexError:
        datelist, cur_pos = datelist_regular_coverage(root, start_date, start, cur_date)
        print 'regular'

    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        d_iso = datetime.fromtimestamp((d-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)	
        print 'NDVI: ', d_iso
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

    ##calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    NVAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    #VCI *= 1000
    #VCI //= (1000 - 0 + 1) / 256.
    #VCI = VCI.astype(numpy.uint8) 
    return NVAI,ds
	
def _NTAI_CAL(date,spl_arr):

    ##request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://192.168.1.104:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']='LST_MOD11C2005_uptodate'#'LST_MOD11C2005'#'trmm_3b42_coverage_1'
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
    #print start

    #tmp_date=datetime(start.year,cur_date.month,cur_date.day)
    #if tmp_date > start :
    #    start=(tmp_date-datetime(1601,01,01)).days
    #else: start=(datetime(start.year+1,cur_date.month,cur_date.day)-datetime(1601,01,01)).days
    #datelist=range(start+1,end_date-1,365)
    #print datelist

    #find the position of the requested date in the datelist
    #cur_epoch=(cur_date-datetime(1601,01,01)).days
    #cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    #print ('Current position:',cur_pos)

    try:    
        datelist, cur_pos = datelist_irregular_coverage(root, start_date, start, cur_date)
        print 'irregular'
    except IndexError:
        datelist, cur_pos = datelist_regular_coverage(root, start_date, start, cur_date)
        print 'regular'
        
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        d_iso = datetime.fromtimestamp((d-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)	
        print 'LST: ', d_iso
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']='LST_MOD11C2005_uptodate'#'LST_MOD11C2005'#'trmm_3b42_coverage_1'
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
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),0)
    ##VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    NTAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    return NTAI, cur_date

def _NDAI_CAL(date,spl_arr,alpha = 0.5):
    print 'NDAI start'
    print 'calculate NTAI'
    NTAI, cur_date = _NTAI_CAL(date,spl_arr)
    print 'calculate NVAI'
    NVAI, ds = _NVAI_CAL(date,spl_arr)
    print 'calculate NDAI'
    print 'IMPROVED NDAI COMPUTATION'	
    NDAI = (alpha * NVAI ) - ((1-alpha) * NTAI)
    NDAI.fill_value = -3000
    print 'NDAI end, lets save'
    #VHI *= 1000
    #VHI //= (1000 - 0 + 1) / 255. #instead of 256
    #VHI = VHI.astype(numpy.uint8)
    #VHI += 1 #so mask values are reserverd for 0 

    ##write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dataset for output
    fmt = 'GTiff'
    vhiFileName = 'NDAI'+cur_date.strftime("%Y%m%d")+'.tif'
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(vhiFileName, ncol, nrow, nband, gdal.GDT_Float32)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.GetRasterBand(1).SetNoDataValue(-3000)
    dst_dataset.GetRasterBand(1).WriteArray(NDAI)
    dst_dataset = None
    return vhiFileName

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
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    Return a reverse colormap from a LinearSegmentedColormaps
    
    cmap: LinearSegmentedColormap instance
    name: name of new cmap (default 'my_cmap_r')

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   
    
    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        
        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    
        
    LinearL = dict(zip(k,reverse))
    my_cmap_r = mcolors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r
	
def getStatsCounty(cnty_array, feat, date):
    """
    Core function to calculate statistics to be applied for each county
    
    Input:  
    cnty_array   = Masked raster array of a single county    
    feat         = feature of shapefile to extract ID
    Output: 
    county_stats = Dictionary containing the stats for the county
    """
    date = str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)
    
    dc=0
    #percentage of no drought
    p0=(cnty_array[(cnty_array <= 0) ]).count()*1.0/cnty_array.count()
    if p0>=0.5: dc=1
    #percentage of no drought
    p1=(cnty_array[(cnty_array<=-0.15)]).count()*1.0/cnty_array.count()
    if p1>0.5: dc=2
    #percentage of no drought
    p2=(cnty_array[(cnty_array<=-0.25) ]).count()*1.0/cnty_array.count()
    if p2>=0.5: dc=3
    #percentage of no drought
    p3=(cnty_array[cnty_array <=-0.35]).count()*1.0/cnty_array.count()
    if p3>=0.5: dc=4
    #print cnty_array.count(),np.nanmin(cnty_array)
    ct=cnty_array.count()
    county_stats = {
        'MINIMUM': np.nan if ct<2 else float(np.nanmin(cnty_array)),
        'MN'+date: np.nan if ct<2 else float(np.ma.mean(cnty_array)),
        'MAX': np.nan if ct<2 else float(np.nanmax(cnty_array)),
        'STD': np.nan if ct<2 else float(np.nanstd(cnty_array)),
        'SUM': np.nan if ct<2 else float(np.nansum(cnty_array)),
        'COUNT': int(cnty_array.count()),
        'FID': int(feat.GetFID()),  
        'P0'+date:p0,
        'P1'+date:p1,
        'P2'+date:p2,
        'P3'+date:p3,
        'DC'+date:dc}
    
    return county_stats
	
def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)
	
def zonal_stats(vector_path, raster_path, nodata_value, date):
        
    # open raster layer
    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()
    
    # set raster nodata value
    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)
    
    # open vector layer
    vds = ogr.Open(vector_path, GA_ReadOnly)  
    assert(vds)
    vlyr = vds.GetLayer(0)    
    
    # compare EPSG values of vector and raster and change projection if necessary
    sourceSR = vlyr.GetSpatialRef()
    sourceSR.AutoIdentifyEPSG()
    EPSG_sourceSR = sourceSR.GetAuthorityCode(None)
    
    targetSR = osr.SpatialReference(wkt=rds.GetProjection())
    targetSR.AutoIdentifyEPSG()
    EPSG_targetSR = targetSR.GetAuthorityCode(None)
    
    if EPSG_sourceSR != EPSG_sourceSR:
        # reproject vector geometry to same projection as raster
        print 'unequal projections'    
        sourceSR = vlyr.GetSpatialRef()
        targetSR = osr.SpatialReference()
        targetSR.ImportFromWkt(rds.GetProjectionRef())
        coordTrans = osr.CreateCoordinateTransformation(sourceSR,targetSR)    
        
    """do the work"""
    global_src_extent = None
    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')
    
    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature() 
    
    while feat is not None:
        # print statement after each hunderds features
        fid = int(feat.GetFID())        
        if fid % 500 == 0:
            print("finished first %s features" % (fid))
    
        if not global_src_extent:
            #print 'bbox county'
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)
        
            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5]
            )
        
        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())
        
        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()
        
        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        try:
            masked = np.ma.MaskedArray(
                src_array,
                mask=np.logical_or(
                    src_array == nodata_value,
                    np.logical_not(rv_array)
                )
            )
            
            #print 'feature ID: ',int(feat.GetFID())
            
            # GET STATISTICS FOR EACH COUNTY
            try:
                county_stats = getStatsCounty(cnty_array = masked, feat=feat, date=date)            
                stats.append(county_stats)

                rvds = None
                mem_ds = None
                feat = vlyr.GetNextFeature()
            except IndexError:
                print 'feature ID: ',fid, 'IndexError, ignore county and lets continue'                
                rvds = None
                mem_ds = None
                feat = vlyr.GetNextFeature()                
                
            
        except np.ma.MaskError: 
            # catch MaskError, ignore feature containing no valid corresponding raster data set
            # in my case the the most southern county of hainan is not totally within the raster extent            
            print 'feature ID: ',fid, ' maskError, ignore county and lets continue'
            
            rvds = None
            mem_ds = None
            feat = vlyr.GetNextFeature()            
    
    vds = None
    rds = None
    return stats#, src_array, rv_array, masked

def rasterize(date_str = '20041015', in_shp = r'D:\Data\ChinaShapefile\Chinese_version_counties\counties_china1.shp'):
    prefix = ['P0','P1','P2','P3','MN', 'DC']
    folder = r'D:\Downloads\Mattijn@Zhou\ChinaDroughtCounty\tif//'
    gdal_rasterize = r'C:\Program Files\GDAL//gdal_rasterize.exe'
    float32 = 'Float32'
    for pre in prefix:
        attribute  = pre + date_str
        out_raster = folder + pre + date_str + '.tif'

        command = [gdal_rasterize, '-a', attribute, '-at', '-l', 'china_NDAI'+date_str, 
                   '-tr', str(0.05),str(0.05), '-ot', float32,'-init',str(7), in_shp, out_raster]
        #, "-a", attribute, "-a_nodata value", str(-1), out_filename, out_raster]

        #logging.info(sp.list2cmdline(command))
        print (sp.list2cmdline(command))

        norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
        norm.communicate()

	
def plot_map(date_str = '20120828', date=datetime(2012,8,28),extent = [-179, 179, -60, 90]):
    #date_str = '20041015'
    prefix = ['P0','P1','P2','P3','MN','DC']
    folder = r'D:\Downloads\Mattijn@Zhou\ChinaDroughtCounty\tif//'
    in_rasters = []
    for pre in prefix:    
        out_raster = folder + pre + date_str + '.tif'
        print out_raster
        in_rasters.append(out_raster)

    data1 = np.ma.masked_equal(gdal.Open(in_rasters[0]).ReadAsArray(),7)
    data2 = np.ma.masked_equal(gdal.Open(in_rasters[1]).ReadAsArray(),7)
    data3 = np.ma.masked_equal(gdal.Open(in_rasters[2]).ReadAsArray(),7)
    data4 = np.ma.masked_equal(gdal.Open(in_rasters[3]).ReadAsArray(),7)
    data5 = np.ma.masked_equal(gdal.Open(in_rasters[4]).ReadAsArray(),7)
    data6 = np.ma.masked_equal(gdal.Open(in_rasters[5]).ReadAsArray(),7)

    in_tif = in_rasters[0]
    ds = gdal.Open(in_tif)
    print 'geotransform', ds.GetGeoTransform()
    print 'raster X size', ds.RasterXSize
    print 'raster Y size', ds.RasterYSize

    data = ds.ReadAsArray()
    data_ma = np.ma.masked_equal(data,7)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    xres = gt[1]
    yres = gt[5]

    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5

    #ds = None
    gridlons = np.mgrid[xmin:xmax+xres:xres]
    gridlats = np.mgrid[ymax+yres:ymin:yres]

    drought_cat_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                          c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#C4DC73')])

    drought_per_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                          c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#FEFFD3')])

    drought_avg_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                                          c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                                          c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                                          c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                                          c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])

    drought_per_tci_cmap_r = reverse_colourmap(drought_per_tci_cmap, name = 'drought_per_tci_cmap_r')
    drought_cat_tci_cmap_r = reverse_colourmap(drought_cat_tci_cmap, name = 'drought_cat_tci_cmap_r')


    #extent = [111.91693268, 123.85693268, 49.43324112, 40.67324112]
    #extent = [73.5,140,14,53.6]    


    fig = plt.figure(figsize=(20,12.9))
    gs = gridspec.GridSpec(3, 3)
    land = cfeature.LAND.scale='50m'
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax1
    # PLOT TOP LEFT
    ax1 = fig.add_subplot(gs[0,0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
    ax1.set_extent(extent)
    #ax1.outline_patch.set_edgecolor('none')
    # gridlines
    gl1 = ax1.gridlines(zorder=3)
    gl1.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
    gl1.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
    gl1.xformatter = LONGITUDE_FORMATTER
    gl1.yformatter = LATITUDE_FORMATTER

    ax1.add_feature(cfeature.LAND, facecolor='0.85')    
    # pcolormesh
    bounds1 = [0.25,0.5,0.75,1]
    cmap1 = cmap_discretize(drought_per_tci_cmap_r,6)   
    norm1 = mcolors.BoundaryNorm(bounds1, cmap1.N)
    im1 = ax1.pcolormesh(gridlons, gridlats, data1, transform=ccrs.PlateCarree(), norm=norm1, cmap=cmap1, vmin=0, vmax=1, zorder=2)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax2
    # PLOT MIDDLE LEFT
    ax2 = fig.add_subplot(gs[1,0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
    ax2.set_extent(extent)
    #ax2.outline_patch.set_edgecolor('none')
    # gridlines
    gl2 = ax2.gridlines(zorder=3)
    gl2.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
    gl2.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
    gl2.xformatter = LONGITUDE_FORMATTER
    gl2.yformatter = LATITUDE_FORMATTER

    ax2.add_feature(cfeature.LAND, facecolor='0.85') 
    # pcolormesh
    bounds2 = [0.25,0.5,0.75,1]
    cmap2 = cmap_discretize(drought_per_tci_cmap_r,6)
    norm2 = mcolors.BoundaryNorm(bounds2, cmap2.N)
    im2 = ax2.pcolormesh(gridlons, gridlats, data2, transform=ccrs.PlateCarree(), norm=norm2, cmap=cmap2, vmin=0, vmax=1, zorder=2)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax3
    # PLOT BOTTOM LEFT
    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
    ax3.set_extent(extent)
    #ax3.outline_patch.set_edgecolor('none')
    # gridlines
    gl3 = ax3.gridlines(zorder=3)
    gl3.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
    gl3.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
    gl3.xformatter = LONGITUDE_FORMATTER
    gl3.yformatter = LATITUDE_FORMATTER

    ax3.add_feature(cfeature.LAND, facecolor='0.85')  
    # pcolormesh
    bounds3 = [0.25,0.5,0.75,1]
    cmap3 = cmap_discretize(drought_per_tci_cmap_r,6)
    norm3 = mcolors.BoundaryNorm(bounds3, cmap3.N)
    im3 = ax3.pcolormesh(gridlons, gridlats, data3, transform=ccrs.PlateCarree(), norm=norm3, cmap=cmap3, vmin=0, vmax=1, zorder=2)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax4
    # PLOT BOTTOM MIDDLE
    ax4 = fig.add_subplot(gs[2,1], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
    ax4.set_extent(extent)
    #ax4.outline_patch.set_edgecolor('none')
    # gridlines
    gl4 = ax4.gridlines(zorder=3)
    gl4.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
    gl4.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
    gl4.xformatter = LONGITUDE_FORMATTER
    gl4.yformatter = LATITUDE_FORMATTER

    ax4.add_feature(cfeature.LAND, facecolor='0.85') 
    # pcolormesh
    bounds4 = [0.25,0.5,0.75,1]
    cmap4 = cmap_discretize(drought_per_tci_cmap_r,6)
    norm4 = mcolors.BoundaryNorm(bounds4, cmap4.N)
    im4 = ax4.pcolormesh(gridlons, gridlats, data4, transform=ccrs.PlateCarree(), norm=norm4, cmap=cmap4, vmin=0, vmax=1, zorder=2)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax5
    # PLOT BOTTOM RIGHT
    ax5 = fig.add_subplot(gs[2,2], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
    ax5.set_extent(extent)
    #ax5.outline_patch.set_edgecolor('none')
    # gridlines
    gl5 = ax5.gridlines(zorder=3)
    gl5.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
    gl5.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
    gl5.xformatter = LONGITUDE_FORMATTER
    gl5.yformatter = LATITUDE_FORMATTER	
    ax5.add_feature(cfeature.LAND, facecolor='0.85')          
    # pcolormesh
    bounds5 = [-1,-0.35,-0.25,-0.15,0,0.15,0.25,0.35,1]
    cmap5 = cmap_discretize(drought_avg_tci_cmap,8)
    norm5 = mcolors.BoundaryNorm(bounds5, cmap5.N)
    im5 = ax5.pcolormesh(gridlons, gridlats, data5, transform=ccrs.PlateCarree(), norm=norm5, cmap=cmap5, vmin=-1, vmax=1, zorder=2)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ax6
    # PLOT CENTER
    ax6 = fig.add_subplot(gs[0:2,1:3], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
    ax6.set_extent(extent)
    #ax6.outline_patch.set_edgecolor('gray')
    #ax6.outline_patch.set_linewidth(1)
    #ax6.outline_patch.set_linestyle(':')
    # features
    coastline = cfeature.COASTLINE.scale='50m'
    borders = cfeature.BORDERS.scale='50m'
    ax6.add_feature(cfeature.COASTLINE,linewidth=0.5, edgecolor='black')
    ax6.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax6.add_feature(cfeature.LAND, facecolor='0.85')	
    # gridlines
    gl6 = ax6.gridlines(zorder=3)
    gl6.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
    gl6.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
    gl6.xformatter = LONGITUDE_FORMATTER
    gl6.yformatter = LATITUDE_FORMATTER
    # pcolormesh
    bounds6 = [0, 1, 2, 3, 4, 5]
    cmap6 = cmap_discretize(drought_cat_tci_cmap_r,5)
    norm6 = mcolors.BoundaryNorm(bounds6, cmap6.N)
    im6 = ax6.pcolormesh(gridlons, gridlats, data6, transform=ccrs.PlateCarree(), norm=norm6, cmap=cmap6, vmin=0, vmax=4, zorder=2)

    #date = i[-7:]
    #year = date[-4::]
    #doy = date[-7:-4]
    #date_out = datetime.datetime.strptime(str(year)+'-'+str(doy),'%Y-%j')
    date_label = 'Date: '+str(date.year) +'-'+str(date.month).zfill(2)+'-'+str(date.day).zfill(2)
    # ADD LABELS FOR EACH PLOT
    #ax1.set_title('Percentage of Slight Drought', weight='semibold', fontsize=12)
    #ax2.set_title('Percentage of Moderate Drought', weight='semibold', fontsize=12)
    #ax3.set_title('Percentage of Severe Drought', weight='semibold', fontsize=12)
    #ax4.set_title('Percentage of Extreme Drought', weight='semibold', fontsize=12)
    #ax5.set_title('Average of NDAI', weight='semibold', fontsize=12)
    #ax6.set_title('Drought Alert at Province Level. '+date_label, fontsize=20, weight='semibold', color='k')
	
    # ADD LABELS FOR EACH PLOT
    ax1.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic(), zorder=3)
    ax1.text(64, 51, 'Percentage of Slight Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic(), zorder=3)
    ax2.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic(), zorder=3)
    ax2.text(64, 51, 'Percentage of Moderate Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic(), zorder=3)
    ax3.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic(), zorder=3)
    ax3.text(64, 51, 'Percentage of Severe Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic(), zorder=3)
    ax4.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic(), zorder=3)
    ax4.text(64, 51, 'Percentage of Extreme Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic(), zorder=3)
    ax5.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic(), zorder=3)
    ax5.text(64, 51, 'Average of NDAI', weight='semibold', fontsize=12, transform=ccrs.Geodetic(), zorder=3)
    ax6.plot(116.4, 39.3, 'ks', markersize=7, transform=ccrs.Geodetic(), zorder=3)
    ax6.text(64, 51, 'Drought Alert at County Level', fontsize=20, weight='semibold', color='k',transform=ccrs.Geodetic(), zorder=3)
    ax6.text(65.5, 49, date_label, fontsize=20, weight='semibold', color='k',transform=ccrs.Geodetic(), zorder=3)
    ax6.text(117, 40., 'Beijing', weight='semibold', transform=ccrs.Geodetic(), zorder=3)

    # ADD LEGEND IN ALL PLOTS
    # -------------------------Ax 1
    #cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
    #cbax1 = fig.add_axes([0.03, 0.7, 0.011, 0.10]) # including tight_layout() GLOBAL
    cbax1 = fig.add_axes([0.28, 0.69, 0.011, 0.10]) # including tight_layout() CHINA ONLY

    #cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap = cmap_discretize(drought_per_tci_cmap,6)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = [1, 2, 3, 4, 5]
    bounds_ticks = [1.5, 2.5, 3.5, 4.5]
    bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb2 = mcb.ColorbarBase(cbax1, cmap=cmap,
                                         norm=norm,
                                         # to use 'extend', you must
                                         # specify two extra boundaries:
                                         #boundaries=[0]+bounds+[13],
                                         #extend='both',
                                         extendfrac='auto',
                                         ticklocation='right',
                                         ticks=bounds_ticks,#_name, # optional
                                         spacing='proportional',
                                         orientation='vertical')
    #cb2.set_label('Discrete intervals, some other units')
    cb2.set_ticklabels(bounds_ticks_name)



    # -------------------------Ax 2
    #cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
    #cbax2 = fig.add_axes([0.03, 0.37, 0.011, 0.10]) # including tight_layout() GLOBAL
    cbax2 = fig.add_axes([0.28, 0.36, 0.011, 0.10]) # including tight_layout() CHINA ONLY

    #cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap = cmap_discretize(drought_per_tci_cmap,6)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = [1, 2, 3, 4, 5]
    bounds_ticks = [1.5, 2.5, 3.5, 4.5]
    bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb2 = mcb.ColorbarBase(cbax2, cmap=cmap,
                                         norm=norm,
                                         # to use 'extend', you must
                                         # specify two extra boundaries:
                                         #boundaries=[0]+bounds+[13],
                                         #extend='both',
                                         extendfrac='auto',
                                         ticklocation='right',
                                         ticks=bounds_ticks,#_name, # optional
                                         spacing='proportional',
                                         orientation='vertical')
    #cb2.set_label('Discrete intervals, some other units')
    cb2.set_ticklabels(bounds_ticks_name)   


    # -------------------------Ax 3
    #cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
    #cbax3 = fig.add_axes([0.03, 0.04, 0.011, 0.10]) # including tight_layout() GLOBAL
    cbax3 = fig.add_axes([0.28, 0.03, 0.011, 0.10]) # including tight_layout() CHINA ONLY

    #cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap = cmap_discretize(drought_per_tci_cmap,6)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = [1, 2, 3, 4, 5]
    bounds_ticks = [1.5, 2.5, 3.5, 4.5]
    bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb2 = mcb.ColorbarBase(cbax3, cmap=cmap,
                                         norm=norm,
                                         # to use 'extend', you must
                                         # specify two extra boundaries:
                                         #boundaries=[0]+bounds+[13],
                                         #extend='both',
                                         extendfrac='auto',
                                         ticklocation='right',
                                         ticks=bounds_ticks,#_name, # optional
                                         spacing='proportional',
                                         orientation='vertical')
    #cb2.set_label('Discrete intervals, some other units')
    cb2.set_ticklabels(bounds_ticks_name)    


    # -------------------------Ax 4
    #cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16]) # without tight_layout()
    #cbax4 = fig.add_axes([0.36, 0.04, 0.011, 0.10]) # including tight_layout() GLOBAL
    cbax4 = fig.add_axes([0.61, 0.03, 0.011, 0.10]) # including tight_layout() CHINA ONLY

    #cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap = cmap_discretize(drought_per_tci_cmap,6)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = [1, 2, 3, 4, 5]
    bounds_ticks = [1.5, 2.5, 3.5, 4.5]
    bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb2 = mcb.ColorbarBase(cbax4, cmap=cmap,
                                         norm=norm,
                                         # to use 'extend', you must
                                         # specify two extra boundaries:
                                         #boundaries=[0]+bounds+[13],
                                         #extend='both',
                                         extendfrac='auto',
                                         ticklocation='right',
                                         ticks=bounds_ticks,#_name, # optional
                                         spacing='proportional',
                                         orientation='vertical')
    #cb2.set_label('Discrete intervals, some other units')
    cb2.set_ticklabels(bounds_ticks_name)        


    # -------------------------Ax 5
    #cbax5 = fig.add_axes([0.85, 0.15, 0.011, 0.16]) # without tight_layout()
    #cbax5 = fig.add_axes([0.6922, 0.04, 0.011, 0.16]) # including tight_layout() GLOBAL
    cbax5 = fig.add_axes([0.9422, 0.03, 0.011, 0.16]) # including tight_layout() CHINA ONLY 

    #cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap = cmap_discretize(drought_avg_tci_cmap,8)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = [1, 2, 3, 4, 5,6,7,8,9]
    bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5,6.6,7.5,8.5]
    bounds_ticks_name = [' ', '-0.35', ' ', '-0.15','0','0.15',' ','0.35',' ']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb2 = mcb.ColorbarBase(cbax5, cmap=cmap,
                                         norm=norm,
                                         # to use 'extend', you must
                                         # specify two extra boundaries:
                                         #boundaries=[0]+bounds+[13],
                                         #extend='both',
                                         extendfrac='auto',
                                         ticklocation='right',
                                         ticks=bounds,#_name, # optional
                                         spacing='proportional',
                                         orientation='vertical')        
    cb2.set_ticklabels(bounds_ticks_name)     

    # ------------------------Ax 6
    #cbax6 = fig.add_axes([0.79, 0.48, 0.020, 0.30]) # without tight_layout()
    #cbax6 = fig.add_axes([0.37, 0.4, 0.020, 0.20]) # without tight_layout() GLOBAL
    cbax6 = fig.add_axes([0.87, 0.4, 0.025, 0.20]) # without tight_layout() CHINA ONLY    

    #cmap = mcolors.ListedColormap(['r', 'g', 'b', 'c'])
    cmap = cmap_discretize(drought_cat_tci_cmap,5)
    cmap.set_over('0.25')
    cmap.set_under('0.75')

    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    bounds = [1, 2, 3, 4, 5,6]
    bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5]
    bounds_ticks_name = ['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Slight Drought', 'No Drought']
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb2 = mcb.ColorbarBase(cbax6, cmap=cmap,
                                         norm=norm,
                                         # to use 'extend', you must
                                         # specify two extra boundaries:
                                         #boundaries=[0]+bounds+[13],
                                         #extend='both',
                                         extendfrac='auto',
                                         ticklocation='right',
                                         ticks=bounds_ticks,#_name, # optional
                                         spacing='proportional',
                                         orientation='vertical')
    #cb2.set_label('Discrete intervals, some other units')
    cb2.set_ticklabels(bounds_ticks_name)
    cb2.ax.tick_params(labelsize=12)
    #         # ADD LAKES AND RIVERS 
    #         #FOR PLOT 1
    #         lakes = cfeature.LAKES.scale='110m'
    #         rivers = cfeature.RIVERS.scale='110m'        
    #         ax1.add_feature(cfeature.LAKES)
    #         ax1.add_feature(cfeature.RIVERS)         

    #         #FOR PLOT 2        
    #         ax2.add_feature(cfeature.LAKES)
    #         ax2.add_feature(cfeature.RIVERS)         

    #         #FOR PLOT 3        
    #         ax3.add_feature(cfeature.LAKES)
    #         ax3.add_feature(cfeature.RIVERS)                 

    #         #FOR PLOT 4        
    #         ax4.add_feature(cfeature.LAKES)
    #         ax4.add_feature(cfeature.RIVERS)         

    #         #FOR PLOT 5
    #         ax5.add_feature(cfeature.LAKES)
    #         ax5.add_feature(cfeature.RIVERS)                 

    #FOR PLOT 6        
    #lakes = cfeature.LAKES.scale='50m'
    #rivers = cfeature.RIVERS.scale='50m'        
    #ax6.add_feature(cfeature.LAKES)
    #ax6.add_feature(cfeature.RIVERS)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black', zorder=3)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black', zorder=3)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black', zorder=3)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black', zorder=3)
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black', zorder=3)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black', zorder=3)
    ax4.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black', zorder=3)
    ax4.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black', zorder=3)    
    ax5.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black', zorder=3)
    ax5.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black', zorder=3)
    ax6.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black', zorder=3)
    ax6.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black', zorder=3)    

    with warnings.catch_warnings():
        # This raises warnings since tight layout cannot
        # handle gridspec automatically. We are going to
        # do that manually so we can filter the warning.
        warnings.simplefilter("ignore", UserWarning)
        gs.tight_layout(fig, rect=[None,None,None,None])

    #gs.update(wspace=0.03, hspace=0.03)
    path_out = r'D:\Downloads\Mattijn@Zhou\ChinaDroughtCounty\png//China_'
    file_out = 'DroughtAlert_'+str(date.timetuple().tm_yday).zfill(3)+str(date.year).zfill(4)+'.png'
    filepath = path_out+file_out 

    fig.savefig(filepath, dpi=200, bbox_inches='tight', pad_inches=1.)
    print filepath
    #plt.show()        
    fig.clf()        
    plt.close()
    #del record#,county
    ram = None    
    return