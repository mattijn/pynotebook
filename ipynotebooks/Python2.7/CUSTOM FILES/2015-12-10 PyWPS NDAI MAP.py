
# coding: utf-8

# In[34]:



# # NDAI calculation process
# This process intend to calculate the Vegetation Condition index (VCI) for a specific area. The fomula of the index is:
# NDAI =a*NVAI-(1-a)*NTAI, where a is 0.5
# where the NDAI is Normalized Droguht Anomaly Index.
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
# http://localhost/cgi-bin/wpswsgi?service=wps&version=1.0.0&request=execute&identifier=WPS_NDAI_DI_CAL&datainputs=[date=2005-02-06;bbox=50,10,120,60]&responsedocument=image=@asReference=true


# coding: utf-8

# In[ ]:

#from pywps.Process import WPSProcess 
import logging
import os
import sys
import urllib
from lxml import etree
from osgeo import gdal
import numpy as np
import numpy
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from datetime import datetime, timedelta
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import subprocess as sp
get_ipython().magic(u'matplotlib inline')

# In[ ]:

def GetCoverageNames():
    file_json = r'D:\Downloads\Mattijn@Zhou\coverages_names.json'
    with open(file_json) as json_data:
        d = json.load(json_data)
    _CoverageID_NDVI = d['COVG_NAME_NDVI_MOD13C1005']
    _CoverageID_LST  = d['COVG_NAME_LST_MOD11C2005']
    return _CoverageID_NDVI, _CoverageID_LST

# In[ ]:

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
    logging.info(datelist)

    #find the position of the requested date in the datelist
    cur_epoch=(cur_date-datetime(1601,1,1)).days
    cur_pos=min(range(len(datelist)),key=lambda x:abs(datelist[x]-cur_epoch))
    print ('Current position:',cur_pos)    
    
    return datelist, cur_pos


# In[ ]:


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
    logging.info(array_selected_ansi)
    
    # get index of nearest date in selection array
    idx_nearest_date_selected = numpy.where(array_selected_dates==nearest_date)[0][0]  
    print idx_nearest_date_selected
    
    # return datelist in ANSI and the index of the nearest date
    return array_selected_ansi, idx_nearest_date_selected


# In[ ]:

def find_nearest(array,value):
    return (np.abs(array-value)).argmin()


# In[ ]:

def _NVAI_CAL(date,spl_arr,CoverageID_NDVI):

    #request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://192.168.1.104:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']=CoverageID_NDVI#'NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
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
        print 'NDVI', d
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']=CoverageID_NDVI#'NDVI_MOD13C1005_uptodate'#'NDVI_MOD13C1005'#'trmm_3b42_coverage_1'
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

    #calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),-3000)
    NVAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    #VCI *= 1000
    #VCI //= (1000 - 0 + 1) / 256.
    #VCI = VCI.astype(numpy.uint8) 
    return NVAI,ds


# In[ ]:


def _NTAI_CAL(date,spl_arr, CoverageID_LST):

    #request image cube for the specified date and area by WCS.
    #firstly we get the temporal length of avaliable NDVI data from the DescribeCoverage of WCS
    endpoint='http://192.168.1.104:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='DescribeCoverage'
    field['COVERAGEID']=CoverageID_LST#'LST_MOD11C2005_uptodate'#'LST_MOD11C2005'#'trmm_3b42_coverage_1'
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
        logging.info('irregular')
    except IndexError:
        datelist, cur_pos = datelist_regular_coverage(root, start_date, start, cur_date)
        print 'regular'
        logging.info('regular')
        
    #retrieve the data cube
    cube_arr=[]
    for d in datelist:
        print 'LST', d
        field={}
        field['SERVICE']='WCS'
        field['VERSION']='2.0.1'
        field['REQUEST']='GetCoverage'
        field['COVERAGEID']=CoverageID_LST#'LST_MOD11C2005_uptodate'#'LST_MOD11C2005'#'trmm_3b42_coverage_1'
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

    #calculate the regional VCI
    cube_arr_ma=ma.masked_equal(numpy.asarray(cube_arr),0) #nan val is 0 for lst
    #VCI=(cube_arr_ma[cur_pos,:,:]-numpy.amin(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    NTAI=(cube_arr_ma[cur_pos,:,:]-numpy.mean(cube_arr_ma,0))*1.0/(numpy.amax(cube_arr_ma,0)-numpy.amin(cube_arr_ma,0))
    
    return NTAI, cur_date

# In[ ]:



def _NDAI_CAL(date,spl_arr,alpha = 0.5):

    CoverageID_NDVI, CoverageID_LST = GetCoverageNames()
    
    NTAI, cur_date = _NTAI_CAL(date, spl_arr, CoverageID_LST)
    NVAI, ds = _NVAI_CAL(date, spl_arr, CoverageID_NDVI)
    NDAI = (alpha * NVAI ) - ((1-alpha) * NTAI)
    
    NDAI_255 = NDAI.copy()
    NDAI_255 += 1
    NDAI_255 *= 1000
    NDAI_255 //= (2000 - 0 + 1) / 255. #instead of 256
    NDAI_255 = NDAI_255.astype(numpy.uint8)
    NDAI_255 += 1 #so mask values are reserverd for 0
    NDAI_255.fill_value = 0    
    try:        
        NDAI_min = numpy.min(NDAI_255[~NDAI_255.mask])
        NDAI_255[ma.getmaskarray(NDAI)] = 0
    except ValueError:
        #only masked values set full array to 0 values
        NDAI_255 = numpy.zeros(NDAI_255.shape)

    #write the result VCI to disk
    # get parameters
    geotransform = ds.GetGeoTransform()
    spatialreference = ds.GetProjection()
    ncol = ds.RasterXSize
    nrow = ds.RasterYSize
    nband = 1

    # create dictionary metadata for datetime
    dict = {}
    dict['TIFFTAG_DATETIME'] = cur_date.strftime("%Y%m%d")
    
    # create dataset for output
    fmt = 'GTiff'
    ndaiFileName = 'NDAI'+cur_date.strftime("%Y%m%d")+'.tif'
    driver = gdal.GetDriverByName(fmt)
    dst_dataset = driver.Create(ndaiFileName, ncol, nrow, nband, gdal.GDT_Byte)
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(spatialreference)
    dst_dataset.SetMetadata(dict)# = cur_date.strftime("%Y%m%d")
    dst_dataset.GetRasterBand(1).SetNoDataValue(int(NDAI_255.fill_value))
    dst_dataset.GetRasterBand(1).WriteArray(NDAI_255)
    dst_dataset = None
    return ndaiFileName, NDAI_255   


# In[ ]:

# class Process(WPSProcess):


#     def __init__(self):

#         #
#         # Process initialization
#         WPSProcess.__init__(self,
#             identifier       = "WPS_NDAI_DI_CAL",
#             title            = "NDAI calculation process",
#             abstract         = """
#                                This process intend to calculate the 
#                                Normalized Drought Anomaly Index (NDAI) 
#                                for a specific area..
#                                """,
#             version          = "1.0",
#             storeSupported   = True,
#             statusSupported  = True)

#         #
#         # Adding process inputs

#         self.boxIn = self.addBBoxInput(      identifier   = "bbox",
#                                              title        = "Spatial region")

#         self.dateIn = self.addLiteralInput(  identifier   = "date",
#                                              title        = "The date to be calcualted",
#                                              type         = type(''))

#         #
#         # Adding process outputs

#         self.dataOut = self.addComplexOutput(identifier   = "map",
#                                              title        = "Output NDAI image",
#                                              useMapscript = True,
#                                              formats      = [{'mimeType':'image/tiff'}])

#     #
#     # Execution part of the process
#     def execute(self):

#         # Get the box value
#         BBOXObject = self.boxIn.getValue()
#         CoordTuple = BBOXObject.coords

#         #Get the date string
#         date = self.dateIn.getValue()

#         logging.info(CoordTuple)
#         logging.info(date)

#         #date='2013-06-30'
#         #spl_arr=[70,30,80,50]
#         spl_arr=[CoordTuple[0][0],CoordTuple[0][1],CoordTuple[1][0],CoordTuple[1][1]]

#         logging.info(date)
#         logging.info(spl_arr)

#         ndaifn=_NDAI_CAL(date,spl_arr)
#         self.dataOut.setValue( ndaifn )
#         #self.textOut.setValue( self.textIn.getValue() )
#         #os.remove(vcifn)
#         logging.info(os.getcwd())
#         return



# In[35]:

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

def setMap(rasterBase):

    # Read the data and metadata
    ds = gdal.Open(rasterBase)
    #band = ds.GetRasterBand(20)
    
    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    #proj = ds.GetProjection()
    
    nan = ds.GetRasterBand(1).GetNoDataValue()
    if nan != None:
        data = np.ma.masked_equal(data,value=nan)
    
    xres = gt[1]
    yres = gt[5]
    
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    
    x = ds.RasterXSize 
    y = ds.RasterYSize  
    extent = [ gt[0],gt[0]+x*gt[1], gt[3],gt[3]+y*gt[5]]
    #ds = None
    img_extent = (extent[0], extent[1], extent[2], extent[3])
    
    # create a grid of xy coordinates in the original projection
    #xy_source = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    
    return extent, img_extent#, xy_source, proj


# In[36]:

def MapThatShit(NDAI_BASE,date,spl_arr,drought_avg_tci_cmap):
    # get array informaiton
    extent, img_extent = setMap(NDAI_BASE)
    ds = gdal.Open(NDAI_BASE)
    array = ds.ReadAsArray()
    #array[ma.getmaskarray(NDAI)] = 0
    #nan = ds.GetRasterBand(1).GetNoDataValue()
    array = ma.masked_equal(array, 0)
    array = np.flipud(array)
    logging.info(extent)

    # set shape of figure
    width = array.shape[0]
    height = array.shape[1]
    base_width = 10.
    base_height = (base_width * height) / width
    print base_width, base_height
    logging.info(base_width, base_height)

    # figure
    fig = plt.figure(figsize=(base_height,base_width)) 
    ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, ccrs.Geodetic())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_right = False
    gl.xlabels_top = False
    # prulletaria on the map
    
    ogr2ogr = r'C:\Program Files\GDAL//ogr2ogr.exe'
    base_geom = r'D:\Data\ChinaShapefile\natural_earth'
    # ocean
    in_file_ocean = base_geom + '\physical//10m_ocean.shp'    
    outfile_ocean = 'ocean.shp'
    # country boudaries
    in_file_bound = base_geom + '\cultural//10m_admin_0_boundary_lines_land.shp' 
    outfile_bound = 'boundaries.shp'
    # costlie
    in_file_coast = base_geom + '\physical//10m_coastline.shp'
    outfile_coast = 'coastline.shp'
    #lakes
    in_file_lakes = base_geom + '\physical//10m_lakes.shp'
    outfile_lakes = 'lakes.shp'
    # run the clip functions
    
    command = [ogr2ogr, '-f', "ESRI Shapefile", outfile_ocean, in_file_ocean,'-clipsrc',
               str(spl_arr[0]),str(spl_arr[1]),str(spl_arr[2]),str(spl_arr[3]),'-overwrite']    
    print (sp.list2cmdline(command))
    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
    norm.communicate()
    
    command = [ogr2ogr, '-f', "ESRI Shapefile", outfile_bound, in_file_bound,'-clipsrc',
               str(spl_arr[0]),str(spl_arr[1]),str(spl_arr[2]),str(spl_arr[3]),'-overwrite']    
    print (sp.list2cmdline(command))
    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
    norm.communicate()    
    
    command = [ogr2ogr, '-f', "ESRI Shapefile", outfile_coast, in_file_coast,'-clipsrc',
               str(spl_arr[0]),str(spl_arr[1]),str(spl_arr[2]),str(spl_arr[3]),'-overwrite']    
    print (sp.list2cmdline(command))
    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
    norm.communicate()    
    
    command = [ogr2ogr, '-f', "ESRI Shapefile", outfile_lakes, in_file_lakes,'-clipsrc',
               str(spl_arr[0]),str(spl_arr[1]),str(spl_arr[2]),str(spl_arr[3]),'-overwrite']    
    print (sp.list2cmdline(command))
    norm = sp.Popen(sp.list2cmdline(command),stdout=sp.PIPE, shell=True)
    norm.communicate()        

    ax.add_geometries(Reader(outfile_ocean).geometries(), ccrs.PlateCarree(), facecolor='lightblue')
    ax.add_geometries(Reader(outfile_bound).geometries(), ccrs.PlateCarree(), 
                      facecolor='',linestyle=':', linewidth=2)
    ax.add_geometries(Reader(outfile_coast).geometries(), ccrs.PlateCarree(), facecolor='')
    ax.add_geometries(Reader(outfile_lakes).geometries(), ccrs.PlateCarree(), facecolor='lightskyblue')
    
    # ticks of classes
    bounds = [   0.   ,   82.875,   95.625,  108.375,  127.5  ,  146.625,
            159.375,  172.125,  255.   ]
    # ticklabels plus colorbar
    ticks = ['-1','-0.35','-0.25','-0.15','+0','+.15','+.25','+.35','+1']
    cmap = cmap_discretize(drought_avg_tci_cmap,8)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(array, origin='upper', extent=img_extent,norm=norm, cmap=cmap, vmin=0, vmax=255, interpolation='nearest')#, transform=ccrs.Mercator())
    title = 'NDAI '+date
    plt.title(title, fontsize=22)
    cb = plt.colorbar(im, fraction=0.0476, pad=0.04, ticks=bounds,norm=norm, orientation='horizontal')
    cb.set_label('Normalized Drought Anomaly Index')
    cb.set_ticklabels(ticks)
    
    spl_arr_str = str(spl_arr)
    spl_arr_str = spl_arr_str.replace('[','')
    spl_arr_str = spl_arr_str.replace(']','')
    spl_arr_str = spl_arr_str.replace(', ','#')
    spl_arr_str = spl_arr_str.replace('.','.')
    # and save the shit
    outpath = 'NDAI_'+date+'_'+spl_arr_str+'.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print outpath
    logging.info(outpath)
    #plt.tight_layout()
    plt.show()
    plt.close(fig)
    fig.clf() 
    return outpath


# In[37]:

#tci_cmap = make_colormap([c('#F29813'), c('#D8DC44'),0.2, c('#D8DC44'), c('#7EC5AD'),0.4, c('#7EC5AD'), c('#5786BE'),0.6, 
#                          c('#5786BE'), c('#41438D'),0.8, c('#41438D')])

drought_avg_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                                      c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                                      c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                                      c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                                      c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])


# In[38]:

date='2015-05-01'
#spl_arr=[73.2,-9.5,140.9,53.9]
spl_arr = [91.8,-14.3,164.4,13] #minlon, minlat, maxlon, maxlat
#spl_arr = [0.3,48.41,8.66,56.46]
#spl_arr = [-69.6,-12.8,-33.6,-0.2]
#spl_arr=[CoordTuple[0][0],CoordTuple[0][1],CoordTuple[1][0],CoordTuple[1][1]]


# In[39]:

NDAI_BASE, NDAI_255 =_NDAI_CAL(date,spl_arr)
#outpath = MapThatShit(NDAI_BASE,date,spl_arr,drought_avg_tci_cmap)


# In[24]:

full_url = 'http://192.168.1.104/wps/wpsoutputs/map-fd5a09d0-a221-11e5-883c-4437e647de9f.tif'
tif = urllib.urlretrieve(full_url, 'tif_test')


# In[41]:

ds = gdal.Open(NDAI_BASE)


# In[42]:

ds.GetMetadata()['TIFFTAG_DATETIME']


# In[28]:

setMap('tif_test')


# In[ ]:

spl_arr_str


# In[ ]:

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
fig = plt.imshow(NDAI, cmap=cmap_discretize(drought_avg_tci_cmap,8), vmin=-1, vmax=1, extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()


# In[ ]:

bounds = np.array([0,6,12,24,36,48,60,72,84,100])
(bounds * 255 / 100 ) 


# In[ ]:

import matplotlib
get_ipython().magic(u'matplotlib inline')
tci_cmap = make_colormap([c('#F29813'), c('#D8DC44'),0.2, c('#D8DC44'), c('#7EC5AD'),0.4, c('#7EC5AD'), c('#5786BE'),0.6, 
                          c('#5786BE'), c('#41438D'),0.8, c('#41438D')])

drought_avg_tci_cma2 = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                                      c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                                      c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                                      c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                                      c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])


# In[ ]:

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
fig = plt.imshow(NDAI_255, cmap=cmap_discretize(drought_avg_tci_cma2,8), vmin=0, vmax=255, extent=extent)#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()


# In[ ]:

plt.imshow(NDAI_255)


# In[ ]:


plt.hist(NDAI.compressed())


# In[ ]:

bounds = (np.array([-1,-0.35,-0.25,-0.15,0,.15,.25,.35,1])+1)*(255/2.)
#
bounds


# In[ ]:

bounds /= 2
bounds = (bounds * 255 / 100)
bounds


# In[ ]:

cb = cmap_discretize(drought_avg_tci_cma2,8)


# In[ ]:

plt.hist(NDAI_255.compressed())


# In[ ]:



