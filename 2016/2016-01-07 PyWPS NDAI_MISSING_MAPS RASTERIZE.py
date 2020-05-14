
# coding: utf-8

# In[1]:

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
sys.path.insert(0, r'D:\GitHub\pynotebook\ipymodules\GDMA')
import gdma


# In[2]:

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
start=datetime.fromtimestamp((start_date-(datetime(1970,1,1)-datetime(1601,1,1)).days)*24*60*60)
#print start


# In[3]:

try:    
    # get sample size coefficients from XML root
    sample_size = root[0][3][0][5][0][1].text #sample size
    #print root[0][3][0][5][0][1].text #sample size

    # use coverage start_date and sample_size array to create all dates in ANSI
    array_stepsize = np.fromstring(sample_size, dtype=int, sep=' ')
    #print np.fromstring(sample_size, dtype=int, sep=' ')
    array_all_ansi = array_stepsize + start_date  
    print 'irregular'
    print array_all_ansi
except IndexError:
    datelist, cur_pos = datelist_regular_coverage(root, start_date, start, cur_date)
    print 'regular'

# create array of all dates in ISO
list_all_dates = []
for stepsize in array_stepsize:
    date_and_stepsize = start + timedelta(stepsize - 1)
    list_all_dates.append(date_and_stepsize)
    #print date_and_stepsize
array_all_dates = np.array(list_all_dates)   

# create array of all dates in string
array_all_date_string = []
for i in array_all_dates:
    date_string = str(i.year).zfill(2)+'-'+str(i.month).zfill(2)+'-'+str(i.day).zfill(2)
    array_all_date_string.append(date_string)
array_all_date_string    


# In[ ]:

x = np.asarray(array_all_date_string)
start = np.where(x == '2006-07-12')[0][0]
end = start + 1
#for date in range(start,end):    
for date in array_all_date_string[start:end]:

    #spl_arr = [70,30,80,50]
    #extent = [73.5,140,14,53.6]
    extent = [-179, 179, -60, 90]
    spl_arr = [extent[0], extent[2], extent[1], extent[3]]
    ndai_wcs= gdma._NDAI_CAL(date, spl_arr)
    #ndai_wcs = 'NDAI20090626.tif'
    array = gdal.Open(ndai_wcs).ReadAsArray()
    #band = raster.GetRasterBand(1)
    #array = band.ReadAsArray()
    #band.GetNoDataValue()

    array_msk = np.ma.masked_equal(array,array.min())
    #plt.imshow(array_msk)    

    vector_path = r'D:\Data\WorldShapefile//world_simplified.shp'
    raster_path = ndai_wcs
    nodata_value = array.min()
    print 'nodata value NDAI: ', nodata_value
    # get date in format DOY+YEAR: eg. 0652011
    # NDAI_2014_008.tif

    year = int(ndai_wcs[-12:-8])
    month = int(ndai_wcs[-8:-6])
    day = int(ndai_wcs[-6:-4])
    date = datetime(year,month,day)
    try: 
        date_str = str(date.year)+str(date.month).zfill(2)+str(date.day).zfill(2)
        print date_str
    except:        
        print date, ' aaahh'    

    stats = gdma.zonal_stats(vector_path, raster_path, nodata_value, date)   

    df_stats = pd.DataFrame(stats)
    #df_stats.set_index('FID', inplace=True)
    #print df_stats.head(2)

    # read shapefile and concatate on index using a 'inner' join
    # meaning counties without statistics info will be ignored
    gdf = gpd.read_file(vector_path)
    gdf.index.rename('FID', inplace=True)
    gdf.reset_index(inplace=True)
    frames  = [df_stats,gdf]
    gdf_df_stats = gdf.merge(df_stats, on='FID')
    gdf_df_stats.set_index('FID', inplace=True)

    # get column names
    ax1_head = gdf_df_stats.columns[10] # P00082014
    ax2_head = gdf_df_stats.columns[11] # P10082014
    ax3_head = gdf_df_stats.columns[12] # P20082014
    ax4_head = gdf_df_stats.columns[13] # P30082014
    ax5_head = gdf_df_stats.columns[8]  # MEAN
    ax6_head = gdf_df_stats.columns[6]  # DC0082014
    print ax1_head, ax2_head, ax3_head, ax4_head, ax5_head, ax6_head
    columns_shp = [ax1_head, ax2_head, ax3_head, ax4_head, ax5_head, ax6_head]
    # drop NaN values for axis
    gdf_df_stats.dropna(inplace=True, subset=columns_shp)
    #gdf_df_stats.head(50)
    #gdf_df_stats = gpd.pd.concat(frames, axis=1, join='inner')
    #gdf_df_stats.index.rename('FID', inplace=True)
    #gdf_df_stats.geometry = gdf_df_stats.geometry.astype(gpd.geoseries.GeoSeries) # overcome bug 
    #gdf_df_stats.head(2) 

    # if necessary save to shapefile
    out_filename = r'D:\GitHub\pynotebook\ipymodules\GDMA//world_NDAI'+date_str+'.shp'
    gdf_df_stats.to_file(out_filename)  

    china_adm3 = out_filename
    china_adm3_shp = shapereader.Reader(china_adm3)
    # rasterize the data
    gdma.rasterize(date_str = date_str, in_shp = out_filename)
    gdma.plot_map(date_str = date_str, date=date, extent = extent)

