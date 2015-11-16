# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import json
import pandas as pd
from osgeo import ogr

# <markdowncell>

# Merge two geopackages of China and Taiwan
# Create a new merge geopackage using China gpkg
# 
#     `ogr2ogr -f GPKG D:\Data\ChinaShapefile\CHN_TWN_adm_gpkg\CHN_TWN_adm.gpkg D:\Data\ChinaShapefile\CHN_adm_gpkg\CHN_adm.gpkg`
# 
# And merge/append the Taiwan geopackage
# 
#     `ogr2ogr -f GPKG -update -append D:\Data\ChinaShapefile\CHN_TWN_adm_gpkg\CHN_TWN_adm.gpkg D:\Data\ChinaShapefile\TWN_adm_gpkg\TWN_adm.gpkg -nln merge`

# <codecell>

# load the merged geopackage and get layer 3 
geopackage = r'D:\Data\ChinaShapefile\CHN_adm_gpkg\CHN_adm.gpkg'
driver = ogr.GetDriverByName("GPKG")
dataSource = driver.Open(geopackage, 0)
layer = dataSource.GetLayer(3)

# <codecell>

# print the headers of the attribute table of this layer
layerDefinition = layer.GetLayerDefn()
for i in range(layerDefinition.GetFieldCount()):
    print layerDefinition.GetFieldDefn(i).GetName()

# <codecell>

province = []
region = []
county = []
for feature in layer:
    province.append( feature.GetField("NAME_1") )
    region.append( feature.GetField("NAME_2") )    
    county.append( feature.GetField("NAME_3") )    

# <codecell>

# job done, but also include the part that iterates over all this to create file used for the javascropt drop-down list

# <codecell>

df = pd.DataFrame([province,region,county]).T
df.columns = ['NAME_1', 'NAME_2','NAME_3']
df.head()

# <codecell>

df.sort('NAME_1', inplace=True)
df.set_index(['NAME_1','NAME_2','NAME_3'], inplace=True)

# <codecell>

df.head()

# <codecell>

provinces_list=[]
regions_list=[]
counties_list=[]

ix_chn = 0
provinces = np.unique(df.index.get_level_values('NAME_1').values)

province_code = str(ix_chn)
province_names = np.unique(df.index.get_level_values('NAME_1').values).tolist()

# append provinces to its list
provinces_list.append(province_code)
provinces_list.append(province_names)

for ix_pr, pr in enumerate(provinces):
    region_list = []
    #print ix_pr
    #print pr

    df_prov = df.xs(pr, level='NAME_1', drop_level=False)    
    regions = np.unique(df_prov.index.get_level_values('NAME_2').values)
    
    region_code = str(ix_chn)+'_'+str(ix_pr)
    region_name = np.unique(df_prov.index.get_level_values('NAME_2').values).tolist()
    
    region_list.append(region_code)
    region_list.append(region_name)
    
    regions_list.append(region_list)
    
    for ix_rg, rg in enumerate(regions):
        
        county_list = []
        #print ix_rg
        #print rg
        
        df_regions = df.xs(rg, level='NAME_2', drop_level=False)
        
        county_code = str(ix_chn)+'_'+str(ix_pr)+'_'+str(ix_rg)
        county_names = np.unique(df_regions.index.get_level_values('NAME_3').values).tolist()
        
        county_list.append(county_code)
        county_list.append(county_names)
        
        counties_list.append(county_list)
        #print county_list
print provinces_list

# <codecell>

with open(r'D:\GoogleChromeDownloads\MyWebSites\CHINA_DROPDOWN\js\counties_china_v03.txt', 'w') as thefile:
    for item in counties_list:
        thefile.write("%s\n" % json.dumps(item))
    for item in regions_list:
        thefile.write("%s\n" % json.dumps(item))
    for item in provinces_list:
        thefile.write("%s\n" % json.dumps(item))        

# <codecell>

# python done

# <markdowncell>

# In notepad++ update the last line, so the provinces are following same structure as the other lines. Then change the first character `[` into `this.add(`, do this by Replacing `^` (zero length match) with `this.add(` using `regular expression` in `search mode`. Next using `normal expression` search for `this.add[` and replace with `this.add(` and replace `]]` with `]);`. Now copy into the javascript file.

# <codecell>


# <codecell>


# <codecell>

def ExtractGeoJSON(NAME_1in, NAME_2in, NAME_3in):
    """
    Function to query OGC GeoPackage of China to extract county boundary as JSON
    input
    NAME_1in = Province/Shang (e.g: 'Anhui')
    NAME_2in = Regon/Shi (e.g: 'Bengbu')
    NAME_3in = County/Qian (e.g: 'Guzhen')
    
    output
    Boundary of County in GeoJSON format    
    """
    path_base = "D:\Data\ChinaShapefile\CHN_adm_gpkg"
    CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")
    CHN_adm_geojson = os.path.join(path_base, "CHN_adm_selection3.geojson")
    if os.path.exists(CHN_adm_geojson):
        os.remove(CHN_adm_geojson)
        print ('removed') 
    print (CHN_adm_geojson)
    
    command = ["ogr2ogr", "-f", "GeoJSON", CHN_adm_geojson, "-sql",
               "SELECT NAME_1, NAME_2, NAME_3 FROM CHN_adm3 WHERE NAME_1 = "+NAME_1in+" and NAME_2 = "+NAME_2in+" and NAME_3 = "+NAME_3in+"",
               CHN_adm_gpkg, "-s_srs", "EPSG:4326","-t_srs","EPSG:900913", "-skipfailures", "-nlt", "LINESTRING"]

    print (sp.list2cmdline(command))
    
    norm = sp.Popen(sp.list2cmdline(command), shell=True)  
    norm.communicate()     

    with open(CHN_adm_geojson) as f:
        geojson2ol = json.load(f)    
    return geojson2ol

# <codecell>

NAME_1in = "'Anhui'"
NAME_2in = "'Bengbu'"
NAME_3in = "'Guzhen'"
geoout = ExtractGeoJSON(NAME_1in, NAME_2in, NAME_3in)

# <codecell>


# <markdowncell>

# Get Coverage using extent and clip based on boundary

# <codecell>

import os
import geojson
import subprocess as sp
import json

import os
import sys
import urllib
from osgeo import gdal
import numpy
import numpy as np
import numpy.ma as ma
from lxml import etree
from datetime import datetime, timedelta
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# <codecell>

%matplotlib inline

# <codecell>

spl_arr=[75,25.5,103.75,39] # left, bottom, right, top
extent = [117.04640962322863,33.00404358318741,117.59765626636589,33.50222015793983]
d = 150842
endpoint='http://192.168.1.104:8080/rasdaman/ows'
field={}
field['SERVICE']='WCS'
field['VERSION']='2.0.1'
field['REQUEST']='GetCoverage'
field['COVERAGEID']='NDAI_1km'#'trmm_3b42_coverage_1'
field['SUBSET']=['ansi('+str(d)+')',
                 'Lat('+str(extent[1])+','+str(extent[3])+')',
                'Long('+str(extent[0])+','+str(extent[2])+')']
field['FORMAT']='image/tiff'
url_values = urllib.urlencode(field,doseq=True)
full_url = endpoint + '?' + url_values
print full_url
tmpfilename='test'+str(d)+'.tif'
f,h = urllib.urlretrieve(full_url,tmpfilename)
print h

ds=gdal.Open(tmpfilename)

# <codecell>

ds=gdal.Open(tmpfilename)
tmpfilename

# <codecell>

#polygon = r'D:\GitHub\pynotebook\ipynotebooks\Python2.7//polygon.geojson'
#output_tif = r'D:\GitHub\pynotebook\ipynotebooks\Python2.7//output.tif'

# <codecell>

#command = ["gdalwarp", "-dstnodata", "-9999", "-co", "COMPRESS=DEFLATE", "-of", "GTiff", "-r", "near", 
#           "-crop_to_cutline", "-cutline", polygon, tmpfilename, output_tif]
#print (sp.list2cmdline(command))

# <codecell>

#norm = sp.Popen(sp.list2cmdline(command), shell=True)  
#norm.communicate() 

# <codecell>

clippedfilename='test'+str(d)+'clip.tif'
clippedfilename

# <codecell>

path_base = "D:\Data\ChinaShapefile\CHN_adm_gpkg"
CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")
CHN_adm_gpkg

# <codecell>

NAME_1in = "'Anhui'"
NAME_2in = "'Bengbu'"
NAME_3in = "'Guzhen'"

# <codecell>

command = ["gdalwarp", "-cutline", CHN_adm_gpkg, "-csql", "SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = "+NAME_1in+" and NAME_2 = "+NAME_2in+" and NAME_3 = "+NAME_3in+"",
           "-crop_to_cutline", "-of", "GTiff", "-dstnodata","-9999",tmpfilename, clippedfilename, "-overwrite"] # 

print (sp.list2cmdline(command))

norm = sp.Popen(sp.list2cmdline(command), shell=True)  
norm.communicate() 

# <codecell>

ds=gdal.Open(clippedfilename)
ds_clip = ds.ReadAsArray()
ds_clip

# <codecell>

ds_clip_ma = np.ma.masked_equal(ds_clip, -9999)
ds_clip_ma

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
fig = plt.imshow(ds_clip_ma, extent=extent, interpolation='nearest')#vmin=-0.4, vmax=0.4
plt.colorbar(fig)
plt.axis('off')
#plt.colorbar()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()

# <codecell>

ds_hist_data = ds_clip[ds_clip != -9999]
y,binEdges=np.histogram(ds_hist_data,bins=100, range=(-1,1), normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-')
plt.show()

# <codecell>


# <codecell>

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier = "WPS_HISTOGRAM",
            title="Histogram computation based on County ",
            abstract="""Module to compute Histograms of numerous NDAI observations""",
            version = "1.0",
            storeSupported = True,
            statusSupported = True)

        ##
        # Adding process inputs
        self.NAME_1 = self.addLiteralInput(identifier="Province",
                    title="Chinese Province",
                    type=type(''))

        self.NAME_2 = self.addLiteralInput(identifier="Prefecture",
                    title="Chinese Prefecture",
                    type=type(''))

        self.NAME_3 = self.addLiteralInput(identifier="County",
                    title = "Chinese County",
                    type=type(''))

        self.bboxCounty = self.addLiteralInput(identifier="ExtentCounty",
                    title = "The Extent of the web-based selected County",
                    type=type(''))   
        
        self.date = self.addLiteralInput(identifier="date",
                    title="The selected date of interest",
                    type=type(''))

        self.no_observations = self.addLiteralInput(identifier="num_observations",
                    title="The number of succeeding observations to consider (output is num_observations+1)",
                    type=type(''))        

        ##
        # Adding process outputs

        self.hist_ts1 = self.addComplexOutput(identifier  = "hist_ts1", 
                                              title       = "Histogram of the first observations",
                                              formats     = [{'mimeType':'text/xml'}]) 
        
        self.hist_ts2 = self.addComplexOutput(identifier  = "hist_ts2", 
                                              title       = "Histogram of the first observations",
                                              formats     = [{'mimeType':'text/xml'}]) 
        
        self.hist_ts3 = self.addComplexOutput(identifier  = "hist_ts3", 
                                              title       = "Histogram of the first observations",
                                              formats     = [{'mimeType':'text/xml'}]) 
        
        self.hist_ts4 = self.addComplexOutput(identifier  = "hist_ts4", 
                                              title       = "Histogram of the first observations",
                                              formats     = [{'mimeType':'text/xml'}]) 
        
        self.hist_ts5 = self.addComplexOutput(identifier  = "hist_ts5", 
                                              title       = "Histogram of the first observations",
                                              formats     = [{'mimeType':'text/xml'}])         


    ##
    # Execution part of the process
    def execute(self):
        # Load the data
        NAME_1 = str(self.NAME_1.getValue())
        NAME_2 = str(self.NAME_2.getValue())
        NAME_3 = str(self.NAME_3.getValue())                
        
        extent = list(self.bboxCounty.getValue())
        date = str(self.date.getValue())        
        no_observations = int(self.no_observations.getValue())
        
        # Do the Work
        # do something
        
        # Save to out
        self.hist_ts1.setValue( hist_ts1 )
        self.hist_ts2.setValue( hist_ts2 )
        self.hist_ts3.setValue( hist_ts3 )
        self.hist_ts4.setValue( hist_ts4 )
        self.hist_ts5.setValue( hist_ts5 )        
        return

# <codecell>

from datetime import datetime, timedelta

# <codecell>

NAME_1 = "Anhui"
NAME_2 = "Bengbu"
NAME_3 = "Guzhen"
extent = [117.04640962322863,33.00404358318741,117.59765626636589,33.50222015793983] # left, bottom, right, top
date = "2014-01-01"
no_observations = 4

# <codecell>

j.toordinal()

# <codecell>

# convert all required dates in ISO date format
date_start = datetime(int(date[0:4]),int(date[5:7]),int(date[8:10]))
date_list = []
date_list.append(date_start)
for i in range(1,no_observations+1):
    #print i
    date_list.append(date_start + (i *timedelta(days=8)))

# request data use WCS service baed on extend and clip based on sql query
array_NDAI = []
endpoint='http://192.168.1.104:8080/rasdaman/ows'
for j in date_list:
    #d = 150842
    date_in_string = '"'+str(j.year)+'-'+str(j.month).zfill(2)+'-'+str(j.day).zfill(2)+'"'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='GetCoverage'
    field['COVERAGEID']='NDAI_1km'#'trmm_3b42_coverage_1'
    field['SUBSET']=['ansi('+date_in_string+')',#['ansi('+str(d)+')',
                     'Lat('+str(extent[1])+','+str(extent[3])+')',
                    'Long('+str(extent[0])+','+str(extent[2])+')']
    field['FORMAT']='image/tiff'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    print full_url
    tmpfilename='test'+str(j.toordinal())+'.tif'
    f,h = urllib.urlretrieve(full_url,tmpfilename)
    print h

    #ds=gdal.Open(tmpfilename)
    clippedfilename='test'+str(j.toordinal())+'clip.tif' 

    path_base = "D:\Data\ChinaShapefile\CHN_adm_gpkg"
    CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")
    
    command = ["gdalwarp", "-cutline", CHN_adm_gpkg, "-csql", "SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = "+NAME_1+" and NAME_2 = "+NAME_2+" and NAME_3 = "+NAME_3+"",
               "-crop_to_cutline", "-of", "GTiff", "-dstnodata","-9999",tmpfilename, clippedfilename, "-overwrite"] # 

    print (sp.list2cmdline(command))

    norm = sp.Popen(sp.list2cmdline(command), shell=True)  
    norm.communicate()   

    ds=gdal.Open(clippedfilename)
    ds_clip = ds.ReadAsArray() 
    
    array_NDAI.append(ds_clip)
array_NDAI = np.asarray(array_NDAI)
array_NDAI_ma = np.ma.masked_equal(array_NDAI, -9999)    

# <codecell>

y_list=[]
bincenters_list=[]
for k in range(0,no_observations):
    ds_hist_data = array_NDAI[k][array_NDAI[k] != -9999]
    y,binEdges=np.histogram(ds_hist_data,bins=100, range=(-1,1), normed=True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    y_list.append(y)
    bincenters_list.append(bincenters)

# <codecell>

y_list[0]

# <codecell>

for l in range(0,no_observations):
    plt.plot(bincenters_list[l],y_list[l])
plt.show()

# <codecell>


# <codecell>

ds_hist_data = array_NDAI[0][array_NDAI[0] != -9999]

# <codecell>

ds_hist_data

# <codecell>


# <codecell>

ds_hist_data = array_NDAI[ds_clip != -9999]
y,binEdges=np.histogram(ds_hist_data,bins=100, range=(-1,1), normed=True)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-')
plt.show()

# <codecell>


# <codecell>

array_NDAI.shape

# <codecell>

date_start+8

# <codecell>


