# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import geojson
import subprocess as sp
import json
import sys
import urllib
from osgeo import gdal
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# <codecell>

%matplotlib inline

# <codecell>

def getExtentCounty(province, prefecture, county, extent, ansidate, coverage):
    """
    extract extent of coverage using OGC WCS and cut to county shape using gdalwarp
    and OGC GeoPackage
    
    example: getExtentCounty("'Anhui'","'Bengbu'","'Guzhen'",
                             [117.04640962322,33.00404358318,117.59765626636,33.50222015793],
                             150842, 'NDAI_1km')  
    
    output is the filepath of the raster layer cliped using the vector layer
    """
    
    extent = [117.04640962322863,33.00404358318741,117.59765626636589,33.50222015793983] # left, bottom, right, top
    d = 150842
    endpoint='http://192.168.1.104:8080/rasdaman/ows'
    field={}
    field['SERVICE']='WCS'
    field['VERSION']='2.0.1'
    field['REQUEST']='GetCoverage'
    field['COVERAGEID']=coverage#'trmm_3b42_coverage_1'
    field['SUBSET']=['ansi('+str(d)+')',
                     'Lat('+str(extent[1])+','+str(extent[3])+')',
                    'Long('+str(extent[0])+','+str(extent[2])+')']
    field['FORMAT']='image/tiff'
    url_values = urllib.urlencode(field,doseq=True)
    full_url = endpoint + '?' + url_values
    print full_url
    wcsCoverage_filename='coverage'+str(d)+'.tif'
    f,h = urllib.urlretrieve(full_url,wcsCoverage_filename)
    print h 
    
    #path_base = "/home/rasdaman/Downloads"
    #CHN_adm_gpkg = os.path.join(path_base, "CHN_adm.gpkg")   
    
    #wcsCoverage_filename_clip = 'coverage'+str(d)+'clip.tif'    

    #command = ["/usr/bin/gdalwarp", "-cutline", CHN_adm_gpkg, "-csql", "SELECT NAME_3 FROM CHN_adm3 WHERE NAME_1 = "+province+" and NAME_2 = "+prefecture+" and NAME_3 = "+county+"",
    #       "-crop_to_cutline", "-of", "GTiff", "-dstnodata","-9999",wcsCoverage_filename, wcsCoverage_filename_clip, "-overwrite"] # 

    #print (sp.list2cmdline(command))

    #norm = sp.Popen(sp.list2cmdline(command), shell=True)  
    #norm.communicate() 
    
    return wcsCoverage_filename #wcsCoverage_filename_clip

# <codecell>

coverage = getExtentCounty("'Anhui'","'Bengbu'","'Guzhen'",
                             [117.04640962322,33.00404358318,117.59765626636,33.50222015793],
                             150842, 'NDAI_1km')

# <codecell>

test_ds = gdal.Open("D:\GitHub\pynotebook\ipynotebooks\Python2.7\test150842clip.tif")

# <codecell>

test_ds

# <codecell>

"http://192.168.1.104/wps/wpsoutputs/Bound_GeoJSON-3c6185ec-7d13-11e5-aa45-4437e647de9f.geojson"

