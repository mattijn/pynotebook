# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
from osgeo import gdal, gdalconst

# <codecell>

def reproject(baseRaster, pathIn, pathOut):
    """
    baseRaster = raster from which to get projection and geotransform info
    pathIn     = raster to reproject, resample and clip
    pathOut    = output raster path    
    """
    # step 1 Open raster that we want to project and resample
    src_filename = pathIn
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()
    print src_proj
    print src_geotrans
    
    # step 2 Open raster from which to get projection and geotransform, will be used for output file
    match_filename = baseRaster
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize
    print match_proj
    print match_geotrans
    
    # step 3 Output / destination save to tiff using 
    dst_filename = pathOut
    print dst_filename
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    print dst
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)
    
    # step 4 Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    
    del dst # Flush

# <codecell>

def FilesFolder(inGSODFolder, format_end=''):
    st_wmo = [os.path.join(root, name)
               for root, dirs, files in os.walk(inGSODFolder)
                 for name in files                 
                 if name.endswith(format_end)]
    return st_wmo

# <codecell>

path = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\dailytrmmoutput'
folderOut = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMClip_Day'
baseRaster = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMClip_Base\3B42_daily.2003.01.01.7.tif'
files = FilesFolder(path, format_end='.tif')
pathOut = folderOut + files[0][-28::]
pathOut

# <codecell>

for pathIn in files:    
    
    print ('pathin\n',pathIn)
    outFile = pathIn[-28::]
    pathOut = folderOut+outFile
    
    reproject(baseRaster, pathIn, pathOut)
    
    print ('pathout\n',pathOut)

# <codecell>

len('gaugemean_apr04.tif')

# <codecell>

path = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\GaugeMean_Month'
folderOut = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\GaugeMean_Month_Resampled'
files = FilesFolder(path, format_end='.tif')
pathOut = folderOut + files[0][-20::]
pathOut

# <codecell>

for pathIn in files:    
    
    print ('pathin\n',pathIn)
    outFile = pathIn[-20::]
    pathOut = folderOut+outFile
    
    reproject(baseRaster, pathIn, pathOut)
    
    print ('pathout\n',pathOut)

# <codecell>

path = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\GaugeStd_Month'
folderOut = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\GaugeStd_Month_Resampled'
baseRaster = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMClip_Base\3B42_daily.2003.01.01.7.tif'
files = FilesFolder(path, format_end='.tif')
pathOut = folderOut + files[0][-19::]
pathOut, files[0][-19::]

# <codecell>

files

# <codecell>

for pathIn in files:    
    
    #print ('pathin\n',pathIn)
    outFile = pathIn[-19::]
    pathOut = folderOut+outFile
    print pathOut +'\n'
    print baseRaster +'\n'
    print pathIn +'\n'
    reproject(baseRaster, pathIn, pathOut)
    
    #print ('pathout\n',pathOut)

# <codecell>

TRMM_LIST = FilesFolder(r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMClip_Day', format_end='.tif')

# <codecell>

jan = []
feb = []
mar = []
apr = []
may = []
jun = []
jul = []
aug = []
sep = []
ocT = []
nov = []
dec = []
for i in TRMM_LIST:
    #year = int(i[-16:-12])
    month = int(i[-11:-9])
    #day = int(i[-8:-6])
    #date = datetime(year,month,day)
    if month == 1:
        jan.append(i)
    elif month == 2:
        feb.append(i)
    elif month == 3:
        mar.append(i)
    elif month == 4:
        apr.append(i)        
    elif month == 5:
        may.append(i)
    elif month == 6:
        jun.append(i)
    elif month == 7:
        jul.append(i)
    elif month == 8:
        aug.append(i)
    elif month == 9:
        sep.append(i)
    elif month == 10:
        ocT.append(i)
    elif month == 11:
        nov.append(i)
    elif month == 12:
        dec.append(i)        

# <codecell>

nov

# <codecell>


# <codecell>


# <codecell>

dec_arr = np.zeros((len(dec),src.ReadAsArray().shape[0],src.ReadAsArray().shape[1]))

for idx, val in enumerate(dec):
    src = gdal.Open(val, gdalconst.GA_ReadOnly)
    dec_arr[idx] = src.ReadAsArray()
    
match_proj = src.GetProjection()
match_geotrans = src.GetGeoTransform()
wide = src.RasterXSize
high = src.RasterYSize

# step 3 Output / destination save to tiff using 
dst_filename = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMMean_Std//trmmstd_dec01.tif'
#dst_filename = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMMean_Month//trmmmean_dec01.tif'
dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
dst.SetGeoTransform( match_geotrans )
dst.SetProjection( match_proj)
dst.GetRasterBand(1).WriteArray(dec_arr.std(axis=0))
#dst.GetRasterBand(1).WriteArray(dec_arr.mean(axis=0))
dst.FlushCache()

dst = None

# step 3 Output / destination save to tiff using 
#dst_filename = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMMean_Std//trmmstd_dec01.tif'
dst_filename = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMMean_Month//trmmmean_dec01.tif'
dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
dst.SetGeoTransform( match_geotrans )
dst.SetProjection( match_proj)
#dst.GetRasterBand(1).WriteArray(dec_arr.std(axis=0))
dst.GetRasterBand(1).WriteArray(dec_arr.mean(axis=0))
dst.FlushCache()

# <codecell>

gauge_mean_month[0]

# <codecell>

import numpy as np

# <codecell>

gauge_mean_month = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\GaugeMean_Month_Resampled'
files_gauge_mean_month = FilesFolder(gauge_mean_month, format_end='.tif')
gauge_std_month = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\GaugeStd_Month_Resampled'
files_gauge_std_month = FilesFolder(gauge_std_month, format_end='.tif')
trmm_mean_month = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMMean_Month'
files_trmm_mean_month = FilesFolder(trmm_mean_month, format_end='.tif')
trmm_std_month = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMMean_Std'
files_trmm_std_month = FilesFolder(trmm_std_month, format_end='.tif')

# <codecell>

src = gdal.Open(files_gauge_mean_month[0], gdalconst.GA_ReadOnly)
match_proj = src.GetProjection()
match_geotrans = src.GetGeoTransform()
wide = src.RasterXSize
high = src.RasterYSize

# <codecell>

folderOut = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMcorr_day'

for i in TRMM_LIST:
    for j in files_trmm_mean_month:
        for k in files_gauge_std_month:
            for l in files_trmm_std_month:
                for m in files_gauge_mean_month:
                    if int(i[-11:-9]) == int(j[-6:-4]) & int(i[-11:-9]) == int(k[-6:-4]) & int(i[-11:-9]) == int(l[-6:-4]) & int(i[-11:-9]) == int(m[-6:-4]):
                        trmm_d = gdal.Open(i,gdalconst.GA_ReadOnly).ReadAsArray()
                        trmm_d = np.ma.masked_equal(trmm_d,0)
                        trmm_m = gdal.Open(j,gdalconst.GA_ReadOnly).ReadAsArray()
                        gaug_s = gdal.Open(k,gdalconst.GA_ReadOnly).ReadAsArray()
                        trmm_s = gdal.Open(l,gdalconst.GA_ReadOnly).ReadAsArray()
                        gaug_m = gdal.Open(k,gdalconst.GA_ReadOnly).ReadAsArray()
                        trmm_c = (trmm_d - trmm_m) * (gaug_s / trmm_s) + gaug_m
                        
                        folderOut = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMcorr_day'
                        pathOut = folderOut + i[-28:-4]+'_cor.tif'
                        dst = gdal.GetDriverByName('GTiff').Create(pathOut, wide, high, 1, gdalconst.GDT_Float32)
                        dst.SetGeoTransform( match_geotrans )
                        dst.SetProjection( match_proj)
                        dst.GetRasterBand(1).WriteArray(trmm_c)
                        dst.FlushCache()                        
                        dst = None
            

# <codecell>

folderOut = r'D:\Downloads\Mattijn@Shiva\2015-07-24 BiasCorrection TRMM\TRMMcorr_day'
pathOut = folderOut + TRMM_LIST[0][-28:-4]+'_cor.tif'
pathOut

# <codecell>

trmm_d

# <codecell>


