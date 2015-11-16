# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.ticker as mticker
from matplotlib import gridspec
from cartopy.io import shapereader
import shapely.geometry as sgeom
import numpy as np
%matplotlib inline

# <codecell>


# <codecell>

china_adm3 = r'D:\Downloads\ZhouJie@Mattijn\20151029_zs\NDAI_2014_2.shp'
china_adm3_shp = shapereader.Reader(china_adm3)

# <codecell>

records = china_adm3_shp.records()
fields = next(records)

# <codecell>

list_classes = sorted(fields.attributes.keys())

# <codecell>

list_classes

# <codecell>

extent = [111.91693268, 123.85693268, 49.43324112, 40.67324112]
extent = [73.5,140,14,53.6]

# <codecell>

for i in list_classes[0:1]:
    if (i == 'ID_3') or (i == 'c_id'):
        print i        
    else:
        #print i
        ax1_head = 'P1'+str(i[-7:])
        ax2_head = 'P2'+str(i[-7:])
        ax3_head = 'P3'+str(i[-7:])        
        ax4_head = 'P4'+str(i[-7:])
        ax5_head = 'N'+str(i[-7:])
        ax6_head = 'C'+str(i[-7:])        
        print ax1_head, ax2_head, ax3_head, ax4_head,ax5_head,ax6_head


        fig = plt.figure(figsize=(20,13))
        gs = gridspec.GridSpec(3, 3)

        #############--------------################-------------#############--------------################-------------

        # PLOT TOP LEFT
        ax1 = fig.add_subplot(gs[0,0], projection=ccrs.AlbersEqualArea(central_longitude=105, central_latitude=15))
        ax1.set_extent(extent)
        ax1.coastlines(resolution='110m')

        gl1 = ax1.gridlines()
        gl1.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl1.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl1.xformatter = LONGITUDE_FORMATTER
        gl1.yformatter = LATITUDE_FORMATTER

        ax1.add_feature(cfeature.LAND, facecolor='0.85')

        # classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax1_head])
            # Classify the records in to groups
            if ID <= .25:
                facecolor = 'k'
                edgecolor = 'k'
            if (ID > .25) and (ID <= .5):
                facecolor = 'b'
                edgecolor = 'b'        
            if (ID > .5) and (ID <= .75):
                facecolor = 'm'
                edgecolor = 'm'         
            if ID > .75:
                facecolor = 'g' 
                edgecolor = 'g' 
            ax1.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

        ax1.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())


        # PLOT MIDDLE LEFT
        ax2 = fig.add_subplot(gs[1,0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
        ax2.set_extent(extent)
        ax2.coastlines(resolution='110m')

        gl2 = ax2.gridlines()
        gl2.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl2.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl2.xformatter = LONGITUDE_FORMATTER
        gl2.yformatter = LATITUDE_FORMATTER

        ax2.add_feature(cfeature.LAND, facecolor='0.85')

        # classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax2_head])
            # Classify the records in to groups
            if ID <= .25:
                facecolor = 'k'
                edgecolor = 'k'
            if (ID > .25) and (ID <= .5):
                facecolor = 'b'
                edgecolor = 'b'        
            if (ID > .5) and (ID <= .75):
                facecolor = 'm'
                edgecolor = 'm'         
            if ID > .75:
                facecolor = 'g' 
                edgecolor = 'g'      
            ax2.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

        ax2.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())



        #############--------------################-------------#############--------------################-------------

        # PLOT BOTTOM LEFT
        ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
        ax3.set_extent(extent)
        ax3.coastlines(resolution='110m')

        gl3 = ax3.gridlines()
        gl3.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl3.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl3.xformatter = LONGITUDE_FORMATTER
        gl3.yformatter = LATITUDE_FORMATTER

        ax3.add_feature(cfeature.LAND, facecolor='0.85')

        # classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax3_head])
            # Classify the records in to groups
            if ID <= .25:
                facecolor = 'k'
                edgecolor = 'k'
            if (ID > .25) and (ID <= .5):
                facecolor = 'b'
                edgecolor = 'b'        
            if (ID > .5) and (ID <= .75):
                facecolor = 'm'
                edgecolor = 'm'         
            if ID > .75:
                facecolor = 'g' 
                edgecolor = 'g'       
            ax3.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

        ax3.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())  


        #############--------------################-------------#############--------------################-------------

        # PLOT BOTTOM MIDDLE
        ax4 = fig.add_subplot(gs[2,1], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
        ax4.set_extent(extent)
        ax4.coastlines(resolution='110m')

        gl4 = ax4.gridlines()
        gl4.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl4.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl4.xformatter = LONGITUDE_FORMATTER
        gl4.yformatter = LATITUDE_FORMATTER

        ax4.add_feature(cfeature.LAND, facecolor='0.85')

        # # classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax4_head])
            if ID <= .25:
                facecolor = 'k'
                edgecolor = 'k'
            if (ID > .25) and (ID <= .5):
                facecolor = 'b'
                edgecolor = 'b'        
            if (ID > .5) and (ID <= .75):
                facecolor = 'm'
                edgecolor = 'm'         
            if ID > .75:
                facecolor = 'g' 
                edgecolor = 'g'     
            ax4.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

        ax4.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())

        #bound = r'D:\MicrosoftEdgeDownloads\Ecoregions_EastAsia//ea_clip.shp'
        #shape_bound = cfeature.ShapelyFeature(Reader(bound).geometries(), ccrs.PlateCarree(), facecolor='b')
        #ax4.add_feature(shape_bound, linewidth='1.0', alpha='1.0')

        #############--------------################-------------#############--------------################-------------

        # PLOT BOTTOM RIGHT
        ax5 = fig.add_subplot(gs[2,2], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
        ax5.set_extent(extent)
        ax5.coastlines(resolution='110m')

        gl5 = ax5.gridlines()
        gl5.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl5.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl5.xformatter = LONGITUDE_FORMATTER
        gl5.yformatter = LATITUDE_FORMATTER
        ax5.add_feature(cfeature.LAND, facecolor='0.85')

        #classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax5_head])
            # Classify the records in to groups
            if ID <= -0.35:
                facecolor = 'k'
                edgecolor = 'k'
            if (ID > -0.35) and (ID <= -0.25):
                facecolor = 'b'
                edgecolor = 'b'        
            if (ID > -0.25) and (ID <= -0.15):
                facecolor = 'm'
                edgecolor = 'm'        
            if (ID > -0.15) and (ID <= 0):
                facecolor = 'r'    
                edgecolor = 'r'                            
            if ID > 0:
                facecolor = 'g' 
                edgecolor = 'g'        
            ax5.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

        ax5.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())


        #############--------------################-------------#############--------------################-------------

        # PLOT CENTER
        ax6 = fig.add_subplot(gs[0:2,1:3], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
        ax6.set_extent(extent)
        ax6.coastlines(resolution='110m')

        gl6 = ax6.gridlines()
        gl6.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl6.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl6.xformatter = LONGITUDE_FORMATTER
        gl6.yformatter = LATITUDE_FORMATTER

        ax6.add_feature(cfeature.LAND, facecolor='0.85')
        # classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            # extract for each row the value corresponding to the column header 
            ID = int(record.attributes[ax6_head])
            # Classify the records in to groups
            if ID == 0:
                facecolor = 'k'
                edgecolor = 'k'
            if ID == 1:
                facecolor = 'b'
                edgecolor = 'b'        
            if ID == 2:
                facecolor = 'm'
                edgecolor = 'm'        
            if ID == 3:
                facecolor = 'r'    
                edgecolor = 'r'        
            if ID == 4:
                facecolor = 'g' 
                edgecolor = 'g'        
            ax6.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

        ax6.plot(116.4, 39.3, 'ks', markersize=7, transform=ccrs.Geodetic())
        ax6.text(117, 40., 'Beijing', weight='semibold', transform=ccrs.Geodetic())    

        gs.update(wspace=0.03, hspace=0.03)
        path_out = r'D:\Downloads\ZhouJie@Mattijn\20151029_zs//'
        file_out = 'test'+str(i[-7:])+'.png'
        filepath = path_out+file_out                              
        fig.savefig(filepath, dpi=100)

# <codecell>


# <codecell>

fig = plt.figure(figsize=(20,13))
gs = gridspec.GridSpec(3, 3)

#############--------------################-------------#############--------------################-------------

# PLOT TOP LEFT
ax1 = fig.add_subplot(gs[0,0], projection=ccrs.AlbersEqualArea(central_longitude=105, central_latitude=15))
ax1.set_extent(extent)
ax1.coastlines(resolution='110m')

gl1 = ax1.gridlines()
gl1.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
gl1.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
gl1.xformatter = LONGITUDE_FORMATTER
gl1.yformatter = LATITUDE_FORMATTER

ax1.add_feature(cfeature.LAND, facecolor='0.85')

# # classify each county based on column ID_3
# for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
#     # extract for each row the value corresponding to the column header 
#     ID = record.attributes['ID_3']    
#     # Classify the records in to groups
#     if ID <= 500:
#         facecolor = 'k'
#         edgecolor = 'k'
#     if (ID > 500) and (x <= 1000):
#         facecolor = 'b'
#         edgecolor = 'b'        
#     if (ID > 1000) and (x <= 1500):
#         facecolor = 'm'
#         edgecolor = 'm'        
#     if (ID > 1500) and (x <= 2000):
#         facecolor = 'r'    
#         edgecolor = 'r'        
#     if ID > 2000:
#         facecolor = 'g' 
#         edgecolor = 'g'        
#     ax1.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

ax1.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())

    
# PLOT MIDDLE LEFT
ax2 = fig.add_subplot(gs[1,0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
ax2.set_extent(extent)
ax2.coastlines(resolution='110m')

gl2 = ax2.gridlines()
gl2.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
gl2.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
gl2.xformatter = LONGITUDE_FORMATTER
gl2.yformatter = LATITUDE_FORMATTER

ax2.add_feature(cfeature.LAND, facecolor='0.85')

# # classify each county based on column ID_3
# for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
#     # extract for each row the value corresponding to the column header 
#     ID = record.attributes['ID_3']    
#     # Classify the records in to groups
#     if ID <= 500:
#         facecolor = 'k'
#         edgecolor = 'k'
#     if (ID > 500) and (x <= 1000):
#         facecolor = 'b'
#         edgecolor = 'b'        
#     if (ID > 1000) and (x <= 1500):
#         facecolor = 'm'
#         edgecolor = 'm'        
#     if (ID > 1500) and (x <= 2000):
#         facecolor = 'r'    
#         edgecolor = 'r'        
#     if ID > 2000:
#         facecolor = 'g' 
#         edgecolor = 'g'        
#     ax2.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

ax2.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())



#############--------------################-------------#############--------------################-------------

# PLOT BOTTOM LEFT
ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
ax3.set_extent(extent)
ax3.coastlines(resolution='110m')

gl3 = ax3.gridlines()
gl3.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
gl3.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
gl3.xformatter = LONGITUDE_FORMATTER
gl3.yformatter = LATITUDE_FORMATTER

ax3.add_feature(cfeature.LAND, facecolor='0.85')

# classify each county based on column ID_3
# for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
#     # extract for each row the value corresponding to the column header 
#     ID = record.attributes['ID_3']    
#     # Classify the records in to groups
#     if ID <= 500:
#         facecolor = 'k'
#         edgecolor = 'k'
#     if (ID > 500) and (x <= 1000):
#         facecolor = 'b'
#         edgecolor = 'b'        
#     if (ID > 1000) and (x <= 1500):
#         facecolor = 'm'
#         edgecolor = 'm'        
#     if (ID > 1500) and (x <= 2000):
#         facecolor = 'r'    
#         edgecolor = 'r'        
#     if ID > 2000:
#         facecolor = 'g' 
#         edgecolor = 'g'        
#     ax3.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)
    
ax3.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())  


#############--------------################-------------#############--------------################-------------

# PLOT BOTTOM MIDDLE
ax4 = fig.add_subplot(gs[2,1], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
ax4.set_extent(extent)
ax4.coastlines(resolution='110m')

gl4 = ax4.gridlines()
gl4.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
gl4.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
gl4.xformatter = LONGITUDE_FORMATTER
gl4.yformatter = LATITUDE_FORMATTER

ax4.add_feature(cfeature.LAND, facecolor='0.85')

# # classify each county based on column ID_3
# for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
#     # extract for each row the value corresponding to the column header 
#     ID = record.attributes['ID_3']    
#     # Classify the records in to groups
#     if ID <= 500:
#         facecolor = 'k'
#         edgecolor = 'k'
#     if (ID > 500) and (x <= 1000):
#         facecolor = 'b'
#         edgecolor = 'b'        
#     if (ID > 1000) and (x <= 1500):
#         facecolor = 'm'
#         edgecolor = 'm'        
#     if (ID > 1500) and (x <= 2000):
#         facecolor = 'r'    
#         edgecolor = 'r'        
#     if ID > 2000:
#         facecolor = 'g' 
#         edgecolor = 'g'        
#     ax4.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)
    
ax4.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())

#bound = r'D:\MicrosoftEdgeDownloads\Ecoregions_EastAsia//ea_clip.shp'
#shape_bound = cfeature.ShapelyFeature(Reader(bound).geometries(), ccrs.PlateCarree(), facecolor='b')
#ax4.add_feature(shape_bound, linewidth='1.0', alpha='1.0')

#############--------------################-------------#############--------------################-------------

# PLOT BOTTOM RIGHT
ax5 = fig.add_subplot(gs[2,2], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
ax5.set_extent(extent)
ax5.coastlines(resolution='110m')

gl5 = ax5.gridlines()
gl5.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
gl5.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
gl5.xformatter = LONGITUDE_FORMATTER
gl5.yformatter = LATITUDE_FORMATTER
ax5.add_feature(cfeature.LAND, facecolor='0.85')

# classify each county based on column ID_3
# for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
#     # extract for each row the value corresponding to the column header 
#     ID = record.attributes['ID_3']    
#     # Classify the records in to groups
#     if ID <= 500:
#         facecolor = 'k'
#         edgecolor = 'k'
#     if (ID > 500) and (x <= 1000):
#         facecolor = 'b'
#         edgecolor = 'b'        
#     if (ID > 1000) and (x <= 1500):
#         facecolor = 'm'
#         edgecolor = 'm'        
#     if (ID > 1500) and (x <= 2000):
#         facecolor = 'r'    
#         edgecolor = 'r'        
#     if ID > 2000:
#         facecolor = 'g' 
#         edgecolor = 'g'        
#     ax5.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)

ax5.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())


#############--------------################-------------#############--------------################-------------

# PLOT CENTER
ax6 = fig.add_subplot(gs[0:2,1:3], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
ax6.set_extent(extent)
ax6.coastlines(resolution='110m')

gl6 = ax6.gridlines()
gl6.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
gl6.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
gl6.xformatter = LONGITUDE_FORMATTER
gl6.yformatter = LATITUDE_FORMATTER

ax6.add_feature(cfeature.LAND, facecolor='0.85')
# classify each county based on column ID_3
# for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
#     # extract for each row the value corresponding to the column header 
#     ID = record.attributes['ID_3']    
#     # Classify the records in to groups
#     if ID <= 500:
#         facecolor = 'k'
#         edgecolor = 'k'
#     if (ID > 500) and (x <= 1000):
#         facecolor = 'b'
#         edgecolor = 'b'        
#     if (ID > 1000) and (x <= 1500):
#         facecolor = 'm'
#         edgecolor = 'm'        
#     if (ID > 1500) and (x <= 2000):
#         facecolor = 'r'    
#         edgecolor = 'r'        
#     if ID > 2000:
#         facecolor = 'g' 
#         edgecolor = 'g'        
#     ax6.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor)
    
ax6.plot(116.4, 39.3, 'ks', markersize=7, transform=ccrs.Geodetic())
ax6.text(117, 40., 'Beijing', weight='semibold', transform=ccrs.Geodetic())    

gs.update(wspace=0.03, hspace=0.03)
fig.savefig(r'D:\Downloads\ZhouJie@Mattijn\20151028_zs/test.png', dpi=100)

# <codecell>


# <codecell>

for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
    # extract for each row the value corresponding to the column header 
    ID = record.attributes['ID_3']    
    # Classify the records in to groups
    if ID <= 500:
        facecolor = 'k'
    if (ID > 500) and (x <= 1000):
        facecolor = 'b'
    if (ID > 1000) and (x <= 1500):
        facecolor = 'm'
    if (ID > 1500) and (x <= 2000):
        facecolor = 'r'    
    if ID > 2000:
        facecolor = 'g' 
    ax5.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor)        

# <codecell>


# <codecell>


       

# <codecell>

from cartopy.io import shapereader

kw = dict(resolution='50m', category='cultural',
          name='admin_1_states_provinces')

states_shp = shapereader.natural_earth(**kw)
shp = shapereader.Reader(states_shp)

# <codecell>

states_shp

# <codecell>

from __future__ import unicode_literals

states = ('Minas Gerais', 'Mato Grosso', 'Goiás',
          'Bahia', 'Rio Grande do Sul', 'São Paulo')

# <codecell>

import cartopy.crs as ccrs
import matplotlib.pyplot as plt


subplot_kw = dict(projection=ccrs.PlateCarree())

fig, ax = plt.subplots(figsize=(7, 11),
                       subplot_kw=subplot_kw)
ax.set_extent([-82, -32, -45, 10])

ax.background_patch.set_visible(False)
ax.outline_patch.set_visible(False)

for record, state in zip(shp.records(), shp.geometries()):    
    name = record.attributes['name'].decode('latin-1')
    print name
    if name in states:
        facecolor = 'DarkOrange'
    else:
        facecolor = 'LightGray'
    ax.add_geometries([state], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor='black')

# <codecell>


