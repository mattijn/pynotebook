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
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib
%matplotlib inline
import datetime

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

#tci_cmap = make_colormap([c('#F29813'), c('#D8DC44'),0.2, c('#D8DC44'), c('#7EC5AD'),0.4, c('#7EC5AD'), c('#5786BE'),0.6, 
#                          c('#5786BE'), c('#41438D'),0.8, c('#41438D')])

# <codecell>

drought_cat_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                      c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#C4DC73')])

drought_per_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.2, c('#D95E0E'), c('#FE9829'),0.4, 
                                      c('#FE9829'), c('#FFD98E'),0.6, c('#FFD98E'), c('#FEFFD3'),0.8, c('#FEFFD3')])

drought_avg_tci_cmap = make_colormap([c('#993406'), c('#D95E0E'),0.1, c('#D95E0E'), c('#FE9829'),0.2, 
                                      c('#FE9829'), c('#FFD98E'),0.3, c('#FFD98E'), c('#FEFFD3'),0.4, 
                                      c('#FEFFD3'), c('#C4DC73'),0.5, c('#C4DC73'), c('#93C83D'),0.6,
                                      c('#93C83D'), c('#69BD45'),0.7, c('#69BD45'), c('#6ECCDD'),0.8,
                                      c('#6ECCDD'), c('#3553A4'),0.9, c('#3553A4')])

# <codecell>

#drought_cat_tci_cmap = make_colormap([c('#FEFFD3'), c('#FFD98E'),0.2, c('#FFD98E'), c('#FE9829'),0.4, 
#                                      c('#FE9829'), c('#D95E0E'),0.6, c('#D95E0E'), c('#993406'),0.8, c('#993406')])

# <codecell>

china_adm3 = r'D:\Downloads\Mattijn@Zhou\Drought_2011/County_2011_065_209.shp'
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

for i in list_classes[0:12]:
    if (i == 'ID_3') or (i == 'c_id'):
        print i        
    else:
        print i

# <codecell>

for i in list_classes[13:18]:
    if (i == 'ID_3') or (i == 'c_id'):
        print i        
    else:
        print i
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
        ax1 = fig.add_subplot(gs[0,0], projection=ccrs.AlbersEqualArea(central_longitude=100, central_latitude=15))
        ax1.set_extent(extent)
        ax1.coastlines(resolution='110m')

        gl1 = ax1.gridlines()
        gl1.xlocator = mticker.FixedLocator([50, 70,90,110,130,150,170])
        gl1.ylocator = mticker.FixedLocator([10,  20,  30,  40,  50, 60])
        gl1.xformatter = LONGITUDE_FORMATTER
        gl1.yformatter = LATITUDE_FORMATTER
        
        ax1.add_feature(cfeature.LAND, facecolor='0.85')      

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
        ax6.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax6.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')   
        linewidth=0.1
#         # classify each county based on column ID_3
        for record, county in zip(china_adm3_shp.records(), china_adm3_shp.geometries()): 
            
            # Ax1 -- Ax1 -- Ax1
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax1_head])
            # Classify the records in to groups
#             if ID == 0:
#                 facecolor = '#C4DC73'
#                 edgecolor = 'k'#'#FEFFD3'
#                 linewidth = 0.05
            if (ID >= .0) and (ID <= .25):
                facecolor = '#FEFFD3'
                edgecolor = '#FEFFD3'
            if (ID > .25) and (ID <= .5):
                facecolor = '#FFD98E'
                edgecolor = '#FFD98E'    
            if (ID > .5) and (ID <= .75):
                facecolor = '#D95E0E'
                edgecolor = '#D95E0E'
            if ID > .75:
                facecolor = '#993406'
                edgecolor = '#993406'
            ax1.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
            
            # Ax2 -- Ax2 -- Ax2            
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax2_head])
            # Classify the records in to groups
#             if ID == 0:
#                 facecolor = '#C4DC73'
#                 edgecolor = 'k'#'#FEFFD3'
#                 linewidth = 0.05
            if (ID >= .0) and (ID <= .25):
                facecolor = '#FEFFD3'
                edgecolor = '#FEFFD3'
            if (ID > .25) and (ID <= .5):
                facecolor = '#FFD98E'
                edgecolor = '#FFD98E'    
            if (ID > .5) and (ID <= .75):
                facecolor = '#D95E0E'
                edgecolor = '#D95E0E'
            if ID > .75:
                facecolor = '#993406'
                edgecolor = '#993406'
            ax2.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)  
            
            # Ax3 -- Ax3 -- Ax3                        
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax3_head])
            # Classify the records in to groups
#             if ID == 0:
#                 facecolor = '#C4DC73'
#                 edgecolor = 'k'#'#FEFFD3'
#                 linewidth = 0.05
            if (ID >= .0) and (ID <= .25):
                facecolor = '#FEFFD3'
                edgecolor = '#FEFFD3'
            if (ID > .25) and (ID <= .5):
                facecolor = '#FFD98E'
                edgecolor = '#FFD98E'    
            if (ID > .5) and (ID <= .75):
                facecolor = '#D95E0E'
                edgecolor = '#D95E0E'
            if ID > .75:
                facecolor = '#993406'
                edgecolor = '#993406'
            ax3.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth) 
            
            # Ax4 -- Ax4 -- Ax4
            # extract for each row the value corresponding to the column header             
            ID = float(record.attributes[ax4_head])
#             if ID == 0:
#                 facecolor = '#C4DC73'
#                 edgecolor = 'k'#'#FEFFD3'
#                 linewidth = 0.05
            if (ID >= .0) and (ID <= .25):
                facecolor = '#FEFFD3'
                edgecolor = '#FEFFD3'
            if (ID > .25) and (ID <= .5):
                facecolor = '#FFD98E'
                edgecolor = '#FFD98E'    
            if (ID > .5) and (ID <= .75):
                facecolor = '#D95E0E'
                edgecolor = '#D95E0E'
            if ID > .75:
                facecolor = '#993406'
                edgecolor = '#993406'
            ax4.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)

            # Ax5 -- Ax5 -- Ax5            
            # extract for each row the value corresponding to the column header 
            ID = float(record.attributes[ax5_head])
            # Classify the records in to groups
            if ID <= -0.35:
                facecolor = '#993406'
                edgecolor = '#993406'
            if (ID > -0.35) and (ID <= -0.25):
                facecolor = '#E26D15'
                edgecolor = '#E26D15'    
            if (ID > -0.25) and (ID <= -0.15):
                facecolor = '#FFB95C'
                edgecolor = '#FFB95C'
            if (ID > -0.15) and (ID <= 0):
                facecolor = '#FEF6C3'
                edgecolor = '#FEF6C3'
            if (ID > 0) and (ID <= 0.15):
                facecolor = '#A0CD4C'
                edgecolor = '#A0CD4C'
            if (ID > 0.15) and (ID <= 0.25):
                facecolor = '#6ABF5A'
                edgecolor = '#6ABF5A'    
            if (ID > 0.25) and (ID <= 0.35):
                facecolor = '#4C85BB'
                edgecolor = '#4C85BB'    
            if (ID > 0.35) and (ID <= 1):
                facecolor = '#3553A4'
                edgecolor = '#3553A4'                    
            ax5.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)            
            
            # Ax6 -- Ax6 -- Ax6             
            ID = int(record.attributes[ax6_head])
            # Classify the records in to groups
            if ID == 0:
                facecolor = '#C4DC73'
                edgecolor = 'k'#'#FEFFD3'
                linewidth = 0.05
            if ID == 1:
                facecolor = '#FEF6C3'
                edgecolor = '#FEF6C3'
            if ID == 2:
                facecolor = '#FFB95C'
                edgecolor = '#FFB95C'
            if ID == 3:
                facecolor = '#E26D15'
                edgecolor = '#E26D15'
            if ID == 4:
                facecolor = '#993406'
                edgecolor = '#993406'
            ax6.add_geometries([county], ccrs.PlateCarree(),facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)

        date = i[-7:]
        year = date[-4::]
        doy = date[-7:-4]
        date_out = datetime.datetime.strptime(str(year)+'-'+str(doy),'%Y-%j')
        date_label = 'Date: '+str(date_out.year) +'-'+str(date_out.month).zfill(2)+'-'+str(date_out.day).zfill(2)
        # ADD LABELS FOR EACH PLOT
        ax1.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())
        ax1.text(64, 51, 'Percentage of Slight Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic())        
        ax2.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())
        ax2.text(64, 51, 'Percentage of Moderate Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic())                
        ax3.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())
        ax3.text(64, 51, 'Percentage of Severe Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic())                
        ax4.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())
        ax4.text(64, 51, 'Percentage of Extreme Drought', weight='semibold', fontsize=12, transform=ccrs.Geodetic())                
        ax5.plot(116.4, 39.3, 'ks', markersize=5, transform=ccrs.Geodetic())        
        ax5.text(64, 51, 'Average of NDAI', weight='semibold', fontsize=12, transform=ccrs.Geodetic())                
        ax6.plot(116.4, 39.3, 'ks', markersize=7, transform=ccrs.Geodetic())
        ax6.text(64, 51, 'Drought Alert at County Level', fontsize=20, weight='semibold', color='k',transform=ccrs.Geodetic())
        ax6.text(65.5, 49, date_label, fontsize=20, weight='semibold', color='k',transform=ccrs.Geodetic())
        ax6.text(117, 40., 'Beijing', weight='semibold', transform=ccrs.Geodetic()) 
        
        # ADD LEGEND IN SOME PLOTS
        # -------------------------Ax 1
        cbax1 = fig.add_axes([0.328, 0.67, 0.011, 0.16])

        #cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
        cmap = cmap_discretize(drought_per_tci_cmap,6)
        cmap.set_over('0.25')
        cmap.set_under('0.75')

        # If a ListedColormap is used, the length of the bounds array must be
        # one greater than the length of the color list.  The bounds must be
        # monotonically increasing.
        bounds = [1, 2, 3, 4, 5]
        bounds_ticks = [1.5, 2.5, 3.5, 4.5]
        bounds_ticks_name = ['>75%', '50-75%', '25-50%', '<25%']
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(cbax1, cmap=cmap,
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
        cbax5 = fig.add_axes([0.85, 0.15, 0.011, 0.16])

        #cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
        cmap = cmap_discretize(drought_avg_tci_cmap,8)
        cmap.set_over('0.25')
        cmap.set_under('0.75')

        # If a ListedColormap is used, the length of the bounds array must be
        # one greater than the length of the color list.  The bounds must be
        # monotonically increasing.
        bounds = [1, 2, 3, 4, 5,6,7,8,9]
        bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5,6.6,7.5,8.5]
        bounds_ticks_name = [' ', '-0.35', ' ', '-0.15','0','0.15',' ','0.35',' ']
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(cbax5, cmap=cmap,
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
        cbax6 = fig.add_axes([0.79, 0.48, 0.020, 0.30])

        #cmap = mpl.colors.ListedColormap(['r', 'g', 'b', 'c'])
        cmap = cmap_discretize(drought_cat_tci_cmap,5)
        cmap.set_over('0.25')
        cmap.set_under('0.75')

        # If a ListedColormap is used, the length of the bounds array must be
        # one greater than the length of the color list.  The bounds must be
        # monotonically increasing.
        bounds = [1, 2, 3, 4, 5,6]
        bounds_ticks = [1.5, 2.5, 3.5, 4.5,5.5]
        bounds_ticks_name = ['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'Slight Drought', 'No Drought']
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(cbax6, cmap=cmap,
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
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax1.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax2.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
        ax3.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax3.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
        ax4.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax4.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')                
        ax5.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax5.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')        
        ax6.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor='black')
        ax6.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='black')                


        gs.update(wspace=0.03, hspace=0.03)
        path_out = r'D:\Downloads\Mattijn@Zhou\Drought_2009//'
        file_out = 'DroughtAlert_'+str(i[-7:])+'.png'
        filepath = path_out+file_out                              
        fig.savefig(filepath, dpi=200)
        #plt.show()        
        fig.clf()        
        plt.close()
        del record,county

# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


# <codecell>


