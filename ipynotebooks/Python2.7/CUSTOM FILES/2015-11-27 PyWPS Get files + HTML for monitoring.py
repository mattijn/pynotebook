# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pywps.Process import WPSProcess 
from os import walk
import logging
from cStringIO import StringIO
from datetime import datetime, timedelta

# <codecell>

def listall(RootFolder, varname='',extension='.png'):
    lists = [os.path.join(root, name)
             for root, dirs, files in os.walk(RootFolder)
             for name in files
             if varname in name
             if name.endswith(extension)]
    return lists

# <codecell>

def getHTML():
    listHTML = []
    path_to_maps = r'D:\MicrosoftEdgeDownloads\drought_monitoring\drought_monitoring\img_small'
    _, _, filenames = next(walk(path_to_maps), (None, None, []))
    #filenames = listall(path_to_maps)
    for file_path in filenames:
        #print file_path
        year = int(file_path[-8:-4])
        days = int(file_path[-11:-8])        
        file_date = datetime(year, 1, 1) + timedelta(days - 1)
        file_date = str(file_date.year)+'-'+str(file_date.month)+'-'+str(file_date.day)
        listHTML.append('<div class="rsContent"><a class="rsImg" href="img_small/'+file_path+
                        '"></a><i class="rsTmb">'+file_date+'</i></div>')
    strHTML = ''.join(listHTML)
    output = StringIO()
    output.write(strHTML)
    id_ = len(filenames) - 1
    return output, id_

# <codecell>

class Process(WPSProcess):


    def __init__(self):

        ##
        # Process initialization
        WPSProcess.__init__(self,
            identifier      = "WPS_GETHTML",
            title           = "Get HTML to be ingested into the drought monitoring viewer",
            abstract        = "Module to get list of all images as HTML",
            version         = "1.0",
            storeSupported  = True,
            statusSupported = True)

        ##
#        # Adding process inputs
#        self.process_input = self.addLiteralInput(identifier="input",
#                                                  title="Chinese Province",
#                                                  type=type(''))
        ##
        # Adding process outputs

        self.flsHTML = self.addComplexOutput(identifier  = "filesHTML", 
                                             title       = "HTML to be loaded into the viewer",
                                             formats     = [{'mimeType':'text/xml'}]) 
        self.slideID = self.addLiteralOutput(identifier  = "slideID", 
                                             title       = "slideID of the most recent observation") 
        
    ##
    # Execution part of the process
    def execute(self):
        # Load the data
        #process_input = str(self.process_input.getValue())
        
        # Do the Work
        filesHTML, slideID = getHTML()
        logging.info('ID number most recent observation: '+slideID)

        # Save to out        
        self.flsHTML.setValue( filesHTML )
        self.slideID.setValue( slideID )        

        return

# <codecell>

a,b = getHTML()

# <codecell>

a.getvalue()

# <codecell>

from cgi import escape

# <codecell>

print escape(a.getvalue())

# <codecell>


