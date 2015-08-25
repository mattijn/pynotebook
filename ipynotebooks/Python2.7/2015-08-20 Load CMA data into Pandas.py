# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
%matplotlib inline

# <codecell>

file_in = r'C:\Users\lenovo\AppData\Roaming\Skype\My Skype Received Files//air_temperature_hour_hour.txt'

# <codecell>

datalist = []
with open(file_in) as f:    
    next(f)
    for line in f:        
        parsed = filter(None, line.split('\t'))
        #print parsed
        
        hour_02 = pd.datetime(int(parsed[1][0:4]),int(parsed[1][4:6]),int(parsed[1][6:8]),2)
        hour_08 = pd.datetime(int(parsed[1][0:4]),int(parsed[1][4:6]),int(parsed[1][6:8]),8)
        hour_14 = pd.datetime(int(parsed[1][0:4]),int(parsed[1][4:6]),int(parsed[1][6:8]),14)        
        hour_20 = pd.datetime(int(parsed[1][0:4]),int(parsed[1][4:6]),int(parsed[1][6:8]),20)
        
        datT_02 = int(parsed[4])
        datT_08 = int(parsed[5])
        datT_14 = int(parsed[6])
        datT_20 = int(parsed[7])        
        
        datP_02 = int(parsed[8])
        datP_08 = int(parsed[9])
        datP_14 = int(parsed[10])
        datP_20 = int(parsed[11])
        
        datalist.append((hour_02, datT_02, datP_02))
        datalist.append((hour_08, datT_08, datP_08))        
        datalist.append((hour_14, datT_14, datP_14))
        datalist.append((hour_20, datT_20, datP_20))

df = pd.DataFrame(datalist, columns=['date','Temp','Pres'])
df.set_index('date', inplace=True)

# <codecell>

df.Temp.plot(color='lightgray')
pd.ewma(df.Temp, 12).plot()

# <codecell>

df.Pres.plot(color='lightgray')
pd.ewma(df.Pres, 12).plot()

# <codecell>


