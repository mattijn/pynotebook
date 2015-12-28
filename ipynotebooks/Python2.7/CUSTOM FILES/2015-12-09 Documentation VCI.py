
# coding: utf-8

# #### **Vegetation Condition Index**

# **Introduction**

# The Vegetation Condition Index (VCI) compares the current NDVI to the range of values observed in the same period in previous years. The VCI is expressed in between 0 and 1 and gives an idea where the observed value is situated between the extreme values (minimum and maximum) in the previous years. Lower and higher values indicate bad and good vegetation state conditions, respectively.

# The VCI is computed as follow:

# $$
# VCI_{ijk} = \frac{NDVI_{ijk} - \max{NDVI}_{ij}}{\max{NDVI}_{ij} - \min{NDVI}_{ij}} 
# $$
# 
# Where $VCI_{ijk}$ is the VCI-value for pixel $i$ during day $j$ for year $k$. $NDVI_{ijk}$ is the daily $NDVI$ value for pixel $i$ during day $j$ for year $k$, $\max{NDVI}_{ij}$ is the maximum $NDVI$ for pixel $i$ during day $j$ over $n$ years, and $\min{NDVI}_{ij}$ is the minumum $NDVI$ for pixel $i$ during day $j$ over $n$ years.

# **References**

# Kogan, F. N. "Application of vegetation index and brightness temperature for drought detection." Advances in Space Research 15.11 (1995): 91-100. [doi: 10.1016/0273-1177(95)00079-T](http://dx.doi.org/10.1016/0273-1177%2895%2900079-T)

# 

# #### **Temperature Condition Index**

# **Introduction**

# The Temperature Condition Index (TCI) compares the current land surface temperature (LST) to the range of values observed in the same period in previous years. The LST is expressed in between 0 and 1 and gives an idea where the observed value is situated between the extreme values (minimum and maximum) in the previous years. Higher values indicate increased temperature condititon and lower values indicate decreased temperature condition. This index was used to determine temperature-related vegetation stress and also stress caused by an excessive wetness

# The TCI is computed as follow:

# $$
# TCI_{ijk} = \frac{\max{LST}_{ij} - LST_{ijk}}{\max{LST}_{ij} - \min{LST}_{ij}} 
# $$
# 
# Where $TCI_{ijk}$ is the TCI-value for pixel $i$ during day $j$ for year $k$. $LST_{ijk}$ is the daily $LST$ value for pixel $i$ during day $j$ for year $k$, $\max{LST}_{ij}$ is the maximum $LST$ for pixel $i$ during day $j$ over $n$ years, and $\min{LST}_{ij}$ is the minumum $LST$ for pixel $i$ during day $j$ over $n$ years.

# **References**

# Kogan, F. N. "Application of vegetation index and brightness temperature for drought detection." Advances in Space Research 15.11 (1995): 91-100. [doi: 10.1016/0273-1177(95)00079-T](http://dx.doi.org/10.1016/0273-1177%2895%2900079-T)

# #### **Vegetation Health Index**

# The Vegetation Health Index, also called the Vegetation-Temperature Index, is based on a combination of Vegetation Condition Index (VCI) and Temperature Condition Index (TCI). It is effective enough to be used as proxy data for monitoring vegetation health, drought, moisture, thermal condition, etc.

# The VHI is computed as follow:

# $$
# VHI_{ijk} = \alpha\cdot VCI_{ijk} + (1-\alpha)\cdot TCI_{ijk}
# $$
# 
# Where $VHI_{ijk}$ is the VHI-value for pixel $i$ during day $j$ for year $k$. $VCI_{ijk}$ is the VCI-value for pixel $i$ during day $j$ for year $k$, $TCI_{ijk}$ is the TCI-value for pixel $i$ during day $j$ for year $k$ and $\alpha$ is used to change the importance of each separated condition index and should have a value between $0$ and $1$ and is normally set to $0.5$.

# $$
# VCI_{ijk} = \frac{NDVI_{ijk} - \max{NDVI}_{ij}}{\max{NDVI}_{ij} - \min{NDVI}_{ij}} 
# $$

# In[ ]:



