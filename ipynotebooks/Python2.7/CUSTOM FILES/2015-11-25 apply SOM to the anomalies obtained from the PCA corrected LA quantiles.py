# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pd.__version__
import sys
sys.path.insert(0, r'D:\Python27x64\Lib\site-packages\sompy')
import sompy as SOM
%matplotlib inline
from sklearn.decomposition import PCA
import matplotlib as mpl
import seaborn as sns
#import seaborn.apionly as sns
sns.set(style="whitegrid", color_codes=True)
from scipy.stats import kendalltau

# <codecell>

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# <codecell>

def ellipsoid(A,center):
    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A, full_matrices=False)
    radii = 1.0/np.sqrt(s)
    
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    return x,y,z

# <codecell>

samples = df_allyears_class.as_matrix()
# mean values
mean_x = np.mean(samples[:,0])
mean_y = np.mean(samples[:,1])
mean_z = np.mean(samples[:,2])
center = [mean_x,mean_y,mean_z]

# <codecell>

df_allyears_class_grouped = df_allyears_class.groupby(lambda x: x.month)
for year, samples in df_allyears_class_grouped:
    print year 
    #samples = df_allyears_class.as_matrix()
    # mean values
    #print samples
    samples = samples.as_matrix()
    mean_x = np.mean(samples[:,0])
    mean_y = np.mean(samples[:,1])
    mean_z = np.mean(samples[:,2])
    center = [mean_x,mean_y,mean_z]    
    
    # calculate eigenvectors
    eigenvectors, eigenvalues, rotation = np.linalg.svd(samples.T, full_matrices=False)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(samples[:,0], samples[:,1], samples[:,2], 'o', markersize=10, color='green', alpha=0.2)
    ax.plot([mean_x],[mean_y],[mean_z],'o',markersize=10,color='red',alpha=0.5)
    ax.plot([eigenvectors[0][0]],[eigenvectors[0][1]],[eigenvectors[0][2]],'.',markersize=10,color='red',alpha=0.5)
    ax.plot([eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2],
            [eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2],
            [eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2],
            '.',markersize=10,color='red',alpha=0.5)
    
    #ax.plot([mean_x, eigenvectors[0][0]], [mean_y, eigenvectors[0][1]], [mean_z, eigenvectors[0][2]], 
    #        color='red', alpha=0.8, lw=3)
    a = Arrow3D([eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2, eigenvectors[0][0]], 
            [eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2, eigenvectors[0][1]], 
            [eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2, eigenvectors[0][2]], 
            lw=3, color="r", arrowstyle="<|-|>",mutation_scale=10)
    ax.add_artist(a)
    #ax = fig.add_axes([0.93,0.25,0.03,0.50])
    ax.set_title(str(year)+' Inner Mongolia', fontsize=16)

    ax.w_xaxis.set_ticklabels(qtls_NDVI_lst)
    ax.w_xaxis.set_label_text('NDVI 5-quantiles')    
    ax.w_yaxis.set_ticklabels(qtls_TRMM_lst)
    ax.w_yaxis.set_label_text('CHIRP 5-quantiles (mm)')   
    ax.w_zaxis.set_ticklabels(qtls_LST_lst)
    ax.w_zaxis.set_label_text('LST 5-quantiles (degrees)')    
    #plt.savefig('r
    plt.tight_layout()
    plt.show()
    plt.clf()    

# <codecell>

l1 = np.array([eigenvectors[0][0],eigenvectors[0][1],eigenvectors[0][2]])
l1

# <codecell>

l2 = np.array([eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2,eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2,eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2])
l2

# <codecell>

center = np.array([mean_x,mean_y,mean_z])
center

# <codecell>

eigenvectors[0][2]

# <codecell>

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot([mean_x],[mean_y],[mean_z],'o',markersize=10,color='red',alpha=0.5)
ax.plot([l1[0]],[l1[1]],[l1[2]],'o',ms=10)
ax.plot([l2[0]],[l2[1]],[l2[2]],'o',ms=10)
ax.plot([samples[4][0]],[samples[4][1]],[samples[4][2]],'o',ms=20)
ax.plot([eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2, eigenvectors[0][0]], 
        [eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2, eigenvectors[0][1]], 
        [eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2, eigenvectors[0][2]], 
        lw=3, color="r")#, arrowstyle="<|-|>",mutation_scale=10)

# <codecell>

p = samples
d = np.linalg.norm(np.cross(l2-l1, l1-p))/np.linalg.norm(l2-l1) 

# <codecell>

import math
def angle_wrt_x(A,B):
    """Return the angle between B-A and the positive x-axis.
    Values go from 0 to pi in the upper half-plane, and from 
    0 to -pi in the lower half-plane.
    """
    ax, ay = A
    bx, by = B
    return math.atan2(by-ay, bx-ax)

# <codecell>

angle_wrt_x(l1,l2)

# <codecell>

np.linalg.norm(l1-l2)

# <codecell>

d_to_center

# <codecell>

math.degrees(math.acos(d_to_line/d_to_center))+90

# <codecell>

for p in samples:
    d_to_line = np.linalg.norm(np.cross(l2-l1, l1-p))/np.linalg.norm(l2-l1)
    print 'distance to line', d_to_line
    d_to_center = np.linalg.norm(p-center)
    print 'distance to center', d_to_center
    deg_center_to_p_from_line = 180-(math.degrees(math.acos(d_to_line/d_to_center))+90)
    print 'degree corner distance to center over distance to line', deg_center_to_p_from_line
    d_to_coord = d_to_line / math.sin(deg_center_to_p_from_line)
    print 'distance to coordinate',d_to_coord,'\n'
    

# <codecell>

d_to_new_coord_from_center = d_to_line / math.sin(deg_center_to_p_from_line)

# <codecell>

math.atan2(

# <codecell>

from math import sin, cos, radians, pi
def point_pos(x0, y0, d, theta):
    theta_rad = pi/2 - radians(theta)
    return x0 + d*cos(theta_rad), y0 + d*sin(theta_rad)

# <codecell>

#points = np.random.multivariate_normal(mean=(1,1), cov=[[1, 0],[3, 10]], size=1000)
#mean = data.mean(axis=0)
#data = data - mean
eigenvectors, eigenvalues, rotation = np.linalg.svd(samples.T, full_matrices=False)

#theta = np.arctan2(*eigenvectors[:,0][::-1])
#height = np.sqrt(eigenvalues)

#point1 = mean + (height) * eigenvectors[0]*-1
#point2 = mean + (height) * eigenvectors[0]*1

#g = sns.jointplot(points[:,0], points[:,1], kind='hex', stat_func=None)
#g.ax_joint.plot([point1[0], point2[0]], [point1[1], point2[1]], '--k')
# your ellispsoid and center in matrix form

#x,y,z = ellipsoid(samples,center)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')


ax.plot(samples[:,0], samples[:,1], samples[:,2], 'o', markersize=10, color='green', alpha=0.2)
ax.plot([mean_x],[mean_y],[mean_z],'o',markersize=10,color='red',alpha=0.5)
ax.plot([eigenvectors[0][0]],[eigenvectors[0][1]],[eigenvectors[0][2]],'.',markersize=10,color='red',alpha=0.5)
ax.plot([eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2],
        [eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2],
        [eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2],
        '.',markersize=10,color='red',alpha=0.5)

#ax.plot([mean_x, eigenvectors[0][0]], [mean_y, eigenvectors[0][1]], [mean_z, eigenvectors[0][2]], 
#        color='red', alpha=0.8, lw=3)
a = Arrow3D([eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2, eigenvectors[0][0]], 
        [eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2, eigenvectors[0][1]], 
        [eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2, eigenvectors[0][2]], 
        lw=3, color="r", arrowstyle="<|-|>",mutation_scale=10)
ax.add_artist(a)
#ellipsoid
#ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
#ax.scatter(points[:,0], points[:,2])
#ax.plot([point1[0], point2[0]], [point2[1], point1[1]])
plt.show()
plt.clf()

# <codecell>

[eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2, eigenvectors[0][0]]

# <codecell>

eigenvectors[0][0]

# <codecell>

mean_z + (height[2]) * eigenvectors[0][2]

# <codecell>

[mean_x, eigenvectors[0][0]]

# <codecell>

eigenvectors[0][0] + (mean_x - eigenvectors[0][0])*2

# <codecell>

[mean_y, eigenvectors[0][1]]

# <codecell>

eigenvectors[0][1] + (mean_y - eigenvectors[0][1])*2

# <codecell>

eigenvectors[0][2] + (mean_z - eigenvectors[0][2])*2

# <codecell>

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# your ellispsoid and center in matrix form
#A = np.array([[1,0,0],[0,2,0],[0,0,2]])
#center = [0,0,0]

def ellipsoid(A,center):
    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A, full_matrices=False)
    radii = 1./s#1.0/np.sqrt(s)
    
    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[1] * np.outer(np.cos(u), np.sin(v))
    y = radii[0] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    return x,y,z,s,rotation

# <codecell>

# your ellispsoid and center in matrix form
A = samples#np.array([[80,0.1,0],[0,1,0],[0,0,1]])
center = [mean_x,mean_y,mean_z]
x,y,z,s,rotation = ellipsoid(A,center)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)

# <codecell>

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')

coefs = (10, 5, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
# Radii corresponding to the coefficients:
rx, ry, rz = 1/np.sqrt(coefs)

# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones_like(u), np.cos(v))

# Plot:
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

# Adjustment of the axes, so that they all have the same span:
max_radius = max(rx, ry, rz)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

plt.show()

# <codecell>


# <codecell>

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# your ellispsoid and center in matrix form
#A = np.array([[1,0,0],[0,2,0],[0,0,2]])
A = np.array([[0.5,0,0],[0.9,1,0],[0,0,1]])
print A
center = [0,0,0]

# find the rotation matrix and radii of the axes
U, s, rotation = linalg.svd(A)
radii = 1.0/np.sqrt(s)

# now carry on with EOL's answer
u = np.linspace(0.0, 2.0 * np.pi, 100)
v = np.linspace(0.0, np.pi, 100)
x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
for i in range(len(x)):
    for j in range(len(x)):
        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()
plt.close(fig)
del fig

# <codecell>


# <codecell>

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import show
sns.set(style="white")

# Generate a random correlated bivariate dataset
rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(2, .25), (.5, 1)] # make it asymmetric as a better test of x=y line
x1, x2 = rs.multivariate_normal(mean, cov, 500).T
x1 = pd.Series(x1, name="$X_1$")
x2 = pd.Series(x2, name="$X_2$")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(x1, x2, kind="kde", size=7, space=0)

x0, x1 = g.ax_joint.get_xlim()
y0, y1 = g.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
g.ax_joint.plot(lims, lims, ':k')    
show()

# <codecell>

df_allyears_raw = pd.read_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_allyears_raw.csv', 
                                index_col=0, parse_dates=True)
df_allyears_class = pd.read_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_allyears_class.csv', 
                                index_col=0, parse_dates=True)
df_avgmonth_allyears = pd.read_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_avgmonth_allyears.csv',
                                   index_col=0, parse_dates=True)
df_allyears_anomalies = pd.read_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_allyears_anomalies.csv',
                                    index_col=0, parse_dates=True)

# <codecell>

qtls_NDVI = np.array([0.1,0.16,0.18,0.21,0.23,0.27,0.35,0.47,0.65])
qtls_TRMM = np.array([1.99,3.18,5.02,7.96,13.34,22.47,36.94,58.43,94.4])
qtls_LST = np.array([13.19,13.59,13.94,14.33,14.66,14.88,15.04,15.21,15.42])

qtls_NDVI_lst = qtls_NDVI.tolist()
qtls_NDVI_lst.insert(0,' ')
qtls_NDVI_lst.append(' ')

qtls_TRMM_lst = qtls_TRMM.tolist()
qtls_TRMM_lst.insert(0,' ')
qtls_TRMM_lst.append(' ')  

qtls_LST_lst = qtls_LST.tolist()
qtls_LST_lst.insert(0,' ')
qtls_LST_lst.append(' ')

# <codecell>

#df_allyears_class.groupby(lambda x: x.year)

# <codecell>

a = df_avgmonth_allyears['NDVI']
b = df_avgmonth_allyears['CHIRP']
c = df_avgmonth_allyears['LST']

# <codecell>

df_allyears_anomalies = pd.DataFrame()

a = df_avgmonth_allyears['NDVI']
b = df_avgmonth_allyears['CHIRP']
c = df_avgmonth_allyears['LST']

# group classified data by year so we can plot the anomalies for each year
df_allyears_class_grouped = df_allyears_class.groupby(lambda x: x.year)
for year, ysel_class in df_allyears_class_grouped:
    print year,    
    x = ysel_class['NDVI']
    y = ysel_class['CHIRP']
    z = ysel_class['LST']    
    
    d = {'NDVI' : pd.Series(x.as_matrix() - a.as_matrix(), index=x.index),
         'CHIRP': pd.Series(y.as_matrix() - b.as_matrix(), index=y.index),
         'LST'  : pd.Series(z.as_matrix() - c.as_matrix(), index=z.index)}    
    df_anomalies = pd.DataFrame(d)
    
    #df_allyears_anomalies = df_allyears_anomalies.append(df_anomalies)   

# <codecell>

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z)

# <codecell>

sklearn_pca = sklearnPCA()#n_components=2)
sklearn_transf = sklearn_pca.fit_transform(ysel_class)

# <codecell>

ysel_class

# <codecell>

plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1],
         'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1],
         '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples using sklearn.decomposition.PCA()')
plt.show()

# <codecell>

#df_allyears_anomalies.to_csv(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//df_allyears_anomalies.csv')

# <codecell>

month = df_allyears_anomalies.index.map(lambda x: x.month)
#df_ay_anom_gs = df_allyears_anomalies_growseason
df_ay_anom_gs = df_allyears_anomalies[(month>=5)& (month <=10)]
df_ay_class_gs = df_allyears_class[(month>=5)& (month <=10)]

# <codecell>

NPOINTS = len(a)
MAP='viridis' # choose carefully, or color transitions will not appear smoooth

x = df_avgmonth_allyears['NDVI']
y = df_avgmonth_allyears['CHIRP']
z = df_avgmonth_allyears['LST']

cm = plt.get_cmap(MAP)

fig = plt.figure(figsize=(12,8))
# --- # --- # --- # --- # --- # ax1 # --- # --- # --- # --- # --- #
ax1 = plt.subplot(231)
ax1.scatter(df_allyears_class.NDVI,df_allyears_class.CHIRP, alpha=0.1, s=450,zorder=3)
ax1.scatter(df_ay_class_gs.NDVI,df_ay_class_gs.CHIRP,alpha=0.2, s=400, facecolor=None,edgecolor='red', lw=2,zorder=3)
#ax1.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
#for i in range(NPOINTS-1):
#    ax1.plot(a[i:i+2],b[i:i+2],zorder=2, lw=3)
#ax1.scatter(a,b,c=np.linspace(0,1,NPOINTS), s=40,zorder=4)

ax1.xaxis.set_ticklabels(qtls_NDVI_lst)
ax1.xaxis.set_label_text('NDVI 5-quantiles')
ax1.set_xlim(0,5)
ax1.yaxis.set_ticklabels(qtls_TRMM_lst)
ax1.yaxis.set_label_text('CHIRP 5-quantiles (mm)')
ax1.set_ylim(0,5)
ax1.grid(True)
ax1.set_title('NDVI vs CHIRP')
# --- # --- # --- # --- # --- # ax2 # --- # --- # --- # --- # --- #
ax2 = plt.subplot(232)
ax2.scatter(df_allyears_class.NDVI,df_allyears_class.LST, alpha=0.1, s=450)
ax2.scatter(df_ay_class_gs.NDVI,df_ay_class_gs.LST,alpha=0.2, s=400, facecolor=None,edgecolor='red', lw=2,zorder=3)
ax2.xaxis.set_ticklabels(qtls_NDVI_lst)
ax2.xaxis.set_label_text('NDVI 5-quantiles')
ax2.set_xlim(0,5)
ax2.yaxis.set_ticklabels(qtls_LST_lst)
ax2.yaxis.set_label_text('LST 5-quantiles (degrees)')
ax2.set_ylim(0,5)
ax2.grid(True)
ax2.set_title('NDVI vs LST')
# --- # --- # --- # --- # --- # ax3 # --- # --- # --- # --- # --- #
ax3 = plt.subplot(233)
ax3.scatter(df_allyears_class.CHIRP,df_allyears_class.LST, alpha=0.1, s=450,zorder=3)
ax3.scatter(df_ay_class_gs.CHIRP,df_ay_class_gs.LST,alpha=0.2, s=400, facecolor=None,edgecolor='red', lw=2,zorder=3)
ax3.yaxis.set_ticklabels(qtls_LST_lst)
ax3.yaxis.set_label_text('LST 5-quantiles (degrees)')
ax3.set_ylim(0,5)
ax3.xaxis.set_ticklabels(qtls_TRMM_lst)
ax3.xaxis.set_label_text('CHIRP 5-quantiles')
ax3.set_xlim(0,5)
ax3.grid(True)
ax3.set_title('CHIRP vs LST')

# --- # --- # --- # --- # --- # ax4 # --- # --- # --- # --- # --- #
ax1 = plt.subplot(234)
# plot matrix
#ax1.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
#for i in range(NPOINTS-1):
#    ax1.plot((x.as_matrix() - a.as_matrix())[i:i+2],(y.as_matrix() - b.as_matrix())[i:i+2],zorder=3)
ax1.scatter(df_allyears_anomalies.NDVI,df_allyears_anomalies.CHIRP,alpha=0.1, s=70,zorder=3)
ax1.scatter(df_ay_anom_gs.NDVI,df_ay_anom_gs.CHIRP,alpha=0.2, s=100, facecolor=None,edgecolor='red', lw=2,zorder=3)

# axes settings
ax1.set_xlim(-1.1,1.1)
ax1.set_ylim(-1.1,1.1)
ax1.vlines(0,-1.1,1.1,zorder=2)
ax1.hlines(0,-1.1,1.1,zorder=2)
ax1.xaxis.set_ticks([-1,-0.5,0,0.5,1])
ax1.yaxis.set_ticks([-1,-0.5,0,0.5,1])
ax1.xaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax1.yaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax1.xaxis.set_label_text('NDVI')    
ax1.yaxis.set_label_text('CHIRP')
plt.grid()
ax1.set_title('NDVI vs CHIRP')

# --- # --- # --- # --- # --- # ax5 # --- # --- # --- # --- # --- #
ax2 = plt.subplot(235)
# plot matrix
#ax2.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
#for i in range(NPOINTS-1):
#    ax2.plot((x.as_matrix() - a.as_matrix())[i:i+2],(z.as_matrix() - c.as_matrix())[i:i+2],zorder=3)
ax2.scatter(df_allyears_anomalies.NDVI,df_allyears_anomalies.LST,alpha=0.1, s=70,zorder=3)
ax2.scatter(df_ay_anom_gs.NDVI,df_ay_anom_gs.LST,alpha=0.2, s=100, facecolor=None,edgecolor='red', lw=2,zorder=3)
# axes settings
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.vlines(0,-1.1,1.1,zorder=2)
ax2.hlines(0,-1.1,1.1,zorder=2)
ax2.xaxis.set_ticks([-1,-0.5,0,0.5,1])
ax2.yaxis.set_ticks([-1,-0.5,0,0.5,1])
ax2.xaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax2.yaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax2.xaxis.set_label_text('NDVI')    
ax2.yaxis.set_label_text('LST')
plt.grid()
ax2.set_title('NDVI vs LST')

# --- # --- # --- # --- # --- # ax6 # --- # --- # --- # --- # --- #
ax3 = plt.subplot(236)
# plot matrix
#ax3.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
#for i in range(NPOINTS-1):
#    ax3.plot((y.as_matrix() - b.as_matrix())[i:i+2],(z.as_matrix() - c.as_matrix())[i:i+2],zorder=3)
ax3.scatter(df_allyears_anomalies.CHIRP,df_allyears_anomalies.LST,alpha=0.1, s=70,zorder=3)
ax3.scatter(df_ay_anom_gs.CHIRP,df_ay_anom_gs.LST,alpha=0.2, s=100, facecolor=None,edgecolor='red', lw=2,zorder=3)
# axes settings
ax3.set_xlim(-1.1,1.1)
ax3.set_ylim(-1.1,1.1)
ax3.vlines(0,-1.1,1.1,zorder=2)
ax3.hlines(0,-1.1,1.1,zorder=2)
ax3.xaxis.set_ticks([-1,-0.5,0,0.5,1])
ax3.yaxis.set_ticks([-1,-0.5,0,0.5,1])
ax3.xaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax3.yaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax3.yaxis.set_label_text('LST')    
ax3.xaxis.set_label_text('CHIRP')
plt.grid()
ax3.set_title('CHIRP vs LST')
plt.tight_layout()
#plt.savefig(r'D:\Downloads\Mattijn@Jia\png\trial_6//distribution_anomalies_IM.png', dpi=200, bbox_inches='tight')

# <codecell>

# classified data
Data = df_allyears_anomalies[['NDVI','CHIRP','LST']].as_matrix()
#Data = df_ay_anom_gs[['NDVI','CHIRP','LST']].as_matrix()
# raw data
#Data = df_allyears_raw[['NDVI','CHIRP','LST']].as_matrix()
Data

# <codecell>

msz11 = 6
msz10 = 6
som1 = SOM.SOM('som1', Data, mapsize = [msz10, msz11], norm_method = 'var', initmethod='pca')
#What you get when you initialize the map with pca
som1.init_map()
som1.view_map(text_size=7)

# <codecell>

#What you get when you train the map
som1.train(n_job = 1, shared_memory = 'no',verbose='off')
som1.view_map(text_size=7)

# <codecell>

#hitmap
som1.hit_map()

# <codecell>

#this gives the cluster label of each node
labels = som1.cluster(method='Kmeans', n_clusters=2)
som1.cluster_labels

# <codecell>

#if you did clusters before
cents  = som1.hit_map_cluster_number()

# <codecell>

#this cents is the array holding the bmu for the training data, the first two cols are x,y in SOM and third one is the node id.
cents  = som1.hit_map_cluster_number(Data)
print 'coordinates of bmus:'
print cents
som_clusters = labels[cents[:,2]]
print "cluster labels for data" ,som_clusters

# <codecell>

cluster0 = df_allyears_anomalies.ix[(som_clusters == 0)]
cluster1 = df_allyears_anomalies.ix[(som_clusters == 1)]
#cluster2 = df_allyears_anomalies.ix[(som_clusters == 2)]

fig = plt.figure(figsize=(12,4))
# --- # --- # --- # --- # --- # ax1 # --- # --- # --- # --- # --- #
ax1 = plt.subplot(131)
# plot matrix
ax1.scatter(cluster0.NDVI,cluster0.CHIRP,alpha=0.2, s=70, color='b',zorder=10)
ax1.scatter(cluster1.NDVI,cluster1.CHIRP,alpha=0.2, s=70, color='r',zorder=10)
#ax1.scatter(cluster2.NDVI,cluster2.CHIRP,alpha=0.2, s=70, color='g',zorder=10)
#ax1.scatter(df_ay_anom_gs.NDVI,df_ay_anom_gs.CHIRP,alpha=0.2, s=100, facecolor=None,edgecolor='red', lw=2)
# axes settings
ax1.set_xlim(-1.1,1.1)
ax1.set_ylim(-1.1,1.1)
ax1.vlines(0,-1.1,1.1,zorder=2)
ax1.hlines(0,-1.1,1.1,zorder=2)
ax1.xaxis.set_ticks([-1,-0.5,0,0.5,1])
ax1.yaxis.set_ticks([-1,-0.5,0,0.5,1])
ax1.xaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax1.yaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax1.xaxis.set_label_text('NDVI')    
ax1.yaxis.set_label_text('CHIRP')
ax1.grid()
ax1.set_title('NDVI vs CHIRP')
plt.grid()

# --- # --- # --- # --- # --- # ax2 # --- # --- # --- # --- # --- #
ax2 = plt.subplot(132)
# plot matrix
ax2.scatter(cluster0.NDVI,cluster0.LST,alpha=0.2, s=70, color='b',zorder=10)
ax2.scatter(cluster1.NDVI,cluster1.LST,alpha=0.2, s=70, color='r',zorder=10)
#ax2.scatter(cluster2.NDVI,cluster2.LST,alpha=0.2, s=70, color='g',zorder=10)
# axes settings
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.vlines(0,-1.1,1.1,zorder=2)
ax2.hlines(0,-1.1,1.1,zorder=2)
ax2.xaxis.set_ticks([-1,-0.5,0,0.5,1])
ax2.yaxis.set_ticks([-1,-0.5,0,0.5,1])
ax2.xaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax2.yaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax2.xaxis.set_label_text('NDVI')    
ax2.yaxis.set_label_text('LST')
ax2.grid()
ax2.set_title('NDVI vs LST')
plt.grid()

# --- # --- # --- # --- # --- # ax3 # --- # --- # --- # --- # --- #
ax3 = plt.subplot(133)
# plot matrix
ax3.scatter(cluster0.LST,cluster0.CHIRP,alpha=0.2, s=70, color='b',zorder=10)
ax3.scatter(cluster1.LST,cluster1.CHIRP,alpha=0.2, s=70, color='r',zorder=10)
#ax3.scatter(cluster2.LST,cluster2.CHIRP,alpha=0.2, s=70, color='g',zorder=10)
# axes settings
ax3.set_xlim(-1.1,1.1)
ax3.set_ylim(-1.1,1.1)
ax3.vlines(0,-1.1,1.1,zorder=2)
ax3.hlines(0,-1.1,1.1,zorder=2)
ax3.xaxis.set_ticks([-1,-0.5,0,0.5,1])
ax3.yaxis.set_ticks([-1,-0.5,0,0.5,1])
ax3.xaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax3.yaxis.set_ticklabels(['-20%','-10%', 'normal', '10%', '20%'])
ax3.yaxis.set_label_text('LST')    
ax3.xaxis.set_label_text('CHIRP')
ax3.grid()
ax3.set_title('CHIRP vs LST')
plt.grid()
plt.tight_layout()
#plt.savefig(r'D:\Downloads\Mattijn@Jia\png\trial_6//classfied_anomalies_IM.png', dpi=200, bbox_inches='tight')

# <codecell>

# combine classified with som clusters
df_ay_class_cluster = pd.concat([df_allyears_class.reset_index(),
                                 pd.Series(som_clusters,name='cluster')], axis=1).set_index(['date'])
df_ay_class_cluster.index = df_ay_class_cluster.index.to_period(freq='M')
year = df_ay_class_cluster.index.map(lambda x: x.year)

# <codecell>

# combine raw with som clusters
df_ay_raw_cluster = pd.concat([df_allyears_raw.reset_index(),
                               pd.Series(som_clusters,name='cluster')], axis=1).set_index(['date'])
df_ay_raw_cluster.index = df_ay_raw_cluster.index.to_period(freq='M')
year = df_ay_raw_cluster.index.map(lambda x: x.year)

# <codecell>

df_ay_raw_cluster.head()

# <codecell>

months_drought_cma = pd.read_excel(r'D:\Downloads\Mattijn@Jia\png\trial_5\data_dataframes_csv//months-drought-sta-box.xlsx',
                                   index_col=0,parse_dates=[[0,1]], freq='M')
months_drought_cma.index = months_drought_cma.index.to_period(freq='M')

# <codecell>

# combine classified som clusters with grounth 'truth'
df_ay_class_cluster_grtruth = pd.concat([df_ay_class_cluster,months_drought_cma], axis=1, join='outer')
year_cma = df_ay_class_cluster_grtruth.index.map(lambda x: x.year)
df_ay_class_cluster_grtruth = df_ay_class_cluster_grtruth[(year_cma>=2003)]
df_ay_class_cluster_grtruth.fillna(0, inplace=True)
df_ay_class_cluster_grtruth_gs = df_ay_class_cluster_grtruth[(month>=5)& (month <=10)]

# <codecell>

# combine raw som clusters with grounth 'truth'
df_ay_raw_cluster_grtruth = pd.concat([df_ay_raw_cluster,months_drought_cma], axis=1, join='outer')
year_cma = df_ay_raw_cluster_grtruth.index.map(lambda x: x.year)
df_ay_raw_cluster_grtruth = df_ay_raw_cluster_grtruth[(year_cma>=2003)]
df_ay_raw_cluster_grtruth.fillna(0, inplace=True)
df_ay_raw_cluster_grtruth_gs = df_ay_raw_cluster_grtruth[(month>=5)& (month <=10)]

# <codecell>

df_ay_raw_cluster_grtruth_gs.head()

# <codecell>

tips = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')
tips.head()

# <codecell>

sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, size=4);

# <codecell>

df_ay_class_cluster_grtruth.sort_values(by='cluster', inplace=True)

# <codecell>

#f, ax = plt.subplots(figsize=(4, 5))
g = sns.violinplot(x="cluster", y="drought_stations", data=df_ay_class_cluster_grtruth, cut=0,
               bw=0.5, scale_hue=False, palette='pastel')#, inner=None)
sns.stripplot(x="cluster", y="drought_stations", data=df_ay_class_cluster_grtruth, jitter=True, size=5)#, alpha=0.5)
g.set(ylim=(0,None))

# <codecell>

# plot classified som clusters with grounth 'truth'
plt.clf()
plt.scatter(df_ay_class_cluster_grtruth.cluster, df_ay_class_cluster_grtruth.drought_stations, alpha=0.2, s=40)
plt.scatter(df_ay_class_cluster_grtruth_gs.cluster, df_ay_class_cluster_grtruth_gs.drought_stations,
            alpha=0.2, s=30, facecolor='none', edgecolor='red', lw=2,zorder=3)
plt.violinplot(df_ay_class_cluster_grtruth_gs.query('cluster==0').drought_stations, positions=[0], showextrema=False,
               showmeans=True, showmedians=True, )
violin_parts = plt.violinplot(df_ay_class_cluster_grtruth_gs.query('cluster==1').drought_stations, positions=[1], showextrema=False,
               showmeans=True, showmedians=True, )
#for pc in violin_parts['bodies']:
#    pc.set_facecolor('c')
#    pc.set_edgecolor('black')
plt.xticks([0,1])
plt.grid(zorder=1, axis='y')
plt.ylabel('No. stations')
plt.xlabel('SOM cluster')
plt.title('Number of stations affected by drought vs its assigned cluster')
#plt.savefig(r'D:\Downloads\Mattijn@Jia\png\trial_5//stations_cluster_IM2.png', dpi=200, bbox_inches='tight')

# <codecell>

# plot raw som clusters with grounth 'truth'
plt.clf()
plt.scatter(df_ay_raw_cluster_grtruth.cluster, df_ay_raw_cluster_grtruth.drought_stations, alpha=0.2, s=40)
plt.scatter(df_ay_raw_cluster_grtruth_gs.cluster, df_ay_raw_cluster_grtruth_gs.drought_stations,
            alpha=0.2, s=30, facecolor='none', edgecolor='red', lw=2,zorder=3)
plt.violinplot(df_ay_raw_cluster_grtruth_gs.query('cluster==0').drought_stations, positions=[0], showextrema=False,
               showmeans=True, showmedians=True, )
violin_parts = plt.violinplot(df_ay_raw_cluster_grtruth_gs.query('cluster==1').drought_stations, positions=[1], showextrema=False,
               showmeans=True, showmedians=True, )
#for pc in violin_parts['bodies']:
#    pc.set_facecolor('c')
#    pc.set_edgecolor('black')
plt.xticks([0,1])
plt.grid(zorder=1, axis='y')
plt.ylabel('No. stations')
plt.xlabel('SOM cluster')
plt.title('Number of stations affected by drought vs its assigned cluster')
plt.savefig(r'D:\Downloads\Mattijn@Jia\png\trial_5//stations_cluster_orgdata_IM2.png', dpi=200, bbox_inches='tight')

# <codecell>

for _year in xrange(2003,2015,1):
    fig = plt.figure(figsize=(12,4))
    NPOINTS = len(df_ay_class_cluster[(year==_year)])
    MAP='jet' # choose carefully, or color transitions will not appear smoooth
    cm = plt.get_cmap(MAP)
    # --- # --- # --- # --- # --- # ax1 # --- # --- # --- # --- # --- #
    ax1 = plt.subplot(131)
    ax1.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
    for i in range(NPOINTS-1):
        ax1.plot(df_ay_class_cluster[(year==_year)].NDVI[i:i+2],
                 df_ay_class_cluster[(year==_year)].CHIRP[i:i+2],zorder=3)
    #ax1.plot(df_ay_class_cluster[(year==_year)].NDVI,df_ay_class_cluster[(year==_year)].CHIRP, zorder=3, alpha=1)
    #ax1.scatter(df_ay_class_cluster[(year==_year)].NDVI,df_ay_class_cluster[(year==_year)].CHIRP,c=df_ay_class_cluster[(year==_year)].cluster, s=40,zorder=4)
    ax1.scatter(df_ay_class_cluster[(year==_year)].NDVI,df_ay_class_cluster[(year==_year)].CHIRP,
                c=np.linspace(0,1,NPOINTS), s=40,zorder=4, alpha=0.5)
    cl0 = ax1.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 0').NDVI, 
                      df_ay_class_cluster[(year==_year)].query('cluster == 0').CHIRP, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='b',zorder=10, label='cluster-0')
    cl1 = ax1.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 1').NDVI,
                      df_ay_class_cluster[(year==_year)].query('cluster == 1').CHIRP, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='r',zorder=10, label='cluster-1')
    cl2 = ax1.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 2').NDVI, 
                      df_ay_class_cluster[(year==_year)].query('cluster == 2').CHIRP, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='g',zorder=10, label='cluster-2')
        
    #ax3.scatter(cluster1.LST,cluster1.CHIRP,alpha=0.2, s=70, color='r',zorder=10)
    #ax3.scatter(cluster2.LST,cluster2.CHIRP,alpha=0.2, s=70, color='g',zorder=10)
    # axis settings
    ax1.xaxis.set_ticklabels(qtls_NDVI_lst)
    ax1.xaxis.set_label_text('NDVI 5-quantiles')
    ax1.set_xlim(0,5)
    ax1.yaxis.set_ticklabels(qtls_TRMM_lst)
    ax1.yaxis.set_label_text('CHIRP 5-quantiles (mm)')
    ax1.set_ylim(0,5)
    ax1.grid(True)
    ax1.set_title('NDVI vs CHIRP')
    
    # --- # --- # --- # --- # --- # ax2 # --- # --- # --- # --- # --- #
    ax2 = plt.subplot(132)
    ax2.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
    for i in range(NPOINTS-1):
        ax2.plot(df_ay_class_cluster[(year==_year)].NDVI[i:i+2],df_ay_class_cluster[(year==_year)].LST[i:i+2],zorder=3)    
    ax2.scatter(df_ay_class_cluster[(year==_year)].NDVI,df_ay_class_cluster[(year==_year)].LST,c=np.linspace(0,1,NPOINTS), 
                s=40,zorder=4, alpha=0.5)
    cl0 = ax2.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 0').NDVI, 
                      df_ay_class_cluster[(year==_year)].query('cluster == 0').LST, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='b',zorder=10, label='cluster-0')
    cl1 = ax2.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 1').NDVI,
                      df_ay_class_cluster[(year==_year)].query('cluster == 1').LST, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='r',zorder=10, label='cluster-1')
    cl2 = ax2.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 2').NDVI, 
                      df_ay_class_cluster[(year==_year)].query('cluster == 2').LST, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='g',zorder=10, label='cluster-2')
    #ax3.scatter(cluster1.LST,cluster1.CHIRP,alpha=0.2, s=70, color='r',zorder=10)
    #ax3.scatter(cluster2.LST,cluster2.CHIRP,alpha=0.2, s=70, color='g',zorder=10)
    # axis settings
    ax2.xaxis.set_ticklabels(qtls_NDVI_lst)
    ax2.xaxis.set_label_text('NDVI 5-quantiles')
    ax2.set_xlim(0,5)
    ax2.yaxis.set_ticklabels(qtls_LST_lst)
    ax2.yaxis.set_label_text('LST 5-quantiles (degrees)')
    ax2.set_ylim(0,5)
    ax2.grid(True)
    ax2.set_title('NDVI vs LST')
    
    # --- # --- # --- # --- # --- # ax3 # --- # --- # --- # --- # --- #
    ax3 = plt.subplot(133)
    # data for single year
    ax3.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
    for i in range(NPOINTS-1):
        ax3.plot(df_ay_class_cluster[(year==_year)].CHIRP[i:i+2],df_ay_class_cluster[(year==_year)].LST[i:i+2],zorder=3)
    ax3.scatter(df_ay_class_cluster[(year==_year)].CHIRP,df_ay_class_cluster[(year==_year)].LST,c=np.linspace(0,1,NPOINTS), 
                s=40,zorder=4, alpha=0.5)
    cl0 = ax3.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 0').CHIRP, 
                      df_ay_class_cluster[(year==_year)].query('cluster == 0').LST, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='b',zorder=10, label='cluster-0')
    cl1 = ax3.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 1').CHIRP,
                      df_ay_class_cluster[(year==_year)].query('cluster == 1').LST, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='r',zorder=10, label='cluster-1')
    cl2 = ax3.scatter(df_ay_class_cluster[(year==_year)].query('cluster == 2').CHIRP, 
                      df_ay_class_cluster[(year==_year)].query('cluster == 2').LST, lw=2,alpha=1, s=125, 
                      facecolors = 'none', edgecolors ='g',zorder=10, label='cluster-2')
    
    # data for mean
    #ax3.plot(c,b, color='gray')
    #ax3.scatter(c,b,c=np.linspace(0,1,NPOINTS), s=20)
    # axis settings
    ax3.yaxis.set_ticklabels(qtls_LST_lst)
    ax3.yaxis.set_label_text('LST 5-quantiles (degrees)')
    ax3.set_ylim(0,5)
    ax3.xaxis.set_ticklabels(qtls_TRMM_lst)
    ax3.xaxis.set_label_text('CHIRP 5-quantiles')
    ax3.set_xlim(0,5)
    ax3.grid(True)
    ax3.set_title('LST vs CHIRP')
    
    ax4 = fig.add_axes([0.135,-0.02,0.803,0.018])
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax4, cmap=cm, ticks=np.linspace(0,1,NPOINTS), 
                                    norm=norm,
                                    orientation='horizontal')
    cb.ax.set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])#'%d-%m-%Y'
    cb.ax.tick_params('both', length=5)
    
    #ax5 = fig.add_axes([0.85,0.02,0.025,0.7])
    ax1.legend(scatterpoints=1, bbox_to_anchor = (3.75, 0.5), frameon=False)
    #cb = mpl.colorbar.ColorbarBase(ax4, cmap=cm, ticks=np.linspace(0,1,NPOINTS), 
    #                                norm=norm,
    #                                orientation='horizontal')
    #cb.ax.set_xticklabels(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])#'%d-%m-%Y'
    #cb.ax.tick_params('both', length=5)
    
    
    ax2.text(-7.0,5.50,str(_year),fontsize=17)
    plt.tight_layout()
    plt.savefig(r'D:\Downloads\Mattijn@Jia\png\trial_5\png_path_anomalies_clusters_gs//'+str(_year)+'path_anomalies_clusters_IM.png', dpi=200, bbox_inches='tight')

# <codecell>


# <codecell>

from sklearn.preprocessing import StandardScaler
X_std = Data#StandardScaler().fit_transform(Data)

# <codecell>

plt.plot(Data)
plt.show()
plt.plot(X_std)

# <codecell>

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

# <codecell>

#you can further, do the following stuff
#this gives you the bmus
som1.project_data(Data[:10])

# <codecell>

som1.project_data(Data)

# <codecell>

#to convert bmu to x,y in som map
som1.ind_to_xy(som1.project_data(Data))

# <codecell>

Data[:10]

# <codecell>

som1.cluster_labels

# <codecell>

msz11 = 1
msz10 = 150
som1 = SOM.SOM('som1', Data, mapsize = [msz10, msz11],norm_method = 'var',initmethod='pca')

som1.train(n_job = 1, shared_memory = 'no',verbose='off')

codebook1 = pd.DataFrame(data= som1.codebook[:])

print 'Done'
codebook1_n = SOM.denormalize_by(som1.data_raw, codebook1, n_method = 'var')

fig = plt.figure()
plt.plot(Data[:,0],Data[:,1],'ob',alpha=0.2, markersize=5)
plt.plot(codebook1_n.values[:,0],codebook1_n.values[:,1],'o-g',linewidth=2,markersize=4)
fig.set_size_inches(10,10)

# <codecell>

#in the previous example, there are few nodes that don't represent any of the trainig data. 
# We can remove them if we like or in another way we can bold each node based on the number of training data that it represents
fig = plt.figure()
a = plt.hist(som1.project_data(Data),bins=msz10)
#Distribution of data for over nodes of som. This can be considered as a probabilty distribution

# <codecell>

fig = plt.figure()
a = plt.hist(som1.project_data(Data),bins=msz10)
plt.close()
fig = plt.figure()
plt.plot(Data[:,0],Data[:,1],'ob',alpha=0.3, markersize=3,label='Data')
plt.plot(codebook1_n.values[:,0],codebook1_n.values[:,1],'o-g',linewidth=.4,markersize=2,label='nodes')
plt.scatter(codebook1_n.values[:,0],codebook1_n.values[:,1], s= 25*a[0], alpha=0.9,c='w',marker='o',cmap='jet',linewidths=3, edgecolor = 'r'
            ,label='node size proportional to their frequency of being bmu')
plt.legend(loc='best',bbox_to_anchor = (1.42, 1.0),fontsize = 'large')
#plt.xlim(0,4.5)
#plt.ylim(-2,2.5)
fig.set_size_inches(10,10)

# <codecell>

#Another interesting thing is that many times we want to know the charachteristics of the clusters
#Let's assume we have a 3dimensional data
Data1 = np.concatenate((Data,np.random.rand(Data.shape[0],2)),axis=1)
# a 1d SOM with 10 nodes can be considered as 10 clusters with a major difference to kmeans that similar cluster labels
#have similar context. For example the weight vector of cluster 1 is similar cluster 2 than cluster than
#while in K-means labels are arbitrary. Here these labels are contextual numbers. For more info:http://arxiv.org/abs/1408.0889 
msz11 = 1
msz10 = 10
som1 = SOM.SOM('som1', Data1, mapsize = [msz10, msz11],norm_method = 'var',initmethod='pca')

som1.train(n_job = 1, shared_memory = 'no',verbose='off')

print 'Done'

# <codecell>

#The following function visualizes the pattern of nodes in a 1d som,
#but since the final som is an ordered set, it creates a nice patters
path = ''
plot_radar(som1,path,save='No',alpha=.4,legend='Yes')

# <codecell>

def denormalize_by(data_by, n_vect, n_method = 'var'):
    #based on the normalization
    if n_method == 'var':
        me = np.mean(data_by, axis = 0)
        st = np.std(data_by, axis = 0)
        vect = n_vect* st + me
        return vect 
    else:
        print 'data is not normalized before'
        return n_vect

# <codecell>

def plot_radar(som,path,save='Yes',legend='No',alpha=.5):
    import matplotlib.pyplot as plt
    titles = som.compname[:][0].copy()
    labels =[]
    denorm_cd= denormalize_by(som.data_raw,som.codebook)
    mx = denorm_cd.max(axis=0)
    mn = denorm_cd.min(axis=0)
    rng = mx-mn
    denorm_cd = (denorm_cd-mn)/rng
    
    for dim in range(som.codebook.shape[1]):
        titles[dim] = titles[dim].replace("; measures: Value","",1)
        titles[dim]= titles[dim].replace(som.name+': ',"",1)
    
        labels.append(np.around(np.linspace(mn[dim],mx[dim],num=5),decimals=3).tolist())

    titles=titles.tolist()


    fig = plt.figure(figsize=(10, 10))

    
    rect = [0.15, 0.1, .7, .7]

    n = len(titles)
    angles = np.arange(0, 360, 360.0/n)
    axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) 
                         for i in range(n)]

    ax = axes[0]
    ax.set_thetagrids(angles,labels=titles, fontsize=5, frac=1.15, rotation=0,multialignment='left',fontweight='demi')
    ax.set_theta_offset(10)

    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)
        ax.set_theta_offset(10)

        

    for ax, angle, label in zip(axes, angles, labels):
        ax.set_rgrids(range(1, 6), angle=angle, labels=label)
        ax.spines["polar"].set_visible(False)
        ax.set_ylim(0, 6)

    
    N = denorm_cd.shape[0]
    n_aspect = denorm_cd.shape[1]
    for dim in range(N):
        angle = np.deg2rad(np.r_[angles, angles[0]])
        values = denorm_cd[dim]*5
        values = np.r_[values, values[0]]
        if n_aspect<=2:
            ax.plot(angle,values,"o", markersize=12, color=plt.get_cmap('RdYlBu_r')(dim/float(N)),alpha=0.8)
#         if n_aspect>2:
        ax.fill(angle,values,"-", lw=4, color=plt.get_cmap('RdYlBu_r')(dim/float(N)), alpha=alpha, label=str(dim))
#         radar.plot(denorm_cd[dim]*5,  "-", lw=4, color=plt.get_cmap('RdYlBu_r')(dim/float(N)), alpha=0.4, label='cluster '+str(dim))

    if legend=='Yes':
        
        ax.legend(bbox_to_anchor= (1.21,0.35),labelspacing=.35,fontsize ='x-small', title='Cluster', 
              fancybox = True, handletextpad=.8, borderpad=1.)
    
    font = {'size'   : 8}
    plt.rc('font', **font)
    plt.figtext(0.5, .9, som.name,
                ha='center', color='black', weight='bold', size='large')
    if save =='Yes':
        plt.savefig(path, dpi=400, transparent=False)
        plt.close() 

# <codecell>


