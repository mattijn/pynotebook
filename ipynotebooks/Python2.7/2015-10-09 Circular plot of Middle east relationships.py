# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.io import Raw
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle

# <codecell>

print(__doc__)

# <codecell>

%matplotlib inline

# <codecell>

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# <codecell>

# Load data
inverse_operator = read_inverse_operator(fname_inv)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# <codecell>

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# <codecell>

# Pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# <codecell>

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# <codecell>

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=True)

# <codecell>

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# <codecell>

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=True)

# <codecell>

# Now we are ready to compute the connectivity in the alpha band. Notice
# from the status messages, how mne-python: 1) reads an epoch from the raw
# file, 2) applies SSP and baseline correction, 3) computes the inverse to
# obtain a source estimate, 4) averages the source estimate to obtain a
# time series for each label, 5) includes the label time series in the
# connectivity computation, and then moves to the next epoch. This
# behaviour is because we are using generators and allows us to
# compute connectivity in computationally efficient manner where the amount
# of memory (RAM) needed is independent from the number of epochs.
fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq']  # the sampling frequency
con_methods = ['pli', 'wpli2_debiased']
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=2)

# <codecell>

# con is a 3D array, get the connectivity for the first (and only) freq. band
# for each method
con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c[:, :, 0]

# <codecell>

# Now, we visualize the connectivity using a circular graph layout

# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]

lh_labels = [name for name in label_names if name.endswith('lh')]

# <codecell>

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# <codecell>

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# <codecell>

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# <codecell>

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

# <codecell>

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig1 = plt.figure(figsize=(20,20))
plot_connectivity_circle(con_res['pli'], label_names, n_lines=300, fontsize_names= 16, colormap='PiYG',
                         node_angles=node_angles2, node_colors=label_colors, fig = fig1,
                         title='All-to-All Connectivity left-Auditory '
                               'Condition (PLI)')
plt.show()

# <codecell>

plt.imshow(con_res['pli'][0::6,0::6])
con_res['pli'][0::6,0::6].shape

# <codecell>

import pandas as pd
df = pd.read_csv(r'C:\Users\lenovo\Documents//middle_east_data2.csv', header=0,index_col=0)

# <codecell>

plt.imshow(df.as_matrix())

# <codecell>

df.index.tolist()

# <codecell>

node_angles_me = np.linspace(90,455,num=len(df.index.tolist())+1)[:-1]
#node_angles_me = node_angles_me+(node_angles_me[1]-node_angles_me[0])

# <codecell>

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
arr = df.as_matrix().astype('float')
fig2 = plt.figure(figsize=(19,19))
plot_connectivity_circle(arr, df.index.tolist(), n_lines=300, fontsize_names= 16, colormap='binary', facecolor='lightgray',                          
                         node_angles=node_angles_me,  fig = fig2, vmin=1, vmax= 2, node_colors=colors,
                         title="Relationships countries Middle-East: 'Friends' (black) vs 'Enemies' (lightgray)", textcolor='black', interactive=True)
fig2.savefig(r'C:\Users\lenovo\Desktop\New folder//figout.png', dpi=400)
plt.show()

# <codecell>

label_colors

# <codecell>

colors = [(0,0.501960784,0,1),
(0.2,0.8,0.2,1),
(0,0.690196078,0.941176471,1),
(0.8,0.8,0,1),
(0.603921569,0.211764706,0.203921569,1),
(0,0.4,0,1),
(1,0,0,1),
(0.08627451,0.211764706,0.360784314,1),
(0.749019608,0.749019608,0.749019608,1),
(0.752941176,0,0,1),
(0.349019608,0.349019608,0.349019608,1),
(0,0,0,1)]

# <codecell>

len(colors)

# <codecell>

arr

# <codecell>

label_colors[0:12]

# <codecell>

plt.imshow(con_res['pli'])

# <codecell>


# <codecell>

label_names

# <codecell>

len(label_colors)

# <codecell>

con_res['pli'].max()

# <codecell>

con_res['wpli2_debiased']

# <codecell>

np.random.shuffle(node_angles.flat)

# <codecell>

node_angles.sort()

# <codecell>

node_angles

# <codecell>


# <codecell>

node_angleslabel_names

# <codecell>


