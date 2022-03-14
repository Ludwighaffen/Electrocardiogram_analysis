# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:25:44 2022

This module analyses an electrocardiogram time-trace and detect P-waves.

Notes:
    - The p-wave annotations are not all centered on the the peak of the wave.
    This might hinder the model performance. There are 2257 p-waves annotated.
    - There are 16 hertbeats that have no p-wave annotated
    
1- Parameters impoacting the model
    - batch size has little influence
    - 2 layers is much better than 1 layer
    - size of first layer can be relatively small compared to the number of data points in window
    best results for size=1.2 to 1.5 * the number of data points (1.5x shows more overfitting,
    looking at the gap between train and test accuracy)
    - size of the rolling window impact model performance?
        => size around signal period gives best result
    - down sampling?
        => slightly worst accuracy for down sampling > 2
    
TO DO:
    DONE- use random seed to generate the same random packets (or plot the packets chronologically to "locate" them more easily)
    - compute a different accuracy metric based on discrete p-wave annotation (not every data point)
    - use 2D-convolutional network on signal spectrogram
    
"""
__author__ = "Ludovic Le Reste"
__credits__ = ["Hisham Ben Hamidane", "Ludovic Le Reste"]
__status__ = "Prototype"

import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import signal
import time
from matplotlib.widgets import Slider

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# tf.config.list_physical_devices('GPU')
# tf.test.gpu_device_name()


# %% LOAD FILES
# path = "D:\Ludo\Docs\programming\CAS_applied_data_science\CAS-Applied-Data-Science-master\Module-6\Electrocardiogram_analysis\Assignments\ECG\www.physionet.org\physiobank\database".replace("\\", "/")
path = r"C:\Users\ludovic.lereste\Documents\CAS_applied_data_science\CAS-Applied-Data-Science-master\Module-6\Electrocardiogram_analysis\Assignments\ECG\www.physionet.org\physiobank\database".replace("\\", "/")
os.chdir(path)

record = wfdb.rdrecord("mitdb/100")
ann = wfdb.rdann("mitdb/100", "pwave")
atr = wfdb.rdann("mitdb/100", "atr")

record.__dict__
ann.__dict__.items()
atr.__dict__.items()

ecg = record.p_signal[:, 0]
ecg_v5 = record.p_signal[:, 1]

# translate discrete p-wave annotation into digital signal
# p-wave is +-width around annotation
p_ts = np.zeros(record.sig_len, dtype=int)
width = 6
for i in ann.sample:
    p_ts[max(i-width, 0):min(i+width, record.sig_len)] = 1
    
# translate discrete r-wave annotation into digital signal
# r-wave is +-width around annotation
r_ts = np.zeros(record.sig_len, dtype=int)
width = 6
for i in atr.sample:
    r_ts[max(i-width, 0):min(i+width, record.sig_len)] = 1

# Generate time vector
fq = 360
t = np.linspace(start=0, stop=ecg.size/fq, num=ecg.size)
tss = np.vstack((t, ecg, p_ts)).T

# %% EXPLORE DATA
# Analyse heartbeats periods
period = np.diff(atr.sample)
period_p = np.diff(ann.sample)

# Fit a normal distribution to period distribution
mu, std = norm.fit(period) 
period_mean = int(mu)
  
mu_p, std_p = norm.fit(period_p) 
period_mean_p = int(mu_p)

# Plot the histogram and Probability Density Function.
fig, axs = plt.subplots(1, 2, sharey=True)

axs[0].hist(period, bins=100, density=True, alpha=0.6, color='b')
x_min, x_max = axs[0].get_xlim()
x_prob = np.linspace(x_min, x_max, 100)
prob = norm.pdf(x_prob, period_mean, std)
axs[0].plot(x_prob, prob, 'k', linewidth=2)
axs[0].set_xlabel('# data points')
axs[0].set_ylabel('density')
axs[0].legend(title=f"Fit mean={period_mean}")
axs[0].set_title("r-waves")

axs[1].hist(period_p, bins=100, density=True, alpha=0.6, color='b')
axs[1].set_xlabel('# data points')
axs[1].set_title('p-waves')

fig.suptitle('Signal periods')
plt.show()

"""
N.B.:
    There are a few (16) heartbeats with no p-wave annotated
    There is only one weird heartbeat (lasting 407 samples: heartbeat # 1907)
"""
print(f"There are {np.count_nonzero(period > 400)} r-wave periods longer than 400 data points")
print(f"There are {np.count_nonzero(period_p > 400)} p-wave periods longer than 400 data points")

# Plot time series
n_samples = period_mean*4
heartbeats = np.where(period>00)[0]
case = 1
fig, ax = plt.subplots()
ax.plot(ecg, label='ecg')
ax.plot(p_ts, label='p-wave')
ax.plot(p_ts*-1, label='-(p-wave)', color='C1')
ax.plot(r_ts, label='r-wave')
ax.set_xlim([atr.sample[heartbeats[case]-1], atr.sample[heartbeats[case]-1]+ n_samples])
# plt.xlim([0, n_samples])
ax.set_xlabel('data points')
ax.set_ylabel('voltage [mV] or boolean')
secax = ax.secondary_xaxis('top', functions=(lambda x: x/fq, lambda x: x*fq))
secax.set_xlabel('time [s]')
plt.legend()
plt.show()

"""first r-wave is wrongly labelled"""
fig, ax = plt.subplots()
ax.plot(ecg, label='ecg')
ax.plot(p_ts, label='p-wave')
ax.plot(r_ts, label='r-wave')
ax.set_xlim([0, 500])
# plt.xlim([0, n_samples])
ax.legend()
plt.show()

# Fourier transform
ecg_ft = np.fft.rfft(ecg)
f = np.fft.rfftfreq(ecg.size, d=1/fq)
fig, ax = plt.subplots()
ax.plot(f[1:], np.abs(ecg_ft[1:]))
ax.set_xlim([0, 60])
ax.set_xlabel('frequency [Hz]')
ax.set_title('real Fourier transform')
plt.show()

# Filtering frequencies
# sos = signal.butter(N=10, Wn=[fq/50000, fq/50], btype='bandpass', output='sos', fs=fq)
cutoff=30
sos = signal.butter(N=10, Wn=cutoff, btype='lowpass', output='sos', fs=fq)
ecg_f = signal.sosfilt(sos, ecg)
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].plot(ecg[:1000])
axs[0].set_xlabel('data points')
axs[0].set_ylabel('voltage [mV]')
axs[0].set_title(f'raw signal (MLII)')
axs[1].plot(ecg_f[:1000])
axs[1].set_xlabel('data points')
axs[1].set_title(f'Low-pass filtered (cut-off={cutoff}Hz)')
plt.show()

# Spectrogram
fig, ax = plt.subplots()
spec_f, spec_t, spec_map = signal.spectrogram(ecg[:20*period_mean], fq)
pcm = ax.pcolormesh(spec_t, spec_f, spec_map, shading='gouraud', cmap='hsv')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
ax.set_title('Spectrogram (first 20 periods)')
fig.colorbar(pcm, ax=ax)

# %% PREP DATA
"""
Split data into packets of same length to feed to the model
The start of the chunk isi randomly chosen
remove the first (weird annotation) and the last two heartbeats (for dimension consistency)
"""
packet_length = 400
sampling = 2
packet_length_ds = ecg[: packet_length : sampling].shape[0]
ecg_packets = np.zeros(shape=(int(ecg.shape[0]/packet_length), packet_length_ds))
p_ts_packets = np.zeros(shape=ecg_packets.shape, dtype=int)

rng = np.random.default_rng(42)
ints = rng.integers(low=0, high=ecg.size-packet_length, size=ecg_packets.shape[0])

for i in range(ecg_packets.shape[0]):
    ecg_packets[i, :] = ecg[ints[i] : ints[i]+packet_length : sampling]
    p_ts_packets[i, :] = p_ts[ints[i] : ints[i]+packet_length : sampling]

# split train, test
perc_split = 0.8
n_inputs = ecg_packets.shape[0]
ecg_packets_train = ecg_packets[0:int(n_inputs*perc_split)]
ecg_packets_test = ecg_packets[int(n_inputs*perc_split):-1]
p_ts_packets_train = p_ts_packets[0:int(n_inputs*perc_split)]
p_ts_packets_test = p_ts_packets[int(n_inputs*perc_split):-1]


# %% MODEL 1: Dense neural network - simplest model
# Build model
d_mult = "single_layer"

d_input = ecg_packets.shape[1]
x = tf.keras.layers.Input(dtype='float64', shape=d_input)
   
lay_1 = tf.keras.layers.Dense(units=d_input, activation='sigmoid', name='L1')(x)

#prediction: probability->integer
# pred = 

#TBD: adding pred to the model output prudces an error (expected int64, instead had float)
model = tf.keras.Model(inputs=x, outputs=[lay_1])

model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# %% MODEL 2: Dense neural network - more complex
# Build model
d_mult = 1.2

d_input = ecg_packets.shape[1]
x = tf.keras.layers.Input(dtype='float64', shape=d_input)

lay_1 = tf.keras.layers.Dense(units=int(d_input*d_mult), activation='relu', name='L1')(x)
lay_2 = tf.keras.layers.Dense(units=d_input, activation='sigmoid', name='L2')(lay_1)

#prediction: probability->integer
# pred = 

#TBD: adding pred to the model output prudces an error (expected int64, instead had float)
model = tf.keras.Model(inputs=x, outputs=[lay_2])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# %% MODEL 3: Convolutional neural network
"""let's try one feature map (1D-line) followed by one dense layer
and maybe try to transform the dense layer into a convolutional layer of correct size

testing:
data = tf.keras.utils.timeseries_dataset_from_array(data=tss,
                                                    targets=None,
                                                    sequence_length=300,
                                                    sequence_stride=300,
                                                    sampling_rate=1)
for el in data:
    inputs = el
    """
# lay_1 = tf.keras.layers.Conv1D()
# tf.keras.layers.MaxPool1D()

# %% TRAIN MODEL
batch_size = 50

start = time.time()
hist = model.fit(x=ecg_packets_train,
                    y=p_ts_packets_train,
                    epochs=50,
                    batch_size=batch_size,
                    validation_data=(ecg_packets_test, p_ts_packets_test))
print(time.time()-start)

# %% COMPUTE NEW METRICS
"""
1- transform raw output to digital (0-1, i.e. p-wave or no p-wave)
2- Figure out a smart way to compare output blocks with p_ts blocks
3- compute precision, recall, and calculate accuracy from those
"""
# Initialise variables
data = ecg_packets_test
data_ann = p_ts_packets_test
pred = model.predict(data)

# Compute blocks corresponding to p-waves ([start, end] indexes pf p-waves)
pred_dig = pred > 0.5
pred_pos = []
data_ann_pos = []
for i in range(data.shape[0]):
    sw_pred = np.argwhere(np.diff(pred_dig[i,:])).squeeze()
    sw_ann = np.argwhere(np.diff(data_ann[i,:])).squeeze()
    # adjust for start and end values
    if pred_dig[i, 0]==True:
        sw_pred = np.insert(sw_pred, 0, 0)
    if pred_dig[i, -1]==True:
        sw_pred = np.append(sw_pred, data.shape[1])
    if data_ann[i, 0]==True:
        sw_ann = np.insert(sw_ann, 0, 0)
    if data_ann[i, -1]==True:
        sw_ann = np.append(sw_ann, data.shape[1])
    pred_pos.append(sw_pred.reshape(-1, 2))
    data_ann_pos.append(sw_ann.reshape(-1, 2))

# True positives
"""
take one packet
take one p-wave interval, is there any number in pred_pos within this interval?
repeat for all p-wave interval
repeat for all packets
"""
i=0
j=0
data_ann_pos[i][j]
pred_pos[i][:].ravel()

# %% PLOT RESULTS
"""N.B.: hist.params['batch_size'] does not work on Thermo's PC"""
fig, axs = plt.subplots(1, 3, figsize=(13,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='upper right')
axs[0].set_xlabel('epoch')

axs[1].plot(hist.epoch, hist.history['binary_accuracy'])
axs[1].plot(hist.epoch, hist.history['val_binary_accuracy'])
axs[1].legend(('training accuracy', 'validation accuracy'),
              title=f"validation accuracy={hist.history['val_binary_accuracy'][-1]:.3f}",
              loc='lower right')
axs[1].set_xlabel('epoch')

axs[2].plot(hist.epoch, hist.history['precision'])
axs[2].plot(hist.epoch, hist.history['val_precision'])
axs[2].plot(hist.epoch, hist.history['recall'])
axs[2].plot(hist.epoch, hist.history['val_recall'])
axs[2].legend(('precision', 'val_precision',
               'recall', 'val_recall'),
              title=f"val_precision={hist.history['val_precision'][-1]:.3f}\n"
              f"val_recall={hist.history['val_recall'][-1]:.3f}",
              loc='lower right')
axs[2].set_xlabel('epoch')

fig.suptitle(f"packet length = {packet_length}\n"
             f"down sampling = {sampling}\n"
             f"layer_1 dimension = {d_input}*{d_mult} = {int(d_input*d_mult)}\n"
             f"batch size={batch_size}")
plt.show()

# %% EXAMINE RESULTS
"""N.B.: run %matplotlib auto in console"""
n_rows = 4
n_cols = 6
n_plots = n_rows * n_cols
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharey=True)
plt.get_current_fig_manager().window.state('zoomed')
plt.subplots_adjust(top=0.90)

# inititate plot lines (ls_xxx) that will later be updated by the slider
ls_data = []
ls_data_ann = []
ls_data_ann2 = []
ls_pred = []
for i, ax in enumerate(axs.flat):
    ls_data.append(ax.plot(data[i, :]))
    ls_data_ann.append(ax.plot(data_ann[i, :]))
    ls_data_ann2.append(ax.plot(data_ann[i, :]*-1, color='orange'))
    ls_pred.append(ax.plot(pred[i, :]))
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylim(ymin=-1.2, ymax=1.3)
plt.subplots_adjust(wspace=0, hspace=0)

ax_slider = plt.axes([0.25, 0.95, 0.65, 0.03])
slider_packet = Slider(ax=ax_slider,
                       label='Test packet',
                       valmin=0,
                       valmax=data.shape[0]-n_plots,
                       valstep=n_plots,
                       valinit=0)

def update_slider(val):
    for i in np.arange(n_plots):
        ls_data[i][0].set_ydata(data[i+val, :])
        ls_data_ann[i][0].set_ydata(data_ann[i+val, :])
        ls_data_ann2[i][0].set_ydata(data_ann[i+val, :]*-1)
        ls_pred[i][0].set_ydata(pred[i+val, :])
        
slider_packet.on_changed(update_slider)
