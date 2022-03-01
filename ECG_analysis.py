# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:25:44 2022

This module analyses an electrocardiogram time-trace and detect P-waves.

Notes:
    - The p-wave annotations are not all centered on the the peak of the wave.
    This might hinder the model performance. There are 2257 p-waves annotated.
    - There are 16 hertbeats that have no p-wave annotated
    - input of 80% of the time trace is not doable since there will be too many parameters on the first layer
    
    
    ann.show_ann_classes()
    
Interrogations:
    1- Is it better to 1-hot encode for training or can we just have one model output ranging from 0 to 1?
    2- Which data to feed into the model?
        - how much does the size of the rolling window impact model performance?
        - strategy:
            + assign beginning of time trace for training, the rest for testing
            + get random chunks of data of size period_mean
    

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
import time

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# tf.config.list_physical_devices('GPU')
# tf.test.gpu_device_name()


# %% LOAD FILES
path = "D:\Ludo\Docs\programming\CAS_applied_data_science\CAS-Applied-Data-Science-master\Module-6\Electrocardiogram_analysis\Assignments\ECG\www.physionet.org\physiobank\database".replace("\\", "/")
os.chdir(path)

record = wfdb.rdrecord("mitdb/100")
ann = wfdb.rdann("mitdb/100", "pwave")
atr = wfdb.rdann("mitdb/100", "atr")

record.__dict__
ann.__dict__.items()
atr.__dict__.items()

ecg = record.p_signal[:, 0]

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


# %% EXPLORE AND PREPARE DATA
# analyse heartbeats periods
period = np.diff(atr.sample)

# Fit a normal distribution to period distribution
# TBD : fit can be improved (pdf vs absolute counts)
mu, std = norm.fit(period) 
period_mean = int(mu)
  
# Plot the histogram and Probability Density Function.
plt.figure()
plt.hist(period, bins=100, density=True, alpha=0.6, color='b')
  
x_min, x_max = plt.xlim()
x_prob = np.linspace(x_min, x_max, 100)
prob = norm.pdf(x_prob, period_mean, std)
  
plt.plot(x_prob, prob, 'k', linewidth=2)
title = f"Fit Values: {period_mean} and {std}"
plt.title(title)
plt.show()

# Plot time series
n_samples = period_mean*6
heartbeats = np.where(period>00)[0]
case = 1
plt.figure()
plt.plot(ecg, label='ecg')
plt.plot(p_ts, label='p-wave')
plt.plot(r_ts, label='r-wave')
plt.xlim([atr.sample[heartbeats[case]-1], atr.sample[heartbeats[case]-1]+ n_samples])
# plt.xlim([0, n_samples])
plt.legend()
plt.show()

# Prep of data input
# METHOD 1
"""
Split data into chunks of periods (use atr annotation as marker)
The start of the chunk isi randomly chosen on the periodic signal
remove the first (weird annotation) and the last two heartbeats (for dimension consistency)

N.B.:
    There are a few (17) heartbeats with no p-wave annotated
    There is only one weird heartbeat (lasting 407 samples: heartbeat # 1907)
"""
print(f"There are {np.count_nonzero(period > 400)} periods longer than 400 original samples, meaning with no r-wave.")
sampling = 2
packet_length = ecg[: period_mean : sampling].shape[0]
ecg_packets = np.zeros(shape=(atr.sample.shape[0]-3, packet_length))
p_ts_packets = np.zeros(shape=ecg_packets.shape, dtype=int)

# for hb, sample in enumerate(atr.sample[1:-2]):
#     offset = np.random.randint(low=0, high=int(period_mean/sampling))
#     ecg_packets[hb, :] = ecg[sample+offset : sample+offset+period_mean : sampling]
#     p_ts_packets[hb, :] = p_ts[sample+offset : sample+offset+period_mean : sampling]

for i in range(ecg_packets.shape[0]):
    offset = np.random.randint(low=0, high=ecg.shape[0]-period_mean)
    ecg_packets[i, :] = ecg[offset : offset+period_mean : sampling]
    p_ts_packets[i, :] = p_ts[offset : offset+period_mean : sampling]

for i in range(0, 100):
    plt.plot(ecg_packets[i, :]) 

# split train, test
perc_split = 0.8
n_inputs = ecg_packets.shape[0]
ecg_packets_train = ecg_packets[0:int(n_inputs*perc_split)]
ecg_packets_test = ecg_packets[int(n_inputs*perc_split):-1]
p_ts_packets_train = p_ts_packets[0:int(n_inputs*perc_split)]
p_ts_packets_test = p_ts_packets[int(n_inputs*perc_split):-1]


# %% MODEL 1: Dense neural network - simplest model
# Build model
d_input = ecg_packets.shape[1]
x = tf.keras.layers.Input(dtype='float64', shape=d_input)

# lay_1 = tf.keras.layers.Dense(units=period_mean, activation='relu')(x)
# lay_2 = tf.keras.layers.Dense(units=50, activation='relu')(lay_1)
# lay_3 = tf.keras.layers.Dense(units=2, activation='softmax')(lay_2)
   
lay_1 = tf.keras.layers.Dense(units=d_input, activation='sigmoid', name='L1')(x)

#prediction: probability->integer
# pred = 

#TBD: adding pred to the model output prudces an error (expected int64, instead had float)
model = tf.keras.Model(inputs=x, outputs=[lay_1])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% MODEL 2: Dense neural network - more complex
# Build model
d_input = ecg_packets.shape[1]
x = tf.keras.layers.Input(dtype='float64', shape=d_input)

# lay_1 = tf.keras.layers.Dense(units=period_mean, activation='relu')(x)
# lay_2 = tf.keras.layers.Dense(units=50, activation='relu')(lay_1)
# lay_3 = tf.keras.layers.Dense(units=2, activation='softmax')(lay_2)
   
lay_1 = tf.keras.layers.Dense(units=d_input*2, activation='relu', name='L1')(x)
lay_2 = tf.keras.layers.Dense(units=d_input, activation='sigmoid', name='L2')(lay_1)

#prediction: probability->integer
# pred = 

#TBD: adding pred to the model output prudces an error (expected int64, instead had float)
model = tf.keras.Model(inputs=x, outputs=[lay_2])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# %% TRAIN MODEL
start = time.time()
hist = model.fit(x=ecg_packets_train,
                    y=p_ts_packets_train,
                    epochs=200,
                    batch_size=100,
                    validation_data=(ecg_packets_test, p_ts_packets_test))
print(time.time()-start)

# %% PLOT RESULTS
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['binary_accuracy'])
axs[1].plot(hist.epoch, hist.history['val_binary_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

# %% EXAMINE RESULTS
pred_test = model.predict(ecg_packets_test)

ax = plt.gca()
for i in range(0, 30):
    color = next(ax._get_lines.prop_cycler)['color']
    # plt.plot(ecg_packets_test[i, :], color = color)
    plt.plot(p_ts_packets_test[i, :], color=color)
    plt.plot(pred_test[i, :], color=color)
    plt.show()

