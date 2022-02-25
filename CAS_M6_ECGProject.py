# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:43:13 2022

@author: hisham.benhamidane
"""

#%%
# 1. Import libraries and data

import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import norm
from scipy.optimize import curve_fit

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Setting the wd
path = "C:/Users/hisham.benhamidane/OneDrive - Thermo Fisher Scientific/Documents/R/projects/CAS2021/Module 6 deep neural network/mitdb"
os.chdir(path)

# Reading the input files
record = wfdb.rdrecord("100")
ann = wfdb.rdann("100", "pwave")

#%%
# 2. Processing and reshaping the data into 1D array of length len(p.signal)

# REshaping the data
## Single channel signal for the timeseries using the mLII signal
x = np.asarray(record.p_signal[:,0])
# Creating an empty numpy array of the same length as x
y = np.zeros(record.p_signal[:,0].shape, dtype=np.int32)
## Creating a vector of annoatations to include the 6 points before and after the manual p-wave annotation
n = x.shape[0]
## defining ext, the number of points before and after the annotated p-wave point
ext = 6
## subsetting y with ann.sample to attribute value 1 else 0
for i,v in zip(ann.sample, ann.symbol):
    if v=='p':
        y[max(i-ext, 0):min(i+ext+1, n)]=1

# Defining the target_sampling value that will define the length of the bits of x that will be passed as input
## down_sampling is the fold reduction of the signal datapoint number; 1st try, no downsampling, var set to 1
down_sampling = 1
## target_sampling is the arbitrary number of signal datapoints that will be fed as input to the NN
target_sampling = 48

# Apply the target_sampling to both x and y
bins = int(x.shape[0]/target_sampling)

# in subsetting the 3 argument is equivalent to a "by" in R; x[start:stop:step]
x_binned = np.zeros(shape=(bins, target_sampling))
y_binned = np.zeros(shape=(bins, target_sampling))
for i in range(bins):
    x_binned[i,:] = x[i*target_sampling : i*target_sampling+target_sampling : down_sampling]
for i in range(bins):
    y_binned[i,:] = y[i*target_sampling : i*target_sampling+target_sampling : down_sampling]

#%%
# 3. Creating the inputs for the NN model

# creating the train/test split: 80/20
## defining the percentage of training data
train_perc = 0.7
## creating a count of indexes belonging to the training data
train_idx_count = int(train_perc*bins)
## creating the training indexes by shuffling through all possible dataset indexes
train_idx = np.random.choice(x_binned.shape[0], size=train_idx_count, replace=False)
## creating the x and y training data set by subsetting with train_idx
train_x = x_binned[train_idx]
train_y = y_binned[train_idx]

## Defining the testing index by excluding from all indexes the ones already selected for training
test_idx = np.where(np.invert(np.isin(range(bins), train_idx)))
## creating the x and y testing data set by subsetting with test_idx
test_x = x_binned[test_idx]
test_y = y_binned[test_idx]


#%% 
# 4. Create the model and NN
# MODEL 1: Dense neural network

# Defining input
x = tf.keras.layers.Input(dtype='float64', shape=train_x.shape[1])
 
# Deinfing the layers
L1 = tf.keras.layers.Dense(units=target_sampling*3, activation='relu', name='L1')(x)
L2 = tf.keras.layers.Dense(units=target_sampling, activation='softmax', name='L2')(L1)

#prediction?
# pred = tf.argmax(L2, axis=1, name='pred')
# print(pred)

# model
## definition
model = tf.keras.Model(inputs=x, outputs=[L2])
## visualization
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
## compiling
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# %%
# test the model accuracy
hist = model.fit(x=train_x,
                    y=train_y,
                    epochs=25,
                    batch_size=int(bins/5),
                    validation_data=(test_x, test_y))

# %% PLOT RESULTS
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['binary_accuracy'])
axs[1].plot(hist.epoch, hist.history['val_binary_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

#%%
model.predict(test_x[1])



# %% EXAMINE RESULTS
pred_test = model.predict(test_y)

ax = plt.gca()
for i in range(0, 5):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(test_y[i, :], color = color)
    plt.plot(test_x[i, :], color = color)
    plt.plot(pred_test[i, :], color = color)
    plt.show()