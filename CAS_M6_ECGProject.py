# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:43:13 2022

@author: hisham.benhamidane
"""

#%% Loading dependencies
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

#%% Signal processing and sampling/down sampling definition
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
target_sampling = 390

# Apply the target_sampling to both x and y
bins = int(x.shape[0]/target_sampling)

# in subsetting the 3 argument is equivalent to a "by" in R; x[start:stop:step]
x_binned = np.zeros(shape=(bins, int(target_sampling/down_sampling)))
y_binned = np.zeros(shape=(bins, int(target_sampling/down_sampling)))
for i in range(bins):
    x_binned[i,:] = x[i*target_sampling : i*target_sampling+target_sampling : down_sampling]
for i in range(bins):
    y_binned[i,:] = y[i*target_sampling : i*target_sampling+target_sampling : down_sampling]

#%% Defining train/test split
# 3. Creating the inputs for the NN model

# creating the train/test split: 80/20.
## defining the percentage of training data
train_perc = 0.80
## creating a count of indexes belonging to the training data
train_idx_count = int(train_perc*bins)
## creating the training indexes by shuffling through all possible dataset indexes
train_idx = np.random.choice(x_binned.shape[0], train_idx_count, replace=False)
## creating the x and y training data set by subsetting with train_idx
train_x = x_binned[train_idx, :]
train_y = y_binned[train_idx, :]

## Defining the testing index by excluding from all indexes the ones already selected for training
test_idx = np.where(np.invert(np.isin(range(bins), train_idx)))
## creating the x and y testing data set by subsetting with test_idx
test_x = x_binned[test_idx]
test_y = y_binned[test_idx]


#%% Simple DNN
# 4.1 Create the model and NN
# MODEL 1: Dense neural network

# Defining input
x = tf.keras.layers.Input(dtype='float64', shape=train_x.shape[1])
 
# Deinfing the layers
L1 = tf.keras.layers.Dense(units=target_sampling*5, activation='relu', name='L1')(x)
L2 = tf.keras.layers.Dense(units=target_sampling, activation='sigmoid', name='L2')(L1)


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

# %% Simple DNN training
# test the model accuracy
hist = model.fit(x=train_x,
                 y=train_y,
                 epochs=20,
                 batch_size=200,
                 validation_data=(test_x, test_y))

# %% Simple DNN accuracy plots
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['binary_accuracy'])
axs[1].plot(hist.epoch, hist.history['val_binary_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

# %% Simple DNN signal/prediction plots visualization
pred_test = model.predict(test_x)

ax = plt.gca()
for i in range(0, 10):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(test_y[i, :], color = color)
    plt.plot(test_x[i, :], color = color)
    plt.plot(pred_test[i, :], color = color)
    plt.show()
#%% Deeper DNN
# 4.2 Create the model and NN
# MODEL 2: Dense neural network, more layers

# Defining input
x = tf.keras.layers.Input(dtype='float64', shape=train_x.shape[1])
 
# Deinfing the layers
L1 = tf.keras.layers.Dense(units=target_sampling*5, activation='relu', name='L1')(x)
L2 = tf.keras.layers.Dense(units=target_sampling*3, activation='relu', name='L2')(L1)
L3 = tf.keras.layers.Dense(units=target_sampling*2, activation='relu', name='L3')(L2)
L4 = tf.keras.layers.Dense(units=target_sampling, activation='relu', name='L4')(L3)
L5 = tf.keras.layers.Dense(units=target_sampling, activation='sigmoid', name='L5')(L4)

# model
## definition
model = tf.keras.Model(inputs=x, outputs=[L5])
#prediction?
pred = model.predict(train_x)
error = abs(train_y - pred) 

## visualization
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)
## compiling
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# %% Deeper DNN training
# test the model accuracy
hist = model.fit(x=train_x,
                 y=train_y,
                 epochs=20,
                 batch_size=200,
                 validation_data=(test_x, test_y))

# %% Deeper DNN accuracy plots
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['binary_accuracy'])
axs[1].plot(hist.epoch, hist.history['val_binary_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()


# %% Deeper DNN signal/prediction plots visualization
pred_test = model.predict(test_x)

ax = plt.gca()
for i in range(0, 10):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(test_y[i, :], color = color)
    plt.plot(test_x[i, :], color = color)
    plt.plot(pred_test[i, :], color = color)
    plt.plot(error[i, :], color = 'red')
    plt.show()

for i in range(0,10):
    plt.plot(test_x[i,:])
    
# %% Convolutional NN
# 4.3 Creating a convolution neural network
# MODEL 3: Convolution neural network

## input x defined as previously with dimensions (number of training bins x target sampling)
x = tf.keras.layers.Input(dtype='float64', shape=(target_sampling, 1))
## Reshaping the training and testing input x so that each vector of n elements is now a list of n elements (in R linguo)
train_x_rs = tf.reshape(train_x,(train_x.shape[0], target_sampling, 1))
train_y_rs = tf.reshape(train_y, (train_y.shape[0], target_sampling, 1))
test_x_rs = tf.reshape(test_x,(test_x.shape[0], target_sampling, 1))
test_y_rs = tf.reshape(test_y, (test_y.shape[0], target_sampling, 1))
# defining global parameters for the convolutional layers
stride = 1
dilation = 1

# Defining the convolutional neural networs
## 1. convolutional layers
## convolution layers accpet input with shape (*,1); 1st argument is number of filters, 2nd argument kernel size (here 1D) 
C1 = tf.keras.layers.Conv1D(int(target_sampling/2), 13, padding = 'same', activation ='relu', input_shape= (None, target_sampling, 1), name = 'C1')(x) 
C2 = tf.keras.layers.Conv1D(int(target_sampling/3), 7, padding = 'same', activation ='relu', name = 'C2')(C1)
# 2. Pooling layer; pooling by groups of pool_size, using padding = 'same' to retain the input dimensions
P1 = tf.keras.layers.MaxPool1D(pool_size=int(target_sampling/3), strides=stride, padding='same', name = 'P1')(C2)
# 3. convolutional layer, higher filter count, reduced kernel size
C3 = tf.keras.layers.Conv1D(int(target_sampling/5), 3, padding = 'same', activation = 'relu', name = 'C3')(P1)
# 4. flatten layer; not needed as long as input (x) dimensions are maintained through the use of padding (argument ='same')
#F1 = tf.keras.layers.Flatten()(C3)
# 5. dense layer 1 and 2 to apply classical NN computations on convoluted input
D1 = tf.keras.layers.Dense(units=target_sampling*3, activation = 'relu', name='D1')(C3)
D2 = tf.keras.layers.Dense(units=target_sampling, activation = 'relu', name='D2')(D1)
D3 = tf.keras.layers.Dense(units=1, activation = 'sigmoid', name = 'D2')(D2)

## definition
model = tf.keras.Model(inputs=x, outputs=[D2])

# compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

## visualization
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

# test the model accuracy
hist = model.fit(x=train_x_rs,
                 y=train_y_rs,
                 epochs=20,
                 batch_size=200,
                 validation_data=(test_x_rs, test_y_rs))

#conv1d = tf.keras.layers.Conv1D(26, 13, padding = 'same', activation ='relu', input_shape= (None, target_sampling, 1))
# That means the last dimention should be 1
#train_x_conv=conv1d(train_x_rs) # output train_x_conv has shape (4000,129,1)
## Defining the convolution operation using a unity filter as a first attempt; simplest case possible (filter is the 1st argument to Conv1D)
## The 2nd argument (kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.)
## the kernel_size value also impacts the output dimension, e.g. : input.shape=(1000, 500, 1), conv1D(x, y)
## output.shape=(1000, 500-y+1, x)
## Unclear to me why or what it implies
#flt = tf.keras.layers.input(dtype='float64', shape=(train_x.shape[1], 1))
## Defining n, a factor to multiply the target_sampling value to create different input lenght for filtering
#n = 1
## An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
#k_size = target_sampling * n
## Default settings for stride and dilation (N.B: both can't be !=1)

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
pred_test = model.predict(test_x_rs)

ax = plt.gca()
for i in range(0, 10):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(test_y_rs[i, :], color = color)
    plt.plot(test_x_rs[i, :], color = color)
    plt.plot(pred_test[i, :], color = color)
    plt.show()
