import os
import sys
sys.path.append(os.path.abspath('./utils'))
from networks import *
from tools import *

from keras.optimizers import SGD, RMSprop, Nadam
import tensorflow as tf
import numpy as np

# ************************* Configuretion **************************
gpu_config(5, 0.99)

# ***************************** path *******************************
weight_path = 'weights/cgnet_fcn.h5'

# *********************** training scheme *************************
batch_size = 5
epochs = 100
lr = 2e-3

# ********************** network parameters ************************
pre_train = False
optimizer = Nadam(lr=lr)
#loss = ['categorical_crossentropy']
loss = ['binary_crossentropy']

# ********************** image configuration ***********************
patch_size = 256
nb_channel = 2
nb_classes = 1
istrain = 'test'
downsample_ratio = 8

# ************************ initialize model ************************
model = sar_building3(patch_size, nb_channel, nb_classes, downsample_ratio, pre_train)
model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
print('model is built.')

# ************************ data preparation ************************
print('loading data...')
_, _, _, X_val1, X_val2, y_val = SetData(istrain=istrain, patch_size=patch_size)
print('data is loaded.')

# *************************** training *****************************
# OA & Kappa
model.load_weights(weight_path, by_name=True)

# ####################### produce outputs ##########################
y_pred = model.predict([np.concatenate([X_val1, X_val2], -1), X_val2])
y_pred = np.uint8(np.round(y_pred))

with h5py.File('y_pred_both_d8_prob_insar.mat', 'w') as f:
    f['y_pred'] = y_pred

print(np.shape(y_pred))
c, h, w, _ = np.shape(y_pred)
num = w*h*c

y_val = np.float32(y_val)
y_pred = np.float32(y_pred)
tp=np.sum((y_val + y_pred)==2)
tn=np.sum((y_val + y_pred)==0)
fp=np.sum((y_val - y_pred)==1)
fn=np.sum((y_val - y_pred)==-1)

oa = np.float32(tp+tn)/np.float32(num)
acc_building = np.float32(tp)/np.float32(tp+fp)
acc_background = np.float32(tn)/np.float32(tn+fn)
iou = np.float32(tp)/np.float32(tp+fp+fn)

print('oa:', oa, ', iou:', iou)
#sio.savemat('X_val.mat', {'x': X_val})
####################################################################
