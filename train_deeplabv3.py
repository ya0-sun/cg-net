import os
import sys
sys.path.append(os.path.abspath('./utils'))
from networks_deeplabv3 import *
from tools import *

from keras.optimizers import SGD, RMSprop, Nadam
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np

# ************************* Configuretion **************************
gpu_config(4, 0.7)

# ***************************** path *******************************
#weight_path = 'weights/fcn_both_d8_xx.h5' # weight name
weight_path = 'weights/fbg_both_insar_deeplabv3_spade.h5' # weight name

# ************************ training scheme *************************
batch_size = 5
epochs = 40
lr = 1e-3 #2e-4, 2e-5 !!!!!!!!!!!!!!!!!!!!!! if loss is large (11.xxx), reduce 10

# ********************** network parameters ************************
pre_train = False
optimizer = Nadam(lr=lr)
#loss = 'categorical_crossentropy'
loss = 'binary_crossentropy'

# ********************** image configuration ***********************
patch_size = 256
nb_channel = 2
nb_classes = 1
downsample_ratio = 8
istrain = 'train'

# ************************ initialize model ************************
model = deeplabv3_SPADE(patch_size, nb_channel, nb_classes, downsample_ratio)
#model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
print('model is built.')

# *************************** monitoring ***************************
history = History()
model_checkpoint= ModelCheckpoint(weight_path, monitor="val_accuracy", save_best_only=True, save_weights_only=True, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-10)
earlystopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0, patience=5, mode='max')

# ************************ data preparation ************************
print('loading data...')
X_tr1, X_tr2, y_tr, X_val1, X_val2, y_val = SetData(istrain=istrain, patch_size=patch_size)
print('data is loaded.')

# *************************** training *****************************
#model.fit([np.concatenate([X_tr1, X_tr2], -1), X_tr2], y_tr, batch_size=batch_size, epochs=epochs, validation_data = ([np.concatenate([X_val1, X_val2], -1), X_val2], y_val), callbacks=[history, model_checkpoint, lr_reducer])
model.fit([np.concatenate([X_tr1, X_tr2], -1), X_tr2], y_tr, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[history, model_checkpoint, lr_reducer])

'''
# OA & Kappa
model.load_weights(weight_path, by_name=True)
test_loss, test_acc = model.evaluate(X_val, y_val1)
print(test_loss)
print(test_acc)
'''

