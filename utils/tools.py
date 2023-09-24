import numpy as np
import cv2
import random
import h5py
import tensorflow.compat.v1 as tf
import keras.backend as K
import os
import scipy.io as sio

#folder_path = '/data/sar_building'# parent folder
#folder_path_fcn = '/data/sar_building_fcn'
folder_path = './data_berlinHS'

def gpu_config(gpu_id, gpu_mem=0.8):
    if gpu_mem:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print('################# gpu memory is fixed to', gpu_mem, 'on', str(gpu_id), '###################')
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        print('gpu memory is flexible.')


def SetData(istrain='test', patch_size=256, early_merge=False):
    # ##################### path #####################################
    tr_sar_path = folder_path+'/train/x1_sarimg.mat'
    tr_gis_path = folder_path+'/train/x3_gisimg.mat'
    tr_gt_wall_path = folder_path+'/train/y1_buildingIMG.mat'
    val_sar_path = folder_path+'/test/x1_sarimg.mat'
    val_gis_path = folder_path+'/test/x3_gisimg.mat'
    val_gt_wall_path = folder_path+'/test/y1_buildingIMG.mat'
    # ################################################################
    X_tr1, X_tr2, y_tr = [], [], []
    X_val1, X_val2, y_val = [], [], []

    # build test samples
    with h5py.File(val_sar_path, 'r') as file_te1:
        x1 = np.float32(file_te1['x2'])
        X_val1 = x1[:, :, :, np.newaxis]

    with h5py.File(val_gis_path, 'r') as file_te2:
        x2 = np.float32(file_te2['x3'])
        X_val2 = x2[:, :, :, np.newaxis]

    with h5py.File(val_gt_wall_path, 'r') as file_te_gt1:
        y1 = np.uint8(file_te_gt1['y1'])
        y_val = y1[:, :, :, np.newaxis]
        #bs, _, _ = np.shape(y1)
        #y_val = np.zeros((bs, patch_size, patch_size, 2))
        #y_val[:, :, :, 0] = np.uint8(y1)
        #y_val[:, :, :, 1] = 1-np.uint8(y1)

    if istrain == 'train':
        with h5py.File(tr_sar_path, 'r') as file_tr1:
            x1 = np.float32(file_tr1['x2'])
            X_tr1 = x1[:, :, :, np.newaxis]

        with h5py.File(tr_gis_path, 'r') as file_tr2:
            x2 = np.float32(file_tr2['x3'])
            X_tr2 = x2[:, :, :, np.newaxis]

        with h5py.File(tr_gt_wall_path, 'r') as file_tr_gt1:
            y1 = np.uint8(file_tr_gt1['y1'])
            y_tr = y1[:, :, :, np.newaxis]
            #bs, _, _ = np.shape(y1)
            #y_tr = np.zeros((bs, patch_size, patch_size, 2))
            #y_tr[:, :, :, 0] = np.uint8(y1)
            #y_tr[:, :, :, 1] = 1-np.uint8(y1)

    if early_merge == True:
        X_val1 = np.concatenate([X_val1, X_val2], axis=-1)
        if istrain == 'train':
            X_tr1 = np.concatenate([X_tr1, X_tr2], axis=-1)

    print(np.shape(X_tr1), np.shape(X_tr2), np.shape(y_tr))
    print(np.shape(X_val1), np.shape(X_val2), np.shape(y_val))
    return X_tr1, X_tr2, y_tr, X_val1, X_val2, y_val

def SetData_fcn(istrain='test', patch_size=256, early_merge=False):

    tr_sar_path = folder_path_fcn+'/train/x1_sarimg.mat'
    tr_gt_wall_path = folder_path_fcn+'/train/y1_buildingmaskA.mat'
    val_sar_path = folder_path_fcn+'/test/x1_sarimg.mat'
    val_gt_wall_path = folder_path_fcn+'/test/y1_buildingmaskA.mat'
    X_tr1, y_tr = [], []
    X_val1, y_val = [], []

    # build test samples
    with h5py.File(val_sar_path, 'r') as file_te1:
        print(val_sar_path)
        x1 = np.float32(file_te1['x1'])
        X_val1 = x1[:, :, :, np.newaxis]

    with h5py.File(val_gt_wall_path, 'r') as file_te_gt1:
        print(val_gt_wall_path)
        y1 = np.uint8(file_te_gt1['y1'])
        y_val = y1[:, :, :, np.newaxis]

    if istrain == 'train':
        with h5py.File(tr_sar_path, 'r') as file_tr1:
            print(tr_sar_path)
            x1 = np.float32(file_tr1['x1'])
            X_tr1 = x1[:, :, :, np.newaxis]

        with h5py.File(tr_gt_wall_path, 'r') as file_tr_gt1:
            print(tr_gt_wall_path)
            y1 = np.uint8(file_tr_gt1['y1'])
            y_tr = y1[:, :, :, np.newaxis]

    if early_merge == True:
        X_val1 = np.concatenate([X_val1, X_val2], axis=-1)
        if istrain == 'train':
            X_tr1 = np.concatenate([X_tr1, X_tr2], axis=-1)

    print(np.shape(X_tr1), np.shape(y_tr))
    print(np.shape(X_val1), np.shape(y_val))
    return X_tr1, y_tr, X_val1, y_val


def TestModel(model, dataset, out_folder='model', patch_size=256, slide_size=128, task='seg'):

    #task: 'seg' -- segmentation
    #      'he' -- height estimation
    if task == 'seg':
        output_channel = 6
    elif task == 'un':
        output_channel = 10
    else:
        output_channel = 1

    if dataset == 'vaihingen':
        test_set = ['11', '15', '28', '30', '34']      
    elif dataset == 'potsdam':
        test_set = ['2_11', '2_12', '4_10', '5_11', '6_7', '7_10', '7_8']
    elif dataset == 'zeebrugge':
        test_set = ['4', '6']
    elif dataset == 'norm_sar_edit' or dataset == 'na_sar_edit' or dataset == 'sar_edit':
        test_set = ['N53E009', 'N53E010', 'N54E009', 'N54E010', 'N55E009', 'N55E010']
    elif dataset == 'dfc18':
        test_set = ['274440_3289689', '273248_3290290', '272652_3290290', '272056_3289689']

    input_path = folder_path + dataset + '/test_images/inputs/' 
    output_path = folder_path + dataset + '/test_images/outputs/'

    if not os.path.isdir(output_path+out_folder):
        os.mkdir(output_path+out_folder)

    for im_id in range(len(test_set)):

        mat_file = input_path + 'im_input' + test_set[im_id] + '.mat'
        print(mat_file)
        with h5py.File(mat_file) as file_tr:
            X_tr = file_tr['im_input']
            print(np.shape(X_tr))
            if task=='sar':
                X_tr = np.expand_dims(X_tr, axis=0)        
            X_tr = np.swapaxes(X_tr, 0, 3)
            X_tr = np.swapaxes(X_tr, 1, 2)
        if task=='he':
            X_tr = X_tr[:,:,:, 0:3]

        print(np.shape(X_tr))
        X_tr = np.float32(X_tr)
        _, im_row, im_col, _ = np.shape(X_tr)
        steps_col = int(np.floor((im_col-(patch_size-slide_size))/slide_size))
        steps_row = int(np.floor((im_row-(patch_size-slide_size))/slide_size))
        #im_out_gt = np.zeros((im_row, im_col))
        im_out_all = np.zeros((im_row, im_col, output_channel))
        im_index = np.zeros((im_row, im_col, output_channel))

        for i in range(steps_row+1):
            for j in range(steps_col+1):
                if i == steps_row:
                    if j == steps_col:
                        patch_input = X_tr[0, -patch_size:im_row, -patch_size:im_col, :]
                        patch_input = patch_input[np.newaxis, :]
                        patch_out_gt = model.predict(patch_input)
                        im_out_all[-patch_size:im_row, -patch_size:im_col, :] += patch_out_gt[0]
                        im_index[-patch_size:im_row, -patch_size:im_col, :] += np.ones((patch_size, patch_size, output_channel))
                    else:
                        patch_input =  X_tr[0, -patch_size:im_row, (j*slide_size):(j*slide_size+patch_size), :]
                        patch_input = patch_input[np.newaxis, :]
                        patch_out_gt = model.predict(patch_input)
                        im_out_all[-patch_size:im_row, (j*slide_size):(j*slide_size+patch_size), :] += patch_out_gt[0]
                        im_index[-patch_size:im_row, (j*slide_size):(j*slide_size+patch_size), :] += np.ones((patch_size, patch_size, output_channel))
                else:
                    if j == steps_col:
                        patch_input =  X_tr[0, (i*slide_size):(i*slide_size+patch_size), -patch_size:im_col, :]
                        patch_input = patch_input[np.newaxis, :]
                        patch_out_gt = model.predict(patch_input)
                        im_out_all[(i*slide_size):(i*slide_size+patch_size), -patch_size:im_col, :] += patch_out_gt[0]
                        im_index[(i*slide_size):(i*slide_size+patch_size), -patch_size:im_col, :] += np.ones((patch_size, patch_size, output_channel))
                    else:
                        patch_input =  X_tr[0, (i*slide_size):(i*slide_size+patch_size), (j*slide_size):(j*slide_size+patch_size), :]
                        patch_input = patch_input[np.newaxis, :]
                        patch_out_gt = model.predict(patch_input)
                        im_out_all[(i*slide_size):(i*slide_size+patch_size), (j*slide_size):(j*slide_size+patch_size), :] += patch_out_gt[0]
                        im_index[(i*slide_size):(i*slide_size+patch_size), (j*slide_size):(j*slide_size+patch_size), :] += np.ones((patch_size, patch_size, output_channel))

        im_out_all = im_out_all/im_index
        print(output_path + out_folder + '/output_' + test_set[im_id] + '.mat')
        sio.savemat(output_path + out_folder + '/output_'+ test_set[im_id] + '.mat', {'out_all': im_out_all})

    return 'all outputs are saved.'


