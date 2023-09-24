from base_model import *
import tensorflow.compat.v1 as tf
#from keras.applications.resnet import ResNet152, ResNet50
#from tensorflow.contrib.slim.nets import resnet_v2
from keras import regularizers, initializers
#from keras.applications.vgg16 import VGG16
#from keras.applications.densenet import DenseNet121
import numpy as np

def deeplabv3_baseline(patch_size, nb_channel, nb_classes, is_training=False, output_stride=8, depth=256):

    if output_stride == 8:
        atrous_rates = [2*rate for rate in [6, 12, 18]]
    else:
        atrous_rates = [6, 12, 18]

    base_model = ResNet50_my(patch_size, nb_channel)
    x = base_model.output
    print(x)

    x_conv_1x1 = Conv2D(depth, (1, 1), strides=1, name='conv_1x1')(x)
    x_conv_3x3_1 = Conv2D(depth, (3, 3), strides=1, dilation_rate=atrous_rates[0], padding='same', name='conv_3x3_1')(x)
    x_conv_3x3_2 = Conv2D(depth, (3, 3), strides=1, dilation_rate=atrous_rates[1], padding='same', name='conv_3x3_2')(x)
    x_conv_3x3_3 = Conv2D(depth, (3, 3), strides=1, dilation_rate=atrous_rates[2], padding='same', name='conv_3x3_3')(x)

    image_level_features = GAP()(x)
    _, w, h, feature_channel = K.int_shape(x)
    image_level_features = Reshape((1, 1, feature_channel))(image_level_features)
    image_level_features = Conv2D(depth, (1, 1), strides=1, name = 'image_level_features_conv_1x1')(image_level_features)
    image_level_features = Lambda(lambda image: tf.image.resize_bilinear(image, [w, h]), name='image_level_features_resize')(image_level_features)

    x = Concatenate(axis=-1, name='concat')([x_conv_1x1, x_conv_3x3_1, x_conv_3x3_2, x_conv_3x3_3, image_level_features])
    x = Conv2D(depth, (1, 1), strides=1, name='conv_1x1_concat')(x)
    x = Conv2D(nb_classes, (1, 1), activation='linear', name='upsampling_conv_1x1')(x)
    x = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='upsampling_resize')(x)
    x = Activation('sigmoid', name='predictions')(x)
    print(x)
    model = Model(base_model.input, x, name='deeplabv3_baseline')

    return model


def SPADE(x, mask, feat_size, filter_num, filter_size, name):
   #x should be normalized
    x_norm = BatchNormalization(momentum=0.01)(x)
    mask = Lambda(lambda image: tf.image.resize_bilinear(image, [feat_size, feat_size]), name=name+'_resize')(mask)
    mask_embedding = Conv2D(int(filter_num/4), (filter_size, filter_size), activation='relu', padding='same', name=name+'_emb')(mask)
    gamma = Conv2D(filter_num, (filter_size, filter_size), activation='linear', padding='same', name=name+'_gamma')(mask_embedding)
    beta = Conv2D(filter_num, (filter_size, filter_size), activation='linear', padding='same', name=name+'_beta')(mask_embedding)
    x = Multiply()([x_norm, gamma])
    x = Add()([x_norm, x, beta])
    return x

def deeplabv3_SPADE(patch_size, nb_channel, nb_classes, downsample_ratio=4, output_stride=8, depth=256):

    if output_stride == 8:
        atrous_rates = [2*rate for rate in [6, 12, 18]]
    else:
        atrous_rates = [6, 12, 18]

    x_gis = Input(shape=(patch_size, patch_size, 1), name='input_gis')
    base_model = ResNet50_my(patch_size, nb_channel)
    x = base_model.output
    print(x)

    x_conv_1x1 = Conv2D(depth, (1, 1), strides=1, name='conv_1x1')(x)
    x_conv_3x3_1 = Conv2D(depth, (3, 3), strides=1, dilation_rate=atrous_rates[0], padding='same', name='conv_3x3_1')(x)
    x_conv_3x3_2 = Conv2D(depth, (3, 3), strides=1, dilation_rate=atrous_rates[1], padding='same', name='conv_3x3_2')(x)
    x_conv_3x3_3 = Conv2D(depth, (3, 3), strides=1, dilation_rate=atrous_rates[2], padding='same', name='conv_3x3_3')(x)

    image_level_features = GAP()(x)
    _, w, h, feature_channel = K.int_shape(x)
    image_level_features = Reshape((1, 1, feature_channel))(image_level_features)
    image_level_features = Conv2D(depth, (1, 1), strides=1, name = 'image_level_features_conv_1x1')(image_level_features)
    image_level_features = Lambda(lambda image: tf.image.resize_bilinear(image, [w, h]), name='image_level_features_resize')(image_level_features)

    x = Concatenate(axis=-1, name='concat')([x_conv_1x1, x_conv_3x3_1, x_conv_3x3_2, x_conv_3x3_3, image_level_features])
    x = Conv2D(depth, (1, 1), strides=1, name='conv_1x1_concat')(x)
    # deeplabv3 already downsample to 256 !!!!!!!!!!!1
    _, w, h, new_channel = K.int_shape(x)
    x = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='upsampling_resize')(x)
    print(x)
    x = SPADE(x, x_gis, patch_size, new_channel, 3, 'after_concat')

    x = Conv2D(nb_classes, (1, 1), activation='linear', name='upsampling_conv_1x1')(x)
    x = Activation('sigmoid', name='predictions')(x)
    print(x)
    model = Model([base_model.input, x_gis], x, name='deeplabv3_spade')

    return model



'''
def SPADE2(x, mask, feat_size, filter_num, filter_size, name):
   #x should be normalized
    x_norm = BatchNormalization(momentum=0.01)(x)
    mask = Lambda(lambda image: tf.image.resize_bilinear(image, [feat_size, feat_size]), name=name+'_resize')(mask)
    mask_embedding = Conv2D(int(filter_num/4), (filter_size, filter_size), activation='relu', padding='same', name=name+'_emb')(mask)
    gamma = Conv2D(1, (filter_size, filter_size), activation='linear', padding='same', name=name+'_gamma')(mask_embedding)
    beta = Conv2D(1, (filter_size, filter_size), activation='linear', padding='same', name=name+'_beta')(mask_embedding)
    x = Multiply()([x_norm, gamma])
    x = Add()([x_norm, x, beta])
    return x


def SPADE_module(x, mask, feat_size, filter_num, filter_size, name):
    x_in = x
    x = SPADE(x, mask, feat_size, filter_num, filter_size, name+'spade1')
    x = Activation('relu')(x)
    x = Conv2D(filter_num, (3, 3), activation='linear', padding='same', name=name+'spade1_conv')(x)
    x = SPADE(x, mask, feat_size, filter_num, filter_size, name+'spade2')
    x = Activation('relu')(x)
    x = Conv2D(filter_num, (3, 3), activation='linear', padding='same', name=name+'spade2_conv')(x)
    x = Add()([x_in, x])
    return x
'''

def sar_building(patch_size, nb_channel, nb_classes, pre_train=False):

    # no early fusion, only SPADE
    x_gis = Input(shape=(patch_size, patch_size, 1), name='input_gis')
    base_model = VGG16_bn(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x = SPADE(x, x_gis, patch_size/8, 256, 3, 'block3')
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_b')(x)
    x3_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x3_up_b')(x_b)

    x = base_model.get_layer('block4_pool').output
    x = SPADE(x, x_gis, patch_size/16, 512, 3, 'block4')
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_b')(x)
    x4_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up_b')(x_b)

    x = base_model.get_layer('block5_pool').output
    x = SPADE(x, x_gis, patch_size/32, 512, 3, 'block5')
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_b')(x)
    x5_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up_b')(x_b)

    x_b = Add(name='sum_345_b')([x3_b, x4_b, x5_b])
    x_b = Activation('sigmoid', name='pred_b')(x_b)

    model = Model([base_model.input, x_gis], x_b, name='fcn_sar_building')

    return model

def sar_building3(patch_size, nb_channel, nb_classes, downsample_ratio=4, pre_train=False):

    # early fusion + SPADE, before classifier
    x_gis = Input(shape=(patch_size, patch_size, 1), name='input_gis')
    base_model = VGG16_bn(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x = Conv2D(int(256/downsample_ratio), (1, 1), activation='linear', padding='same', name='block3_red')(x)
    x = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x3_up_b')(x)
    x = SPADE(x, x_gis, patch_size, int(256/downsample_ratio), 3, 'block3')
    x3_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_b')(x)

    x = base_model.get_layer('block4_pool').output
    x = Conv2D(int(512/downsample_ratio), (1, 1), activation='linear', padding='same', name='block4_red')(x)
    x = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up_b')(x)
    x = SPADE(x, x_gis, patch_size, int(512/downsample_ratio), 3, 'block4')
    x4_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_b')(x)

    x = base_model.get_layer('block5_pool').output
    x = Conv2D(int(512/downsample_ratio), (1, 1), activation='linear', padding='same', name='block5_red')(x)
    x = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up_b')(x)
    x = SPADE(x, x_gis, patch_size, int(512/downsample_ratio), 3, 'block5')
    x5_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_b')(x)

    x_b = Add(name='sum_345_b')([x3_b, x4_b, x5_b])
    x_b = Activation('sigmoid', name='pred_b')(x_b)

    model = Model([base_model.input, x_gis], x_b, name='fcn_sar_building')

    return model


def sar_building_baseline(patch_size, nb_channel, nb_classes, pre_train=False):

    base_model = VGG16_bn(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_b')(x)
    x3_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x3_up_b')(x_b)

    x = base_model.get_layer('block4_pool').output
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_b')(x)
    x4_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up_b')(x_b)

    x = base_model.get_layer('block5_pool').output
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_b')(x)
    x5_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up_b')(x_b)

    x_b = Add(name='sum_345_b')([x3_b, x4_b, x5_b])
    x_b = Activation('sigmoid', name='pred_b')(x_b)

    model = Model(base_model.input, x_b, name='fcn_sar_building')

    return model

def sar_building_baseline_v2(patch_size, nb_channel, nb_classes, pre_train=False):

    base_model = VGG16_bn(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_b')(x)
    x3_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x3_up_b')(x_b)

    x = base_model.get_layer('block4_pool').output
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_b')(x)
    x4_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up_b')(x_b)

    x = base_model.get_layer('block5_pool').output
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_b')(x)
    x5_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up_b')(x_b)

    x_b = Add(name='sum_345_b')([x3_b, x4_b, x5_b])
    x_b = Activation('softmax', name='pred_b')(x_b)

    model = Model(base_model.input, x_b, name='fcn_sar_building')

    return model


def sar_building_module(patch_size, nb_channel, nb_classes, downsample_ratio=4, pre_train=False):

    x_gis = Input(shape=(patch_size, patch_size, 1), name='input_gis')
    base_model = VGG16_bn(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x = Conv2D(int(256/downsample_ratio), (1, 1), activation='linear', padding='same', name='block3_red')(x)
    x3_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x3_up_b')(x_b)
    x = SPADE_module(x, x_gis, patch_size/8, int(256/downsample_ratio), 3, 'block3')
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_b')(x)

    x = base_model.get_layer('block4_pool').output
    
    x = SPADE_module(x, x_gis, patch_size/16, 512, 3, 'block4')
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_b')(x)
    x4_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up_b')(x_b)

    x = base_model.get_layer('block5_pool').output
    x = SPADE_module(x, x_gis, patch_size/32, 512, 3, 'block5')
    x_b = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_b')(x)
    x5_b = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up_b')(x_b)

    x_b = Add(name='sum_345_b')([x3_b, x4_b, x5_b])
    x_b = Activation('sigmoid', name='pred_b')(x_b)

    model = Model([base_model.input, x_gis], x_b, name='fcn_sar_building')

    return model

