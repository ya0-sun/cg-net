from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, UpSampling2D, ZeroPadding2D, Flatten, LSTM, TimeDistributed, Reshape, Bidirectional, Permute, Lambda, Add, Concatenate, Dot, Dropout, Multiply, RepeatVector
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D as GAP
from tensorflow.keras.layers import GlobalMaxPooling2D as GMP
from tensorflow.keras import regularizers, layers
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant
import tensorflow.compat.v1 as tf


#print tf.__version__

cnn_weights = '/work/hua/weights/classification/baseline_models/'
fcn_weights = '/work/hua/weights/segmentation/FCN_baseline/'

def VGG16(patch_size, nb_channel, pre_train=False):
    img_input = Input(shape=(patch_size, patch_size, nb_channel), name='input1')
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(img_input, x, name='vgg16')
    if pre_train:
        model.load_weights(cnn_weights+'vgg16_weights.h5', by_name=True)
        print('pre-trained vgg16 is loaded!')

    return model


def VGG16_bn(patch_size, nb_channel, pre_train=False, input_norm=False):
   
    img_input = Input(shape=(patch_size, patch_size, nb_channel), name='input1')
    if input_norm:
        x = BatchNormalization(name='input_norm')(img_input)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
    else:
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    print(img_input)
    x = BatchNormalization(momentum=0.01, name='block1_bn1')(x)
    x = Activation('relu', name='block1_ac1')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization(momentum=0.01, name='block1_bn2')(x)
    x = Activation('relu', name='block1_ac2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization(momentum=0.01, name='block2_bn1')(x)
    x = Activation('relu', name='block2_ac1')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization(momentum=0.01, name='block2_bn2')(x)
    x = Activation('relu', name='block2_ac2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization(momentum=0.01, name='block3_bn1')(x)
    x = Activation('relu', name='block3_ac1')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization(momentum=0.01, name='block3_bn2')(x)
    x = Activation('relu', name='block3_ac2')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization(momentum=0.01, name='block3_bn3')(x)
    x = Activation('relu', name='block3_ac3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization(momentum=0.01, name='block4_bn1')(x)
    x = Activation('relu', name='block4_ac1')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization(momentum=0.01, name='block4_bn2')(x)
    patch_sizex = Activation('relu', name='block4_ac2')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization(momentum=0.01, name='block4_bn3')(x)
    x = Activation('relu', name='block4_ac3')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization(momentum=0.01, name='block5_bn1')(x)
    x = Activation('relu', name='block5_ac1')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization(momentum=0.01, name='block5_bn2')(x)
    x = Activation('relu', name='block5_ac2')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization(momentum=0.01, name='block5_bn3')(x)
    x = Activation('relu', name='block5_ac3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(img_input, x, name='vgg16_bn')
    if pre_train:
        model.load_weights(cnn_weights+'vgg16_weights.h5', by_name=True)
        print('pre-trained vgg16 is loaded!')

    return model


def FCN_blog(patch_size, nb_channel, nb_classes, pre_train=False):

    base_model = VGG16(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_out')(x)
    x3 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]))(x)

    x = base_model.get_layer('block4_pool').output
    x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_out')(x)
    x4 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]))(x)

    x = base_model.get_layer('block5_pool').output
    x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_out')(x)
    x5 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]))(x)

    x = Add(name='sum_345')([x3, x4, x5])
    x = Activation('softmax')(x)

    model = Model(base_model.input, x, name='fcn_blog')

    return model


def FCN_blog_bn(patch_size, nb_channel, nb_classes, pre_train=False):

    base_model = VGG16_bn(patch_size, nb_channel, pre_train)

    x = base_model.get_layer('block3_pool').output
    x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block3_out')(x)
    x3 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x3_up')(x)

    x = base_model.get_layer('block4_pool').output
    x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block4_out')(x)
    x4 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x4_up')(x)

    x = base_model.get_layer('block5_pool').output
    x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', name='block5_out')(x)
    x5 = Lambda(lambda image: tf.image.resize_bilinear(image, [patch_size, patch_size]), name='x5_up')(x)

    x = Add(name='sum_345')([x3, x4, x5])
    print(x)
    x = Activation('softmax')(x)

    model = Model(base_model.input, x, name='fcn_blog_bn')
    
    return model


def identity_block(input_tensor, kernel_size, filters, stage, block, atro = 1):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', dilation_rate=atro)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b', dilation_rate=atro)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', dilation_rate=atro)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), atro=1):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a', dilation_rate=atro)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b', dilation_rate=atro)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', dilation_rate=atro)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1', dilation_rate=atro)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50_my(im_size, nb_channel):

    img_input = Input(shape=(im_size, im_size, nb_channel))

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1))
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', atro=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', atro=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', atro=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', atro=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', atro=2)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), atro=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', atro=4)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', atro=4)

    #x = GAP()(x)
    print(x)

    model = Model(img_input, x, name='resnet50')
   
    return model 
