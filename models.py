'''
File created by Reza Kalantar - 29/11/2022
'''

import numpy as np
import tensorflow as tf
from tensorflow import pad
import tensorflow_addons as tfa
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.activations import *

# reflection padding taken from: https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        size_increase = [0, 2*self.padding[0], 2*self.padding[1], 2*self.padding[2], 0]
        output_shape = list(s)

        for i in range(len(s)):
            if output_shape[i] == None:
                continue
            output_shape[i] += size_increase[i]

        return tuple(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class volumePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_vols = 0
            self.volumes = []

    def query(self, volumes):
        if self.pool_size == 0:
            return volumes
        return_volumes = []
        for volume in volumes:
            if len(volume.shape) == 4:
                volume = volume[np.newaxis, :, :, :, :]

            if self.num_vols < self.pool_size:  # fill up the volume pool
                self.num_vols = self.num_vols + 1
                if len(self.volumes) == 0:
                    self.volumes = volume
                else:
                    self.volumes = np.vstack((self.volumes, volume))

                if len(return_volumes) == 0:
                    return_volumes = volume
                else:
                    return_volumes = np.vstack((return_volumes, volume))

            else:  # 50% chance that we replace an old synthetic volume
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.volumes[random_id, :, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :, :]
                    self.volumes[random_id, :, :, :, :] = volume[0, :, :, :, :]
                    if len(return_volumes) == 0:
                        return_volumes = tmp
                    else:
                        return_volumes = np.vstack((return_volumes, tmp))
                else:
                    if len(return_volumes) == 0:
                        return_volumes = volume
                    else:
                        return_volumes = np.vstack((return_volumes, volume))

        return return_volumes

# First generator layer
def conv_block_g(x, k):
    x = Conv3D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x, training=True)
    x = Activation('relu')(x)
    return x

# Downsampling
def downsample(x, k):  # Should have reflection padding
    x = Conv3D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=True)(x)
    x = tfa.layers.InstanceNormalization()(x, training=True)
    x = Activation('relu')(x)
    return x

# Residual block
def residualblock(x0, use_dropout=False, use_bias=True):
    k = int(x0.shape[-1])

    # First layer
    x = ReflectionPadding3D((1,1,1))(x0)
    x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization()(x, training=True)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    # Second layer
    x = ReflectionPadding3D((1, 1, 1))(x)
    x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization()(x, training=True)
    # Merge
    x = add([x, x0])
    return x

# Upsampling
def upsample(x, k, use_bias=True, use_resize_convolution=False):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    if use_resize_convolution:
        x = UpSampling3D(size=(2, 2, 2))(x)  # Nearest neighbor upsampling
        x = ReflectionPadding3D((1, 1, 1))(x)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=use_bias)(x)
    else:
        x = Conv3DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=use_bias)(x)  # this matches fractionally stided with stride 1/2
    x = tfa.layers.InstanceNormalization()(x, training=True)
    x = Activation('relu')(x)
    return x

def modelGenerator(input_shape, generator_residual_blocks):

    input_img = Input(shape=input_shape)

    x = ReflectionPadding3D((3, 3, 3))(input_img)
    x = conv_block_g(x, 32)
    x = downsample(x, 64)
    x = downsample(x, 128)

    for _ in range(generator_residual_blocks):
        x = residualblock(x)

    x = upsample(x, 64)
    x = upsample(x, 32)
    x = ReflectionPadding3D((3, 3, 3))(x)
    x = Conv3D(1, kernel_size=7, strides=1)(x)
    x = Activation('tanh')(x)
    return Model(inputs=input_img, outputs=x)

def conv_block_d(x, k, use_normalization, stride):
    x = Conv3D(filters=k, kernel_size=4, strides=stride, padding='same')(x)
    if use_normalization:
        x = tfa.layers.InstanceNormalization()(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def modelDiscriminator(input_shape):

    input_img = Input(shape=input_shape)

    x = conv_block_d(input_img, 64, False, 2) #Instance normalization is not used for this layer
    x = conv_block_d(x, 128, True, 2)
    x = conv_block_d(x, 256, True, 2)
    x = conv_block_d(x, 512, True, 1)
    x = Conv3D(filters=1, kernel_size=4, strides=1, padding='same')(x)
    x = Activation('sigmoid')(x)
    return Model(inputs=input_img, outputs=x)