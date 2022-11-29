'''
File created by Reza Kalantar - 29/11/2022
'''

import tensorflow as tf

def least_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))

def cycle_loss(y_true, y_pred):
    return 0.5*(1 - tf.image.ssim(y_pred, y_true, max_val=2.0)[0]) + 0.5*tf.reduce_mean(tf.abs(y_pred - y_true))

def discriminator_loss(real, generated):
    real_loss = least_squared_error(tf.ones_like(real), real)
    generated_loss = least_squared_error(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5

def generator_loss(generated):
    return least_squared_error(tf.ones_like(generated), generated)

def identity_loss(real_image, same_image, LAMBDA=10):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))