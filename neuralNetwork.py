# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:59:39 2023

@author: Guilherme
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2DTranspose, UpSampling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD
from tensorflow.keras import backend as K
from keras.saving import register_keras_serializable, get_custom_objects
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss

class TotalLoss(Loss):
    def __init__(self, model, lambda_mse, lambda_gs, lambda_l2, lambda_huber, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.lambda_mse = lambda_mse
        self.lambda_gs = lambda_gs
        self.lambda_l2 = lambda_l2
        self.lambda_huber = lambda_huber

    def call(self, y_true, y_pred):
        return total_loss(y_true, y_pred, self.model, self.lambda_mse, self.lambda_gs, self.lambda_l2, self.lambda_huber)


class MaskingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_pred, mask = inputs
        inverse_mask = 1 - mask
        return y_pred * mask + inverse_mask * 0
    
def MSEshared(y_true, y_pred, lambda_mse):
    mask = tf.cast(tf.greater(y_true, 0), dtype='float32')
    m = tf.shape(y_true)[0]
    nx = tf.shape(y_true)[1]
    ny = tf.shape(y_true)[2]
    U_true, V_true = tf.split(y_true * mask, 2, axis=-1)
    U_pred, V_pred = tf.split(y_pred * mask, 2, axis=-1)
    U_diff = K.square(U_true - U_pred)
    V_diff = K.square(V_true - V_pred)
    sum_diff = K.sum(U_diff[:, 1:-1, 1:-1] + V_diff[:, 1:-1, 1:-1])
    loss = lambda_mse * sum_diff / tf.cast(m * (nx - 2) * (ny - 2), tf.float32)
    return loss

def GSshared(y_true, y_pred, lambda_gs):
    mask = tf.cast(tf.greater(y_true, 0), dtype='float32')
    m = tf.shape(y_true)[0]
    nx = tf.shape(y_true)[1]
    ny = tf.shape(y_true)[2]
    U_true, V_true = tf.split(y_true * mask, 2, axis=-1)
    U_pred, V_pred = tf.split(y_pred * mask, 2, axis=-1)
    dUdx_true = U_true[:, 2:, :] - U_true[:, :-2, :]
    dUdx_pred = U_pred[:, 2:, :] - U_pred[:, :-2, :]
    dVdx_true = V_true[:, 2:, :] - V_true[:, :-2, :]
    dVdx_pred = V_pred[:, 2:, :] - V_pred[:, :-2, :]
    dU_diff = K.square(dUdx_true - dUdx_pred)
    dV_diff = K.square(dVdx_true - dVdx_pred)
    sum_diff = K.sum(dU_diff[:, 1:-1, 1:-1] + dV_diff[:, 1:-1, 1:-1])
    loss = lambda_gs * sum_diff / tf.cast(6 * m * (nx - 2) * (ny - 2), tf.float32)
    return loss


def L2regularization(theta, lambda_l2):
    l2_reg = 0
    for weight in theta:
        l2_reg += tf.reduce_sum(tf.square(weight))
    return lambda_l2 * l2_reg / 2

def huber_loss(y_true, y_pred,lambda_huber):
    delta = 3
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = tf.square(error) / 2
    linear_loss  = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss*lambda_huber, linear_loss*lambda_huber)




@register_keras_serializable(package="my_package", name="my_loss_fn")
def total_loss(y_true, y_pred, model, lambda_mse, lambda_gs, lambda_l2, lambda_huber):
    return MSEshared(y_true, y_pred, lambda_mse)  + L2regularization(model.trainable_weights, lambda_l2)+ GSshared(y_true, y_pred, lambda_gs) + huber_loss(y_true, y_pred, lambda_huber)

@register_keras_serializable(package="my_package", name="my_loss_fn_wrapper")
def get_total_loss(model, lambda_mse, lambda_gs, lambda_l2,lambda_huber):
    def loss_fn(y_true, y_pred):
        return total_loss(y_true, y_pred, model, lambda_mse, lambda_gs, lambda_l2,lambda_huber)
    return loss_fn

# Registrar a função de perda personalizada
get_custom_objects().update({'my_loss_fn_wrapper': get_total_loss})


def trainNeuralNetwork(lambda_mse = 0.03, lambda_gs = 0.1, lambda_l2=1e-5, lambda_huber=0.9, lr=0.3, filters=300):
    get_custom_objects().clear()
    # Definindo o codificador
    input_img = Input(shape=(300, 300, 1))  # adaptar isso para o tamanho da sua imagem
    x = Conv2D(filters, (5, 5), activation=swish, padding='same')(input_img)
    x = MaxPooling2D((5, 5))(x)
    x = Conv2D(filters, (5, 5), activation=swish, padding='same')(x)
    x = MaxPooling2D((5, 5))(x)
    x = Conv2D(filters, (3, 3), activation=swish, padding='same')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(filters, (2, 2), activation=swish, padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    encoded = Flatten()(x)
    # Adicionando o ângulo de ataque e o número de Reynolds como entrada
    input_conditions = Input(shape=(2,))  # adaptar isso para o tamanho do seu vetor de condições
    input2Flat = Flatten()(input_conditions)
    merged = Concatenate()([input2Flat,encoded ])    
    # Definindo o decodificador
    x = Dense(filters*2*2, activation=swish)(merged)
    x = Reshape((2,2,filters))(x)
    x = Conv2D(filters, (2, 2), padding='same', activation=swish)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(filters, (3, 3), padding='same', activation=swish)(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(filters, (5, 5), padding='same', activation=swish)(x)
    x = UpSampling2D((5, 5))(x)
    x = Conv2D(filters, (5, 5), padding='same', activation=swish)(x)
    decoded = UpSampling2D((5, 5))(x)
    output = Conv2D(1, (1, 1), padding='same')(decoded)
    mask = tf.cast(tf.greater(input_img, 0), dtype='float32')
    masked_output = MaskingLayer()([output, mask])

    # Definindo o modelo
    autoencoder = Model(inputs=[input_img, input_conditions], outputs=[masked_output])
    #autoencoder.summary()


    autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gs, lambda_l2,lambda_huber),metrics = tf.keras.metrics.MeanAbsolutePercentageError())

    #autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gs, lambda_l2))
    
    return autoencoder