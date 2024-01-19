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
    def __init__(self, model, lambda_mse, lambda_gdl, lambda_l2, lambda_huber, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.lambda_mse = lambda_mse
        self.lambda_gdl = lambda_gdl
        self.lambda_l2 = lambda_l2
        self.lambda_huber = lambda_huber

    def call(self, y_true, y_pred):
        return total_loss(y_true, y_pred, self.model, self.lambda_mse, self.lambda_gdl, self.lambda_l2, self.lambda_huber)


class MaskingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_pred, mask = inputs
        inverse_mask = 1 - mask
        return y_pred * mask + inverse_mask * 0
    
def mse_loss(y_true, y_pred, lambda_mse):
    mask = tf.cast(tf.greater(y_true, 0), dtype='float32')
    y_true = y_true*mask
    y_pred = y_pred*mask
    
    loss = (y_pred-y_true)**2 * lambda_mse
    
    return loss

def gdl_loss(y_true, y_pred,lambda_gdl):
    # Calculate the difference in the x direction
    diff_x_true = tf.abs(y_true[:, :, 1:] - y_true[:, :, :-1])
    diff_x_pred = tf.abs(y_pred[:, :, 1:] - y_pred[:, :, :-1])    
    loss_x = tf.reduce_sum(tf.abs(diff_x_true - diff_x_pred),axis=1)

    # Calculate the difference in the y direction
    diff_y_true = tf.abs(y_true[:, 1:, :] - y_true[:, :-1, :])
    diff_y_pred = tf.abs(y_pred[:, 1:, :] - y_pred[:, :-1, :])
    loss_y = tf.reduce_sum(tf.abs(diff_y_true - diff_y_pred),axis=2)
    
    
    # Sum the losses in both directions
    return (loss_x + loss_y) * lambda_gdl



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
def total_loss(y_true, y_pred, model, lambda_mse, lambda_gdl, lambda_l2, lambda_huber):
    return mse_loss(y_true, y_pred, lambda_mse)  + L2regularization(model.trainable_weights, lambda_l2)+ gdl_loss(y_true, y_pred, lambda_gdl) + huber_loss(y_true, y_pred, lambda_huber)

@register_keras_serializable(package="my_package", name="my_loss_fn_wrapper")
def get_total_loss(model, lambda_mse, lambda_gdl, lambda_l2,lambda_huber):
    def loss_fn(y_true, y_pred):
        return total_loss(y_true, y_pred, model, lambda_mse, lambda_gdl, lambda_l2,lambda_huber)
    return loss_fn

# Registrar a função de perda personalizada
get_custom_objects().update({'my_loss_fn_wrapper': get_total_loss})


def trainNeuralNetwork(lambda_mse = 0.03, lambda_gdl = 0.1, lambda_l2=1e-5, lambda_huber=0.9, lr=0.3, filters=300):
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


    autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gdl, lambda_l2,lambda_huber),metrics = tf.keras.metrics.MeanAbsolutePercentageError())

    #autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gdl, lambda_l2))
    
    return autoencoder