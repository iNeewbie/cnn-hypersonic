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

class PredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, N):
        super(PredictionCallback, self).__init__()
        self.N = N

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.N == 0:  # N is the number of epochs after which you want to predict
            y_pred = self.model.predict(self.validation_data[0])
            print('prediction: {} at epoch: {}'.format(y_pred, epoch))

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

    loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)

    return lambda_mse * loss

"""

def gdl_loss(y_true, y_pred, lambda_gdl):
    alpha=1


    y_true = tf.expand_dims(y_true, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    # Calculate gradients for true and predicted images
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    # Compute the squared difference of the partial derivatives
    diff_dXdx = tf.square(dx_true - dx_pred)
    diff_dXdy = tf.square(dy_true - dy_pred)
    
    mean_x = tf.reduce_mean(diff_dXdx)
    mean_y = tf.reduce_mean(diff_dXdy)
    # Compute the GSseparated value
    GSseparated = mean_x+mean_y
    
    # Return the GSseparated value
    return GSseparated * lambda_gdl

    
    

    # Compute gradient difference loss
    #gd_loss = tf.reduce_sum(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred), axis=[1, 2]) ** alpha

    #return tf.reduce_mean(gd_loss)*lambda_gdl"""

"""def gdl_loss(y_true, y_pred, lambda_gdl):
    mask = tf.cast(tf.greater(y_true, 0), dtype='float32')
    m = tf.shape(y_true)[0]
    nx = tf.shape(y_true)[1]
    ny = tf.shape(y_true)[2]
    U_true, V_true = tf.split(y_true * mask, 2, axis=-1)
    U_pred, V_pred = tf.split(y_pred * mask, 2, axis=-1)
    dUdx_true = U_true[:, 1:, :] - U_true[:, :-1, :]
    dUdx_pred = U_pred[:, 1:, :] - U_pred[:, :-1, :]
    dVdx_true = V_true[:, 1:, :] - V_true[:, :-1, :]
    dVdx_pred = V_pred[:, 1:, :] - V_pred[:, :-1, :]
    dU_diff = K.square(dUdx_true - dUdx_pred)
    dV_diff = K.square(dVdx_true - dVdx_pred)
    sum_diff = K.sum(dU_diff[:, 1:-1, 1:-1] + dV_diff[:, 1:-1, 1:-1])
    loss = lambda_gdl * sum_diff / tf.cast(2 * m * (nx - 2) * (ny - 2), tf.float32)
    return loss"""

def gdl_loss(y_true, y_pred, lambda_gdl):
    y_true = tf.expand_dims(y_true, -1)  
    y_pred = tf.expand_dims(y_pred, -1)
    m = tf.shape(y_true)[0]
    nx = tf.shape(y_true)[1]
    ny = tf.shape(y_true)[2]

    # Calculate the gradients
    grad_y_true_x, grad_y_true_y = tf.image.image_gradients(y_true)
    grad_y_pred_x, grad_y_pred_y = tf.image.image_gradients(y_pred)

    # Calculate the loss according to the provided formula
    loss = tf.cast(1 / (2 * m * (nx - 2) * (ny - 2)),tf.float32) * K.sum(K.square(grad_y_true_x - grad_y_pred_x) + K.square(grad_y_true_y - grad_y_pred_y))
    return loss*lambda_gdl

    
    # Return the GSseparated value
    #return GSseparated * lambda_gdl



def L2regularization(theta, lambda_l2):
    l2_reg = 0
    for weight in theta:
        l2_reg += tf.reduce_sum(tf.square(weight))
    return lambda_l2 * l2_reg / 2

def huber_loss(y_true, y_pred,lambda_huber):
    delta = 2
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
    x = Conv2D(filters, (5, 5), activation=swish, padding='same')(input_img)
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
    x = Conv2DTranspose(filters, (2, 2), padding='same', activation=swish)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(filters, (3, 3), padding='same', activation=swish)(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2DTranspose(filters, (5, 5), padding='same', activation=swish)(x)
    x = UpSampling2D((5, 5))(x)
    x = Conv2DTranspose(filters, (5, 5), padding='same', activation=swish)(x)
    decoded = UpSampling2D((5, 5))(x)
    output = Conv2DTranspose(1, (5, 5), padding='same')(decoded)
    output = Reshape((300, 300))(output)

    #mask = tf.cast(tf.greater(input_img, 0), dtype='float32')
    #masked_output = MaskingLayer()([output, mask])

    # Definindo o modelo
    autoencoder = Model(inputs=[input_img, input_conditions], outputs=output)
    #autoencoder.summary()

    autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gdl, lambda_l2,lambda_huber),metrics = tf.keras.metrics.MeanAbsolutePercentageError())
    #autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=tf.keras.losses.MeanSquaredError(),metrics = tf.keras.metrics.MeanAbsolutePercentageError())
    #get_total_loss(autoencoder, lambda_mse, lambda_gdl, lambda_l2,lambda_huber),metrics = tf.keras.metrics.MeanAbsolutePercentageError())
    #autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gdl, lambda_l2))
    
    return autoencoder
