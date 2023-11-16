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




@register_keras_serializable(package="my_package", name="my_loss_fn")
def total_loss(y_true, y_pred, model, lambda_mse, lambda_gs, lambda_l2):
    return MSEshared(y_true, y_pred, lambda_mse)  + L2regularization(model.trainable_weights, lambda_l2)+ GSshared(y_true, y_pred, lambda_gs)

# Define a custom loss function that wraps the total_loss function
def get_total_loss(model, lambda_mse, lambda_gs, lambda_l2):
    def loss_fn(y_true, y_pred):
        return total_loss(y_true, y_pred, model, lambda_mse, lambda_gs, lambda_l2)
    return loss_fn

def trainNeuralNetwork(x_train,conditions_train,y1,y2,epochs_N,batch_size_N, lr = 0.01, autoencoder = False):
    get_custom_objects().clear()
        

    

    
    if autoencoder == False:
        # Definindo o codificador
        input_img = Input(shape=(150, 150, 1))  # adaptar isso para o tamanho da sua imagem
        x = Conv2D(150, (5, 5), activation=swish, padding='same')(input_img)
        x = MaxPooling2D((5, 5))(x)
        x = Conv2D(150, (5, 5), activation=swish, padding='same')(x)
        x = MaxPooling2D((5, 5))(x)
        x = Conv2D(150, (3, 3), activation=swish, padding='same')(x)
        x = MaxPooling2D((3, 3))(x)
        encoded = Flatten()(x)
        # Adicionando o ângulo de ataque e o número de Reynolds como entrada
        input_conditions = Input(shape=(2,))  # adaptar isso para o tamanho do seu vetor de condições
        input2Flat = Flatten()(input_conditions)
        merged = Concatenate()([input2Flat,encoded ])    
        # Definindo o decodificador
        x = Dense(150*2*2, activation=swish)(merged)
        x = Reshape((2,2,150))(x)
        x = Conv2D(150, (3, 3), padding='same', activation=swish)(x)
        x = UpSampling2D((3, 3))(x)
        x = Conv2D(150, (5, 5), padding='same', activation=swish)(x)
        x = UpSampling2D((5, 5))(x)
        x = Conv2D(150, (5, 5), padding='same', activation=swish)(x)
        decoded = UpSampling2D((5, 5))(x)
        output = Conv2D(2, (1, 1), padding='same')(decoded)
        mask = tf.cast(tf.greater(input_img, 0), dtype='float32')
        masked_output = MaskingLayer()([output, mask])
        # Split the output tensor into three 150x150x1 tensors
        temperature = Lambda(lambda x: x[:, :, :, 0:1])(masked_output)
        pressure = Lambda(lambda x: x[:, :, :, 1:2])(masked_output)

        # Definindo o modelo
        autoencoder = Model(inputs=[input_img, input_conditions ], outputs=[temperature,pressure])
        autoencoder.summary()


    my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.8,patience=200)]
    lambda_mse = 0.9
    lambda_gs = 0.1
    lambda_l2 = 1e-5
    autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gs, lambda_l2))

    #autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(autoencoder, lambda_mse, lambda_gs, lambda_l2))
    
    history = autoencoder.fit([x_train,conditions_train], [y1,y2], epochs=epochs_N, batch_size=batch_size_N,callbacks=my_callbacks)
    
    return history, autoencoder