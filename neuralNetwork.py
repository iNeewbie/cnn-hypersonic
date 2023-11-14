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
sgd = SGD(lr=0.01)


def trainNeuralNetwork(x_train,conditions_train,y1,y2,epochs_N,batch_size_N):

    # Definindo o codificador
    input_img = Input(shape=(150, 150, 1))  # adaptar isso para o tamanho da sua imagem
    x = Conv2D(300, (5, 5), activation=swish, padding='same')(input_img)
    x = MaxPooling2D((5, 5))(x)
    x = Conv2D(300, (5, 5), activation=swish, padding='same')(x)
    x = MaxPooling2D((5, 5))(x)
    x = Conv2D(300, (3, 3), activation=swish, padding='same')(x)
    x = MaxPooling2D((3, 3))(x)
    encoded = Flatten()(x)
    # Adicionando o ângulo de ataque e o número de Reynolds como entrada
    input_conditions = Input(shape=(2,))  # adaptar isso para o tamanho do seu vetor de condições
    input2Flat = Flatten()(input_conditions)
    merged = Concatenate()([input2Flat,encoded ])    
    # Definindo o decodificador
    x = Dense(1200, activation=swish)(merged)
    x = Reshape((2,2,300))(x)
    x = Conv2DTranspose(300, (3, 3), padding='same', activation=swish)(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2DTranspose(300, (5, 5), padding='same', activation=swish)(x)
    x = UpSampling2D((5, 5))(x)
    x = Conv2DTranspose(300, (5, 5), padding='same', activation=swish)(x)
    decoded = UpSampling2D((5, 5))(x)
    output = Conv2D(2, (1, 1), activation='sigmoid', padding='same')(decoded)
    # Split the output tensor into three 150x150x1 tensors
    temperature = Lambda(lambda x: x[:, :, :, 0:1])(output)
    pressure = Lambda(lambda x: x[:, :, :, 1:2])(output)

    
    lambda_l2 = 0.01
    
    
    def MSEshared(y_true, y_pred):
        m = tf.shape(y_true)[0]
        nx = tf.shape(y_true)[1]
        ny = tf.shape(y_true)[2]
        U_true, V_true = tf.split(y_true, 2, axis=-1)
        U_pred, V_pred = tf.split(y_pred, 2, axis=-1)
        U_diff = K.square(U_true - U_pred)
        V_diff = K.square(V_true - V_pred)
        sum_diff = K.sum(U_diff[:, 1:-1, 1:-1] + V_diff[:, 1:-1, 1:-1])
        loss = sum_diff / tf.cast(m * (nx - 2) * (ny - 2), tf.float32)
        return loss
    
    def GSshared(y_true, y_pred):
        m = tf.shape(y_true)[0]
        nx = tf.shape(y_true)[1]
        ny = tf.shape(y_true)[2]
        U_true, V_true = tf.split(y_true, 2, axis=-1)
        U_pred, V_pred = tf.split(y_pred, 2, axis=-1)
        dUdx_true = U_true[:, 2:, :] - U_true[:, :-2, :]
        dUdx_pred = U_pred[:, 2:, :] - U_pred[:, :-2, :]
        dVdx_true = V_true[:, 2:, :] - V_true[:, :-2, :]
        dVdx_pred = V_pred[:, 2:, :] - V_pred[:, :-2, :]
        dU_diff = K.square(dUdx_true - dUdx_pred)
        dV_diff = K.square(dVdx_true - dVdx_pred)
        sum_diff = K.sum(dU_diff[:, 1:-1, 1:-1] + dV_diff[:, 1:-1, 1:-1])
        loss = sum_diff / tf.cast(6 * m * (nx - 2) * (ny - 2), tf.float32)

        return loss
    
    def L2regularization(theta, lambda_l2):
        l2_reg = 0
        for weight in theta:
            l2_reg += tf.reduce_sum(tf.square(weight))
        return lambda_l2 * l2_reg / 2


    
    def get_total_loss(model, lambda_l2):
        def total_loss(y_true, y_pred):
            return MSEshared(y_true, y_pred) + GSshared(y_true, y_pred) + L2regularization(model.trainable_weights, lambda_l2)
        return total_loss


    """
    def MSEshared(y_true, y_pred):
        Ttruth, Ptruth = tf.split(y_true, 2, axis=-1)
        Tpred, Ppred = tf.split(y_pred, 2, axis=-1)
        return tf.reduce_mean(tf.square(Ttruth - Tpred) + tf.square(Ptruth - Ppred))
    
    def Gshared(y_true, y_pred):
        m = tf.shape(y_true)[0]
        n = tf.shape(y_true)[1]
        dxdy_truth = tf.gradients(y_true, [input_img])[0]
        dxdy_pred = tf.gradients(y_pred, [input_img])[0]
        return 6 * m * (n - 1) * tf.reduce_mean(tf.square(dxdy_truth - dxdy_pred))
    
    def L2regularization(theta, lambda_l2):
        l2_reg = 0
        for weight in theta:
            l2_reg += tf.reduce_sum(tf.square(weight))
    return lambda_l2 * l2_reg
    
    def total_loss(y_true, y_pred):
        return MSEshared(y_true, y_pred) + GSshared(y_true, y_pred) + L2regularization(autoencoder.trainable_weights, lambda_l2)
    """
    #+ Gshared(y_true, y_pred)
    
    # Compilando o modelo
    autoencoder = Model(inputs=[input_img, input_conditions ], outputs=[temperature,pressure])
    autoencoder.summary()

    autoencoder.compile(optimizer=sgd, loss=get_total_loss(autoencoder, lambda_l2)) # Usando o otimizador SGD e a função de perda total definida no artigo
    
    
    history = autoencoder.fit([x_train,conditions_train], [y1,y2], epochs=epochs_N, batch_size=batch_size_N)
    
    return history, autoencoder