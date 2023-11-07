# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:59:39 2023

@author: Guilherme
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
import tensorflow as tf

def trainNeuralNetwork(x_train,conditions_train,y_train,epochs_N,batch_size_N):

    # Definindo o codificador
    input_img = Input(shape=(150, 150, 1))  # adaptar isso para o tamanho da sua imagem
    x = Conv2D(300, (5, 5), activation=swish, strides=(1, 1))(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(300, (5, 5), activation=swish, strides=(1, 1))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(300, (3, 3), activation=swish, strides=(1, 1))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(1200, activation=swish)(x)
    encoded = Dense(1200, activation=swish)(x)
    
    # Adicionando o ângulo de ataque e o número de Reynolds como entrada
    input_conditions = Input(shape=(2,))  # adaptar isso para o tamanho do seu vetor de condições
    merged = Concatenate()([encoded, input_conditions])
    
    # Definindo o decodificador
    x = Dense(1202, activation=swish)(merged)
    x = Dense(1200, activation=swish)(x)
    x = Reshape((2,2,300))(x)
    x = Conv2DTranspose(300, (3, 3), strides=(1, 1), activation=swish)(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2DTranspose(300, (5, 5), strides=(1, 1), activation=swish)(x)
    x = UpSampling2D((5, 5))(x)
    x = Conv2DTranspose(300, (5, 5), strides=(1, 1), activation=swish)(x)
    x = UpSampling2D((5, 5))(x)
    decoded = Conv2D(2, (3, 3), activation='sigmoid')(x)
    
    
    
    
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
    
    def L2regularization(theta):
        m = tf.shape(theta)[0]
        return tf.reduce_sum(tf.square(theta)) / m
    
    def total_loss(y_true, y_pred):
        theta = autoencoder.trainable_weights
        return MSEshared(y_true, y_pred) + Gshared(y_true, y_pred) + L2regularization(theta)
    
    
    
    # Compilando o modelo
    autoencoder = Model([input_img, input_conditions], decoded)
    autoencoder.compile(optimizer='sgd', loss=total_loss) # Usando o otimizador SGD e a função de perda total definida no artigo
    
    
    history = autoencoder.fit([x_train, conditions_train], y_train, epochs=epochs_N, batch_size=batch_size_N)
    
    return history