# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:59:39 2023

@author: Guilherme
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_custom_objects, register_keras_serializable

@register_keras_serializable(package="my_package")
class CustomTotalLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_mse, lambda_gdl, lambda_huber, **kwargs):
        super().__init__(**kwargs)
        self.lambda_mse = lambda_mse
        self.lambda_gdl = lambda_gdl
        self.lambda_huber = lambda_huber

    def call(self, y_true, y_pred):
        return total_loss(y_true, y_pred, self.lambda_mse, self.lambda_gdl, self.lambda_huber)

    def get_config(self):
        config = super().get_config()
        config.update({
            'lambda_mse': self.lambda_mse,
            'lambda_gdl': self.lambda_gdl,
            'lambda_huber': self.lambda_huber
        })
        return config


# Funções de perda personalizadas registradas para serialização
@register_keras_serializable(package="my_package")
def mse_loss(y_true, y_pred, lambda_mse):
    loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return lambda_mse * loss

@register_keras_serializable(package="my_package")
def gdl_loss(y_true, y_pred, lambda_gdl):
    # y_true e y_pred têm shape (batch_size, 800, 800, 1)
    # Calcular os gradientes ao longo dos eixos x e y
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    # Calcular a diferença dos gradientes
    grad_diff_x = tf.abs(dx_true - dx_pred)
    grad_diff_y = tf.abs(dy_true - dy_pred)
    
    # Calcular a perda média
    loss = tf.reduce_mean(grad_diff_x + grad_diff_y)
    return lambda_gdl * loss

@register_keras_serializable(package="my_package")
def huber_loss(y_true, y_pred, lambda_huber):
    loss = tf.keras.losses.Huber(delta=2.0)(y_true, y_pred)
    return lambda_huber * loss

@register_keras_serializable(package="my_package")
def total_loss(y_true, y_pred, lambda_mse, lambda_gdl, lambda_huber):
    return (mse_loss(y_true, y_pred, lambda_mse) +
            gdl_loss(y_true, y_pred, lambda_gdl) +
            huber_loss(y_true, y_pred, lambda_huber))



def trainNeuralNetwork(lambda_mse=0.03, lambda_gdl=0.1, lambda_l2=1e-5, lambda_huber=0.9, lr=0.001, filters=300):
    # Limpar objetos personalizados
    
    # Definir as entradas
    input_img = Input(shape=(150, 150, 1))
    input_conditions = Input(shape=(2,))

    # Codificador
    x = Conv2D(filters, kernel_size=5, strides=5, padding='same', kernel_regularizer=l2(lambda_l2), activation=swish)(input_img)


    x = Conv2D(filters, kernel_size=5, strides=5, padding='same', kernel_regularizer=l2(lambda_l2), activation=swish)(x)


    x = Conv2D(filters, kernel_size=3, strides=3, padding='same', kernel_regularizer=l2(lambda_l2), activation=swish)(x)


    x = Flatten()(x)

    # Combinar com as condições de entrada
    x_conditions = Flatten()(input_conditions)
    x = Concatenate()([x_conditions, x])

    # Decodificador
    x = Dense(filters * 2 * 2, kernel_regularizer=l2(lambda_l2), activation=swish)(x)
    x = Reshape((2, 2, filters))(x)  # Correção: aplicar a camada Reshape em x

    # Upsampling usando Conv2DTranspose
    x = Conv2DTranspose(filters, kernel_size=3, strides=3, padding='same', kernel_regularizer=l2(lambda_l2), activation=swish)(x)

    x = Conv2DTranspose(filters, kernel_size=5, strides=5, padding='same', kernel_regularizer=l2(lambda_l2), activation=swish)(x)

    x = Conv2DTranspose(filters, kernel_size=5, strides=5, padding='same', kernel_regularizer=l2(lambda_l2), activation=swish)(x)

    # Camada de saída
    output = Conv2D(1, kernel_size=1, strides=1,padding='same')(x)     # Sem função de ativação, assumindo tarefa de regressão

    # Definir o modelo
    autoencoder = Model(inputs=[input_img, input_conditions], outputs=output)

    
    # Compilar o modelo
    loss = CustomTotalLoss(lambda_mse, lambda_gdl, lambda_huber)
    autoencoder.compile(optimizer=Adam(learning_rate=lr),
                        loss=loss,
                        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

    
    return autoencoder
