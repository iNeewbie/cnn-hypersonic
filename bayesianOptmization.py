# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:01:14 2024

@author: Guilherme
"""

# Importações
from neuralNetwork import trainNeuralNetwork, CustomTotalLoss
from bayes_opt import BayesianOptimization
from geraDadosTreino import geraDadosTreino
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import joblib

# =========================================================
# 1. Carregar os dados, ou gerar se não estiverem disponíveis
# =========================================================
try:
    data = np.load('arquivo.npz')
    x1_train, x2_train = data['array1'], data['array2']
    x1_test, x2_test = data['array3'], data['array4']
    y_train, y_test = data['array5'], data['array6']
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    tempo_gerarDados = time.time()
    x1, x2, y1, _, label, scaler = geraDadosTreino()
    fim_gerarDados = time.time()
    joblib.dump(scaler, 'scaler.pkl')
    x1_train, x1_test, y_train, y_test = train_test_split(
        x1, y1, test_size=0.15, shuffle=True, random_state=19
    )
    x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=19)
    np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
             array5=y_train, array6=y_test)

x1_train = np.expand_dims(x1_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
x1_test = np.expand_dims(x1_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

x1_train = x1_train.astype('float32')
y_train = y_train.astype('float32')
x1_test = x1_test.astype('float32')
y_test = y_test.astype('float32')

# =========================================================
# 2. Criação dos datasets de treino e teste com tf.data
# =========================================================
train_dataset = tf.data.Dataset.from_tensor_slices(((x1_train, x2_train), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices(((x1_test, x2_test), y_test))

BUFFER_SIZE = len(x1_train)
BATCH_SIZE = 39
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# =========================================================
# 3. Configurar e Inicializar o Modelo
# =========================================================

epochs_N = 10000
lambda_mse = 0.0
lambda_gdl = 0.2
lambda_l2 = 1e-5
lambda_huber = 0.8
delta_huber = 0.5
lr = 0.0001
filtros = 100

model = trainNeuralNetwork(lambda_mse, lambda_gdl, lambda_l2, lambda_huber, lr, filtros, delta_huber)
initial_weights = model.get_weights()
my_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mean_absolute_percentage_error',
            patience=500,
            min_delta=0.0001
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

# =========================================================
# 4. Função de Otimização
# =========================================================
def optimizeParameters(lambda_huber, lambda_gdl, lambda_l2, delta_huber):
    # Relacionar lambda_gdl
    lambda_mse = 0  # Fixo

    

    # Resetar pesos do modelo
    model.set_weights(initial_weights)

    # Recompilar modelo com novos hiperparâmetros
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CustomTotalLoss(lambda_mse, lambda_gdl, lambda_huber, delta_huber),
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()]
    )

    # Treinar modelo
    with tf.device('/GPU:0'):
        history = model.fit(
                train_dataset,
            epochs=epochs_N,
            validation_data=test_dataset,
            callbacks=my_callbacks,
            verbose=0
    )

    # Calcular métrica de avaliação
    val_loss = np.mean(history.history['val_mean_absolute_percentage_error'][-200:])
    if np.isnan(val_loss) or np.isinf(val_loss):
        return -1e10
    return -val_loss  # Negativo para maximização

# =========================================================
# 5. Configurar Limites e Otimizar
# =========================================================
pbounds = {
    'lambda_huber': (0.1, 10),       # λ_huber entre 0 e 1
    'lambda_gdl': (0.1, 10),         # λ_gdl entre 
    'lambda_l2': (1e-7, 1e-4),      # Regularização L2
    'delta_huber': (0.1, 5)        # Delta do Huber Loss
}

optimizer = BayesianOptimization(
    f=optimizeParameters,
    pbounds=pbounds,
)

opt_time_start = time.time()

optimizer.maximize(
    init_points=30,  # Pontos iniciais de exploração
    n_iter=60      # Iterações para otimização
)

opt_time_end = time.time()
joblib.dump(optimizer, 'optimizer_results.pkl')


# Resultados
print(optimizer.max)
print(f"Otimização demorou {(opt_time_end - opt_time_start) / 60:.2f} minutos")

# Salvar resultados em CSV
df = pd.DataFrame(optimizer.res)
df.to_csv('optimization_results.csv', index=False)

# =========================================================
# 6. Treinar com os Melhores Parâmetros
# =========================================================
best_params = optimizer.max['params']
lambda_huber = best_params['lambda_huber']
lambda_gdl = best_params['lambda_gdl']
lambda_l2 = best_params['lambda_l2']
delta_huber = best_params['delta_huber']

model.set_weights(initial_weights)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=CustomTotalLoss(0, lambda_gdl, lambda_huber, delta_huber),
    metrics=[tf.keras.metrics.MeanAbsolutePercentageError()]
)

with tf.device('/GPU:0'):
	history = model.fit(
    		train_dataset,
		epochs=30000,
		validation_data=test_dataset,
		callbacks=my_callbacks,
		verbose=0
)
model.save('best_model.keras')
print("Modelo treinado salvo como 'best_model.keras'")
