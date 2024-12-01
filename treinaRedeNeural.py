# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:01:14 2024

@author: Guilherme
"""
# Importações
from neuralNetwork import trainNeuralNetwork, total_loss, mse_loss, gdl_loss, huber_loss, CustomTotalLoss
from geraDadosTreino import geraDadosTreino
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
import joblib
import datetime  # Para logs do TensorBoard

plt.close('all')

# =========================================================
# 1. Carregar os dados, ou gerar se não estiverem disponíveis
# =========================================================
try:
    data = np.load('arquivo.npz')
    x1_train = data['array1']
    x2_train = data['array2']
    x1_test = data['array3']
    x2_test = data['array4']
    y_train = data['array5']
    y_test = data['array6']
    label_train = data['array7']
    label_test = data['array8']
    scaler = joblib.load('scaler.pkl')
except:
    tempo_gerarDados = time.time()
    x1, x2, y1, _, label, scaler = geraDadosTreino()    
    fim_gerarDados = time.time()
    joblib.dump(scaler, 'scaler.pkl')
    if len(x1) == 1:
        ar0  = np.array([0])
        np.savez('arquivo.npz', array1=x1, array2=x2, array3=ar0, array4=ar0,
                 array5=y1, array6=ar0, array7=label, array8=ar0)
    else:
        print(f"Passou {(fim_gerarDados - tempo_gerarDados) / 60} minutos para gerar dados")
        x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=19)
        x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=19)
        y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=19)
        label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=19)
        np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
                 array5=y_train, array6=y_test, array7=label_train, array8=label_test)

# ==========================================
# 2. Pré-processamento dos dados
# ==========================================
x1_train = np.expand_dims(x1_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
x1_test = np.expand_dims(x1_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

x1_train = x1_train.astype('float32')
y_train = y_train.astype('float32')
x1_test = x1_test.astype('float32')
y_test = y_test.astype('float32')

# =========================================================
# 3. Criação dos datasets de treino e teste com tf.data
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
# 4. Hiperparâmetros do modelo
# =========================================================
epochs_N = 30000
lambda_mse = 0.8
lambda_gdl = 0.2
lambda_l2 = 1e-5
lambda_huber = 0
lr = 0.0001
filtros = 100

# =========================================================
# 5. Configuração dos callbacks (TensorBoard incluído)
# =========================================================
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint(
    'meu_modelo.keras',  # sem especificar uma extensão; `.tf` será o padrão
    save_best_only=True,
    save_weights_only=False,  # Salva o modelo completo, não apenas os pesos
    monitor='val_mean_absolute_percentage_error',  # Monitorando a perda de validação, ajuste conforme necessário
    mode='min'  # Minimizando a perda, ajuste conforme necessário
)


my_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
]


# =========================================================
# 6. Carregamento ou inicialização do modelo
# =========================================================
try:
    custom_objects = {
        'CustomTotalLoss': CustomTotalLoss,
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'gdl_loss': gdl_loss,
        'huber_loss': huber_loss
    }

    model = load_model('meu_modelo.keras', custom_objects=custom_objects)
    print("Modelo carregado com sucesso.")

    loss = CustomTotalLoss(lambda_mse, lambda_gdl, lambda_huber)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=loss,
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
except:
    print("NÃO carregou modelo, inicializando um novo modelo.")
    model = trainNeuralNetwork(lambda_mse, lambda_gdl, lambda_l2, lambda_huber, lr, filtros)

model.summary()

# =========================================================
# 7. Treinamento do modelo com TensorBoard
# =========================================================

print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))


start_time = time.time()
with tf.device('/GPU:0'):
	history = model.fit(
    		train_dataset,
		epochs=epochs_N,
		validation_data=test_dataset,
		callbacks=my_callbacks,
		verbose=1
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo de treinamento: {elapsed_time / 60:.2f} minutos.")

# =========================================================
# 8. Salvamento do modelo e histórico de treinamento
# =========================================================
model.save('meu_modelo.keras')
print("Modelo salvo em 'meu_modelo.keras'.")

new_hist_df = pd.DataFrame(history.history)
try:
    hist_df = pd.read_csv('history.csv')
    hist_df = pd.concat([hist_df, new_hist_df], ignore_index=True)
except FileNotFoundError:
    hist_df = new_hist_df

hist_df.to_csv('history.csv', index=False)
print("Histórico de treinamento salvo em 'history.csv'.")

# =========================================================
# 9. Plotagem das métricas de perda e MAPE
# =========================================================
plt.figure(figsize=(10, 6))

plt.semilogy(new_hist_df['loss'], label='Loss')
plt.semilogy(new_hist_df['mean_absolute_percentage_error'], label='MAPE')

plt.title('Perda de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda (log)')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()
