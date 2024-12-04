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
# Supondo que seus dados estejam em um array numpy 150x150

# Converter o array numpy para um DataFrame
df = pd.DataFrame(y_test.flatten(), columns=['valores'])


# Definir os intervalos de valores

bins = [-1, 0, 1, 2, 3, 4, 5,6]
labels = ['-1-0', '0-1', '1-2', '2-3', '3-4', '4-5','5-6']

# Criar uma nova coluna com os intervalos
df['intervalo'] = pd.cut(df['valores'], bins=bins, labels=labels, include_lowest=True)

# Calcular a porcentagem de valores em cada intervalo
distribuicao = df['intervalo'].value_counts(normalize=True) * 100

# Exibir a distribuição
print(distribuicao)
