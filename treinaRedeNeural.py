# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:01:14 2024

@author: Guilherme
"""
from neuralNetwork import trainNeuralNetwork, get_total_loss
from geraDadosTreino import geraDadosTreino
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard

class MaskingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_pred, mask = inputs
        inverse_mask = 1 - mask
        return y_pred * mask + inverse_mask * 0


plt.close('all')

try:
    data = np.load('arquivo.npz')
    x1_train = data['array1']
    x2_train = data['array2']
    x1_test = data['array3']
    x2_test = data['array4']
    y_train = data['array5']
    y_test = data['array6']
except:
    tempo_gerarDados = time.time()
    x1, x2, y1, _ = geraDadosTreino()    
    fim_gerarDados = time.time()
    
    print(f"Passou {(fim_gerarDados-tempo_gerarDados)/60} minutos para gerar dados")
    x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=13)
    x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=13)
    y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=13)
    np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test, array5=y_train, array6=y_test)

    


epochs_N = 800
batch_size_N = 17


tensorboard_callback = TensorBoard(log_dir='logs')

#my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.8,patience=200), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100,min_delta = 0.001), tf.keras.callbacks.TerminateOnNaN()]
my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50,min_delta = 0.0001), tf.keras.callbacks.TerminateOnNaN(), tensorboard_callback]


try:
    # Carregar o modelo
    model = load_model('meu_modelo.keras', custom_objects={'MaskingLayer': MaskingLayer, 'my_loss_fn_wrapper': get_total_loss})
    print("carregou modelo")
    # Criar uma instância da função de perda personalizada
    lambda_mse=0
    lambda_gs=0.6
    lambda_l2=1e-6
    lambda_huber=0.9
    lr = 0.1
    loss = get_total_loss(model, lambda_mse, lambda_gs, lambda_l2, lambda_huber)

    # Compilar o modelo com a função de perda personalizada
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=loss, metrics=tf.keras.metrics.MeanAbsolutePercentageError())
    
except:
    print("NÃO carregou modelo")
    # Se o modelo ainda não existe, inicialize-o
    model = trainNeuralNetwork(0, 0.6, 1e-6, 0.9, 0.1, 150)

# Treinar o modelo
start_time = time.time()
history = model.fit([x1_train,x2_train], y_train,validation_data=([x1_test,x2_test],y_test), epochs=epochs_N, batch_size=batch_size_N,callbacks=my_callbacks,verbose=1,use_multiprocessing=True)
end_time = time.time()

elapsed_time = end_time-start_time

# Obter o loss e o tempo
loss = history.history['loss'][-1]  # Substitua se você tiver uma maneira diferente de calcular o loss
val_loss = history.history['val_loss'][-1]
# Tente ler o arquivo CSV
try:
    df = pd.read_csv('dados_treinamento.csv')
    last_day = df['Dia'].iloc[-1]
except:
    # Se o arquivo CSV ainda não existe, inicialize o DataFrame e defina o último dia como 0
    df = pd.DataFrame(columns=['Dia', 'Épocas', 'Loss', 'Val_loss' 'Tempo'])
    last_day = 0


# Adicionar os dados ao DataFrame
df = pd.concat([df, pd.DataFrame([{'Dia': last_day+1, 'Épocas': len(loss), 'Loss': loss, 'Val_loss':val_loss, 'Tempo': elapsed_time}])], ignore_index=True)
# Salvar o modelo
model.save('meu_modelo.keras')

# Salvar o DataFrame como um arquivo CSV
df.to_csv('dados_treinamento.csv',index=False)


new_hist_df = pd.DataFrame(history.history)

try:
    hist_df = pd.read_csv('history.csv')
    hist_df = pd.concat([hist_df, new_hist_df], ignore_index=True)
except:
    hist_df = new_hist_df  

hist_df.to_csv('history.csv', index=False)



# Criar uma nova figura
plt.figure()

# Plotar a perda de treinamento
plt.semilogy(hist_df['loss'], label='Treinamento')

# Plotar a perda de validação
plt.semilogy(hist_df['val_loss'], label='Validação')

# Adicionar um título e rótulos aos eixos
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda (log)')

# Adicionar uma legenda
plt.legend()

# Mostrar o gráfico
plt.show()

temp = model.predict([np.array([x1_test[0]]),np.array([x2_test[0]])])

# Calculate color map limits
vmin_temp = min(temp[0,:,:,0].min(), y_test[0].min())
vmax_temp = max(temp[0,:,:,0].max(), y_test[0].max())

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Subfigure for predicted temperature
data = temp
mask = np.zeros_like(data[0,:,:,0], dtype=bool)
mask[x1_test[0] < 0] = True
masked_data = np.ma.masked_array(data[0,:,:,0], mask)
c = axs[0].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(c, ax=axs[0])
axs[0].set_title('Temperatura Prevista')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# Subfigure for real temperature
data = y_test
mask = np.zeros_like(data[0,:,:], dtype=bool)
mask[x1_test[0] < 0] = True
masked_data = np.ma.masked_array(data[0,:,:], mask)
c = axs[1].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(c, ax=axs[1])
axs[1].set_title('Temperatura Real')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()
