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

class MaskingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaskingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_pred, mask = inputs
        inverse_mask = 1 - mask
        return y_pred * mask + inverse_mask * 0


# Inicializar o DataFrame
df = pd.DataFrame(columns=['Dia', 'Épocas', 'Loss', 'Tempo'])


try:
    data = np.load('arquivo.npz')
    x1_train = data['array1']
    x2_train = data['array2']
    x1_test = data['array3']
    x2_test = data['array4']
    y_train = data['array5']
    y_test = data['array5']
except:
    tempo_gerarDados = time.time()
    
    x1, x2, y1, _ = geraDadosTreino()
    
    fim_gerarDados = time.time()
    
    print(f"Passou {(fim_gerarDados-tempo_gerarDados)/60} minutos para gerar dados")
    x1_train, x1_test, y_train, y_test = train_test_split(x1, y1, test_size=0.15, shuffle=True, random_state=13)
    x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=13)


epochs_N = 100
batch_size_N = 154/2



my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.8,patience=200), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100,min_delta = 0.001), tf.keras.callbacks.TerminateOnNaN()]


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
history = model.fit([x1_train,x2_train], y_train,validation_data=([x1_test,x2_test],y_test), epochs=epochs_N, batch_size=batch_size_N,callbacks=my_callbacks,verbose=1)
end_time = time.time()

elapsed_time = end_time-start_time

# Obter o loss e o tempo
loss = history.history['loss'][-1]  # Substitua se você tiver uma maneira diferente de calcular o loss

# Tente ler o arquivo CSV
try:
    df = pd.read_csv('dados_treinamento.csv')
    last_day = df['Dia'].iloc[-1]
except:
    # Se o arquivo CSV ainda não existe, inicialize o DataFrame e defina o último dia como 0
    df = pd.DataFrame(columns=['Dia', 'Épocas', 'Loss', 'Tempo'])
    last_day = 0


# Adicionar os dados ao DataFrame
df = pd.concat([df, pd.DataFrame([{'Dia': last_day+1, 'Épocas': epochs_N, 'Loss': loss, 'Tempo': elapsed_time}])], ignore_index=True)
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


plt.close('all')
# Criar uma nova figura
plt.figure()

# Plotar a perda de treinamento
plt.plot(hist_df['loss'], label='Treinamento')

# Plotar a perda de validação
plt.plot(hist_df['val_loss'], label='Validação')

# Adicionar um título e rótulos aos eixos
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')

# Adicionar uma legenda
plt.legend()

# Mostrar o gráfico
plt.show()



#model = tf.keras.saving.load_model("model.keras",safe_mode=False)

#model = False
#trained, model = trainNeuralNetwork(sdfFile, conditionsFile, outputTemp, outputPress ,30000, 3, 0.05, model,filters=100)

#model.save('model2111.keras')




























"""temp, press = model.predict([np.array([sdfFile[0]]),np.array([[-5,5]])])
        
plt.close('all')

# Create arrays for X and Y coordinates
#X, Y = np.meshgrid(np.arange(150), np.arange(150))

# Divide X and Y by your desired values
#X = X / 150*2 - 0.5
#Y = Y / 150 - 0.5

# Calculate color map limits
vmin_temp = min(temp[0,:,:,0].min(), outputTemp[0].min())
vmax_temp = max(temp[0,:,:,0].max(), outputTemp[0].max())
vmin_press = min(press[0,:,:,0].min(), outputPress[0].min())
vmax_press = max(press[0,:,:,0].max(), outputPress[0].max())

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Subfigure for predicted temperature
data = temp[0,:,:,0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[0].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(c, ax=axs[0])
axs[0].set_title('Temperatura Prevista')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# Subfigure for real temperature
data = outputTemp[0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[1].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(c, ax=axs[1])
axs[1].set_title('Temperatura Real')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Subfigure for predicted pressure
data = press[0,:,:,0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[0].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_press, vmax=vmax_press)
fig.colorbar(c, ax=axs[0])
axs[0].set_title('Pressao Prevista')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# Subfigure for real pressure
data = outputPress[0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[1].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_press, vmax=vmax_press)
fig.colorbar(c, ax=axs[1])
axs[1].set_title('Pressao Real')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()


plt.figure()
plt.plot(trained.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()"""