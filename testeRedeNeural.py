
from geraDadosTreino import geraDadosTreino
from neuralNetwork import MaskingLayer, get_total_loss
import tensorflow as tf
from keras.models import load_model
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np



"""Processo para treinar rede neural
1. Cria SDF 
2. Importa dados ansys
3. Interpola SDF e cartesian grid com dados do ansys
4. Normaliza dados
4. Envia dados p rede neural

"""



plt.close('all')

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
    media = data['array9'][0]
    std_dev = data['array9'][1]
except:
    tempo_gerarDados = time.time()
    x1, x2, y1, _, label, mean, std = geraDadosTreino()    
    fim_gerarDados = time.time()
    
    print(f"Passou {(fim_gerarDados-tempo_gerarDados)/60} minutos para gerar dados")
    x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=13)
    x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=13)
    y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=13)
    label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=13)
    
    media_std = np.array([mean,std])
    
    np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
             array5=y_train, array6=y_test, array7=label_train, array8=label_test,array9=media_std)


# Carregar o modelo
#weights = model.get_weights(); new_model.set_weights(weights)
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


