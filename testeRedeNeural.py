
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

except:
    tempo_gerarDados = time.time()
    x1, x2, y1, _, label = geraDadosTreino()    
    fim_gerarDados = time.time()
    
    print(f"Passou {(fim_gerarDados-tempo_gerarDados)/60} minutos para gerar dados")
    x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=13)
    x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=13)
    y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=13)
    label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=13)  
    np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
             array5=y_train, array6=y_test, array7=label_train, array8=label_test)


    

# Carregar o modelo
#weights = model.get_weights(); new_model.set_weights(weights)
model = load_model('meu_modelo.keras', custom_objects={'MaskingLayer': MaskingLayer, 'my_loss_fn_wrapper': get_total_loss})
print("carregou modelo")
# Criar uma instância da função de perda personalizada
lambda_mse=0
lambda_gs=0#0.6
lambda_l2=0#1e-6
lambda_huber=0#0.9
lr = 0.01
loss = get_total_loss(model, lambda_mse, lambda_gs, lambda_l2, lambda_huber)

# Compilar o modelo com a função de perda personalizada
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=loss, metrics=tf.keras.metrics.MeanAbsolutePercentageError())


temp = model.predict([x1_train,x2_train])

mask = np.zeros_like(temp, dtype=bool)
mask[temp <= 0] = True
tempMasked = (np.ma.masked_array(temp, mask))
y_testMasked = (np.ma.masked_array(y_train, mask))

model.evaluate([x1_train,x2_train], y_train)

for i in range(len(temp)):
  # Create a new figure




  # Aplicação da máscara ao array de temperatura


  vmin_temp = (min(temp[i,:,:,0].min(), y_train[i].min()))
  vmax_temp = (max(temp[i,:,:,0].max(), y_train[i].max()))
  plt.figure(figsize=(10, 5))



  # Subfigure for temp[i]
  plt.subplot(2, 2, 1)
  #plt.contour(temp_denormalizada[i,:,:,0],levels=11,colors='black')
  plt.contourf((tempMasked[i,:,:,0]),levels=200, cmap='jet')
  plt.colorbar()
  plt.title('Temp[i]')

  # Subfigure for y_test[i]
  plt.subplot(2, 2, 2)
  #plt.contour(y_denormalizado[i],levels=11,colors='black')
  plt.contourf((y_testMasked[i]),levels=200, cmap='jet')
  plt.colorbar()
  plt.title('y_test[i]')

  # Calculate the difference
  diff =(y_testMasked[i]) - (tempMasked[i,:,:,0])


  # Subfigure for the difference
  plt.subplot(2, 2, 3)
  plt.contourf(diff, levels=200, cmap='jet')
  plt.colorbar()
  plt.title('y_test[i] - Temp[i]')

  # Show the figure
  plt.tight_layout()
  plt.show()



