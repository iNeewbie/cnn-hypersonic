# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:01:14 2024

@author: Guilherme
"""
from neuralNetwork import trainNeuralNetwork
from geraDadosTreino import geraDadosTreino
import tensorflow as tf
from keras.models import load_model


model = False
sdfFile, conditionsFile, outputTemp, outputPress = geraDadosTreino()

model = trainNeuralNetwork(lambda_mse=0, lambda_gs=0.6, lambda_l2=1e-6, lambda_huber=0.9, lr=0.1, filters=150)

epochs_N = 2000
batch_size_N = 32

x_train,conditions_train,y1,y2,epochs_N,batch_size_N = sdfFile, conditionsFile, outputTemp, outputPress ,10000, 25


my_callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.8,patience=200), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100,min_delta = 0.001), tf.keras.callbacks.TerminateOnNaN()]

history = model.fit([x_train,conditions_train], [y1,y2], epochs=epochs_N, batch_size=batch_size_N,callbacks=my_callbacks,verbose=1)


#model = tf.keras.saving.load_model("model.keras",safe_mode=False)

#model = False
#trained, model = trainNeuralNetwork(sdfFile, conditionsFile, outputTemp, outputPress ,30000, 3, 0.05, model,filters=100)

model.save('model2111.keras')




























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