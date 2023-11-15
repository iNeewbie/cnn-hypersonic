import numpy as np
import matplotlib.pyplot as plt
from signalDistanceFunction import getSDF
from importCFDResults import importResults
from interpolationSDFCartGrid import interpSDFCart
from normalizaDados import normalizaDadosFunc
from neuralNetwork import trainNeuralNetwork
import tensorflow as tf
import keras

"""Processo para treinar rede neural
1. Cria SDF 
2. Importa dados ansys
3. Interpola SDF e cartesian grid com dados do ansys
4. Normaliza dados
4. Envia dados p rede neural

"""

plt.close('all')


#Importa dados
datFile = np.genfromtxt('dw5.dat', delimiter='', skip_header=0)
simulationFile = np.genfromtxt('DataCFD/FFF-27-005002', delimiter=',', skip_header=1)
interior = np.genfromtxt('DataCFD/FFF-27-005003', delimiter=',', skip_header=1)


#Obtém SDF
sdf5deg, X, Y = getSDF(datFile, 5)

#Obtém resultados em malha cartesiana
results = importResults(interior,simulationFile,datFile,5)

#Interpola SDF e malha cartesiana
dadosTemperatura,grid_pressure,_,_ = interpSDFCart(sdf5deg, X, Y, results)

#Normaliza os dados interpolados 
tempNormalizada = normalizaDadosFunc(dadosTemperatura)
pressNormal = normalizaDadosFunc(grid_pressure)


train_conditions = np.array([5,10])
output1 = np.array([tempNormalizada,tempNormalizada, tempNormalizada])
output2 = np.array([pressNormal,pressNormal, pressNormal])

x = np.array([sdf5deg,sdf5deg, sdf5deg])
train_c = np.array([[5,10], [5,10], [5,10]])



model = keras.models.load_model('model.keras')

x_pred = [np.array([sdf5deg]),np.array([[5,10]])]

temp,press = model.predict(x_pred)


# Plot the first image
X = 2 * temp[:, :, :, 0] / 150 - 0.5
Y = temp[:, :, :, 1] / 150 - 0.5

plt.figure()
c = plt.contourf(X, Y, temp[0, :, :, 2], cmap=plt.cm.jet, levels=200)
plt.colorbar(c)
plt.title('temp Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')

# Adjust the x-limits to range from -0.5 to 1.5
plt.xlim([-0.5, 1.5])

# Adjust the y-limits to range from -0.5 to 0.5
plt.ylim([-0.5, 0.5])

plt.show()



plt.figure()
c = plt.contourf(press[0,:,:,0], cmap=plt.cm.jet, levels=200)
plt.colorbar(c)
plt.title('pre Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.xlim([-0.5,1.5])
plt.ylim([-0.5,0.5])
plt.show()


