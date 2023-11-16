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
results = np.genfromtxt("H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD\\-5-AoA\\5-WedgeAngle\\5-Mach\\solData", delimiter=',', skip_header=1)

#Obtém SDF
sdf5deg, X, Y = getSDF(datFile, -5)

#Obtém resultados em malha cartesiana
results = importResults(results)

#Interpola SDF e malha cartesiana
dadosTemperatura,grid_pressure,_ = interpSDFCart(sdf5deg, X, Y, results,True)

#Normaliza os dados interpolados 
tempNormalizada = normalizaDadosFunc(dadosTemperatura)
pressNormal = normalizaDadosFunc(grid_pressure)



model = keras.models.load_model('model.keras',safe_mode=False)

x_pred = [np.array([sdf5deg]),np.array([[-5,5]])]

temp,press = model.predict(x_pred)


# Create arrays for X and Y coordinates
X, Y = np.meshgrid(np.arange(150), np.arange(150))

# Divide X and Y by your desired values
X = X / 150*2 - 0.5
Y = Y / 150 - 0.5


# Now you can plot it using X and Y for coordinates and press for values
plt.figure()
c = plt.contourf(X, Y, temp[0,:,:,0], cmap=plt.cm.jet, levels=200)
plt.colorbar(c)
plt.title('Temperatura Prevista')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()


# Now you can plot it using X and Y for coordinates and press for values
plt.figure()
c = plt.contourf(X, Y, press[0,:,:,0], cmap=plt.cm.jet, levels=200)
plt.colorbar(c)
plt.title('Pressão prevista')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()




