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



trained, model = trainNeuralNetwork(x, train_c, output1, output2 , 1000, 64)
model.save('model.keras')

def testModel(model, x_test, y_test):
    
        # Assuming x_test is your test images, y_test is your test labels
    score = model.evaluate(x_test, y_test, verbose=1)


