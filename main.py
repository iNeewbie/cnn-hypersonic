import numpy as np
import matplotlib.pyplot as plt
from signalDistanceFunction import getSDF
from importCFDResults import importResults
from interpolationSDFCartGrid import interpSDFCart
from normalizaDados import normalizaDadosFunc
from neuralNetwork import trainNeuralNetwork

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
dadosTemperatura,_,_,_ = interpSDFCart(sdf5deg, X, Y, results)

#Normaliza os dados interpolados 
tempNormalizada = normalizaDadosFunc(dadosTemperatura)



#trained = trainNeuralNetwork(x_train, conditions_train, y_train, epochs_N, batch_size_N)

