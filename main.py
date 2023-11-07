import numpy as np
import matplotlib.pyplot as plt
from signalDistanceFunction import getSDF
from importCFDResults import importResults
from interpolationSDFCartGrid import interpSDFCart
from normalizaDados import normalizaDadosFunc

"""Processo para treinar rede neural
1. Cria SDF 
2. Importa dados ansys
3. Interpola SDF e cartesian grid com dados do ansys
4. Normaliza dados
4. Envia dados p rede neural

"""

plt.close('all')
datFile = np.genfromtxt('dw5.dat',
                     delimiter='',
                     skip_header=0)

simulationFile = np.genfromtxt('DataCFD/FFF-27-005002',
                     delimiter=',',
                     skip_header=1)


interior = np.genfromtxt('DataCFD/FFF-27-005003',
                     delimiter=',',
                     skip_header=1)


#1
sdf5deg, X, Y = getSDF(datFile, 5)



results = importResults(interior,simulationFile,datFile,5)


dadosTemperatura,_,_,_ = interpSDFCart(sdf5deg, X, Y, results)

tempNormalizada = normalizaDadosFunc(dadosTemperatura)

plt.figure()
c = plt.contourf(X, Y, tempNormalizada, cmap=plt.cm.jet, levels=200)
plt.colorbar(c)
plt.title('E Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
