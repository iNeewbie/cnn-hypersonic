# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:30:08 2024

@author: Guilherme
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Dados fornecidos
MachNumbers = [5, 6, 7, 8, 9, 10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]

# Criando uma grade para o gráfico
X,Y,Z = np.meshgrid(AoAs,MachNumbers,WedgeAngles)

# Achatando a grade para o gráfico de dispersão
x = X.flatten()
y = Y.flatten()
z = Z.flatten()

# Criando um mapa de cores com esferas vermelhas e esferas azuis selecionadas aleatoriamente (15% do total)
cores = ['r'] * len(x)
np.random.seed(13) # Para reprodutibilidade 
indices_azuis = np.random.choice(range(len(x)), size=int(0.15*len(x)), replace=False)
for i in indices_azuis:
    cores[i] = 'b'
    
plt.close('all')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c=cores,linewidth=10,alpha=1)

ax.set_xlabel('Ângulo de Ataque')
ax.set_ylabel('Número de Mach')
ax.set_zlabel('Ângulo de Cunha')

plt.show()