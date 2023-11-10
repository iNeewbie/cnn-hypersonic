import os
import numpy as np

#Iterações:
MachNumbers = [5,6,7,8,9,10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]


pasta = [os.path.join('DataCFD', str(AoA) + '-AoA', str(WedgeAngle) + '-WedgeAngles', 'meshFile.msh') 
         for AoA in AoAs for WedgeAngle in WedgeAngles]


interior = np.genfromtxt('DataCFD/FFF-27-005003', delimiter=',', skip_header=1)
