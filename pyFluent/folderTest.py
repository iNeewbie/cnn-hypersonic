import os
import numpy as np

MachNumbers = [5,6,7,8,9,10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]

base_path = r'H:\Meu Drive\TCC\Programming\cnn-hypersonic\DataCFD'

pasta = [os.path.join(base_path, str(AoA) + '-AoA', str(WedgeAngle) + '-WedgeAngle', 'meshFile.msh') 
         for AoA in AoAs for WedgeAngle in WedgeAngles]

#Simulação tem que rodar:
    