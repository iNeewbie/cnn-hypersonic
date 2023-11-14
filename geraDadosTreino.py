import numpy as np
import os
from signalDistanceFunction import getSDF
from importCFDResults import importResults
from interpolationSDFCartGrid import interpSDFCart
from normalizaDados import normalizaDadosFunc
from neuralNetwork import trainNeuralNetwork
from tqdm import tqdm



MachNumbers = [5, 6, 7, 8, 9, 10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]

def genDatFile(WA):
    datFile = np.array([[0,0],[0.5,0.5*np.tan(WA/2*np.pi/180)],[1,0],[0.5,-0.5*np.tan(WA/2*np.pi/180)],[0,0]])
    return datFile


base_path = "H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD"
pasta2 = [os.path.join(base_path, str(AoA) + "-AoA", str(WedgeAngle) + "-WedgeAngle", str(MachNumber) + "-Mach","solData") 
         for AoA in AoAs for WedgeAngle in WedgeAngles for MachNumber in MachNumbers]

simFiles = []


for i in tqdm(pasta2):
    if i == "H:\\Meu Drive\\TCC\Programming\\cnn-hypersonic\\DataCFD\\5-AoA\\7-WedgeAngle\\5-Mach\\solData":
        break
    simFiles.append(np.genfromtxt(i, delimiter=',', skip_header=1))


index = 0
index_mach = 0


sdfFile = []
conditionsFile = []
outputTemp = []
outputPress = []
for aoa_it in range(len(AoAs)):
    if aoa_it == 3:
        break
    for wa_it in range(len(WedgeAngles)):
        for mn_it in tqdm(range(len(MachNumbers))):       
            conditionsFile.append([AoAs[aoa_it],MachNumbers[mn_it]])
            sdf,X,Y = getSDF(genDatFile(WedgeAngles[wa_it]), AoAs[aoa_it])
            sdfFile.append(sdf)
            results = importResults(simFiles[index_mach])
            dadosTemperatura,grid_pressure,_ = interpSDFCart(sdf, X, Y, results)
            outputTemp.append(normalizaDadosFunc(dadosTemperatura))
            outputPress.append(normalizaDadosFunc(grid_pressure))
            
            index_mach+=1
        index += 1
        
conditionsFile = np.array(conditionsFile)    
sdfFile = np.array(sdfFile) 
outputTemp = np.array(outputTemp)
outputPress = np.array(outputPress)

trained, model = trainNeuralNetwork(sdfFile, conditionsFile, outputTemp, outputPress , 100, 1)
model.save('model.keras')
        