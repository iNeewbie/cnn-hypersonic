import numpy as np
import os
from signalDistanceFunction import getSDF
from importCFDResults import importResults
from interpolationSDFCartGrid import interpSDFCart
from normalizaDados import normalizaDadosFunc
from neuralNetwork import trainNeuralNetwork
from tqdm import tqdm
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

def geraDadosTreino():
        
    
    
    MachNumbers = [5, 6, 7, 8, 9, 10]
    WedgeAngles = [5, 7, 10, 12, 15]
    AoAs = [-5, -3, 0, 5, 10, 15]
    
    def genDatFile(WA):
        datFile = np.array([[0,0],[0.5,0.5*np.tan(WA*np.pi/180)],[1,0],[0.5,-0.5*np.tan(WA*np.pi/180)],[0,0]])
        return datFile
    
    
    pasta2 = [os.path.join("DataCFD", str(WedgeAngle) + "-WedgeAngle", str(AoA) + "-AoA", str(MachNumber) + "-Mach","solData") 
            for WedgeAngle in WedgeAngles for AoA in AoAs for MachNumber in MachNumbers]
    
    simFiles = []
    
    
    """for i in tqdm(pasta2):
        if i == "H:\\Meu Drive\\TCC\Programming\\cnn-hypersonic\\DataCFD\\5-AoA\\7-WedgeAngle\\5-Mach\\solData":
            break
        simFiles.append(np.genfromtxt(i, delimiter=',', skip_header=1))
    
    """
    index = 0
    index_mach = 0
    
    label = []
    sdfFile = []
    conditionsFile = []
    outputTemp = []
    outputPress = []
    for wa_it in tqdm(range(len(WedgeAngles))):
        sdf,X,Y = getSDF(genDatFile(WedgeAngles[wa_it]), 0)
        for aoa_it in range(len(AoAs)):    
            for mn_it in range(len(MachNumbers)):       
                if index_mach==112:
                    if os.path.isfile(pasta2[index_mach]):
                        simFiles.append(np.genfromtxt(pasta2[index_mach], delimiter=',', skip_header=1))
                        conditionsFile.append([AoAs[aoa_it],MachNumbers[mn_it]])                    
                        sdfFile.append(sdf)
                        results = importResults(simFiles[-1],True)#importResults(simFiles[index_mach])
                        dadosTemperatura,_,_ = interpSDFCart(sdf, X, Y, results,True)

                        
                        outputTemp.append(dadosTemperatura)


                        #outputPress.append(normalizaDadosFunc(grid_pressure))
                        label.append(f'Wedge: {WedgeAngles[wa_it]}, AoA: {AoAs[aoa_it]}, Mach: {MachNumbers[mn_it]}')
    
                        index_mach+=1

                    else:
                        print("Arquivo não existe")
                else:
                    
                    index_mach+=1
                        
        index += 1
        
    outputTemp, scaler = normalizaDadosFunc(outputTemp,True) 
            
    conditionsFile = np.array(conditionsFile)    
    outputPress = np.array(outputPress)
    
    
    
    sdfFile = np.array(sdfFile)
    outputTemp = np.array(outputTemp)
    
    return sdfFile, conditionsFile, outputTemp, outputPress, label, scaler
