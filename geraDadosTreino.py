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
    
    index = 0
    index_mach = 0
    
    label = []
    sdfFile = []
    conditionsFile = []
    outputTemp = []
    outputPress = []
    
    for wa_it in tqdm(range(len(WedgeAngles))):
        sdf, X, Y = getSDF(genDatFile(WedgeAngles[wa_it]), 0)
        for aoa_it in range(len(AoAs)):    
            for mn_it in range(len(MachNumbers)):       
                if os.path.isfile(pasta2[index_mach]):
                    simFiles.append(np.genfromtxt(pasta2[index_mach], delimiter=',', skip_header=1))
                    conditionsFile.append([AoAs[aoa_it], MachNumbers[mn_it]])                    
                    sdfFile.append(sdf)
                    results = importResults(simFiles[index_mach])
                    dadosTemperatura, _, _ = interpSDFCart(sdf, X, Y, results)
                    
                    outputTemp.append(dadosTemperatura)
                    label.append(f'Wedge: {WedgeAngles[wa_it]}, AoA: {AoAs[aoa_it]}, Mach: {MachNumbers[mn_it]}')

                    # Interrompe o loop logo após a primeira iteração - MOD TEMP (excluir linha)
                    return np.array(sdfFile), np.array(conditionsFile), np.array(outputTemp), np.array(outputPress), label
                    
                else:
                    print("Arquivo não existe")
                index_mach += 1
        index += 1