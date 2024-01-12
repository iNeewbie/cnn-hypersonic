import numpy as np
import os
from signalDistanceFunction import getSDF
from importCFDResults import importResults
from interpolationSDFCartGrid import interpSDFCart
from tqdm import tqdm
from denormalizaDados import denormalizaDadosFunc
import time
from sklearn.model_selection import train_test_split






def geraDadosParaPostProcessing():
    MachNumbers = [5, 6, 7, 8, 9, 10]
    WedgeAngles = [5, 7, 10, 12, 15]
    AoAs = [-5, -3, 0, 5, 10, 15]
    
    def genDatFile(WA):
        datFile = np.array([[0,0],[0.5,0.5*np.tan(WA/2*np.pi/180)],[1,0],[0.5,-0.5*np.tan(WA/2*np.pi/180)],[0,0]])
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
    
    
    sdfFile = []

    mean_list = []
    std_dev_list = []
    label = []
    
    for wa_it in tqdm(range(len(WedgeAngles))):
        sdf,X,Y = getSDF(genDatFile(WedgeAngles[wa_it]), 0)
        for aoa_it in range(len(AoAs)):    
            for mn_it in range(len(MachNumbers)):       
                #if index_mach <3:
                if os.path.isfile(pasta2[index_mach]):
                    simFiles.append(np.genfromtxt(pasta2[index_mach], delimiter=',', skip_header=1))
                    label.append(f'Wedge: {WedgeAngles[wa_it]}, AoA: {AoAs[aoa_it]}, Mach: {MachNumbers[mn_it]}')
                    sdfFile.append(sdf)
                    results = importResults(simFiles[index_mach])
                    dadosTemperatura,_,_ = interpSDFCart(sdf, X, Y, results)
                    
                                        
                    mean, std_dev = denormalizaDadosFunc(dadosTemperatura)
                    
                    mean_list.append(mean)
                    std_dev_list.append(std_dev)
                    
                    index_mach+=1
                else:
                    print("Arquivo não existe")
        index += 1
        

    return np.array(label), np.array(mean_list), np.array(std_dev_list)

try:
    data = np.load('postProc.npz')
    label_train = data['array1']
    label_test = data['array2']
    mean_train = data['array3']
    mean_test = data['array4']
    std_train = data['array5']
    std_test = data['array6']
except:
    tempo_gerarDados = time.time()
    label, mean, std = geraDadosParaPostProcessing()   
    fim_gerarDados = time.time()
    
    print(f"Passou {(fim_gerarDados-tempo_gerarDados)/60} minutos para gerar dados")
    label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=13)
    mean_train, mean_test = train_test_split(mean, test_size=0.15, shuffle=True, random_state=13)
    std_train, std_test = train_test_split(std, test_size=0.15, shuffle=True, random_state=13)
    np.savez('postProc.npz', array1=label_train, array2=label_test, array3=mean_train, array4=mean_test, array5=std_train, array6=std_test)
    


