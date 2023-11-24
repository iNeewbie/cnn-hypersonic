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


"""for i in tqdm(pasta2):
    if i == "H:\\Meu Drive\\TCC\Programming\\cnn-hypersonic\\DataCFD\\5-AoA\\7-WedgeAngle\\5-Mach\\solData":
        break
    simFiles.append(np.genfromtxt(i, delimiter=',', skip_header=1))

"""
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
           # if index_mach > 0:
                
            simFiles.append(np.genfromtxt(pasta2[index_mach], delimiter=',', skip_header=1))
            conditionsFile.append([AoAs[aoa_it],MachNumbers[mn_it]])
            sdf,X,Y = getSDF(genDatFile(WedgeAngles[wa_it]), AoAs[aoa_it])
            sdfFile.append(sdf)
            results = importResults(simFiles[index_mach])
            dadosTemperatura,grid_pressure,_ = interpSDFCart(sdf, X, Y, results)
            outputTemp.append(normalizaDadosFunc(dadosTemperatura))
            outputPress.append(normalizaDadosFunc(grid_pressure))
            
            index_mach+=1
            break
        break
    index += 1
        
        
conditionsFile = np.array(conditionsFile)    
sdfFile = np.array(sdfFile) 
outputTemp = np.array(outputTemp)
outputPress = np.array(outputPress)

#model = tf.keras.saving.load_model("model.keras",safe_mode=False)

model = False
trained, model = trainNeuralNetwork(sdfFile, conditionsFile, outputTemp, outputPress ,1000, 3, 0.05, model,filters=150)
model.save('model2111.keras')

temp, press = model.predict([np.array([sdfFile[0]]),np.array([[-5,5]])])
        
plt.close('all')

# Create arrays for X and Y coordinates
#X, Y = np.meshgrid(np.arange(150), np.arange(150))

# Divide X and Y by your desired values
#X = X / 150*2 - 0.5
#Y = Y / 150 - 0.5

# Calculate color map limits
vmin_temp = min(temp[0,:,:,0].min(), outputTemp[0].min())
vmax_temp = max(temp[0,:,:,0].max(), outputTemp[0].max())
vmin_press = min(press[0,:,:,0].min(), outputPress[0].min())
vmax_press = max(press[0,:,:,0].max(), outputPress[0].max())

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Subfigure for predicted temperature
data = temp[0,:,:,0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[0].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(c, ax=axs[0])
axs[0].set_title('Temperatura Prevista')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# Subfigure for real temperature
data = outputTemp[0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[1].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(c, ax=axs[1])
axs[1].set_title('Temperatura Real')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Subfigure for predicted pressure
data = press[0,:,:,0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[0].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_press, vmax=vmax_press)
fig.colorbar(c, ax=axs[0])
axs[0].set_title('Pressao Prevista')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

# Subfigure for real pressure
data = outputPress[0]
mask = np.zeros_like(data, dtype=bool)
mask[sdfFile[0] < 0] = True
masked_data = np.ma.masked_array(data, mask)
c = axs[1].contourf(masked_data, cmap=plt.cm.jet, levels=200, vmin=vmin_press, vmax=vmax_press)
fig.colorbar(c, ax=axs[1])
axs[1].set_title('Pressao Real')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

plt.tight_layout()
plt.show()


plt.figure()
plt.plot(trained.history['lambda_6_loss'])
plt.title('Model lambda_6_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Plot training & validation lambda_7_loss values
plt.figure()
plt.plot(trained.history['lambda_7_loss'])
plt.title('Model lambda_7_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Plot training lambda_6_mean_absolute_percentage_error values
plt.figure()
plt.plot(trained.history['lambda_6_mean_absolute_percentage_error'])
plt.title('Model lambda_6_mean_absolute_percentage_error')
plt.ylabel('Mean Absolute Percentage Error')
plt.xlabel('Epoch')
plt.show()
