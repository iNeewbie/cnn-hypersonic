import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalizaDadosFunc(dados,plot=False):
    
    dados = np.array(dados)
    dados[dados <= 216.65] = 216.65
    dados = np.log(dados)
    scaled_data = dados
    
    # Redimensiona os dados para duas dimensões
    dados_2d = dados.reshape(-1, 1)
    
    # Agora você pode usar o StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dados_2d)
    
    # Redimensiona os dados de volta para a forma original
    scaled_data = scaled_data.reshape(-1, 300, 300)
    #scaler = 0
    
    #print(np.mean(scaled_data), np.std(scaled_data))
        
    if plot != False:
    
        # Plot the grid_mach_number
        plt.figure()
        c = plt.contourf(scaled_data, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Dados normalizados')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return scaled_data, scaler
