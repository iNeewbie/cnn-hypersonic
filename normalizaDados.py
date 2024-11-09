import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def normalizaDadosFunc(dados, plot=False):
    dados = np.array(dados)
    dados[dados <= 2.166500000E+02] = 2.166500000E+02
    dados = np.log(dados)
    scaled_data = dados

    # Redimensiona os dados para duas dimensões
    dados_2d = dados.reshape(-1, 1)

    # Aplica a normalização com StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dados_2d)

    # Redimensiona os dados de volta para a forma original
    scaled_data = scaled_data.reshape(-1, 150, 150)

    if plot:
        # Usando apenas a primeira camada para visualização
        plt.figure()
        c = plt.contourf(scaled_data[0], cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Dados normalizados')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return scaled_data, scaler
