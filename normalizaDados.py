import numpy as np
import matplotlib.pyplot as plt

def normalizaDadosFunc(dados,plot=False):

    dados = np.array(dados)
    dados = np.where(dados < 0, 0, dados)
    normalized_data = np.log(dados)
    
    if plot != False:
    
                # Plot the grid_mach_number
        plt.figure()
        c = plt.contourf(normalized_data, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Dados normalizados')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        
    return normalized_data