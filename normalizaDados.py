import numpy as np
import matplotlib.pyplot as plt

def normalizaDadosFunc(dados,plot=False):
    # Calculate the mean and standard deviation of the data
    mean = np.mean(dados)
    std_dev = np.std(dados)
    
    # Normalize the data
    normalized_data = (dados - mean) / std_dev
    
    if plot != False:
    
                # Plot the grid_mach_number
        plt.figure()
        c = plt.contourf(normalized_data, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Dados normalizados')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        
    return normalized_data, mean, std_dev