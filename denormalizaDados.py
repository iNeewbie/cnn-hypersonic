import numpy as np

def denormalizaDadosFunc(dados):
    # Calculate the mean and standard deviation of the data
    mean = np.mean(dados)
    std_dev = np.std(dados)
        
    return mean, std_dev