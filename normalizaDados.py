import numpy as np

def normalizaDadosFunc(dados):
    # Calculate the mean and standard deviation of the data
    mean = np.mean(dados)
    std_dev = np.std(dados)
    
    # Normalize the data
    normalized_data = (dados - mean) / std_dev
    return normalized_data