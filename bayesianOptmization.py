# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:01:17 2023

@author: Guilherme
"""
from neuralNetwork import trainNeuralNetwork
from bayes_opt import BayesianOptimization
from geraDadosTreino import geraDadosTreino
import numpy as np

sdfFile, conditionsFile, outputTemp, outputPress = geraDadosTreino()

def optimizeParameters(lambda_mse, lambda_gs, lambda_huber):    
    trained, model = trainNeuralNetwork(sdfFile, conditionsFile, outputTemp, outputPress ,1000, 1, False,lambda_mse, lambda_gs, lambda_l2=0, lambda_huber=lambda_huber, lr=0.1, filters=50)
    loss = trained.history['loss'][-1]
    if np.isnan(loss) or np.isinf(loss):
        return -1e10
    return -loss
# Define the bounds of the hyperparameters
pbounds = {
    'lambda_mse': (0,0.5),
    'lambda_gs': (0.6, 1.5),
    'lambda_huber': (0, 1),
}

optimizer = BayesianOptimization(
    f=optimizeParameters,
    pbounds=pbounds,
)

# Optimize
optimizer.maximize(
    init_points=10,
    n_iter=20,
)

# Print the best parameters

print(optimizer.max)
