# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:01:17 2023

@author: Guilherme
"""
from neuralNetwork import trainNeuralNetwork, get_total_loss, MaskingLayer
from bayes_opt import BayesianOptimization
from geraDadosTreino import geraDadosTreino
import numpy as np
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import pandas as pd
import joblib



try:
    data = np.load('arquivo.npz')
    x1_train = data['array1']
    x2_train = data['array2']
    x1_test = data['array3']
    x2_test = data['array4']
    y_train = data['array5']
    y_test = data['array6']
    label_train = data['array7']
    label_test = data['array8']
    scaler = joblib.load('scaler.pkl')
except:
    tempo_gerarDados = time.time()
    x1, x2, y1, _, label, scaler = geraDadosTreino()    
    fim_gerarDados = time.time()
    joblib.dump(scaler, 'scaler.pkl')
    if len(x1) == 1:
        ar0  = np.array([0])
        np.savez('arquivo.npz', array1=x1, array2=x2, array3=ar0, array4=ar0,
             array5=y1, array6=ar0, array7=label, array8=ar0)
        
    else:
            
        print(f"Passou {(fim_gerarDados-tempo_gerarDados)/60} minutos para gerar dados")
        x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=13)
        x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=13)
        y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=13)
        label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=13)  
        np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
                 array5=y_train, array6=y_test, array7=label_train, array8=label_test)


# Initialize the model and weights outside the function
model = trainNeuralNetwork(lambda_mse=0, lambda_gdl=0, lambda_l2=0, lambda_huber=0, lr=00, filters=150)
initial_weights = model.get_weights()


def optimizeParameters(lambda_gdl, lambda_mse):    
    
    epochs_N = 5000
    lr = 0.01
    lambda_huber = 0
    lambda_l2 = 0
    batch_size_N = 100
    
    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_percentage_error', patience=200,min_delta = 0.0001), tf.keras.callbacks.TerminateOnNaN()]

    # Reset the weights of the model
    model.set_weights(initial_weights)
    
    # Recompile the model with the new lambda values
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=get_total_loss(model, lambda_mse, lambda_gdl, lambda_l2,lambda_huber),metrics = tf.keras.metrics.MeanAbsolutePercentageError())
    
    history = model.fit([x1_train,x2_train], y_train, epochs=epochs_N, batch_size=batch_size_N,callbacks=my_callbacks,verbose=1,use_multiprocessing=True)
    
      
    
    loss = np.mean(history.history['mean_absolute_percentage_error'][-10:])
    if np.isnan(loss) or np.isinf(loss):
        return -1e10
    return -loss

# Define the bounds of the hyperparameters
pbounds = {
    'lambda_gdl': (0, 1),
    'lambda_mse': (0, 1)
}

optimizer = BayesianOptimization(
    f=optimizeParameters,
    pbounds=pbounds,
)

opt_time_start = time.time()

# Optimize
optimizer.maximize(
    init_points=4,
    n_iter=20,
)

opt_time_end = time.time()
# Print the best parameters

print(optimizer.max)
print(f"Otimização demorou {(opt_time_end - opt_time_start)/60} minutos")

# Convert the optimization results to a DataFrame
df = pd.DataFrame(optimizer.res)

# Save the DataFrame to a CSV file
df.to_csv('optimization_results.csv', index=False)



