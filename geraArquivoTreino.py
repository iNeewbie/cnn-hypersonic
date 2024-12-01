# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:01:14 2024

@author: Guilherme
"""
# Importações
from geraDadosTreino import geraDadosTreino
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import time



tempo_gerarDados = time.time()
x1, x2, y1, _, label, scaler = geraDadosTreino()    
fim_gerarDados = time.time()
joblib.dump(scaler, 'scaler.pkl')
x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=19)
x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=19)
y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=19)
label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=19)
np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
            array5=y_train, array6=y_test, array7=label_train, array8=label_test)
