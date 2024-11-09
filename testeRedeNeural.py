from geraDadosTreino import geraDadosTreino
from neuralNetwork import total_loss, mse_loss, gdl_loss, huber_loss, CustomTotalLoss
from keras.models import load_model
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Carregar os dados
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
    # Gerar os dados caso o arquivo não exista
    tempo_gerarDados = time.time()
    x1, x2, y1, _, label, scaler = geraDadosTreino()    
    fim_gerarDados = time.time()
    joblib.dump(scaler, 'scaler.pkl')
    if len(x1) == 1:
        ar0  = np.array([0])
        np.savez('arquivo.npz', array1=x1, array2=x2, array3=ar0, array4=ar0,
                 array5=y1, array6=ar0, array7=label, array8=ar0)
    else:
        print(f"Passou {(fim_gerarDados - tempo_gerarDados) / 60} minutos para gerar dados")
        x1_train, x1_test = train_test_split(x1, test_size=0.15, shuffle=True, random_state=13)
        x2_train, x2_test = train_test_split(x2, test_size=0.15, shuffle=True, random_state=13)
        y_train, y_test = train_test_split(y1, test_size=0.15, shuffle=True, random_state=13)
        label_train, label_test = train_test_split(label, test_size=0.15, shuffle=True, random_state=13)  
        np.savez('arquivo.npz', array1=x1_train, array2=x2_train, array3=x1_test, array4=x2_test,
                 array5=y_train, array6=y_test, array7=label_train, array8=label_test)

#x1_test= x1_train
#x2_test=x2_train
#y_test=y_train
# Adicionar a dimensão do canal aos dados de teste
x1_test = np.expand_dims(x1_test, axis=-1)  # Shape: (num_samples, height, width, 1)
y_test = np.expand_dims(y_test, axis=-1)    # Shape: (num_samples, height, width, 1)

# Carregar o modelo com os objetos personalizados necessários
custom_objects = {
    'CustomTotalLoss': CustomTotalLoss,
    'total_loss': total_loss,
    'mse_loss': mse_loss,
    'gdl_loss': gdl_loss,
    'huber_loss': huber_loss
}

model = load_model('meu_modelo.keras', custom_objects=custom_objects)
print("Modelo carregado com sucesso.")

# Fazer predições nos dados de teste
temp_test = model.predict([x1_test, x2_test])  # Saída com shape (num_samples, height, width, 1)

# Remover a dimensão do canal para processamento
temp_test = temp_test[..., 0]      # Shape: (num_samples, height, width)
y_test = y_test[..., 0]            # Shape: (num_samples, height, width)
x1_test = x1_test[..., 0]

# Remover bordas indesejadas (se necessário)
#temp_test = temp_test[:, 1:-1, 1:-1]
#x1_test = x1_test[:, 1:-1, 1:-1]
#y_test = y_test[:, 1:-1, 1:-1]

# Redimensionar os dados para o formato apropriado para denormalização
y_test_flat = y_test.reshape(-1, 1)
temp_test_flat = temp_test.reshape(-1, 1)

# Denormalização usando inversão do `StandardScaler` e exponenciação
y_test_inv = scaler.inverse_transform(y_test_flat)
temp_test_inv = scaler.inverse_transform(temp_test_flat)

# Redimensionar de volta para o formato original
height, width = y_test.shape[1], y_test.shape[2]
y_test_inv = y_test_inv.reshape(-1, height, width)
temp_test_inv = temp_test_inv.reshape(-1, height, width)

# Exponenciação para desfazer a transformação logarítmica (se aplicável)
y_test_inv = np.exp(y_test_inv)
temp_test_inv = np.exp(temp_test_inv)




# Aplicar máscara nos dados de teste
mask_test = np.zeros_like(temp_test_inv, dtype=bool)
mask_test[x1_test <= 0] = True

# Criar arrays mascarados
y_testMasked_inv = np.ma.masked_array(y_test_inv, mask_test)
tempMasked_inv = np.ma.masked_array(temp_test_inv, mask_test)

# Loop para plotagem
for i in range(1):#len(tempMasked_inv)):  # Ajuste o range conforme necessário
    vmin_temp = min(tempMasked_inv[i].min(), y_testMasked_inv[i].min())
    vmax_temp = max(tempMasked_inv[i].max(), y_testMasked_inv[i].max())
    plt.figure(figsize=(10, 5))

    # Gráfico da predição denormalizada e exponenciada
    plt.subplot(2, 2, 1)
    plt.contourf(tempMasked_inv[i], levels=200, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    plt.colorbar()
    plt.title('Predição (Denormalizada e Exponenciada)')

    # Gráfico do ground truth denormalizado e exponenciado
    plt.subplot(2, 2, 2)
    plt.contourf(y_testMasked_inv[i], levels=200, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    plt.colorbar()
    plt.title('Ground Truth (Denormalizada e Exponenciada)')

    # Calcule a diferença (erro percentual) entre predição e ground truth
    diff = np.abs(tempMasked_inv[i] - y_testMasked_inv[i]) /(y_testMasked_inv[i]) *100 # Evitar divisão por zero

    # Gráfico do erro percentual
    plt.subplot(2, 2, 3)
    plt.contourf(diff, levels=200, cmap='jet')
    plt.colorbar()
    plt.title('Erro Percentual')

    # Exibir a figura
    plt.tight_layout()
    plt.show()

    

    # Preparar a SDF para plotagem
    sdf_sample = x1_test[i]
    """
    # Plotar a SDF para visualizar o perfil
    plt.figure(figsize=(12, 6))
    plt.imshow(sdf_sample, origin='lower', cmap='gray')
    plt.colorbar(label='Valor da SDF')
    plt.title(f'Visualização da SDF - Amostra {i}')
    
    # Sobrepor a borda do perfil (contorno onde SDF = 0)
    contours = plt.contour(sdf_sample, levels=[0], colors='black', linewidths=2)
    plt.clabel(contours, inline=True, fontsize=8, fmt='Borda do Perfil')"""
    
    # Agora encontrar os pontos imediatamente acima e abaixo da borda da SDF
    num_rows, num_cols = sdf_sample.shape# Inicializar listas para armazenar os valores para os dorsos superior e inferior
    x_indices_upper = []
    y_indices_upper = []
    pred_values_upper = []
    true_values_upper = []
    
    x_indices_lower = []
    y_indices_lower = []
    pred_values_lower = []
    true_values_lower = []
    
    # Definir a linha média em y=200 (ajuste conforme necessário)
    y_mid = 199.5
    
    # Obter o número de linhas (altura) e colunas (largura) da imagem
    num_rows, num_cols = x1_test.shape[1], x1_test.shape[2]
 
    # Obter a SDF para a amostra atual
    sdf_sample = x1_test[i]

    # Loop sobre cada coluna x
    for x in range(num_cols):
        sdf_column = sdf_sample[:, x]
        s = sdf_column
        zero_crossings = np.where(s[:-1] * s[1:] <= 0)[0]

        # Se houver cruzamentos de zero nesta coluna
        if len(zero_crossings) > 0:
            for y_borda in zero_crossings:
                # Garantir que os índices estão dentro dos limites
                if y_borda >= 1 and y_borda + 1 < num_rows:
                    # Verificar se é o dorso superior ou inferior com base na posição y_borda
                    if y_borda < y_mid:
                        # Dorso Superior
                        # Determinar se a SDF muda de negativo para positivo ou vice-versa
                        s1 = s[y_borda]
                        s2 = s[y_borda + 1]
                        if s1 <= 0 and s2 > 0:
                            # De dentro (negativo) para fora (positivo)
                            y_inside = y_borda
                            y_outside = y_borda + 1
                        elif s1 >= 0 and s2 < 0:
                            # De fora (positivo) para dentro (negativo)
                            y_inside = y_borda + 1
                            y_outside = y_borda
                        else:
                            continue  # Não é um cruzamento válido
                        # Armazenar os valores para o dorso superior
                        x_indices_upper.append(x)
                        y_indices_upper.append(y_outside)
                        pred_values_upper.append(temp_test_inv[i, y_outside, x])
                        true_values_upper.append(y_test_inv[i, y_outside, x])
                    elif y_borda > y_mid:
                        # Dorso Inferior
                        # Determinar se a SDF muda de negativo para positivo ou vice-versa
                        s1 = s[y_borda]
                        s2 = s[y_borda + 1]
                        if s1 <= 0 and s2 > 0:
                            # De dentro (negativo) para fora (positivo)
                            y_inside = y_borda
                            y_outside = y_borda + 1
                        elif s1 >= 0 and s2 < 0:
                            # De fora (positivo) para dentro (negativo)
                            y_inside = y_borda + 1
                            y_outside = y_borda
                        else:
                            continue  # Não é um cruzamento válido
                        # Armazenar os valores para o dorso inferior
                        x_indices_lower.append(x)
                        y_indices_lower.append(y_outside)
                        pred_values_lower.append(temp_test_inv[i, y_outside, x])
                        true_values_lower.append(y_test_inv[i, y_outside, x])
        # Plotar os valores imediatamente acima do dorso superior
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices_upper, pred_values_upper, label='Predição Acima do Dorso Superior', color='red', linestyle='--')
    plt.plot(x_indices_upper, true_values_upper, label='Ground Truth Acima do Dorso Superior', color='darkred')
    plt.xlabel('Índice X')
    plt.ylabel('Temperatura (K)')
    plt.title(f'Valores Imediatamente Acima do Dorso Superior - Amostra {i}')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotar os valores imediatamente abaixo do dorso inferior
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices_lower, pred_values_lower, label='Predição Abaixo do Dorso Inferior', color='blue', linestyle='--')
    plt.plot(x_indices_lower, true_values_lower, label='Ground Truth Abaixo do Dorso Inferior', color='darkblue')
    plt.xlabel('Índice X')
    plt.ylabel('Temperatura (K)')
    plt.title(f'Valores Imediatamente Abaixo do Dorso Inferior - Amostra {i}')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plotar a SDF para visualizar o perfil
    plt.figure(figsize=(12, 6))
    #plt.imshow(sdf_sample, origin='lower', cmap='gray')
    #plt.colorbar(label='Valor da SDF')
    plt.title(f'Visualização da SDF - Amostra {i}')

    # Sobrepor a borda do perfil (contorno onde SDF = 0)
    contours = plt.contour(sdf_sample, levels=[0], colors='black', linewidths=2)
    #plt.clabel(contours, inline=True, fontsize=8, fmt='Borda do Perfil')

    # Sobrepor os pontos no gráfico da SDF
    plt.scatter(x_indices_upper, y_indices_upper, c='red', s=10, label='Pontos Acima do Dorso Superior')
    plt.scatter(x_indices_lower, y_indices_lower, c='blue', s=10, label='Pontos Abaixo do Dorso Inferior')
    plt.legend()
    plt.xlabel('Índice X')
    plt.ylabel('Índice Y')
    plt.title(f'SDF e Pontos Acima/Abaixo do Perfil - Amostra {i}')
    plt.show()
    
        