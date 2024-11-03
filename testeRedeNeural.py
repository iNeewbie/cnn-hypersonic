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

x1_test= x1_train
x2_test=x2_train
y_test=y_train
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
temp_test = temp_test[:, 1:-1, 1:-1]
x1_test = x1_test[:, 1:-1, 1:-1]
y_test = y_test[:, 1:-1, 1:-1]

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
for i in range(len(tempMasked_inv)):  # Ajuste o range conforme necessário
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
    diff = np.abs(tempMasked_inv[i] - y_testMasked_inv[i]) / np.abs(y_testMasked_inv[i])  # Evitar divisão por zero

    # Gráfico do erro percentual
    plt.subplot(2, 2, 3)
    plt.contourf(diff, levels=200, cmap='jet')
    plt.colorbar()
    plt.title('Erro Percentual')

    # Exibir a figura
    plt.tight_layout()
    plt.show()



# Loop sobre as três primeiras amostras de teste
for i in range(len(tempMasked_inv)):  # Ajuste o range conforme necessário
    # Defina a coordenada x para a linha de corte vertical
    cut_x = 200

    # Extrair valores ao longo da linha vertical para a predição e o ground truth
    pred_line = tempMasked_inv[i, :, cut_x]
    truth_line = y_testMasked_inv[i, :, cut_x]

    # Plot da linha vertical de corte
    plt.figure(figsize=(10, 6))
    plt.plot(pred_line, label='Predição (Denormalizada e Exponenciada)', color='blue')
    plt.plot(truth_line, label='Ground truth (Denormalizado e Exponenciado)', color='orange')
    plt.xlabel('Índice y')
    plt.ylabel('Valor')
    plt.title(f'Linha de Corte Vertical em x={cut_x} - {label_test[i]}')
    plt.legend()
    plt.grid()
    plt.show()

    # Inicializar listas para armazenar os valores acima e abaixo do perfil
    above_profile_pred = []
    below_profile_pred = []
    above_profile_truth = []
    below_profile_truth = []

    # Loop sobre cada coluna x
    for x in range(tempMasked_inv.shape[2]):
        # Encontre o índice y onde a SDF está mais próxima de zero (próximo à borda do perfil)
        sdf_column = x1_test[i, :, x]  # Use a amostra atual
        y_borda = np.argmin(np.abs(sdf_column))

        # Pega valores imediatamente acima e abaixo da borda
        if y_borda > 0 and y_borda < tempMasked_inv.shape[1] - 1:  # Verifica limites da borda
            above_profile_pred.append(tempMasked_inv[i, y_borda + 1, x])
            below_profile_pred.append(tempMasked_inv[i, y_borda - 1, x])
            above_profile_truth.append(y_testMasked_inv[i, y_borda + 1, x])
            below_profile_truth.append(y_testMasked_inv[i, y_borda - 1, x])

    # Plot dos valores imediatamente acima do perfil
    plt.figure(figsize=(12, 6))
    plt.plot(above_profile_pred, label='Predição Acima do Perfil', color='blue', linestyle='--')
    plt.plot(above_profile_truth, label='Ground Truth Acima do Perfil', color='orange', linestyle='--')
    plt.xlabel('Índice x')
    plt.ylabel('Valor')
    plt.title(f'Comparação de Valores Acima do Perfil Aerodinâmico - {label_test[i]}')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot dos valores imediatamente abaixo do perfil
    plt.figure(figsize=(12, 6))
    plt.plot(below_profile_pred, label='Predição Abaixo do Perfil', color='blue')
    plt.plot(below_profile_truth, label='Ground Truth Abaixo do Perfil', color='orange')
    plt.xlabel('Índice x')
    plt.ylabel('Valor')
    plt.title(f'Comparação de Valores Abaixo do Perfil Aerodinâmico - {label_test[i]}')
    plt.legend()
    plt.grid()
    plt.show()