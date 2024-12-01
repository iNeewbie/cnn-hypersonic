import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
history_df = pd.read_csv('history.csv')

# Criar subplots (2 subgráficos verticais)
fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plotando a Loss
axs[0].semilogy(history_df['loss'], label='Loss', color='blue')
axs[0].semilogy(history_df['val_loss'], label='Val Loss', color='orange')
axs[0].set_title('Perda de Treinamento e Validação')
axs[0].set_ylabel('Perda (log)')
axs[0].legend()
axs[0].grid(True, which="both", ls="--")

# Plotando o MAPE
axs[1].semilogy(history_df['mean_absolute_percentage_error'], label='MAPE', color='blue')
axs[1].semilogy(history_df['val_mean_absolute_percentage_error'], label='Val MAPE', color='orange')
axs[1].set_title('MAPE de Treinamento e Validação')
axs[1].set_xlabel('Épocas')
axs[1].set_ylabel('MAPE (log)')
axs[1].legend()
axs[1].grid(True, which="both", ls="--")

# Ajustar layout para não sobrepor os gráficos
plt.tight_layout()

# Exibir os gráficos
plt.show()
