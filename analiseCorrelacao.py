import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
data = pd.read_csv("optimization_results.csv")

# Expandir a coluna 'params' para separar os parâmetros em colunas individuais
params = data['params'].apply(eval).apply(pd.Series)
data = pd.concat([data.drop(columns=['params']), params], axis=1)

# Análise inicial: Correlação simples
print("\nCorrelação entre cada parâmetro e o target:")
correlations = data.corr()['target'].drop('target')
print(correlations)

# Visualizar as correlações com mapas de calor
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Mapa de calor das correlações")
plt.show()

# Divisão de dados para modelagem
X = data[['delta_huber', 'lambda_huber', 'lambda_l2']]
y = data['target']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Regressão Linear Múltipla
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Avaliar o modelo de regressão linear
y_pred_linear = linear_model.predict(X_test)
print("\nRegressão Linear Múltipla:")
print("Coeficientes do modelo:")
for param, coef in zip(X.columns, linear_model.coef_):
    print(f"{param}: {coef:.4f}")
print(f"Intercepto: {linear_model.intercept_:.4f}")
print(f"Erro quadrático médio (MSE): {mean_squared_error(y_test, y_pred_linear):.4f}")
print(f"R² (variância explicada): {r2_score(y_test, y_pred_linear):.4f}")

# 2. Regressão Polinomial (grau 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_poly_train)

# Avaliar o modelo de regressão polinomial
y_pred_poly = poly_model.predict(X_poly_test)
print("\nRegressão Polinomial (grau 2):")
print(f"R² no conjunto de teste: {r2_score(y_poly_test, y_pred_poly):.4f}")
print(f"Erro quadrático médio (MSE): {mean_squared_error(y_poly_test, y_pred_poly):.4f}")

# 3. Visualizações
# Parâmetros individuais e target
sns.pairplot(data, vars=['delta_huber', 'lambda_huber', 'lambda_l2', 'target'], kind='reg')
plt.suptitle("Relação entre parâmetros e target", y=1.02)
plt.show()

# Interação entre dois parâmetros e o target
sns.lmplot(x='delta_huber', y='target', hue='lambda_huber', data=data, scatter_kws={'alpha':0.6}, palette='coolwarm')
plt.title("Interação: delta_huber e lambda_huber com target")
plt.show()

sns.lmplot(x='lambda_l2', y='target', hue='delta_huber', data=data, scatter_kws={'alpha':0.6}, palette='coolwarm')
plt.title("Interação: lambda_l2 e delta_huber com target")
plt.show()
