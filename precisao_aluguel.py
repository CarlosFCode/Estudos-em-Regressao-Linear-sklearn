import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Dataframe
dataset = {
    'Tamanho': [40, 60, 80, 100, 120],
    'Número de quartos': [1, 2, 3, 3, 4],
    'Preço do Aluguel': [1200, 2000, 3200, 4000, 5000]
}

#Transformar o dicionário em um dataframe
df = pd.DataFrame(dataset)

#separar os dados em variáveis dependentes e independentes
x = df[['Tamanho', 'Número de quartos']]
y = df['Preço do Aluguel']

#Aplicar a regressão linear
modelo = LinearRegression()
modelo.fit(x,y)

#Previsão sobre o preço
y_pred = modelo.predict(x)

#Avaliar o modelo
mse = mean_squared_error(y, y_pred)

#Quanto mais o R2 tiver perto do número 1, mais o modelo de regressão linear está acertando.
r2 = r2_score(y, y_pred)
print('MSE: {:.2f}'.format(mse))
print('R2: {:.3f}'.format(r2))

#Visualizar o modelo

plt.scatter(y, y_pred, color='blue', label="Previsões")
plt.plot(y, y_pred, color='red', label="Linha de Regressão")
plt.title('Previsão do Preço do Aluguel')
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.legend()
plt.show()