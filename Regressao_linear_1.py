# Importando as bibliotecas

import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns

# Leitura dos dados

dados = pd.read_csv('.../content/Consumo_cerveja.csv', sep=';')
dados.head()


# Estatísticas descritivas

dados.describe().round(2)

# Matriz de correlação
dados.corr().round(4)

# Plotando a variável dependente (y)

fig, ax = plt.subplots(figsize=(20,6))

ax.set_title('Consumo de Cerveja', fontsize=20)
ax.set_ylabel('Litros', fontsize=16)
ax.set_xlabel('Dias', fontsize=16)
ax = dados['consumo'].plot(fontsize=14)

# Boxplot da variável dependente (y)

ax = sns.boxplot(data=dados['consumo'], orient='v', width=0.2)
ax.figure.set_size_inches(12,6)
ax.set_title('Consumo de Cerveja', fontsize=20)
ax.set_ylabel('Litros', fontsize=16)
ax

#  Boxplot com duas variáveis

ax = sns.boxplot(y ='consumo', x='fds', data=dados, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Consumo de Cerveja', fontsize=20)
ax.set_ylabel('Litros', fontsize=16)
ax.set_xlabel('Final de Semana', fontsize=16)
ax

# Paletas de cores

sns.set_palette("Accent_r")

sns.set_style("darkgrid")

ax = sns.boxplot(y ='consumo', x='fds', data=dados, orient='v', width=0.5)
ax.figure.set_size_inches(12,6)
ax.set_title('Consumo de Cerveja', fontsize=20)
ax.set_ylabel('Litros', fontsize=16)
ax.set_xlabel('Final de Semana', fontsize=16)
ax

# Gráficos de dispersão entre as variáveis do dataset

ax = sns.pairplot(dados)

# Plotando o pairplot fixando somente uma variável no eixo y

ax =sns.pairplot(dados, y_vars='consumo', x_vars=['temp_min', 'temp_media', 'temp_max', 'chuva', 'fds'])
ax.fig.suptitle('Dispersão entre as variáveis', fontsize=20, y=1.1)
ax

# Seaborn jointplot

ax = sns.jointplot(x='temp_max', y='consumo', data=dados)

ax.fig.suptitle('Dispersão - consumo X Temperatura', fontsize=18, y=1.05)
ax.set_axis_labels('Temperatura Máxima', 'Consumo de Cerveja', fontsize=14)
ax

# Seaborn jointplot com a reta de regressão estimada

ax = sns.jointplot(x='temp_max', y='consumo', data=dados, kind= "reg" )

ax.fig.suptitle('Dispersão - consumo X Temperatura', fontsize=18, y=1.05)
ax.set_axis_labels('Temperatura Máxima', 'Consumo de Cerveja', fontsize=14)
ax

# Seaborn lmplot

ax =sns.lmplot(x='temp_max', y='consumo', data=dados)

ax.fig.suptitle('Reta de Regressão - consumo X Temperatura', fontsize=16, y=1.02)
ax.set_xlabels('Temperatura Máxima (°C)', fontsize=14)
ax.set_ylabels('Consumo de Cerveja (Litros)', fontsize=14)
ax

# Plotando um lmplot utilizando uma terceira variável na análise (TIPO I)

ax =sns.lmplot(x='temp_max', y='consumo', data=dados, hue='fds', markers=['o', '*'], legend=False )

ax.fig.suptitle('Reta de Regressão - Consumo X Temperatura X Final de Semana', fontsize=16, y=1.02)
ax.set_xlabels('Temperatura Máxima (°C)', fontsize=14)
ax.set_ylabels('Consumo de Cerveja (Litros)', fontsize=14)
ax.add_legend(title='Fim de Semana')

# Plotando um lmplot utilizando uma terceira variável na análise (TIPO II)

ax =sns.lmplot(x='temp_max', y='consumo', data=dados, col='fds' )

ax.fig.suptitle('Reta de Regressão - Consumo X Temperatura X Final de Semana', fontsize=16, y=1.02)
ax.set_xlabels('Temperatura Máxima (°C)', fontsize=14)
ax.set_ylabels('Consumo de Cerveja (Litros)', fontsize=14)
ax

# Estimando um modelo de regressão linear

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
modelo = LinearRegression()
modelo.fit(X_train, y_train)

y = dados['consumo']

print('R² = {}'. format(modelo.score(X_train, y_train).round(2)))

y_previsto = modelo.predict(X_test )

print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(2))

X = dados[['temp_max', 'chuva', 'fds']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)