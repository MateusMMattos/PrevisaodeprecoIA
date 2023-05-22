import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tabela = pd.read_csv('./barcos_ref.csv')
print(tabela.corr()['Preco'])

sns.heatmap(tabela.corr()[['Preco']], annot=True, cmap='Blues')
plt.show()

y = tabela['Preco']
x = tabela.drop('Preco', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.4, random_state=1)

#cria as inteligencias artificiais

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#treina as inteligencias artificiais

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

#cria as previsoes

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#compara os modelos

print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar['y_teste'] = y_teste
tabela_auxiliar['Previsoes ArvoreDecisao'] = previsao_arvoredecisao
tabela_auxiliar['Previsoes Regressao Linear'] = previsao_regressaolinear

sns.lineplot(data=tabela_auxiliar)
plt.show()

nova_tabela = pd.read_csv('novos_barcos.csv')
print(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)