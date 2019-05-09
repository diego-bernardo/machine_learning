# Importando pacotes
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor


class Modelo(object):

	"""docstring for Modelo"""
	def __init__(self):
		warnings.filterwarnings("ignore")
		print('Iniciando Modelo')
		self.prepararDados()
		self.treinarModelo()
		

	def prepararDados(self):

		print('Preparando os Dados...')

		# Lendo conjunto de dados
		df = pd.read_csv('dados_indice_m5.csv', sep=';')

		########################
		# Criando novas features
		df['type_candle'] = np.where(df['open'] > df['close'], 1, -1) # 1 Alta, -1 Baixa
		df['dif_open_close'] = df['open'] - df['close']
		df['dif_open_high'] = df['open'] - df['high']
		df['dif_open_low'] = df['open'] - df['low']
		df['time'] = pd.to_datetime(df['time'])
		df['hour'] = [data.minute for data in df['time']]
		df['dia_semana'] = [data.isoweekday() for data in df['time']]
		df['std_open_close'] = df['dif_open_close'].std()
		df['target'] = df['close']
		df['target'] = df['target'].shift(-1)
		df.drop(columns=['time'], inplace=True)


		##########################################
		# Separando dados para treinamento e teste
		self.train = df.iloc[500:2900,:]
		self.test  = df.iloc[2900:-1,:]
		self.target_train = self.train['target']
		self.target_test = self.test['target']
		self.train.drop(columns=['target'], inplace=True)
		self.test.drop(columns=['target'], inplace=True)

		del df # liberar espaco na memoria


	def treinarModelo(self):
		print('Treinando o modelo...')
		self.model = RandomForestRegressor(n_estimators=10000, n_jobs=6)
		self.model.fit(self.train, self.target_train)


	def predicao(self, dados):
		alvo = self.model.predict(dados)

		return alvo[0]
