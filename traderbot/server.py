# Importando os pacotes
import modelo as mdl
import pandas as pd
import numpy as np
import socket
import pytz
from MetaTrader5 import *
from datetime import datetime
from sklearn.externals import joblib


rates_frame = pd.DataFrame()
modelo = mdl.Modelo()

# load the model
#model = joblib.load('modelo_rf_indice_easy.pkl')

# Obtem os dados do ultimo candle: time, open, close, high, low, volume
def getRatesFromMT5():
	# Initializing MT5 connection 
	MT5Initialize()
	MT5WaitForTerminal()

	# Lendo dados do mercado
	timezone = pytz.timezone("America/Sao_Paulo")
	data = datetime.now(tz=timezone)
	utc_from = datetime(data.year, data.month, data.day, data.hour, data.minute, tzinfo=timezone)
	#utc_from = datetime(data.year, 4, 30, 17, 50, tzinfo=timezone)
	rates = MT5CopyRatesFrom("WINM19", MT5_TIMEFRAME_M5, utc_from, 1)
	MT5Shutdown()

	rates_frame = pd.DataFrame(list(rates), 
	                           columns=['time', 'open', 'low', 'high', 'close', 'tick_volume', 'spread', 'real_volume'])
	return rates_frame

  
# criamos uma função simples que diretamente corrija o deslocamento 
def local_to_utc(dt): 
	# para o computador local, obtemos o deslocamento da hora UTC 
	UTC_OFFSET_TIMEDELTA = datetime.utcnow() - datetime.now() 

	return dt + UTC_OFFSET_TIMEDELTA


# Ajusta fuso-horario da coluna time
def ajustaFusoHorario(rates_frame):
	
	  
	# aplicamos o deslocamento para a coluna time no dataframe rates_frame 
	rates_frame['time'] = rates_frame.apply(lambda rate: local_to_utc(rate['time']), axis=1)

	return rates_frame


# Insere dados de indicadores no DataFrame
def insertDataIntoDataFrame(rates_frame, dados=''):
	if dados != '':
		rsi, mm_7, mm_21 = dados.split(';')
		
		rates_frame['rsi'] = rsi
		rates_frame['mm_7'] = mm_7
		rates_frame['mm_21'] = mm_21

		rates_frame['type_candle'] = np.where(rates_frame['open'] > rates_frame['close'], 1, -1) # 1 Alta, -1 Baixa
		rates_frame['dif_open_close'] = rates_frame['open'] - rates_frame['close']
		rates_frame['dif_open_high'] = rates_frame['open'] - rates_frame['high']
		rates_frame['dif_open_low'] = rates_frame['open'] - rates_frame['low']

		rates_frame['time'] = pd.to_datetime(rates_frame['time'])
		rates_frame['hour'] = [data.minute for data in rates_frame['time']]
		rates_frame['dia_semana'] = [data.isoweekday() for data in rates_frame['time']]

		rates_frame['std_open_close'] = 105.953004 # Desvio Padrao

		rates_frame.rename(columns={'real_volume':'volume'}, inplace=True)
		rates_frame.drop(columns=['tick_volume', 'spread'], inplace=True)
		rates_frame.drop(columns=['time'], inplace=True)

	return rates_frame


# Ordenas as colunas da mesma maneira na qual o modelo foi treinado
def ordenaColunas(rates_frame):
	columns = ['open', 'close', 'high', 'low', 'volume', 'rsi', 'mm_7', 'mm_21', 'type_candle', 'dif_open_close', 'dif_open_high', 'dif_open_low', 'hour', 'dia_semana', 'std_open_close']
	rates_frame = rates_frame.reindex(columns=columns)

	return rates_frame



def processa(dados=''):
	df = getRatesFromMT5()
	df = ajustaFusoHorario(df)
	df = insertDataIntoDataFrame(df, dados)
	df = ordenaColunas(df)
	projecao = modelo.predicao(df)
	price_close = df['close'][0]
	print('projecao: ', projecao)
	print('price_close: ', price_close)

	return projecao, price_close



class socketserver:
    def __init__(self, address = '', port = 9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''
        
    def recvmsg(self):
      print('')
      print('Aguardando dados...')
      self.sock.listen(1)
      self.conn, self.addr = self.sock.accept()
      print('connected to', self.addr)
      self.cummdata = ''

      while True:
	      data = self.conn.recv(10000)
	      self.cummdata += data.decode("utf-8")
	      if not data:
	      	break
	      projecao, price_close = processa(self.cummdata)
	      sinal = enviaSinal(projecao, price_close)
	      self.conn.send(bytes(sinal, "utf-8"))
	      return self.cummdata
            
    def __del__(self):
        self.sock.close()

	



def enviaSinal(projecao, price_close):
	if projecao > price_close:
		print('Sinal: BUY')
		return 'BUY'

	if projecao < price_close:
		print('Sinal: SELL')
		return 'SELL'

	return ''


def startServer():
	serv = socketserver('', 9090)

	while True:  
		msg = serv.recvmsg()


startServer()