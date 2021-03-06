//+------------------------------------------------------------------+
//|                                             EA_Random_Forest.mq5 |
//|                        Copyright 2018, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"


input ENUM_TIMEFRAMES _periodo = PERIOD_M5;
input int periodo_rsi = 24;
input int media_movel_rapida = 7;
input int media_movel_lenta = 21;
input int stop_loss = 30;
input int qtde_contrato = 1;


// Indicadro RSI
int RSI_Handle;
double RSI_Buffer[];

// Indicador Média Móvel
int MA_Fast_Handle;
int MA_Slow_Handle;
double MA_Fast_Buffer[];
double MA_Slow_Buffer[];

int socket;
string dados;
string sinal;
string linha;
string NOME_ROBO = "EA_Random_Forest";
string file_path = "historico_operacoes.csv";
int handle_file;
double profit;

MqlTradeRequest request; // nossa ordem
MqlTradeResult  result;  // resposta do envio da nossa ordem
MqlDateTime mqlDateTime;
MqlTick tick;

int OnInit()
  {
      //EventSetTimer(5);
      
      RSI_Handle = iRSI("WIN$", _periodo, periodo_rsi, PRICE_CLOSE);
      MA_Fast_Handle = iMA("WIN$", _periodo, media_movel_rapida, 0, MODE_EMA, PRICE_CLOSE);
      MA_Slow_Handle = iMA("WIN$", _periodo, media_movel_lenta, 0, MODE_EMA, PRICE_CLOSE);
      
      return(INIT_SUCCEEDED);
  }


void OnDeinit(const int reason)
  {
    
  }


void OnTick()
  {
      
      if(isNewBar())
        {
            SymbolInfoTick(_Symbol, tick);
            
            if(horarioDeNegociacao())
              {
                  SinalTrade();
              }
            else
              {
                  // Encerra Posicao as 17h
                  TimeCurrent(mqlDateTime);
                  if(mqlDateTime.hour == 17)
                  {
                     encerraPosicao();
                  }
              }
        }
      else
        {
            if(PositionSelect(_Symbol))
            {
               profit = PositionGetDouble(POSITION_PROFIT);
               if(profit < -stop_loss)
                 {
                     encerraPosicao();
                 }
            }
        }
   
  }




void encerraPosicao()
{

   // Se tem posicao, encerra
   if(PositionSelect(_Symbol))
     {
         ENUM_POSITION_TYPE tipoPosicaoAberta = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         if(tipoPosicaoAberta == POSITION_TYPE_BUY)
           {
               Venda();
               registrarTrade("close");
           }
         else
           {
               Compra();
               registrarTrade("close");
           }
     }
}

void Trade() {

   Print("Sinal: ", sinal);

   if(PositionSelect(_Symbol))
     {
         // Caso o sinal seja na posicao oposta a posicao atual entao:
         // Encerra a posicao atual e abre uma nova na posicao do sinal
         // Caso contrario nao faz nada, continua com a posicao aberta
     
         ENUM_POSITION_TYPE tipoPosicaoAberta = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         if(tipoPosicaoAberta == POSITION_TYPE_BUY)
           {
               if(sinal == "SELL")
                 {
                     // Encerra posicao
                     Venda();
                     registrarTrade("close");
                     
                     // Abre uma nova na direcao do sinal
                     Venda();
                     registrarTrade("open");
                 }
           }
         else
           {
               if(sinal == "BUY")
                 {
                     // Encerra posicao
                     Compra();
                     registrarTrade("close");
                     
                     // Abre uma nova na direcao do sinal
                     Compra();
                     registrarTrade("open");
                 }
           }
     }
   else
     {
         if(sinal == "BUY")
           {
               Compra();
               registrarTrade("open");
           }
         
         if(sinal == "SELL")
           {
               Venda();
               registrarTrade("open");
           }
     }

}

void SinalTrade()
{
   
   int socket = SocketCreate();
   if(socket!=INVALID_HANDLE) {
   	if(SocketConnect(socket,"127.0.0.1",9090,1000)) {
   		Print("Connected to "," localhost",":",9090);
       
         CopyBuffer(RSI_Handle, 0, 0, 2, RSI_Buffer);
         CopyBuffer(MA_Fast_Handle, 0, 0, 2, MA_Fast_Buffer);
         CopyBuffer(MA_Slow_Handle, 0, 0, 2, MA_Slow_Buffer);
       
   		dados = RSI_Buffer[1] + ";" + MA_Fast_Buffer[1] + ";" + MA_Slow_Buffer[1];
   		
   		sinal = socksend(socket, dados) ? socketreceive(socket, 5000) : ""; 
   		Trade();
   		
   		SocketClose(socket);
   	}
   	else {
   		Print("Connection ","localhost",":",9090," error ",GetLastError());
   	}
   	
   }
   else {
   	Print("Socket creation error ",GetLastError());
   }
   
}

void OnTimer()
  {
      //SinalTrade();
  }




bool horarioDeNegociacao()
{
   TimeCurrent(mqlDateTime);
   if(mqlDateTime.hour < 9 || mqlDateTime.hour > 16)
     {
         return(false);
     }
   if(mqlDateTime.hour == 9 && mqlDateTime.min <= 15)
     {
         return(false);
     }
     
    return(true);
}


void registrarTrade(string momento)
{

   if(momento == "open")
     {
         linha = TimeCurrent();
         if(sinal == "BUY")
           {
                linha = linha + ";" + "1";
           }
         else
           {
               linha = linha + ";" + "-1";
           }
           linha = linha + ";" + tick.last;
     }
   else
     {
         linha = linha + ";" + TimeCurrent() + ";" + tick.last;
         handle_file = FileOpen(file_path,FILE_WRITE|FILE_ANSI|FILE_CSV);
         if(handle_file == INVALID_HANDLE){
            Alert("Error opening file");
            return;
         }
         FileWrite(handle_file, linha);
         FileClose(handle_file);
         
     }
}

bool isNewBar()
  {
//--- memorize the time of opening of the last bar in the static variable
   static datetime last_time=0;
//--- current time
   datetime lastbar_time=(datetime)SeriesInfoInteger(Symbol(),_periodo,SERIES_LASTBAR_DATE);

//--- if it is the first call of the function
   if(last_time==0)
     {
      //--- set the time and exit
      last_time=lastbar_time;
      return(false);
     }

//--- if the time differs
   if(last_time!=lastbar_time)
     {
      //--- memorize the time and return true
      last_time=lastbar_time;
      return(true);
     }
//--- if we passed to this line, then the bar is not new; return false
   return(false);
  }





// Envia dados para o servidor
bool socksend(int sock,string request) 
  {
   char req[];
   int  len=StringToCharArray(request,req)-1;
   if(len<0) return(false);
   return(SocketSend(sock,req,len)==len); 
  }
  

// Fica escutando a porta esperando uma resposta do servidor
string socketreceive(int sock,int timeout)
  {
   char rsp[];
   string result="";
   uint len;
   uint timeout_check=GetTickCount()+timeout;
   do
     {
      len=SocketIsReadable(sock);
      if(len)
        {
         int rsp_len;
         rsp_len=SocketRead(sock,rsp,len,timeout);
         if(rsp_len>0) 
           {
            result+=CharArrayToString(rsp,0,rsp_len); 
           }
        }
     }
   while((GetTickCount()<timeout_check) && !IsStopped());
   return result;
  }
  
  
  

void Compra()
{
   
   
   // Limpa os objetos
   ZeroMemory(request);
   ZeroMemory(result);
   
   //--- Definir todas as características da ordem
   request.action = TRADE_ACTION_DEAL;
   request.magic = 1234;
   request.symbol = _Symbol; // Ativo
   request.volume = qtde_contrato; // Quantidade de contrato ou quantidade de ações
   request.type = ORDER_TYPE_BUY; // Compra/Venda
   request.type_filling = ORDER_FILLING_FOK; //Executa tudo ou nada
   request.type_time = ORDER_TIME_DAY; // Quando a ordem expira
   request.expiration = 0; // Hora que a ordem expira
   request.comment = "Order enviada pelo "+ NOME_ROBO;
   
   // Limpa os erros
   ResetLastError();
   
   // Comando para enviar order para negociação
   bool returnOrder = OrderSend(request, result);
   
   // Se false significa que houve um erro
   // e sua order não foi enviada para a corretora
   if(!returnOrder)
     {
      Print("Erro ao enviar a ordem: ", GetLastError());
      ExpertRemove(); // Remove o robo caso dê algum erro
      // O que fazer quando dá erro?
      /*
         1- Remover o EA
         2- Tentar enviar a ordem novamente
         3- Enviar um alerta/notificação/e-mail
         
      */
     }
    else
      {
       // Se chegar aqui significa que a ordem foi enviada para a corretora
       // Mas não sabemos se a ordem foi executa
       // para isso precisamos chegar a variavel MqlTradeResult
       
       // code 10008 --> Ordem colocada
       // code 10009 --> Solicitação concluída
       if(result.retcode == 10008 || result.retcode == 10009)
         {
            Print("Ordem corretamente colocada ou executada - COMPRA");
         }
         else
           {
               // Se entrar aqui significa que houve algum erro ao tentar executar a ordem na bolsa
               // Este é um código de erro da corretora
               Print("Erro ao enviar a ordem: ", result.retcode);
             
               ExpertRemove(); // Remove o robo caso dê algum erro  
             
               // O que fazer quando dá erro?
               /*
                  1- Remover o EA
                  2- Tentar enviar a ordem novamente
                  3- Enviar um alerta/notificação/e-mail
                  
               */
           }
       
      }
}

void Venda()
{
   
   //--- Definir todas as características da ordem
   request.action = TRADE_ACTION_DEAL;
   request.magic = 1234;
   request.symbol = _Symbol; // Ativo
   request.volume = qtde_contrato; // Quantidade de contrato ou quantidade de ações
   request.type = ORDER_TYPE_SELL; // Compra/Venda
   request.type_filling = ORDER_FILLING_FOK; //Executa tudo ou nada
   request.type_time = ORDER_TIME_DAY; // Quando a ordem expira
   request.expiration = 0; // Hora que a ordem expira
   request.comment = "Order enviada pelo "+ NOME_ROBO;
   
   // Limpa os erros
   ResetLastError();
   
   // Comando para enviar order para negociação
   bool returnOrder = OrderSend(request, result);
   
   // Se false significa que houve um erro
   // e sua order não foi enviada para a corretora
   if(!returnOrder)
     {
      Print("Erro ao enviar a ordem: ", GetLastError());
      ExpertRemove(); // Remove o robo caso dê algum erro
      // O que fazer quando dá erro?
      /*
         1- Remover o EA
         2- Tentar enviar a ordem novamente
         3- Enviar um alerta/notificação/e-mail
         
      */
     }
    else
      {
       // Se chegar aqui significa que a ordem foi enviada para a corretora
       // Mas não sabemos se a ordem foi executa
       // para isso precisamos chegar a variavel MqlTradeResult
       
       // code 10008 --> Ordem colocada
       // code 10009 --> Solicitação concluída
       if(result.retcode == 10008 || result.retcode == 10009)
         {
            Print("Ordem corretamente colocada ou executada - VENDA");
         }
         else
           {
               // Se entrar aqui significa que houve algum erro ao tentar executar a ordem na bolsa
               // Este é um código de erro da corretora
               Print("Erro ao enviar a ordem: ", result.retcode);
             
               ExpertRemove(); // Remove o robo caso dê algum erro  
             
               // O que fazer quando dá erro?
               /*
                  1- Remover o EA
                  2- Tentar enviar a ordem novamente
                  3- Enviar um alerta/notificação/e-mail
                  
               */
           }
       
      }
}
