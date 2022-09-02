import pandas as pd
import streamlit as st 
import yfinance as yf
import numpy as np
import json


from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go


############################# PAGINA INICIAL ##############################
st.title('Analisador Financeiro')
st.subheader('Bem vindo!!!')
st.subheader('O que deseja analisar hoje?')

acoes = st.checkbox('Ações')
crypto = st.checkbox('Cryptomoedas')



if not acoes and not crypto:
    opcao = st.write('')

    #PLOTANDO DADOS DO DOLAR
    st.subheader('Dolar Hoje')
    #PEGANDO DADOS ONLINE
    dados_dolar = yf.Ticker('USDBRL=X').history(period='1d', interval='1m' )
    dados_dolar.reset_index(inplace=True)

    #PLOATANDO GRAFICO E TABELA
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados_dolar['Datetime'],
                    y=dados_dolar['Close'],
                    name='Preço de Fechamento',
                    line_color='blue'))
    st.plotly_chart(fig)
    st.write(dados_dolar)
    

######################### ANALIZANDO AÇÕES ##############################
elif acoes:
    #CRIANDO SIDEBAR
    st.sidebar.header('Configurações')


    #PEGANDO OS DADOS DAS AÇÕES
    def pegar_dados_acoes():
        path = 'acoes.csv'
        return pd.read_csv(path, delimiter=';')

    df = pegar_dados_acoes()

    #COLOCANDO OS DADOS NO STREAMLIT
    acao = df['snome']
    nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação:', acao)
    df_acao = df[df['snome'] == nome_acao_escolhida]
    acao_escolhida = df_acao.iloc[0]['sigla_acao']
    acao_escolhida = acao_escolhida + '.SA'

    #CONFIGURANDO DATA PARA PREVISÃO
    DATA_INICIO = st.sidebar.date_input('Data de inicio', date(2017,1,1))
    DATA_FIM = st.sidebar.date_input('Data de fim')

    #CONFIGURANDO A QTD DE DIAS PARA PREVISÃO
    n_dias = st.sidebar.slider('Quantidade de dias de previsão',10,365)


    #PEGANDO DADOS DAS AÇÕES ONLINE
    @st.cache
    def pegar_valores_online(sigla_acao):
        df = yf.download(sigla_acao,DATA_INICIO,DATA_FIM)
        df.reset_index(inplace=True)
        return df

    df_valores = pegar_valores_online(acao_escolhida)


    #PLOTANDO UMA TABELA COM OS VALORES ADQUIRIDOS
    st.subheader('Tabela de valores de - ' + nome_acao_escolhida)
    st.write(df_valores.tail(10))

    #PLOTANDO GRAFICO COM PREÇOS
    st.subheader('Gráfico de preço')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_valores['Date'],
                            y=df_valores['Close'],
                            name='Preço de Fechamento',
                            line_color='red'))

    fig.add_trace(go.Scatter(x=df_valores['Date'],
                            y=df_valores['Open'],
                            name='Preço de Abertura',
                            line_color='blue'))
    st.plotly_chart(fig)



    #COMEÇANDO A PREVISÃO, PEGANDO DADOS DE TREINO
    df_treino = df_valores[['Date','Close']]

    #RENOMEADNDO AS COLUNAS QUE O PROPHET EXIGE QUE MUDEMOS
    df_treino = df_treino.rename(columns={'Date': 'ds', 'Close': 'y'})

    #TREINANDO O MODELO
    modelo = Prophet()
    modelo.fit(df_treino)

    #FAZENDO A PREVISÃO
    futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
    previsao = modelo.predict(futuro)


    #PLOTANDO UMA TABELA COM OS PREÇOS PREVISTOS NO STREAMLIT
    st.subheader('Previsão')
    st.write(previsao[['ds','yhat','yhat_lower','yhat_upper','additive_terms']].tail(n_dias))


    #PLOTANDO O GRAFICO 1 NO STREAMLIT
    st.subheader('Gráfico de Preços e Previsões')
    grafico1 = plot_plotly(modelo,previsao)
    st.plotly_chart(grafico1)


    #PLOTANDO O GRAFICO 2 NO STREAMLIT
    st.subheader('Gráfico de Tendências')
    grafico2 = plot_components_plotly(modelo,previsao)
    st.plotly_chart(grafico2)

    st.subheader('PREÇO DE FECHAMENTO')
    ontem,hoje,amanha = st.columns(3)

    price = df_valores['Close'] - df_valores['Open']

    ontem.metric(label = nome_acao_escolhida + '- ONTEM ',
                value = df_valores['Close'][len(df_valores)-2],
                delta = price[len(df_valores)-2]
                )

    hoje.metric(label = nome_acao_escolhida + '- HOJE ',
            value = df_valores['Close'][len(df_valores)-1],
            delta = price[len(df_valores)-1]
            )
    amanha.metric(label = nome_acao_escolhida + '- AMANHA ',
            value = previsao['yhat'][(len(previsao)- n_dias)],
            delta = previsao['additive_terms'][(len(previsao)- n_dias)]  )

    


######################### ANALIZANDO CRYPTO ##############################
elif crypto:
    
    #LENDO JSON
    with open('data.json', 'r') as _json:
        data_string = _json.read()

    obj = json.loads(data_string)

    crypto_names = obj["crypto_names"]
    crypto_symbols = obj["crypto_symbols"]

    #CRIANDO DICIONARIO COM NOMES E SIMBOLOS
    crypto_dict = dict(zip(crypto_names, crypto_symbols))

    #CRIANDO SIDEBAR
    st.sidebar.header('Configurações')
    crypto_selected = st.sidebar.selectbox(label = 'Escolha uma crypto', 
                                    options = crypto_dict.keys())

    start_date = st.sidebar.date_input("Data de inicio", date(2017,1,1))
    final_date = st.sidebar.date_input("Data Final")

    n_dias = st.sidebar.slider('Quantidade de dias de previsão', 10,365)

    #CHAMANDO SIMBOLO NO YFINANCE
    _symbol = crypto_dict[crypto_selected] + '-USD'

    df = yf.Ticker(_symbol).history(interval='1d', 
                                    start=start_date, 
                                    end=final_date)

    
    #PLOTANDO UMA TABELA COM OS VALORES ADQUIRIDOS
    st.subheader('Tabela de valores de - ' + crypto_selected)
    st.write(df.tail(10))

  
     #PLOTANDO GRAFICO COM PREÇOS
    st.subheader('Gráfico de preço')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,
                            y=df['Close'],
                            name='Preço de Fechamento',
                            line_color='red'))

    fig.add_trace(go.Scatter(x=df.index,
                            y=df['Open'],
                            name='Preço de Abertura',
                            line_color='blue'))
    st.plotly_chart(fig)


    #PREVISÃO
    df_prophet = pd.DataFrame()

    df_cut = df.loc[start_date:final_date]['Close']

    df_prophet['ds'] = df_cut.index
    df_prophet['y'] = df_cut.values

    #TREINANDO O MODELO
    m = Prophet()
    m.fit(df_prophet)

    #FAZENDO A PREVISÃO
    future = m.make_future_dataframe(periods=n_dias)
    future.tail()
    forecast = m.predict(future)

    #PLOTANDO A PREVISÃO NO STREAMLIT
    st.subheader('Previsão')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

    df_to_plot = pd.DataFrame(index = forecast['ds'])
    df_to_plot['trend_prevision'] = forecast['yhat'].values

    df_to_plot['close_price'] = np.nan
    df_to_plot['close_price'].loc[start_date:final_date] = df_cut

    
    #PLOTANDO GRAFICO DA PREVISÃO
    st.subheader('Grafico com dias de previsão')
    grafico1 = plot_plotly(m,forecast)
    st.plotly_chart(grafico1)

    #PLOTANDO GRAFICO DE TENDÊNCIA
    st.subheader('Grafico Tendência')
    grafico2 = plot_components_plotly(m,forecast)
    st.plotly_chart(grafico2)


    st.subheader('PREÇO DE FECHAMENTO')
    ontem,hoje,amanha = st.columns(3)

    price = df['Close'] - df['Open']

    ontem.metric(label = crypto_selected + '- ONTEM ',
            value = df['Close'][len(df)-2],
            delta = price[len(df)-2]
            )

    hoje.metric(label = crypto_selected + '- HOJE ',
        value = df['Close'][len(df)-1],
        delta = price[len(df)-1]
        )
    amanha.metric(label = crypto_selected + '- AMANHA ',
        value = forecast['yhat'][(len(forecast) - n_dias)],
        delta = forecast['additive_terms'][(len(forecast)- n_dias)]  )
