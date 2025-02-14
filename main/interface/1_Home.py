import streamlit as st
import pandas as pd
import os
import sys

# Adiciona o caminho da pasta `main` ao sys.path para usar bibliotecas locais dentro do Streamlit
sys.path.append(os.path.abspath("../"))

# configurações da página
st.set_page_config(page_title="Home", page_icon="📊", layout="wide")

# Carregar o DataFrame com todas as ações
if "df_acoes" not in st.session_state:
    df_acoes = pd.read_csv("../datasets/b3_dados/processed/acoes_concat.csv")
    df_acoes["Date"] = pd.to_datetime(df_acoes["Date"])
    df_acoes.set_index("Date", inplace=True)
    st.session_state["df_acoes"] = df_acoes  # Salvar no session_state

# Carregar o DataFrame com todas as ações normalizadas
if "df_acoes_norm" not in st.session_state:
    path_norm = "../datasets/b3_dados/processed/acoes_concat.csv"
    st.session_state["df_acoes_norm"] = path_norm  # Salvar no session_state

st.write("# DADOS DE AÇÕES NEGOCIADAS NA B3 ENTRE 2010 E 2022 📈")
st.divider()
st.markdown(""" 
    Nosso projeto consiste em analisar dados de ações negociadas na B3 entre 2010 e 2022 e, usando modelos de Machine Learning e, realizar previsões dos mesmos.   
    Neste projeto, nos atemos aos valores de fechamento das ações, mas você pode expandir o projeto para incluir outros valores, como valores de abertura, máximos e mínimos, volumes de negociação, etc.   
    Na barra lateral temos a página **Stocks** onde você pode analisar os dados de cada ação individualmente e na página **Models** você pode treinar modelos de Machine Learning para prever o valor de fechamento das ações.   
    Escolhemos 11 ações para análise, o criteiro é que elas tenham 10 anos ou mais de negociação na B3. São elas: 
      - Vale (VALE3)
      - Itaú Unibanco (ITUB4)
      - Eletrobras (ELET3)
      - Petrobras (PETR3)
      - Banco do Brasil (BBAS3)
      - Bradespar (BRAP3) 
      - Equatorial (EQTL3)
      - Cemig (CMIG4)
      - Cyrela (CYRE3)
      - Direcional Engenharia (DIRR3)
      - Construtora Tenda (TEND3)   
    
    **Objetivo**: Análise de dados de ações negociadas na B3 entre 2010 e 2022.   
    **Fonte dos dados**: Yahoo Finance
""")
st.sidebar.markdown(""" 
                    Autores do Projeto: 
                     - [Diogo Brasil Da Silva](https://www.github.com/digobrasil) 
                     - [Matheus Costa Alves](https://github.com/matheus2049alves)
                     - [Emanuel Lopes Silva](https://github.com/EmanuelSilva69)
                    """)
