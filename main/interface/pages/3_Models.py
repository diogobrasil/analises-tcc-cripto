import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from classes.neural_networks.training.train_linear_regression import train_and_evaluate

# configurações da página
st.set_page_config(page_title="Modelos de Machine Learning", page_icon="🧠", layout="wide")

# título da página
st.title("🧠 Modelos de Machine Learning")
st.divider()
st.markdown(""" 
Podemos visualizar algumas métricas dos madelos treinados para diferentes ações.
 """)
st.divider()

# Verifica se os dados estão carregados
if "df_acoes_norm" not in st.session_state:
    st.error("Os dados das ações ainda não foram carregados. Volte para a página inicial!")
    st.stop()

# Verifica se os dados estão carregados
if "df_acoes" not in st.session_state:
    st.error("Os dados das ações ainda não foram carregados. Volte para a página inicial!")
    st.stop()

# Caminho para csv com valores de fechamento de todas as ações normalizados
csv_path = st.session_state["df_acoes_norm"]

# Carrega os dados para obter as datas
df = st.session_state["df_acoes"]
dates = df.index.values

# Seleção do modelo
modelo_selecionado = st.sidebar.selectbox("Escolha um modelo:", ["Regressão Linear", "Random Forest", "Rede Neural"])

# Lista das ações disponíveis
acoes = {
    "Vale (VALE3)":"VALE3",
    "Itaú Unibanco (ITUB4)":"ITUB4",
    "Eletrobras (ELET3)":"ELET3",
    "Petrobras (PETR3)":"PETR3",
    "Banco do Brasil (BBAS3)":"BBAS3",
    "Bradespar (BRAP3)":"BRAP3",
    "Equatorial (EQTL3)":"EQTL3",
    "Cyrela (CYRE3)":"CYRE3",
    "Direcional Engenharia (DIRR3)":"DIRR3",
    "Construtora Tenda (TEND3)":"TEND3", 
}

# Seleção do target
target = acoes[st.sidebar.selectbox("Escolha uma ação:", list(acoes.keys()))]

# Janela de tempo
window = 3

if modelo_selecionado == "Regressão Linear":
    # Treinar e avaliar o modelo
    model, theta, train_metrics, val_metrics, test_metrics, y_test, test_predictions = train_and_evaluate(csv_path, target, window)

    # Exibir métricas
    st.write("## Métricas de Treino")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Ação:** {next((k for k, v in acoes.items() if v == target), None)}")
    col2.markdown(f"**Normalização:** Min-Max")
    col3.markdown(f"**Tamanho da Janela:** {window}")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="MSE", value=f"{train_metrics['MSE']:.4f}")
    col2.metric(label="RMSE", value=f"{train_metrics['RMSE']:.4f}")
    col3.metric(label="MAE", value=f"{train_metrics['MAE']:.4f}")
    col4.metric(label="R2", value=f"{train_metrics['R2']:.4f}")
    st.divider()

    st.write("## Métricas de Teste")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Ação:** {next((k for k, v in acoes.items() if v == target), None)}")
    col2.markdown(f"**Normalização:** Min-Max")
    col3.markdown(f"**Tamanho da Janela:** {window}")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="MSE", value=f"{test_metrics['MSE']:.4f}")
    col2.metric(label="RMSE", value=f"{test_metrics['RMSE']:.4f}")
    col3.metric(label="MAE", value=f"{test_metrics['MAE']:.4f}")
    col4.metric(label="R2", value=f"{test_metrics['R2']:.4f}")
    st.divider()

    st.write("## Métricas de Validação")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Ação:** {next((k for k, v in acoes.items() if v == target), None)}")
    col2.markdown(f"**Normalização:** Min-Max")
    col3.markdown(f"**Tamanho da Janela:** {window}")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="MSE", value=f"{val_metrics['MSE']:.4f}")
    col2.metric(label="RMSE", value=f"{val_metrics['RMSE']:.4f}")
    col3.metric(label="MAE", value=f"{val_metrics['MAE']:.4f}")
    col4.metric(label="R2", value=f"{val_metrics['R2']:.4f}")

    # Criar a figura com Plotly
    fig = go.Figure()

    # Adicionar linha para os valores reais
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_test,
        mode="lines+markers",
        name="Real",
        line=dict(color="blue", width=2),
        marker=dict(size=6)
    ))

    # Adicionar linha para os valores preditos
    fig.add_trace(go.Scatter(
        x=dates,
        y=test_predictions,
        mode="lines+markers",
        name="Predito",
        line=dict(color="green", width=2),
        marker=dict(size=6)
    ))

    # Configurar o eixo X para exibir apenas algumas datas (evita poluição visual)
    num_dates = len(y_test)
    step = num_dates // 5  # Intervalo para 5 datas
    fig.update_xaxes(
        tickvals=dates[::step],  # Define os pontos do eixo X
        ticktext=[pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates[::step]],  # Converte para string formatada
        tickangle=45
    )

    # Melhorar layout
    fig.update_layout(
        title="Comparação entre Valores Reais e Preditos",
        xaxis_title="Datas",
        yaxis_title="Preço (R$)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x",
        legend=dict(x=0, y=1)
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Modelo não implementado ainda. Por favor, selecione 'Regressão Linear'.")
