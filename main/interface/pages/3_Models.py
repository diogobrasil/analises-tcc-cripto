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
Podemos visualizar algumas métricas dos madelos treinados para diferentes nomarlizações e ações.
 """)
st.divider()

# Verifica se os dados estão carregados
if "df_acoes_norm" not in st.session_state:
    st.error("Os dados das ações ainda não foram carregados. Volte para a página inicial!")
    st.stop()

# Caminho para csv com valores de fechamento de todas as ações normalizados
csv_path = st.session_state["df_acoes_norm"]

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

# Lista de normalizações disponíveis
norms = {"Z-Score":"z_score", "MinMax":"min_max", "Robust Scaler":"robust"}

# Normalização
normalization = norms[st.sidebar.radio("Normalização:", ["MinMax", "Z-Score", "Robust Scaler"])]

# Janela de tempo
window = 1

if modelo_selecionado == "Regressão Linear":
    # Treinar e avaliar o modelo
    model, theta, train_metrics, test_metrics, y_test, test_predictions = train_and_evaluate(csv_path, target, normalization, window)

    # Definir um limiar para considerar uma previsão como correta
    threshold = 0.05

    # Calcular a acurácia
    accuracy = sum(abs(test_predictions - y_test) < threshold) / len(y_test)
    
    # Exibir métricas
    st.write("## Métricas de Treino")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Ação:** {next((k for k, v in acoes.items() if v == target), None)}")
    col2.markdown(f"**Normalização:** {next((k for k, v in norms.items() if v == normalization), None)}")
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
    col2.markdown(f"**Normalização:** {next((k for k, v in norms.items() if v == normalization), None)}")
    col3.markdown(f"**Tamanho da Janela:** {window}")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="MSE", value=f"{test_metrics['MSE']:.4f}")
    col2.metric(label="RMSE", value=f"{test_metrics['RMSE']:.4f}")
    col3.metric(label="MAE", value=f"{test_metrics['MAE']:.4f}")
    col4.metric(label="R2", value=f"{test_metrics['R2']:.4f}")

    #Acuracia do modelo
    st.subheader(f"Acurácia: {accuracy:.2%}")
    st.progress(value=accuracy)

    # Gráfico de previsões vs valores reais
    # Criar um índice numérico sequencial
    indice = np.arange(len(y_test))

    # Criar DataFrame com os valores reais e preditos
    df = pd.DataFrame({
        "Índice": indice,
        "Real": y_test,
        "Predito": test_predictions
    })

    # Criar a figura do Plotly manualmente para personalizar as cores
    fig = go.Figure()

    # Adicionar linha para os valores reais (Azul)
    fig.add_trace(go.Scatter(
        x=indice,
        y=y_test,
        mode="lines+markers",
        name="Real",
        line=dict(color="blue", width=2),
        marker=dict(size=6)
    ))

    # Adicionar linha para os valores preditos (Vermelho)
    fig.add_trace(go.Scatter(
        x=indice,
        y=test_predictions,
        mode="lines+markers",
        name="Predito",
        line=dict(color="green", width=2),
        marker=dict(size=6)
    ))

    # Melhorar layout
    fig.update_layout(
        title="Comparação entre Valores Reais e Preditos",
        xaxis_title="Observação",
        yaxis_title="Preço (R$)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        hovermode="x",  # Tooltip alinhado ao eixo X
        legend=dict(x=0, y=1)  # Posição da legenda
    )

    # Exibir o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Modelo não implementado ainda. Por favor, selecione 'Regressão Linear'.")
