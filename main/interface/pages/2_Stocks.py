import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="Ações", page_icon="📈", layout="wide")

st.title("📈 Análise de Ações")
st.divider()
st.markdown("""
            Aqui poderemos analisar o nosso conjunto de dados e também visualizar informações para uma ação específica, em um ano especifco e com certa granularidade.
          """)
st.divider()

# Verifica se os dados estão carregados
if "df_acoes" not in st.session_state:
    st.error("Os dados das ações ainda não foram carregados. Volte para a página inicial!")
    st.stop()

# Recupera os dados carregados na Home
df_acoes = st.session_state["df_acoes"]

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

# Inicializar a variável no session_state se ainda não existir
if "toggle" not in st.session_state:
    st.session_state["toggle"] = False
    
# Definir o rótulo dinamicamente
label = "Todo o período" if not st.session_state["toggle"] else "Selecionar ano específico"

# Criar botão
if st.sidebar.button(label):
    st.session_state["toggle"] = not st.session_state["toggle"]  # Alterna entre True e False
    st.rerun() # Recarrega a página imediatamente para refletir a mudança

# Alternar entre seleção de ano ou todo o príodo
if st.session_state["toggle"]:
    df = df_acoes.copy()
else:
    # Criar seletor de anos
    anos_disponiveis = sorted(df_acoes.index.year.unique(), reverse=True)
    ano_selecionado = st.sidebar.selectbox("Selecionar ano:", anos_disponiveis)
    # Filtrar os dados pelo ano selecionado
    df = df_acoes[df_acoes.index.year == ano_selecionado].copy()

# Criar seletor de granularidade
granularidade = st.sidebar.radio("Selecionar granularidade:", ["Diário", "Semanal", "Mensal"])

# Aplicar a granularidade escolhida
if granularidade == "Semanal":
    df = df.resample("W").last()  # Pega o último valor de cada semana
elif granularidade == "Mensal":
    df = df.resample("ME").last()  # Pega o último valor de cada mês

# Mostrar informações gerais do dataset
st.write(f"### 📊 Fechamento {granularidade} Entre 2010 e 2022" if st.session_state["toggle"] else f"### 📊 Fechamento {granularidade} Em {ano_selecionado}")

st.divider()

# Formatar o índice para exibir apenas a data
df_exibicao = df.copy()
df_exibicao.index = df_exibicao.index.strftime('%Y-%m-%d')

# Exibir a tabela formatada no Streamlit
st.dataframe(df_exibicao, height=400)

st.divider()

# Criar gráfico interativo com todas as ações
st.write("### 📈 Comparação de Todas as Ações")
col1, col2 = st.columns(2)
col1.markdown("**Período:** 2010-2022" if st.session_state["toggle"] else f"**Ano:** {ano_selecionado}")
col2.markdown(f"**Granularidade:** {granularidade}")
st.divider()

# Criar a figura com Plotly para todas as ações
fig_all = px.line(
    df,
    x=df.index,  # Datas no eixo X
    y=df.columns,  # Todas as ações no eixo Y
    title=f"Comparação de Todas as Ações",
    labels={"value": "Preço (R$)", "index": "Data", "variable": "Ação"},
    markers=False  # Sem marcadores para não poluir o gráfico
)

# Melhorar layout
fig_all.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço (R$)",
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
    hovermode="x",  # Tooltip aparece alinhado ao eixo X
)

# Exibir o gráfico no Streamlit
st.plotly_chart(fig_all, use_container_width=True)

# Exibir métricas comparativas
st.divider()
st.write("### 📊 Comparação Estatística Entre Ações")
col1, col2 = st.columns(2)
col1.markdown("**Período:** 2010-2022" if st.session_state["toggle"] else f"**Período:** {ano_selecionado}")
col2.markdown(f"**Granularidade:** {granularidade}")
st.divider()

# Criar colunas para seletores
col1, col2 = st.columns(2)

with col1:
    # Criar seletor da ação 1
    acao1 = st.selectbox("Ação 1:", list(acoes.keys()), index=0)
    # Calcular estatísticas
    media1 = df[acoes[acao1]].mean()
    mediana1 = df[acoes[acao1]].median()
    desvio1 = df[acoes[acao1]].std()
    # Exibir metricas
    st.metric(label="Média", value=f"R$ {media1:.2f}")
    st.metric(label="Mediana", value=f"R$ {mediana1:.2f}")
    st.metric(label="Desvio padrão", value=f"R$ {desvio1:.2f}")
    
    fig1 = go.Figure()

    # Adicionar Gráfico de Violino
    fig1.add_trace(go.Violin(
        y=df[acoes[acao1]],
        name="Violin",
        box_visible=True,  # Mostra boxplot dentro do violino
        meanline_visible=True,  # Adiciona linha da média
        marker_color='green',
        points="suspectedoutliers"  # Mostra todos os pontos
    ))

    # Melhorar layout
    fig1.update_layout(
        yaxis_title="Preço (R$)",
        xaxis_title="",
        showlegend=True,
        title=f"Distribuição de Preços de {acao1}",
        height=500
    )

    # Exibir gráfico no Streamlit
    st.plotly_chart(fig1, use_container_width=True)


with col2:
    # Criar seletor da ação 2
    acao2 = st.selectbox("Ação 2:", list(acoes.keys()), index=1)
    # Calcular estatísticas
    media2 = df[acoes[acao2]].mean()
    mediana2 = df[acoes[acao2]].median()
    desvio2 = df[acoes[acao2]].std()
    # Exibir metricas
    st.metric(label="Média", value=f"R$ {media2:.2f}")
    st.metric(label="Mediana", value=f"R$ {mediana2:.2f}")
    st.metric(label="Desvio padrão", value=f"R$ {desvio2:.2f}")

    fig2 = go.Figure()

    # Adicionar Gráfico de Violino
    fig2.add_trace(go.Violin(
        y=df[acoes[acao2]],
        name="Violin",
        box_visible=True,  # Mostra boxplot dentro do violino
        meanline_visible=True,  # Adiciona linha da média
        marker_color='green',
        points="suspectedoutliers"  # Mostra todos os pontos
    ))

    # Melhorar layout
    fig2.update_layout(
        yaxis_title="Preço (R$)",
        xaxis_title="",
        showlegend=True,
        title=f"Distribuição de Preços de {acao2}",
        height=500
    )

    # Exibir gráfico no Streamlit
    st.plotly_chart(fig2, use_container_width=True)
