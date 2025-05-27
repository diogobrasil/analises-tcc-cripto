import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../main/classes/neural_networks/training/")))
from train_linear_regression import train_and_evaluate

# Função para treinar os 11 ativos usando train_and_evaluate
def train_all_assets(csv_path, model_dir, window=3, version="1.1"):
    ativos = ['ITUB4', 'BBAS3', 'CYRE3', 'TEND3', 'DIRR3', 'ELET3', 'EQTL3', 'CMIG4', 'PETR3', 'VALE3', 'BRAP3']
    for ativo in ativos:
        print(f"\n🚀 Treinando modelo para {ativo} (versão {version})...")
        model, theta, train_metrics, val_metrics, test_metrics, y_test, test_predictions = train_and_evaluate(
            csv_path=csv_path,
            target=ativo,
            window=window,
            model_dir=model_dir,
            version=version
        )
    print("\n✅ Todos os modelos treinados com sucesso!")

# Função para análise das hipóteses da regressão linear
def analyze_assumptions(y_test, y_pred, X_test_norm, target):
    residuals = y_test - y_pred

    # 1️⃣ Linearidade e Homoscedasticidade
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Preditos')
    plt.ylabel('Resíduos')
    plt.title(f'Resíduos vs Preditos - {target}')
    plt.show()

    # Teste de Breusch-Pagan
    X_test_sm = sm.add_constant(y_pred)
    bp_test = het_breuschpagan(residuals, X_test_sm)
    print(f"\n📊 Breusch-Pagan para {target}:\nEstatística: {bp_test[0]:.4f} | p-valor: {bp_test[1]:.4f}")

    # 2️⃣ Normalidade dos Resíduos
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f'Histograma dos Resíduos - {target}')
    plt.show()

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot dos Resíduos - {target}')
    plt.show()

    shapiro = stats.shapiro(residuals)
    print(f"Shapiro-Wilk p-valor para {target}: {shapiro.pvalue:.4f}")

    # 3️⃣ Multicolinearidade (se aplicável)
    if X_test_norm.shape[1] > 1:
        X_test_df = pd.DataFrame(X_test_norm, columns=[f'lag_{i}' for i in range(1, X_test_norm.shape[1]+1)])
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X_test_df.columns
        vif_data['VIF'] = [variance_inflation_factor(X_test_df.values, i) for i in range(X_test_df.shape[1])]
        print(f"\n📊 VIF para {target}:\n", vif_data)

# Função para rodar a análise de hipóteses para todos os ativos
def analyze_all_assets(model_dir, version="1.1"):
    ativos = ['ITUB4', 'BBAS3', 'CYRE3', 'TEND3', 'DIRR3', 'ELET3', 'EQTL3', 'CMIG4', 'PETR3', 'VALE3', 'BRAP3']
    for ativo in ativos:
        print(f"\n🔍 Analisando hipóteses para {ativo} (versão {version})...")
        y_test = np.load(os.path.join(model_dir, f"{ativo}_y_test_v{version}.npy"))
        y_pred = np.load(os.path.join(model_dir, f"{ativo}_y_pred_v{version}.npy"))
        X_test_norm = np.load(os.path.join(model_dir, f"{ativo}_X_test_norm_v{version}.npy"))
        analyze_assumptions(y_test, y_pred, X_test_norm, ativo)

if __name__ == "__main__":
    # Definir caminhos
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../main/datasets/b3_dados/processed/acoes_concat.csv"))
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models"))

    # Passo 1: Treinar todos os modelos (versão 1.1)
    train_all_assets(csv_path, model_dir, window=3, version="1.1")

    # Passo 2: Analisar hipóteses para todos os ativos
    analyze_all_assets(model_dir, version="1.1")
