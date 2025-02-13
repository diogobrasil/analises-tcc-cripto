import pandas as pd
import numpy as np
import argparse


from classes.neural_networks.architectures.linear_regression import LinearRegression

def create_window_data(df: pd.DataFrame, target: str, window_size: int = 3):
    """
    Cria dados com janela para uma coluna alvo.
    
    Args:
        df: DataFrame com as colunas dos dados.
        target: Nome da coluna alvo.
        window_size: Tamanho da janela (número de lags).
        
    Returns:
        X: Features com os valores lags.
        y: Target (valores atuais) para previsão.
    """
    df_target = df[[target]].copy()
    for lag in range(1, window_size + 1):
        df_target[f'lag_{lag}'] = df_target[target].shift(lag)
    df_target.dropna(inplace=True)
    X = df_target[[f'lag_{lag}' for lag in range(1, window_size + 1)]].values
    y = df_target[target].values
    return X, y

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carrega os dados do arquivo CSV.
    
    Args:
        csv_path: Caminho do arquivo CSV.
        
    Returns:
        DataFrame carregado.
    """
    return pd.read_csv(csv_path)

def main(args):
    # Carrega o dataset
    df = load_data(args.csv_path)
    
    # Se houver coluna 'Date', ordena os dados
    if "Date" in df.columns:
        df.sort_values("Date", inplace=True)
    
    # Monta o sufixo de normalização (ex.: "_min_max")
    norm_suffix = "_" + args.normalization.strip().lower()
    
    # Define a coluna alvo com base no argumento target + o sufixo da normalização
    target_col = args.target + norm_suffix
    
    if target_col not in df.columns:
        available_cols = list(df.columns)
        print(f"Coluna alvo '{target_col}' não encontrada. Colunas disponíveis: {available_cols}")
        return

    # Cria os dados com janela
    X, y = create_window_data(df, target=target_col, window_size=args.window)
    
    # Instancia e treina o modelo usando a equação normal
    model = LinearRegression()
    theta = model.normal_equation(X, y)
    predictions = model.predict(X)
    
    print("Parâmetros (theta):", theta)
    print("Previsões:", predictions)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento de regressão linear com janela usando dados normalizados")
    parser.add_argument('--csv_path', type=str, default="main/datasets/b3_dados/processed/selected_stocks_normalized.csv",
                        help="Caminho para o arquivo CSV com TODAS as ações normalizadas.")
    parser.add_argument('--target', type=str, default="ITUB4",
                        help="Nome base da ação alvo para previsão (ex.: ITUB4).")
    parser.add_argument('--normalization', type=str, default="min_max",
                        help="Tipo de normalização a usar: 'min_max', 'z_score' ou 'robust'.")
    parser.add_argument('--window', type=int, default=3,
                        help="Tamanho da janela (número de lags) a ser considerado.")
    
    args = parser.parse_args()
    main(args)