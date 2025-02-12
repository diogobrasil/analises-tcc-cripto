import pandas as pd
import numpy as np
import argparse
import os
import sys

# Adiciona o caminho da pasta `main` ao sys.path
sys.path.append(os.path.abspath("../"))

from neural_networks.architectures.linear_regression import LinearRegression

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
    if args.target not in df.columns:
        available_cols = list(df.columns)
        print(f"Coluna alvo '{args.target}' não encontrada. Colunas disponíveis: {available_cols}")
        return

    # Cria os dados com janela
    X, y = create_window_data(df, args.target, args.window)
    
    # Instancia e treina o modelo usando a equação normal
    model = LinearRegression()
    theta = model.normal_equation(X, y)
    predictions = model.predict(X)
    
    print("Parâmetros:", theta)
    print("Previsões:", predictions)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento de regressão linear com janela para ações")
    parser.add_argument('--csv_path', type=str, default="main/datasets/b3_dados/processed/acoes_concat.csv",
                        help="Caminho para o arquivo CSV com os dados")
    parser.add_argument('--target', type=str, default="ITUB4",
                        help="Coluna alvo (nome da ação) para previsão")
    parser.add_argument('--window', type=int, default=3,
                        help="Tamanho da janela (número de lags) a ser considerado")
    
    args = parser.parse_args()
    main(args)