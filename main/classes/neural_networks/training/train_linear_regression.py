import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from classes.neural_networks.architectures.linear_regression import LinearRegression

def create_window_data(df: pd.DataFrame, target: str, window_size: int = 3):
    df_target = df[[target]].copy()
    for lag in range(1, window_size + 1):
        df_target[f'lag_{lag}'] = df_target[target].shift(lag)
    df_target.dropna(inplace=True)
    X = df_target[[f'lag_{lag}' for lag in range(1, window_size + 1)]].values
    y = df_target[target].values
    return X, y

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def train_and_evaluate(csv_path: str, target: str, normalization: str, window: int):
    df = load_data(csv_path)
    
    if "Date" in df.columns:
        df.sort_values("Date", inplace=True)
    
    norm_suffix = "_" + normalization.strip().lower()
    target_col = target + norm_suffix
    
    if target_col not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada. Colunas disponíveis: {available_cols}")

    X, y = create_window_data(df, target=target_col, window_size=window)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = LinearRegression()
    theta = model.normal_equation(X_train, y_train)
    
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, train_predictions)
    test_metrics = evaluate_model(y_test, test_predictions)
    
    print("\nMétricas de Treino:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nMétricas de Teste:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model, theta, train_metrics, test_metrics, y_test, test_predictions

if __name__ == '__main__':
    import argparse
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
    train_and_evaluate(args.csv_path, args.target, args.normalization, args.window)