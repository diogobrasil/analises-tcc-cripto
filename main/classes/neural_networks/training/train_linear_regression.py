import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from main.classes.neural_networks.architectures.linear_regression import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def split_data_by_date(df: pd.DataFrame, target: str):
    # Define os períodos de treino, validação e teste
    train_end_date = '2017-12-31'
    val_end_date = '2018-12-31'
    test_end_date = '2022-09-30'
    
    # Filtra os dados para cada período
    train_data = df[df['Date'] <= train_end_date]
    val_data = df[(df['Date'] > train_end_date) & (df['Date'] <= val_end_date)]
    test_data = df[df['Date'] > val_end_date]
    
    # Cria os conjuntos de treino, validação e teste
    X_train, y_train = create_window_data(train_data, target)
    X_val, y_val = create_window_data(val_data, target)
    X_test, y_test = create_window_data(test_data, target)
   
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_norm = scaler_X.fit_transform(X_train)
    X_val_norm = scaler_X.transform(X_val)
    X_test_norm = scaler_X.transform(X_test)

    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_norm = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_norm = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_X, scaler_y

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

def train_and_evaluate(csv_path: str, target: str, window: int, model_dir: str, version: str):
    logging.info(f"Carregando dados do CSV para a ação {target}")
    df = load_data(csv_path)
    
    if "Date" in df.columns:
        df.sort_values("Date", inplace=True)
    
    if target not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Coluna alvo '{target}' não encontrada. Colunas disponíveis: {available_cols}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_date(df, target)
    
    logging.info("Normalizando dados")
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, scaler_X, scaler_y = normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    model = LinearRegression()
    logging.info("Treinando modelo usando a equação normal")
    theta = model.normal_equation(X_train_norm, y_train_norm)
    
    train_predictions = model.predict(X_train_norm)
    val_predictions = model.predict(X_val_norm)
    test_predictions = model.predict(X_test_norm)
    
    logging.info("Desnormalizando previsões")
    train_predictions = scaler_y.inverse_transform(train_predictions.reshape(-1, 1)).ravel()
    val_predictions = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).ravel()
    test_predictions = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).ravel()
    
    logging.info("Avaliando modelo")
    train_metrics = evaluate_model(y_train, train_predictions)
    val_metrics = evaluate_model(y_val, val_predictions)
    test_metrics = evaluate_model(y_test, test_predictions)
    
    logging.info("\nMétricas de Treino:")
    for metric, value in train_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
        
    logging.info("\nMétricas de Validação:")
    for metric, value in val_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
        
    logging.info("\nMétricas de Teste:")
    for metric, value in test_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Salvar o modelo
    model_path = os.path.join(model_dir, f"{target}_model_v{version}.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Modelo para {target} salvo em {model_path}")
    
    # Salvar os scalers
    scaler_X_path = os.path.join(model_dir, f"{target}_scaler_X_v{version}.pkl")
    scaler_y_path = os.path.join(model_dir, f"{target}_scaler_y_v{version}.pkl")
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    logging.info(f"Scalers para {target} salvos em {scaler_X_path} e {scaler_y_path}")
    
    # Salvar as métricas
    metrics_path = os.path.join(model_dir, f"{target}_metrics_v{version}.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }, f)
    logging.info(f"Métricas para {target} salvas em {metrics_path}")

    # Salvar y_test, test_predictions e X_test_norm para análise de hipóteses
    np.save(os.path.join(model_dir, f"{target}_y_test_v{version}.npy"), y_test)
    np.save(os.path.join(model_dir, f"{target}_y_pred_v{version}.npy"), test_predictions)
    np.save(os.path.join(model_dir, f"{target}_X_test_norm_v{version}.npy"), X_test_norm)
    logging.info(f"Arquivos para análise de hipóteses salvos para {target}")

    # Salvar os coeficientes em JSON
    coefs = {
        "intercept": float(theta[0]),
        "lag_1": float(theta[1]),
        "lag_2": float(theta[2]),
        "lag_3": float(theta[3])
    }
    with open(os.path.join(model_dir, f"{target}_coefs_v{version}.json"), "w") as f:
        json.dump(coefs, f)
    
    return model, theta, train_metrics, val_metrics, test_metrics, y_test, test_predictions

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Treinamento de regressão linear com janela usando dados normalizados")
    parser.add_argument('--csv_path', type=str, default="main/datasets/b3_dados/processed/acoes_concat.csv",
                        help="Caminho para o arquivo CSV com os dados originais das ações.")
    parser.add_argument('--target', type=str, default="VALE3",
                        help="Nome base da ação alvo para previsão (ex.: ITUB4).")
    parser.add_argument('--window', type=int, default=3,
                        help="Tamanho da janela (número de lags) a ser considerado.")
    parser.add_argument('--model_dir', type=str, default="saved_models",
                        help="Diretório para salvar os modelos treinados.")
    parser.add_argument('--version', type=str, default="1.0",
                        help="Versão do modelo.")
    
    args = parser.parse_args()
    
    # Criar o diretório para salvar os modelos, se não existir
    os.makedirs(args.model_dir, exist_ok=True)
    
    logging.info(f"\nTreinando modelo para a ação: {args.target}")
    train_and_evaluate(args.csv_path, args.target, args.window, args.model_dir, args.version)
