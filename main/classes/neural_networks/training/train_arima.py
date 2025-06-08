import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from main.classes.neural_networks.architectures.arima_model import ArimaModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.asfreq('B', method='pad')  # Preenche dias úteis ausentes
    df.index.freq = pd.tseries.offsets.BDay()  # Força a frequência
    return df

def split_data(df: pd.DataFrame, target: str, split_date: str = "2019-01-01"):
    y = df[target].dropna()
    train = y[y.index < split_date]
    test = y[y.index >= split_date]
    train.index.freq = df.index.freq
    test.index.freq = df.index.freq
    return train, test

def evaluate_model(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def train_and_evaluate_arima(csv_path, target, order, model_dir, version, split_date="2019-01-01"):
    logging.info(f"Carregando dados para {target}")
    df = load_data(csv_path)

    if target not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Coluna alvo '{target}' não encontrada. Colunas disponíveis: {available_cols}")

    train, test = split_data(df, target, split_date)
    logging.info(f"Tamanho do treino: {len(train)} | Tamanho do teste: {len(test)}")
    
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Conjunto de treino ou teste está vazio. Verifique as datas de divisão.")

    arima = ArimaModel(order)
    arima.fit(train)
    
    predictions = []
    for t in range(len(test)):
        yhat = arima.predict_one_step()
        predictions.append(yhat)

        # Corrigir o formato de new_value para update()
        new_value = pd.Series([test.iloc[t]], index=[test.index[t]], name=target)
        arima.update(new_value)


    metrics = evaluate_model(test.values, predictions)

    logging.info("\nMétricas no conjunto de teste:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{target}_arima_rolling_v{version}.pkl")
    joblib.dump(arima, model_path)
    
    metrics_path = os.path.join(model_dir, f"{target}_arima_rolling_metrics_v{version}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    np.save(os.path.join(model_dir, f"{target}_arima_rolling_y_true_v{version}.npy"), test.values)
    np.save(os.path.join(model_dir, f"{target}_arima_rolling_y_pred_v{version}.npy"), predictions)

    logging.info(f"Modelo, métricas e previsões salvos em {model_dir}")
    return arima, metrics, predictions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Treinamento ARIMA Rolling para série temporal de ações da B3")
    parser.add_argument('--csv_path', type=str, default="main/datasets/b3_dados/processed/acoes_concat.csv",
                        help="Caminho para o CSV")
    parser.add_argument('--target', type=str, default="ITUB4", help="Coluna da ação alvo (ex.: ITUB4)")
    parser.add_argument('--order', type=str, default="2,1,1", help="Parâmetros ARIMA (p,d,q) separados por vírgula")
    parser.add_argument('--model_dir', type=str, default="saved_models", help="Diretório de saída")
    parser.add_argument('--version', type=str, default="1.0", help="Versão do modelo")

    args = parser.parse_args()
    order = tuple(map(int, args.order.split(',')))

    train_and_evaluate_arima(args.csv_path, args.target, order, args.model_dir, args.version)
