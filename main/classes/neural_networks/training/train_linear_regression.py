import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from classes.neural_networks.architectures.linear_regression import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carrega o CSV, parseia a coluna Date como datetime e define como índice.
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df


def create_window_data(
    df: pd.DataFrame,
    target: str,
    window_size: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera X e y usando lag features de forma vetorizada.
    """
    # Concatena coluna target + lags
    lags = [df[target].shift(lag).rename(f'lag_{lag}') for lag in range(1, window_size + 1)]
    df_lagged = pd.concat([df[target]] + lags, axis=1).dropna()
    X = df_lagged[[f'lag_{lag}' for lag in range(1, window_size + 1)]].values
    y = df_lagged[target].values
    return X, y


def split_data_by_date(
    df: pd.DataFrame,
    target: str,
    train_end: str = '2017-12-31',
    val_end: str = '2018-12-31'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide em treino, validação e teste com base em datas.
    """
    train_df = df.loc[:train_end]
    val_df   = df.loc[train_end:val_end]
    test_df  = df.loc[val_end:]

    X_train, y_train = create_window_data(train_df, target)
    X_val,   y_val   = create_window_data(val_df,   target)
    X_test,  y_test  = create_window_data(test_df,  target)

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Ajusta MinMaxScaler em X_train e y_train, e transforma val/test.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_norm = scaler_X.fit_transform(X_train)
    X_val_norm   = scaler_X.transform(X_val)
    X_test_norm  = scaler_X.transform(X_test)

    y_train_norm = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_norm   = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
    y_test_norm  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return (
        X_train_norm, X_val_norm, X_test_norm,
        y_train_norm, y_val_norm, y_test_norm,
        scaler_X, scaler_y
    )


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de regressão.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def train_and_evaluate(
    csv_path: str,
    target: str,
    window: int,
    model_dir: str,
    version: str
) -> dict:
    """
    Pipeline de treinamento de regressão linear:
      - carrega dados
      - divide treino/val/test
      - normaliza
      - treina pela equação normal
      - avalia e salva artefatos
    """
    logging.info(f"Loading data for {target} from {csv_path}")
    df = load_data(csv_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_date(df, target)
    logging.info("Normalizing data")
    (
        X_train_n, X_val_n, X_test_n,
        y_train_n, y_val_n, y_test_n,
        scaler_X, scaler_y
    ) = normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

    model = LinearRegression()
    logging.info("Training model (normal equation)")
    theta = model.normal_equation(X_train_n, y_train_n)

    # previsões (normalizadas)
    y_train_pred_n = model.predict(X_train_n)
    y_val_pred_n   = model.predict(X_val_n)
    y_test_pred_n  = model.predict(X_test_n)

    logging.info("Denormalizing predictions")
    y_train_pred = scaler_y.inverse_transform(y_train_pred_n.reshape(-1, 1)).ravel()
    y_val_pred   = scaler_y.inverse_transform(y_val_pred_n.reshape(-1, 1)).ravel()
    y_test_pred  = scaler_y.inverse_transform(y_test_pred_n.reshape(-1, 1)).ravel()

    logging.info("Evaluating model")
    train_metrics = evaluate_model(y_train, y_train_pred)
    val_metrics   = evaluate_model(y_val,   y_val_pred)
    test_metrics  = evaluate_model(y_test,  y_test_pred)

    for phase, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        logging.info(f"{phase} metrics:")
        for name, val in metrics.items():
            logging.info(f"  {name}: {val:.4f}")

    # prepara diretórios
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    # salva modelo e scalers (com compressão)
    model_path    = model_dir_path / f"{target}_model_v{version}.pkl"
    scaler_X_path = model_dir_path / f"{target}_scaler_X_v{version}.pkl"
    scaler_y_path = model_dir_path / f"{target}_scaler_y_v{version}.pkl"

    joblib.dump(model, model_path,    compress=('gzip', 3))
    joblib.dump(scaler_X, scaler_X_path, compress=('gzip', 3))
    joblib.dump(scaler_y, scaler_y_path, compress=('gzip', 3))
    logging.info(f"Saved model and scalers to {model_dir_path}")

    # salva métricas em JSON legível
    metrics_path = model_dir_path / f"{target}_metrics_v{version}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train': train_metrics,
            'val':   val_metrics,
            'test':  test_metrics
        }, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved metrics to {metrics_path}")

    # salva arrays para análises posteriores
    np.save(model_dir_path / f"{target}_y_test_v{version}.npy",   y_test)
    np.save(model_dir_path / f"{target}_y_pred_v{version}.npy",   y_test_pred)
    np.save(model_dir_path / f"{target}_X_test_norm_v{version}.npy", X_test_n)
    logging.info("Saved test arrays for hypothesis analysis")

    return {
        'model': model,
        'theta': theta,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        'y_test': y_test,
        'y_pred': y_test_pred
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Treinamento de regressão linear com janelas e normalização"
    )
    parser.add_argument('--csv_path', type=str, default="main/datasets/b3_dados/processed/acoes_concat.csv")
    parser.add_argument('--target',   type=str, required=True)
    parser.add_argument('--window',   type=int, default=3)
    parser.add_argument('--model_dir',type=str, default="main/saved_models/linear_regression")
    parser.add_argument('--version',  type=str, default="1.0")

    args = parser.parse_args()
    logging.info(f"Starting training for {args.target}")
    train_and_evaluate(
        args.csv_path, args.target,
        args.window, args.model_dir,
        args.version
    )
