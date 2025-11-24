#!/usr/bin/env python3

import json
import logging
import argparse
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from classes.neural_networks.architectures.linear_regression import LinearRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------------
# Utilities / Validações
# -------------------------
def validate_config(cfg: dict):
    """Validação mínima do config para evitar KeyErrors silenciosos."""
    required = [
        ('data', 'filepath'),
        ('data', 'date_col'),
        ('data', 'target_col'),
        ('training', 'window_size'),
        ('training', 'test_split_ratio'),
        ('output', 'model_dir'),
        ('output', 'version_tag')
    ]
    missing = []
    for section, key in required:
        if section not in cfg or key not in cfg[section]:
            missing.append(f"{section}.{key}")
    if missing:
        raise KeyError(f"Config missing required keys: {missing}")

    test_ratio = cfg['training']['test_split_ratio']
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("training.test_split_ratio must be between 0 and 1 (exclusive).")


# -------------------------
# 1. Load (CSV/Parquet) + timezone
# -------------------------
def load_data(filepath: str, date_col: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Carrega CSV ou Parquet, converte a coluna de data, define índice e normaliza timezone.
    Raises se date_col não existir.
    """
    logging.info(f"Loading data from: {filepath}")
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Leitura dependendo do formato
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    else:
        # Para CSV usamos infer_datetime_format mais tarde
        df = pd.read_csv(path)

    # Validação da coluna de data
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in file. Available columns: {list(df.columns)}")

    # Converter e indexar
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    # Timezone handling
    if df.index.tz is None:
        logging.info(f"Localizing naive datetimes to timezone: {tz}")
        try:
            df.index = df.index.tz_localize(tz)
        except Exception as e:
            logging.error(f"Failed to localize index to {tz}: {e}")
            raise
    else:
        if str(df.index.tz) != tz:
            logging.info(f"Converting index timezone from {df.index.tz} to {tz}")
            try:
                df.index = df.index.tz_convert(tz)
            except Exception as e:
                logging.error(f"Failed to convert timezone to {tz}: {e}")
                raise

    return df


# -------------------------
# 2. Windowing (protege cross-day)
# -------------------------
def create_window_data(
    df: pd.DataFrame,
    target: str,
    window_size: int,
    filter_cross_day: bool = True
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Gera janelas (X) e targets (y) usando lags:
      - X shape (n_samples, window_size) se univariado
      - y shape (n_samples,)
    Retorna também os timestamps que representam o instante associado a cada sample (timestamp de y_next).
    Se filter_cross_day=True, remove janelas cujo lag mais antigo pertence a outro dia (evita leakage overnight).
    """

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não existe no dataframe. Colunas: {list(df.columns)}")

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    # Cria lags (do mais antigo para o mais recente)
    lags = {}
    for lag in range(window_size - 1, -1, -1):
        lags[f'lag_{lag}'] = df[target].shift(lag)

    df_features = pd.DataFrame(lags, index=df.index)
    df_features['y_next'] = df[target].shift(-1)

    # Colunas auxiliares para detecção de cross-day
    df_features['date_current'] = df_features.index.date
    df_features['date_oldest_lag'] = pd.Series(df_features.index.date, index=df_features.index).shift(window_size - 1)

    # Drop NaNs gerados pelos shifts
    df_clean = df_features.dropna().copy()
    if df_clean.empty:
        raise ValueError("No valid windows produced after shifting. Try a smaller window_size or check data length.")

    if filter_cross_day: 
        initial_len = len(df_clean)
        mask_same_day = df_clean['date_current'] == df_clean['date_oldest_lag']
        df_clean = df_clean[mask_same_day]
        removed = initial_len - len(df_clean)
        if removed > 0:
            logging.warning(f"Cross-day filter removed {removed} windows (overnight gaps).")
        if df_clean.empty:
            raise ValueError("All windows removed by cross-day filter. Adjust window_size or disable filter_cross_day.")

    feature_cols = [f'lag_{lag}' for lag in range(window_size - 1, -1, -1)]
    X = df_clean[feature_cols].values    # (n_samples, window_size) for univariate
    y = df_clean['y_next'].values
    timestamps = df_clean.index         # timestamps aligned to y_next

    return X, y, timestamps


# -------------------------
# 3. Split temporally com embargo (Timedelta)
# -------------------------
def split_data_time_anchored(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    test_ratio: float,
    embargo_str: str = "0s"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    """
    Divide em treino/test com base em tempo:
      - test_ratio é a fração destinada ao teste (0..1)
      - embargo_str pode ser '0s', '5min', '30s', etc.
    Retorna X_train, X_test, y_train, y_test e o train_end_time (Timestamp) usado.
    """

    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1 (exclusive).")

    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Empty X provided to split_data_time_anchored.")

    train_size = int(n_samples * (1.0 - test_ratio))
    if train_size <= 0:
        raise ValueError("Train size computed as 0 — reduce test_ratio or provide more data.")

    # O último timestamp do treino é o índice train_size - 1 (off-by-one corrigido)
    train_end_time = timestamps[train_size - 1]

    embargo_delta = pd.to_timedelta(embargo_str)
    test_start_time_threshold = train_end_time + embargo_delta

    # Busca o primeiro índice cujo timestamp >= threshold
    # (use searchsorted; timestamps já é Index ordenado)
    test_start_idx = timestamps.searchsorted(test_start_time_threshold)

    # Se embargo purgar todo o teste, lançar erro
    if test_start_idx >= n_samples:
        raise ValueError(
            f"Embargo too large — no test samples left. "
            f"train_end_time={train_end_time}, embargo={embargo_str}, threshold={test_start_time_threshold}"
        )

    # Partições
    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[test_start_idx:]
    y_test = y[test_start_idx:]

    purge_count = test_start_idx - train_size
    logging.info("--- Split Info ---")
    logging.info(f"Total samples: {n_samples}, Train size: {train_size}, Test start idx: {test_start_idx}")
    logging.info(f"Train end time: {train_end_time}, embargo: {embargo_str}, purge_count: {purge_count}")
    if len(X_test) > 0:
        logging.info(f"Test start time: {timestamps[test_start_idx]}")
    else:
        logging.error("No test data after applying embargo.")

    return X_train, X_test, y_train, y_test, train_end_time


# -------------------------
# 4. Normalização (2D / nota para 3D)
# -------------------------
def normalize_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
):
    """
    Normaliza com MinMaxScaler. Suporta X 2D (n_samples, features).
    Se futuramente X for 3D para LSTM, adapte para escalonar por feature via reshape.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # assumimos X 2D para Regressão Linear simples
    X_train_n = scaler_X.fit_transform(X_train)
    X_test_n = scaler_X.transform(X_test)

    y_train_n = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_n = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_n, X_test_n, y_train_n, y_test_n, scaler_X, scaler_y


# -------------------------
# 5. Pipeline principal
# -------------------------
def run_training_pipeline(config: dict):
    validate_config(config)

    data_cfg = config['data']
    train_cfg = config['training']
    out_cfg = config['output']

    target = data_cfg['target_col']

    # 1. Load
    df = load_data(data_cfg['filepath'], data_cfg['date_col'], data_cfg.get('timezone', 'UTC'))

    # Confirma target
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # 2. Windowing
    X, y, timestamps = create_window_data(
        df,
        target,
        train_cfg['window_size'],
        filter_cross_day=train_cfg.get('filter_cross_day', True)
    )

    # 3. Split (Train/Test) com embargo de tempo
    X_train, X_test, y_train, y_test, train_end_time = split_data_time_anchored(
        X, y, timestamps,
        test_ratio=train_cfg['test_split_ratio'],
        embargo_str=train_cfg.get('embargo', "0s")
    )

    # 4. Normalize
    X_train_n, X_test_n, y_train_n, y_test_n, scaler_X, scaler_y = normalize_data(
        X_train, X_test, y_train, y_test
    )

    # 5. Train
    logging.info("Training Linear Regression...")
    model = LinearRegression()
    # LinearRegression.normal_equation deve aceitar X 2D
    theta = model.normal_equation(X_train_n, y_train_n)

    # 6. Evaluate
    y_pred_n = model.predict(X_test_n)
    y_pred = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()

    metrics = {
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'MAE': float(mean_absolute_error(y_test, y_pred)),
        'R2': float(r2_score(y_test, y_pred))
    }
    logging.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    # 7. Save artifacts (atomic save via arquivos temporários)
    save_path = Path(out_cfg['model_dir'])
    save_path.mkdir(parents=True, exist_ok=True)
    ver = out_cfg['version_tag']

    # caminhos finais
    model_path = save_path / f"{target}_model_{ver}.pkl"
    scalerX_path = save_path / f"{target}_scalerX_{ver}.pkl"
    scalerY_path = save_path / f"{target}_scalerY_{ver}.pkl"
    metadata_path = save_path / f"{target}_metadata_{ver}.json"

    # atomic-ish dumps (write to temp file then rename)
    with tempfile.NamedTemporaryFile(delete=False, dir=save_path) as tmp_m:
        joblib.dump(model, tmp_m.name)
        tmp_m.flush()
    Path(tmp_m.name).replace(model_path)

    with tempfile.NamedTemporaryFile(delete=False, dir=save_path) as tmp_sx:
        joblib.dump(scaler_X, tmp_sx.name)
        tmp_sx.flush()
    Path(tmp_sx.name).replace(scalerX_path)

    with tempfile.NamedTemporaryFile(delete=False, dir=save_path) as tmp_sy:
        joblib.dump(scaler_y, tmp_sy.name)
        tmp_sy.flush()
    Path(tmp_sy.name).replace(scalerY_path)

    # Metadata completo para reprodutibilidade
    metadata = {
        "metrics": metrics,
        "config": config,
        "training_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_shape": X_train.shape,
            "input_columns": [f'lag_{i}' for i in range(train_cfg['window_size'] - 1, -1, -1)],
            "train_end_time": str(train_end_time),
            "timezone": str(df.index.tz)
        },
        "artifact_paths": {
            "model": str(model_path),
            "scaler_X": str(scalerX_path),
            "scaler_y": str(scalerY_path)
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=save_path, encoding='utf-8') as tmp_meta:
        json.dump(metadata, tmp_meta, indent=4, ensure_ascii=False)
        tmp_meta.flush()
    Path(tmp_meta.name).replace(metadata_path)

    logging.info(f"Pipeline Finished. Artifacts saved to {save_path.resolve()}")


# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento intraday - pipeline")
    parser.add_argument('--config', type=str, required=True, help="Caminho para arquivo JSON de configuração")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    run_training_pipeline(cfg)
