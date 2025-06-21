import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pandas_market_calendars as mcal
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from main.classes.neural_networks.architectures.arima_model import ArimaModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# calendário oficial da B3
b3_cal = mcal.get_calendar('B3')


def load_series(csv_path: str, target: str) -> pd.Series:
    """
    Carrega a série de preços do `target` indexada somente em dias úteis
    pelo calendário da B3. Retorna uma pd.Series com índice datetime.
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    schedule = b3_cal.schedule(start_date=df.index.min(), end_date=df.index.max())
    full_idx = schedule.index
    ts = df[target].reindex(full_idx).dropna()
    return ts


def split_series(
    ts: pd.Series,
    train_end: str = "2017-12-31",
    val_end:   str = "2018-12-31"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Divide `ts` em três pedaços:
      - train: até `train_end` (inclusive)
      - val: de `train_end` até `val_end`
      - test: após `val_end`
    """
    train_series = ts[:train_end]
    val_series   = ts[train_end:val_end]
    test_series  = ts[val_end:]
    return train_series, val_series, test_series


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de regressão: MSE, RMSE, MAE e R².
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[: len(y_true)]
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def rolling_forecast(model_fit, series: pd.Series, step_size: int) -> np.ndarray:
    """
    Executa previsão em janela deslizante, estendendo `model_fit`
    com valores reais conforme avançamos.
    """
    preds = []
    start = 0
    while start < len(series):
        h = min(step_size, len(series) - start)
        yhat = model_fit.forecast(steps=h)
        preds.extend(yhat)
        if start + h < len(series):
            model_fit = model_fit.extend(series.iloc[start : start + h].values)
        start += h
    return np.array(preds)


def train_and_evaluate_arima(
    csv_path: str,
    target: str,
    order: tuple[int, int, int],
    model_dir: str,
    version: str,
    train_end: str = "2017-12-31",
    val_end:   str = "2018-12-31",
    rolling:   bool = True,
    step_size: int = 1
) -> dict:
    """
    Treina um ARIMA em `train`, avalia em train/val/test e salva artefatos.
    Retorna dict com chaves: model, metrics, predictions.
    """
    logger.info(f"Loading series for target {target}")
    ts = load_series(csv_path, target)

    # valida cortes de data
    min_date = ts.index.min()
    max_date = ts.index.max()
    if pd.to_datetime(train_end) < min_date or pd.to_datetime(val_end) > max_date:
        raise ValueError(f"Cutoff dates must lie within series range [{min_date}, {max_date}]")

    train_series, val_series, test_series = split_series(ts, train_end, val_end)
    logger.info(f"Split sizes → train: {len(train_series)}, val: {len(val_series)}, test: {len(test_series)}")
    if train_series.empty or val_series.empty or test_series.empty:
        raise ValueError("One of train/val/test series is empty after split.")

    # ajusta ARIMA no conjunto de treino
    arima = ArimaModel(order)
    arima.fit(train_series)

    # previsões in-sample (fittedvalues)
    train_pred = arima.model_fit.fittedvalues

    # previsões out-of-sample
    if rolling:
        model_fit_copy = arima.model_fit
        val_pred  = rolling_forecast(model_fit_copy, val_series,  step_size)
        test_pred = rolling_forecast(model_fit_copy, test_series, step_size)
    else:
        val_pred  = arima.model_fit.forecast(steps=len(val_series))
        test_pred = arima.model_fit.forecast(steps=len(test_series))

    # cálculo de métricas
    metrics = {
        'train': evaluate_model(train_series.values,    train_pred),
        'val':   evaluate_model(val_series.values,      val_pred),
        'test':  evaluate_model(test_series.values,     test_pred)
    }

    # log sem sobrescrever variáveis de séries
    for phase, phase_metrics in metrics.items():
        logger.info(f"\n{phase.upper()} metrics:")
        for metric_name, metric_value in phase_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    # salvar artefatos
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{target}_arima_v{version}"

    joblib.dump(arima,    out_dir / f"{prefix}.pkl",           compress=('gzip', 3))
    with open(out_dir / f"{prefix}_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.save(out_dir / f"{prefix}_y_train.npy",      train_series.values)
    np.save(out_dir / f"{prefix}_y_train_pred.npy", train_pred)
    np.save(out_dir / f"{prefix}_y_val.npy",        val_series.values)
    np.save(out_dir / f"{prefix}_y_val_pred.npy",   val_pred)
    np.save(out_dir / f"{prefix}_y_test.npy",       test_series.values)
    np.save(out_dir / f"{prefix}_y_test_pred.npy",  test_pred)

    logger.info(f"Artifacts saved under {out_dir}")
    return {
        'model': arima,
        'metrics': metrics,
        'predictions': {
            'train': train_pred,
            'val':   val_pred,
            'test':  test_pred
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train & evaluate ARIMA with rolling forecast")
    parser.add_argument('--csv_path',  type=str, required=True)
    parser.add_argument('--target',    type=str, required=True)
    parser.add_argument('--order',     type=str, default="2,1,1",
                        help="ARIMA order p,d,q as comma-separated")
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--version',   type=str, default="1.0")
    parser.add_argument('--train_end', type=str, default="2017-12-31")
    parser.add_argument('--val_end',   type=str, default="2018-12-31")
    parser.add_argument('--rolling',   action='store_true')
    parser.add_argument('--step_size', type=int, default=1)
    args = parser.parse_args()
    order = tuple(map(int, args.order.split(',')))

    train_and_evaluate_arima(
        csv_path=args.csv_path,
        target=args.target,
        order=order,
        model_dir=args.model_dir,
        version=args.version,
        train_end=args.train_end,
        val_end=args.val_end,
        rolling=args.rolling,
        step_size=args.step_size
    )
