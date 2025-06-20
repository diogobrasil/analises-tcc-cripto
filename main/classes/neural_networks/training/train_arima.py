import os
import json
import numpy as np
import pandas as pd
import joblib
import pandas_market_calendars as mcal
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from main.classes.neural_networks.architectures.arima_model import ArimaModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega o calendário oficial da B3 via pandas-market-calendars
b3_cal = mcal.get_calendar('B3')

def load_series(csv_path: str, target: str) -> pd.Series:
    # 1) Carrega DataFrame e indexa por Date
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    # 2) Obtém o schedule de pregões da B3 entre as datas do CSV
    schedule = b3_cal.schedule(start_date=df.index.min().strftime('%Y-%m-%d'),
                                end_date=df.index.max().strftime('%Y-%m-%d'))
    full_idx = schedule.index

    # 3) Reindexa só a série do target, criando NaN nos dias sem pregão
    ts = df[target].reindex(full_idx)

    # 4) Elimina esses NaN (caso haja datas de pregão faltando no CSV)
    ts = ts.dropna()

    # 5) Retorna série com datas de pregão reais (freq inferida)
    return ts


def split_data(ts: pd.Series, split_date: str = "2019-01-01"):
    train = ts[ts.index < split_date]
    test  = ts[ts.index >= split_date]
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


def train_and_evaluate_arima(csv_path, target, order, model_dir, version,
                             split_date="2019-01-01", rolling=True, step_size=1):
    logging.info(f"Carregando dados para {target}")
    ts = load_series(csv_path, target)

    train, test = split_data(ts, split_date)
    logging.info(f"Tamanho do treino: {len(train)} | Tamanho do teste: {len(test)}")

    if train.empty or test.empty:
        raise ValueError("Conjunto de treino ou teste está vazio. Verifique a data de divisão.")

    arima = ArimaModel(order)
    arima.fit(train)

    if rolling:
        predictions = []
        for t in range(0, len(test), step_size):
            yhat = arima.model_fit.forecast(steps=step_size)
            predictions.extend(yhat[:min(step_size, len(test) - len(predictions))])

            if t + step_size < len(test):
                new_values = test.iloc[t : t + step_size].values
                arima.model_fit = arima.model_fit.extend(new_values)

        predictions = predictions[:len(test)]
    else:
        predictions = arima.model_fit.forecast(steps=len(test)).tolist()

    metrics = evaluate_model(test.values, predictions)
    logging.info("\nMétricas no conjunto de teste:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(arima, os.path.join(model_dir, f"{target}_arima_rolling_v{version}.pkl"))
    with open(os.path.join(model_dir, f"{target}_arima_rolling_metrics_v{version}.json"), 'w') as f:
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
    parser.add_argument('--model_dir', type=str, default="main/saved_models/arima", help="Diretório de saída")
    parser.add_argument('--version', type=str, default="1.0", help="Versão do modelo")

    args = parser.parse_args()
    order = tuple(map(int, args.order.split(',')))

    train_and_evaluate_arima(args.csv_path, args.target, order,
                             args.model_dir, args.version)
