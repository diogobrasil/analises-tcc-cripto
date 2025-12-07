import sys
from pathlib import Path

# Adiciona a pasta 'main' ao caminho de busca do Python
# Ajuste o número de '.parent' se necessário até apontar para a pasta que contém 'classes'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent # Sobe até a raiz do projeto
sys.path.append(str(project_root)) # Adiciona a raiz geral
sys.path.append(str(project_root / 'main')) # Adiciona a pasta main especificamente

# Importar a classe ArimaModel
from classes.neural_networks.architectures.arima_model import ArimaModel

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_transform_data(csv_path: str, target_col: str):
    # ... (código de carregamento igual ao anterior) ...
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    # 1. Calcular Retorno Log (Target)
    log_ret = np.log(df[target_col] / df[target_col].shift(1))
    
    # 2. Preparar Exógena (Volume)
    # Log Volume é melhor que Volume bruto para normalizar a escala
    # Adicionamos +1 para evitar log(0)
    vol_log = np.log(df['tickvol'] + 1)
    
    # Alinhar os dados (o shift do retorno cria um NaN na primeira linha)
    data = pd.DataFrame({'ret': log_ret, 'vol': vol_log}).dropna()
    
    return data['ret'], data[['vol']] # Retorna Target e Exog

def split_series_by_date(ts: pd.Series, train_end: str, val_end: str):
    """
    Divide baseada em datas strings. Suporta formato ISO (YYYY-MM-DD HH:MM).
    """
    # Converter strings para timestamp para garantir comparação correta
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    train = ts.loc["2025":train_end]

    # Adicione este print/log para conferir no terminal:
    logger.info(f"Treino iniciado em: {train.index.min()} | Finalizado em: {train.index.max()}")

    val = ts.loc[train_end:val_end].iloc[1:] # Evitar sobreposição exata
    test = ts.loc[val_end:].iloc[1:]
    
    return train, val, test

def evaluate_model(y_true, y_pred):
    """
    Métricas adaptadas. Inclui 'Directional Accuracy' (Sinal).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Alinha tamanhos
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Directional Accuracy: O modelo acertou se subiu ou caiu?
    # (Sinal do predito == Sinal do real)
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    accuracy = accuracy_score(true_sign, pred_sign)

    return {
        'MSE': mse, 
        'RMSE': rmse, 
        'MAE': mae, 
        'DIR_ACC': accuracy # 0.50 = moeda, > 0.55 = bom para intraday
    }

def rolling_forecast_optimized(model_wrapper, series: pd.Series, exog: pd.DataFrame, step_size: int = 1):
    preds = []
    # Loop ajustado para passar o Exog
    # Nota: Em tempo real, você não tem o volume FUTURO. 
    # Em ARIMAX puro, você precisa prever o volume ou usar o volume passado (lagged).
    # Para este teste, usaremos o volume do candle atual para prever o fechamento (assumindo que temos o fluxo intra-candle).
    
    # IMPORTANTE: Usar exog[i] para prever series[i] implica que sabemos o volume antes do fechamento.
    
    for i in range(0, len(series), step_size):
        # Pegamos o volume correspondente ao passo que queremos prever
        # O modelo precisa de input 2D para exog [[valor]]
        current_exog = exog.iloc[i:i+step_size]
        
        try:
            mu, sigma = model_wrapper.predict_next(steps=step_size, exog_future=current_exog)
            preds.append(mu)
        except:
            preds.append(0) # Fallback

        # Ignoramos update online por enquanto pela complexidade do exog
        
    return np.array(preds)

def train_pipeline(args):
    # 1. Carregar (agora retorna tupla)
    ts, exog = load_and_transform_data(args.csv_path, args.target)
    
    # 2. Split (precisamos dividir o exog igual à série)
    # Lógica simplificada de split baseada nas suas datas
    train_mask = (ts.index >= "2025-01-01") & (ts.index <= args.train_end)
    val_mask = (ts.index > args.train_end) & (ts.index <= args.val_end)
    test_mask = (ts.index > args.val_end)
    
    y_train, ex_train = ts.loc[train_mask], exog.loc[train_mask]
    y_val,   ex_val   = ts.loc[val_mask],   exog.loc[val_mask]
    y_test,  ex_test  = ts.loc[test_mask],  exog.loc[test_mask]
    logger.info(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

    if len(y_train) < 100:
        raise ValueError("Dataset de treino muito pequeno para ARIMA/GARCH.")

    # 3. Setup
    model = ArimaModel(arima_order=args.order, use_garch=args.use_garch)

    # 4. Treino com EXOG
    logger.info(f"Treinando ARIMAX{args.order} com Volume...")
    model.fit(y_train, exog_train=ex_train)

    # 5. Previsão com EXOG
    val_pred = rolling_forecast_optimized(model, y_val, exog=ex_val)
    test_pred = rolling_forecast_optimized(model, y_test, exog=ex_test)
    
    # In-sample (apenas para referência)
    train_pred = model.arima_result.fittedvalues

    # 6. Métricas
    metrics = {
        'train': evaluate_model(y_train.values, train_pred),
        'val':   evaluate_model(y_val.values, val_pred),
        'test':  evaluate_model(y_test.values, test_pred)
    }

    # Exibir
    print("\n=== RESULTADOS FINAIS ===")
    print(json.dumps(metrics, indent=2))

    # 7. Salvar (Opcional - usando joblib para o objeto e numpy para arrays)
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{args.target}_hybrid_model.pkl")
    # Salvar predições pode ser feito com np.save conforme seu script original

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--target', required=True, help="Nome da coluna de PREÇO (Close)")
    parser.add_argument('--order', type=str, default="2,0,2", help="p,d,q")
    parser.add_argument('--train_end', type=str, required=True, help="Ex: 2023-01-01 12:00")
    parser.add_argument('--val_end', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default="./artifacts")
    parser.add_argument('--use_garch', action='store_true', help="Ativar GARCH nos resíduos")
    
    args = parser.parse_args()
    args.order = tuple(map(int, args.order.split(',')))
    
    train_pipeline(args)
