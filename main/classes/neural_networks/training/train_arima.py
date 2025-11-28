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

def load_and_transform_data(csv_path: str, target_col: str) -> pd.Series:
    """
    Carrega dados intradiários e converte para Retornos Logarítmicos.
    Crítico para ARIMA funcionar em 15min.
    """
    logger.info(f"Carregando dataset: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    # Verifica se a coluna existe
    if target_col not in df.columns:
        raise ValueError(f"Coluna '{target_col}' não encontrada no CSV.")

    # --- TRANSFORMAÇÃO ---
    # Log Return: ln(Pt / Pt-1)
    # Adicionamos fillna(0) ou dropna() para o primeiro elemento
    ts_log_ret = np.log(df[target_col] / df[target_col].shift(1))
    ts_log_ret.dropna(inplace=True)

    # Remover retornos infinitos (caso de preço zero, raro mas possível em bugs de dados)
    ts_log_ret = ts_log_ret.replace([np.inf, -np.inf], np.nan).dropna()
    
    logger.info(f"Dados transformados para Log-Returns. Total de linhas: {len(ts_log_ret)}")
    return ts_log_ret

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

def rolling_forecast_optimized(model_wrapper, series: pd.Series, step_size: int = 1):
    """
    Executa previsão em janela deslizante.
    Para intraday, usamos 'update' (filtro) em vez de 'fit' completo para performance.
    """
    preds = []
    # Usamos os valores da série para atualizar o histórico
    history = series.values
    
    # Loop for chunks if step_size > 1, or simple loop
    # Atenção: Iterar pandas series é lento, usar numpy values
    series_values = series.values
    
    # Estado inicial já está treinado no wrapper
    
    for i in range(0, len(series_values), step_size):
        # 1. Prever
        mu, sigma = model_wrapper.predict_next(steps=step_size)
        
        # Se step_size > 1, o modelo retorna array, aqui simplificamos para step=1 ou pegamos o primeiro
        # Para simplificar este exemplo, assumimos step_size=1
        preds.append(mu)
        
        # 2. Atualizar modelo com o dado que "acabou de acontecer" (Real)
        # O dado real atual é series_values[i]
        obs = series_values[i:i+step_size]
        model_wrapper.update(obs)

    return np.array(preds)

def train_pipeline(args):
    # 1. Carregar
    ts = load_and_transform_data(args.csv_path, args.target)
    
    # 2. Split
    train, val, test = split_series_by_date(ts, args.train_end, args.val_end)
    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    if len(train) < 100:
        raise ValueError("Dataset de treino muito pequeno para ARIMA/GARCH.")

    # 3. Setup Modelo
    # Se d=1 foi passado, alertamos. Para LogReturns, d=0 é o correto.
    p, d, q = args.order
    if d > 0:
        logger.warning("ALERTA: 'd' > 0 detectado. Para retornos, use d=0. Forçando d=0.")
        d = 0
    
    arima_order = (p, d, q)
    model = ArimaModel(arima_order=arima_order, use_garch=args.use_garch)

    # 4. Treino Inicial
    logger.info(f"Treinando ARIMA{arima_order}...")
    model.fit(train)
    logger.info("Treino inicial concluído.")

    # 5. Previsões (Rolling)
    logger.info("Iniciando Validação Rolling...")
    val_pred = rolling_forecast_optimized(model, val, step_size=1)
    
    logger.info("Iniciando Teste Rolling...")
    test_pred = rolling_forecast_optimized(model, test, step_size=1)
    
    # In-sample (apenas para referência)
    train_pred = model.arima_result.fittedvalues

    # 6. Métricas
    metrics = {
        'train': evaluate_model(train.values, train_pred),
        'val':   evaluate_model(val.values, val_pred),
        'test':  evaluate_model(test.values, test_pred)
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
