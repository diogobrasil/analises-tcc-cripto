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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- FEATURE ENGINEERING ---
def calculate_rsi(series, period=14):
    """Calcula RSI manualmente para não depender de bibliotecas extras."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_features(df):
    """
    Adiciona features exógenas para ajudar o ARIMA.
    """
    df = df.copy()
    
    # 1. Feature de Tempo (Sazonalidade Intradiária)
    # Transforma hora/minuto em ciclos de Seno e Cosseno
    # Um dia tem 24h * 60m = 1440 minutos.
    minutes_in_day = df.index.hour * 60 + df.index.minute
    df['time_sin'] = np.sin(2 * np.pi * minutes_in_day / 1440)
    df['time_cos'] = np.cos(2 * np.pi * minutes_in_day / 1440)
    
    # 2. RSI (Indicador de Momento Estacionário)
    # O RSI varia de 0 a 100, ideal para ARIMAX.
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # 3. Volatilidade Recente (Log Range)
    # Ajuda o modelo a saber se o mercado está calmo ou agitado
    df['log_vol'] = np.log(df['high'] / df['low'])
    
    # 4. Volume Logarítmico
    df['log_volume'] = np.log(df['tickvol'] + 1)

    return df.dropna()

def load_and_transform_data(csv_path: str, target_col: str):
    logger.info(f"Carregando e processando: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    
    # Gerar Features Técnicas (Exógenas)
    df_features = add_technical_features(df)
    
    # Calcular Target (Log Returns)
    # Shiftamos o retorno para alinhar com as features DO MOMENTO ANTERIOR se quiséssemos prever t+1 com dados de t.
    # Mas no ARIMAX padrão, usamos exog contemporâneo ou defasado. 
    # Aqui, usaremos features atuais para explicar o retorno atual (análise) 
    # ou features defasadas para prever. Vamos alinhar tudo.
    log_ret = np.log(df_features[target_col] / df_features[target_col].shift(1))
    
    # As features exógenas devem ser as que temos disponíveis NO MOMENTO da previsão.
    # Para prever o retorno de (t+1), usamos o RSI de (t).
    # Portanto, precisamos alinhar: Target(t) vs Features(t-1)
    # Shiftamos as features exógenas 1 para frente para alinhar com o retorno futuro
    exog_cols = ['time_sin', 'time_cos', 'rsi', 'log_vol', 'log_volume']
    exog_data = df_features[exog_cols].shift(1) 
    
    # Juntar e limpar NaNs gerados pelos shifts
    data = pd.DataFrame({'target': log_ret}).join(exog_data).dropna()
    
    return data['target'], data[exog_cols]

# --- SPLIT POR DATA ---
def split_data(ts, exog, train_end, val_end):
    # Recorte temporal rígido (ex: só 2025)
    # Se quiser forçar inicio 2025:
    start_date = "2025-01-01"
    
    mask_train = (ts.index >= start_date) & (ts.index <= train_end)
    mask_val   = (ts.index > train_end) & (ts.index <= val_end)
    mask_test  = (ts.index > val_end)
    
    return (ts[mask_train], exog[mask_train]), \
           (ts[mask_val],   exog[mask_val]), \
           (ts[mask_test],  exog[mask_test])

# --- METRICAS ---
def evaluate_model(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Acurácia de Sinal (Direção)
    acc_dir = accuracy_score(np.sign(y_true), np.sign(y_pred))
    # Acurácia Invertida (Caso o modelo esteja invertido)
    acc_inv = accuracy_score(np.sign(y_true), -np.sign(y_pred))
    
    return {'RMSE': rmse, 'DIR_ACC': acc_dir, 'INV_ACC': acc_inv}

# --- ROLLING FORECAST COM EXOG ---
def rolling_forecast(model, series, exog_data, step_size=1):
    """
    Simula a previsão passo a passo.
    Atenção: exog_data aqui deve ser o dado 'futuro' alinhado. 
    Como shiftamos no load, a linha 'i' de exog contém os dados conhecidos para prever 'i' de target.
    """
    preds = []
    # Loop otimizado: prever em blocos se step_size > 1
    # Para teste rápido, step_size=1 é o mais preciso
    
    for i in range(0, len(series), step_size):
        # Pegar exógenas correspondentes ao passo atual
        # O modelo espera formato (n_steps, n_features)
        current_exog = exog_data.iloc[i : i+step_size]
        
        try:
            # Prever
            mu, _ = model.predict_next(steps=step_size, exog_future=current_exog)
            if step_size == 1:
                preds.append(mu)
            else:
                preds.extend(mu)
        except Exception as e:
            # Fallback em caso de erro numérico
            preds.append(0)
            
    return np.array(preds)

# --- PIPELINE ---
def train_pipeline(args):
    # 1. Carregar
    y, X = load_and_transform_data(args.csv_path, args.target)
    
    # 2. Split
    (y_train, X_train), (y_val, X_val), (y_test, X_test) = split_data(y, X, args.train_end, args.val_end)
    
    logger.info(f"Treino: {len(y_train)} linhas | Exog Features: {X_train.columns.tolist()}")
    
    # 3. Modelo
    # Se d=1, alertar. Retornos geralmente d=0.
    model = ArimaModel(arima_order=args.order, use_garch=args.use_garch)
    
    # 4. Fit
    logger.info("Iniciando treinamento ARIMAX...")
    model.fit(y_train, exog_train=X_train)
    logger.info("Treinamento concluído.")
    
    # Ver coeficientes
    print("\n--- COEFICIENTES DO ARIMAX ---")
    print(model.arima_result.summary().tables[1])
    print("------------------------------\n")

    # 5. Predict Rolling
    logger.info("Prevendo Validação...")
    val_pred = rolling_forecast(model, y_val, X_val)
    
    logger.info("Prevendo Teste...")
    test_pred = rolling_forecast(model, y_test, X_test)
    
    # 6. Avaliar
    metrics = {
        'train': evaluate_model(y_train, model.arima_result.fittedvalues),
        'val':   evaluate_model(y_val, val_pred),
        'test':  evaluate_model(y_test, test_pred)
    }
    
    print("\n=== RESULTADOS FINAIS ===")
    print(json.dumps(metrics, indent=2))
    
    # Salvar
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"arimax_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--target', default='close')
    parser.add_argument('--order', type=str, default="1,0,0") # AR(1) simples + Features
    parser.add_argument('--train_end', required=True)
    parser.add_argument('--val_end', required=True)
    parser.add_argument('--model_dir', default="./artifacts")
    parser.add_argument('--use_garch', action='store_true')
    
    args = parser.parse_args()
    args.order = tuple(map(int, args.order.split(',')))
    
    train_pipeline(args)
