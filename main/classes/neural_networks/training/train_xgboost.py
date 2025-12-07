import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import argparse
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering(df, threshold=0.0000):
    df = df.copy()
    
    # Target com Threshold (Filtro de Ruído)
    # Só consideramos alta se for maior que o custo (spread)
    # Tente ajustar threshold para 0.0005 (0.05%) se quiser sinais mais fortes
    df['future_ret'] = np.log(df['close'].shift(-1) / df['close'])
    df['target'] = np.where(df['future_ret'] > threshold, 1, 0)
    
    # Features
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_vol'] = np.log(df['high'] / df['low'])
    df['log_volume'] = np.log(df['tickvol'] + 1)
    df['rsi'] = calculate_rsi(df['close'])
    
    # Time
    minutes = df.index.hour * 60 + df.index.minute
    df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)
    
    # Lags (Reduzido para 3 para diminuir complexidade)
    for lag in range(1, 4):
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
        df[f'vol_lag_{lag}'] = df['log_vol'].shift(lag)
        
    return df.dropna()

def train_optimized(args):
    logger.info(f"Carregando: {args.csv_path}")
    df = pd.read_csv(args.csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    
    # Engenharia (Threshold zero para comparar com anterior, ou mude para 0.0005)
    df_processed = feature_engineering(df, threshold=0.0000)
    
    drop_cols = ['open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread', 'future_ret', 'target']
    features = [c for c in df_processed.columns if c not in drop_cols]
    
    X = df_processed[features]
    y = df_processed['target']
    
    # Split Rígido (2025)
    train_mask = (X.index >= "2025-01-01") & (X.index <= args.train_end)
    test_mask = (X.index > args.val_end) # Vamos usar o resto tudo como teste final
    
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    logger.info(f"Treino Tamanho: {len(X_train_full)} | Teste Tamanho: {len(X_test)}")
    
    # --- OTIMIZAÇÃO DE HIPERPARÂMETROS ---
    num_zeros = y_train_full.value_counts()[0]
    num_ones = y_train_full.value_counts()[1]
    scale_weight = num_zeros / num_ones
    
    logger.info(f"Ratio de Classes (0/1): {scale_weight:.2f}")

    # --- OTIMIZAÇÃO DE HIPERPARÂMETROS AJUSTADA ---
    # Mudanças:
    # 1. learning_rate mais alto (0.01 deixou ele lento demais, vamos tentar 0.05 a 0.1)
    # 2. scale_pos_weight incluído para forçar a prever classe 1
    # 3. max_depth reduzido para evitar decorar o ruído
    
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 4],           # Árvores rasas funcionam melhor em ruído
        'learning_rate': [0.05, 0.1],  # Aumentando para ser menos conservador
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'scale_pos_weight': [scale_weight, scale_weight * 1.2], # Forçar atenção aos "1"s
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        n_jobs=-1, 
        random_state=42,
        eval_metric='logloss' # Removemos reg_alpha/lambda fixos para deixar o grid decidir
    )    
    # TimeSeriesSplit garante que não haja vazamento de dados no GridSearch
    tscv = TimeSeriesSplit(n_splits=3)
    
    logger.info("Iniciando RandomizedSearchCV (isso pode demorar 1-2 min)...")
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,  # Tenta 20 combinações aleatórias
        scoring='accuracy',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train_full, y_train_full)
    
    best_model = search.best_estimator_
    logger.info(f"Melhores Parâmetros: {search.best_params_}")
    
# --- AVALIAÇÃO FINAL ---
    logger.info("Avaliando no Teste (Out-of-Sample)...")
    
    # Previsão Padrão
    preds = best_model.predict(X_test)
    
    # Previsão de Probabilidades (Ver se ele está quase acertando)
    probs = best_model.predict_proba(X_test)[:, 1] # Probabilidade de ser 1
    
    # Teste com Threshold Personalizado (ex: > 0.45 vira 1)
    threshold_custom = 0.45
    preds_custom = (probs > threshold_custom).astype(int)
    
    acc = accuracy_score(y_test, preds)
    acc_custom = accuracy_score(y_test, preds_custom)
    
    logger.info(f"--- RESULTADO PADRÃO (Thresh 0.50) ---")
    logger.info(f"Acurácia: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))

    logger.info(f"--- RESULTADO AJUSTADO (Thresh {threshold_custom}) ---")
    logger.info(f"Acurácia: {acc_custom:.4f}")
    logger.info("\n" + classification_report(y_test, preds_custom))
    
    # Salvar
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_dir / "xgboost_best.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--train_end', default="2025-09-21 19:00")
    parser.add_argument('--val_end', default="2025-10-21 19:00")
    parser.add_argument('--model_dir', default="./models_xgboost_opt")
    
    args = parser.parse_args()
    train_optimized(args)
