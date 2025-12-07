import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import argparse

# Reutilizando funções de engenharia (copie as funções calculate_rsi e feature_engineering do script anterior)
# ... (Cole aqui as funções calculate_rsi e feature_engineering do train_xgboost_optimized.py) ...
# Para economizar espaço, assumo que você as tem. Se não, avise que colo tudo.

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering(df, threshold=0.0000):
    df = df.copy()
    df['future_ret'] = np.log(df['close'].shift(-1) / df['close'])
    df['target'] = np.where(df['future_ret'] > threshold, 1, 0)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_vol'] = np.log(df['high'] / df['low'])
    df['log_volume'] = np.log(df['tickvol'] + 1)
    df['rsi'] = calculate_rsi(df['close'])
    minutes = df.index.hour * 60 + df.index.minute
    df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)
    for lag in range(1, 4):
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
        df[f'vol_lag_{lag}'] = df['log_vol'].shift(lag)
    return df.dropna()

def test_inference(args):
    # 1. Carregar Modelo e Dados
    print(f"Carregando modelo de: {args.model_path}")
    model = joblib.load(args.model_path)
    
    df = pd.read_csv(args.csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    df_processed = feature_engineering(df)
    
    # 2. Filtrar apenas o Teste (Outubro/Novembro 2025)
    test_mask = (df_processed.index > args.val_end)
    
    # Selecionar colunas corretas (igual ao treino)
    # XGBoost precisa da ordem exata das colunas
    feature_names = model.get_booster().feature_names
    X_test = df_processed[test_mask][feature_names]
    y_test = df_processed[test_mask]['target']
    
    # 3. Probabilidades
    probs = model.predict_proba(X_test)[:, 1]
    
    # 4. Loop de Thresholds
    print(f"\n--- TESTANDO THRESHOLDS (Total Amostras: {len(y_test)}) ---")
    for thresh in [0.50, 0.53, 0.55, 0.60]:
        preds = (probs > thresh).astype(int)
        acc = accuracy_score(y_test, preds)
        
        # Calcular Precision da Classe 1 manualmente para resumo rápido
        # (Evita imprimir relatório gigante para cada um)
        tp = ((preds == 1) & (y_test == 1)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        print(f"Threshold > {thresh:.2f} | Acc: {acc:.2%} | Precision Compra: {prec_1:.2%} | Trades: {sum(preds)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--model_path', default="./models_xgboost_opt/xgboost_best.pkl")
    parser.add_argument('--val_end', default="2025-10-21 19:00")
    args = parser.parse_args()
    test_inference(args)
