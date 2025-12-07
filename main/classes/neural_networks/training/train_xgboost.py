import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import logging
from pathlib import Path
import joblib

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering(df):
    df = df.copy()
    
    # 1. Target: Classificação Binária (1 se subiu, 0 se caiu/igual)
    # Shiftamos -1 porque queremos prever o retorno do PRÓXIMO candle usando dados ATUAIS
    df['future_ret'] = np.log(df['close'].shift(-1) / df['close'])
    df['target'] = np.where(df['future_ret'] > 0, 1, 0)
    
    # 2. Features Técnicas (Baseadas no candle ATUAL e passados)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_vol'] = np.log(df['high'] / df['low'])
    df['log_volume'] = np.log(df['tickvol'] + 1)
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # Sazonalidade
    minutes = df.index.hour * 60 + df.index.minute
    df['time_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['time_cos'] = np.cos(2 * np.pi * minutes / 1440)
    
    # 3. Lags (Memória de Curto Prazo)
    # Adiciona os valores dos últimos 5 candles como colunas
    for lag in range(1, 6):
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
        df[f'vol_lag_{lag}'] = df['log_vol'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
    return df.dropna()

def train_xgboost(args):
    # 1. Carregar
    logger.info(f"Carregando: {args.csv_path}")
    df = pd.read_csv(args.csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    
    # 2. Engenharia
    df_processed = feature_engineering(df)
    
    # Definir Features (X) e Target (y)
    drop_cols = ['open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread', 'future_ret', 'target']
    # Garante que só sobram as colunas numéricas criadas
    features = [c for c in df_processed.columns if c not in drop_cols]
    
    X = df_processed[features]
    y = df_processed['target']
    
    logger.info(f"Features ({len(features)}): {features}")
    
    # 3. Split Temporal (Janela Curta 2025)
    train_mask = (X.index >= "2025-01-01") & (X.index <= args.train_end)
    val_mask = (X.index > args.train_end) & (X.index <= args.val_end)
    test_mask = (X.index > args.val_end)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 4. Treinar XGBoost
    # Scale pos weight: ajuda se houver desbalanceamento (muitos 0s ou 1s)
    pos_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    model = xgb.XGBClassifier(
        n_estimators=100,      # Número de árvores
        learning_rate=0.05,    # Velocidade de aprendizado
        max_depth=4,           # Profundidade da árvore (evita overfitting)
        subsample=0.8,         # Usa 80% dos dados por árvore
        colsample_bytree=0.8,  # Usa 80% das features por árvore
        scale_pos_weight=pos_ratio,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10 # Para se não melhorar em 10 rodadas
    )
    
    logger.info("Treinando XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # 5. Avaliar
    def evaluate(name, X_set, y_set):
        preds = model.predict(X_set)
        acc = accuracy_score(y_set, preds)
        logger.info(f"--- {name} Results ---")
        logger.info(f"Acurácia: {acc:.4f}")
        logger.info("\n" + classification_report(y_set, preds))
        return acc
        
    evaluate("Treino", X_train, y_train)
    evaluate("Validação", X_val, y_val)
    acc_test = evaluate("Teste", X_test, y_test)
    
    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top 10 Features Importantes ---")
    print(importance.head(10))
    
    # Salvar
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "xgboost_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--train_end', default="2025-09-21 19:00")
    parser.add_argument('--val_end', default="2025-10-21 19:00")
    parser.add_argument('--model_dir', default="./models_xgboost")
    
    args = parser.parse_args()
    train_xgboost(args)
