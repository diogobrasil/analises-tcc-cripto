import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse
import logging

# Configuração de estilo para os gráficos
plt.style.use('bmh') # Estilo limpo e profissional

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering(df, threshold=0.0000):
    df = df.copy()
    # Recriando as features exatamente como no treino
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

def run_backtest(args):
    print(f"--- INICIANDO BACKTEST (Threshold: {args.threshold}) ---")
    
    # 1. Carregar Modelo e Dados
    model = joblib.load(args.model_path)
    df = pd.read_csv(args.csv_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    
    # 2. Preparar Dados
    df_processed = feature_engineering(df)
    
    # Filtrar apenas o período de Teste (Out-of-Sample)
    test_mask = (df_processed.index > args.val_end)
    df_test = df_processed[test_mask].copy()
    
    # Features para previsão
    feature_names = model.get_booster().feature_names
    X_test = df_test[feature_names]
    
    # 3. Gerar Sinais
    # Probabilidade de ser Classe 1 (Alta)
    probs = model.predict_proba(X_test)[:, 1]
    
    # SINAL: 1 se Prob > Threshold (Compra), 0 caso contrário (Fica fora)
    # Estratégia: Entra no fechamento atual, sai no fechamento do próximo (1 candle hold)
    df_test['signal'] = (probs > args.threshold).astype(int)
    
    # 4. Calcular Retornos
    # Retorno da Estratégia = Sinal * Retorno do Próximo Candle
    # Se Sinal for 0, retorno é 0 (estamos líquidos/caixa)
    df_test['strategy_gross'] = df_test['signal'] * df_test['future_ret']
    
    # Custos (Spread + Emolumentos)
    # Assumindo 0.05% por trade (entrada + saída)
    cost_per_trade = 0.0005 
    df_test['costs'] = df_test['signal'] * cost_per_trade
    
    df_test['strategy_net'] = df_test['strategy_gross'] - df_test['costs']
    
    # Retorno Acumulado (Equity Curve)
    df_test['cum_ret_model'] = df_test['strategy_net'].cumsum()
    df_test['cum_ret_bh'] = df_test['future_ret'].cumsum() # Buy & Hold
    
    # 5. Métricas
    total_trades = df_test['signal'].sum()
    final_return = df_test['cum_ret_model'].iloc[-1]
    bh_return = df_test['cum_ret_bh'].iloc[-1]
    
    # Win Rate
    # Um trade é vencedor se o retorno bruto for positivo
    winning_trades = df_test[(df_test['signal'] == 1) & (df_test['future_ret'] > 0)]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    print("\n=== RESULTADOS FINANCEIROS ===")
    print(f"Período: {df_test.index.min()} até {df_test.index.max()}")
    print(f"Trades Executados: {total_trades}")
    print(f"Taxa de Acerto (Win Rate): {win_rate:.2%}")
    print(f"Retorno Modelo (Líquido): {final_return:.4f} ({final_return*100:.2f}%)")
    print(f"Retorno Buy & Hold: {bh_return:.4f} ({bh_return*100:.2f}%)")
    
    # 6. Plotagem Profissional
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Gráfico 1: Curva de Capital
    ax1.plot(df_test.index, df_test['cum_ret_model'], label=f'Modelo AI (Thresh > {args.threshold})', color='green', linewidth=2)
    ax1.plot(df_test.index, df_test['cum_ret_bh'], label='Buy & Hold (ABEV3)', color='gray', linestyle='--', alpha=0.6)
    
    # Marcar onde ocorreram os trades
    trades = df_test[df_test['signal'] == 1]
    ax1.scatter(trades.index, trades['cum_ret_model'], marker='^', color='lime', s=30, label='Entrada (Compra)', zorder=5)
    
    ax1.set_title(f'Backtest: Estratégia Sniper XGBoost (Trades: {total_trades})', fontsize=14)
    ax1.set_ylabel('Retorno Log Acumulado')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Gráfico 2: Drawdown (Risco)
    # Drawdown = (Valor Atual - Pico Máximo)
    cumulative = np.exp(df_test['cum_ret_model']) # Convertendo log para preço base 1
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    ax2.fill_between(df_test.index, drawdown, 0, color='red', alpha=0.3)
    ax2.plot(df_test.index, drawdown, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Data')
    ax2.set_title('Risco (Drawdown)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/backtest_chart.png")
    print(f"\nGráfico salvo em: {args.output_dir}/backtest_chart.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--model_path', default="./models_xgboost_opt/xgboost_best.pkl")
    parser.add_argument('--val_end', default="2025-10-21 19:00")
    parser.add_argument('--threshold', type=float, default=0.60) # O SEU NÚMERO MÁGICO
    parser.add_argument('--output_dir', default="./models_xgboost_opt")
    
    args = parser.parse_args()
    run_backtest(args)
