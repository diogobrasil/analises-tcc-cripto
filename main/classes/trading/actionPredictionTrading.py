import numpy as np
import pandas as pd
import joblib

class ActionPredictionTrading:
    def __init__(self, df, ticker, window=3, model_path=None):
        self.df = df[['Date', ticker]].dropna().reset_index(drop=True)
        self.df.columns = ['date', 'actual']
        self.window = window
        self.ticker = ticker
        self.model = None
        self.model_path = model_path
        self.scaler = None

    def create_windows(self):
        prices = self.df['actual'].values
        X, y = [], []
        for i in range(self.window, len(prices)):
            X.append(prices[i - self.window:i])
            y.append(prices[i])
        return np.array(X), np.array(y)

    def load_model(self):
        if not self.model_path:
            raise ValueError("Model path not specified.")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
       

    def load_scaler(self, scaler_path):
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")

    def generate_predictions(self):
        if self.model is None:
            raise ValueError("Model not loaded. Run `load_model()` first.")
        X, _ = self.create_windows()
        if self.scaler:
            X = self.scaler.transform(X)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Usar os coeficientes do modelo (self.model.theta)
        if self.model.theta is None:
            raise ValueError("Model coefficients not found. Ensure the model was trained and saved correctly.")
        
        y_pred = X_bias @ self.model.theta  # Multiplicação usando os coeficientes
        self.df = self.df.iloc[self.window:].copy()
        self.df['predicted'] = y_pred

    def simulate_trading(self, stop_loss=False, initial_capital=100000, shares_per_trade=100, stop_type='percent', stop_value=0.02):
        capital = initial_capital
        capital_history = [capital]
        hits = 0
        total_trades = 0
        profits = []
        stop_triggered = 0
        limits = []

        for i in range(len(self.df) - 1):
            price_today = self.df.iloc[i]['actual']
            price_tomorrow = self.df.iloc[i + 1]['actual']
            prediction = self.df.iloc[i]['predicted']

            # Define stop limit
            if stop_type == 'percent':
                limit = stop_value * price_today  # valor absoluto baseado na porcentagem
            elif stop_type == 'fixed':
                limit = stop_value  # valor fixo
            else:
                raise ValueError("Invalid stop_type. Use 'percent' or 'fixed'.")
            
            limits.append(limit)  # Armazena o limite usado

            if prediction > price_today:
                # Buy signal (posição longa)
                price_change = price_tomorrow - price_today
                profit = price_change * shares_per_trade
                
                # Aplica stop loss apenas se houver perda E ela exceder o limite
                if stop_loss and price_change < 0 and abs(price_change) > limit:
                    profit = -limit * shares_per_trade
                    stop_triggered += 1
                    
                capital += profit
                profits.append(profit)
                total_trades += 1
                
                # Hit apenas se previsão estava correta (preço subiu)
                if price_tomorrow > price_today:
                    hits += 1

            elif prediction < price_today:
                # Sell signal (posição curta)
                price_change = price_today - price_tomorrow  # lucro quando preço cai
                profit = price_change * shares_per_trade
                
                # Aplica stop loss apenas se houver perda E ela exceder o limite
                if stop_loss and price_change < 0 and abs(price_change) > limit:
                    profit = -limit * shares_per_trade
                    stop_triggered += 1

                capital += profit
                profits.append(profit)
                total_trades += 1
                
                # Hit apenas se previsão estava correta (preço caiu)
                if price_tomorrow < price_today:
                    hits += 1
            
            # Adiciona capital atual ao histórico
            capital_history.append(capital)

        # Métricas finais
        hit_rate = hits / total_trades if total_trades > 0 else 0
        total_return = (capital - initial_capital) / initial_capital
        sharpe_ratio = np.mean(profits) / np.std(profits) if len(profits) > 0 and np.std(profits) != 0 else 0
        
        # Cálculo correto do max drawdown
        peak = np.maximum.accumulate(capital_history)
        drawdown = (peak - capital_history) / peak
        max_drawdown = np.max(drawdown)

        return {
            'total_return': total_return,
            'hit_rate': hit_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'total_trades': total_trades,
            'stop_triggered': stop_triggered,
        }

    def simulate_buy_and_hold(self, initial_capital=100000, shares=100):
        if self.df.empty:
            raise ValueError("DataFrame is empty. Ensure the data was loaded correctly.")

        price_buy = self.df.iloc[0]['actual']
        price_sell = self.df.iloc[-1]['actual']

        profit = (price_sell - price_buy) * shares
        final_capital = initial_capital + profit
        total_return = profit / initial_capital

        capital_history = [
            initial_capital + (self.df.iloc[i]['actual'] - price_buy) * shares
            for i in range(len(self.df))
        ]

        return {
            'total_return': total_return,
            'initial_price': price_buy,
            'final_price': price_sell,
            'final_capital': final_capital,
            'shares_held': shares,
            'days_held': len(self.df),
        }
