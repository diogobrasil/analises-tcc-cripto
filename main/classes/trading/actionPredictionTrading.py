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
                limit = stop_value * price_today  # ex: 2% do preço
            elif stop_type == 'fixed':
                limit = stop_value  # ex: R$ 0.50
            else:
                raise ValueError("Invalid stop_type. Use 'percent' or 'fixed'.")

            if prediction > price_today:
                # Buy signal
                profit = (price_tomorrow - price_today) * shares_per_trade
                if stop_loss and abs(price_tomorrow - price_today) > limit:
                    profit = -limit * shares_per_trade if price_tomorrow < price_today else profit
                    stop_triggered += 1
                    
                capital += profit
                profits.append(profit)
                capital_history.append(capital)
                total_trades += 1
                if price_tomorrow > price_today:
                    hits += 1

            elif prediction < price_today:
                # Sell signal (short)
                profit = (price_today - price_tomorrow) * shares_per_trade
                if stop_loss and abs(price_tomorrow - price_today) > limit:
                    profit = -limit * shares_per_trade if price_tomorrow > price_today else profit
                    stop_triggered += 1

                capital += profit
                profits.append(profit)
                capital_history.append(capital)
                total_trades += 1
                if price_tomorrow < price_today:
                    hits += 1
            else:
                capital_history.append(capital)

        hit_rate = hits / total_trades if total_trades else 0
        total_return = (capital - initial_capital) / initial_capital
        sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) != 0 else 0
        max_drawdown = max(np.maximum.accumulate(capital_history) - capital_history)

        return {
            'total_return': total_return,
            'hit_rate': hit_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'stop_triggered': stop_triggered,
            'limits': limits,
        }
