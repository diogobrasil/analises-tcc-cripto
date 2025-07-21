import numpy as np
import pandas as pd
import joblib

class ActionPredictionTrading:
    """
    Classe para criar previsões de preço, simular operações de trading
    e comparar com buy-and-hold, usando regressão linear via equação normal.
    Agora com suporte a MinMaxScaler para X e para y, e conversão correta de previsões.
    """

    def __init__(self, df: pd.DataFrame, ticker: str, window: int = 3,
                 model_path: str = None,
                 scaler_x_path: str = None,
                 scaler_y_path: str = None):
        """
        Args:
            df: DataFrame com colunas ['Date', ticker]
            ticker: nome da coluna de preços
            window: tamanho da janela de lag para features
            model_path: caminho para o arquivo do modelo serializado (joblib)
            scaler_x_path: caminho para o scaler de X (joblib)
            scaler_y_path: caminho para o scaler de y (joblib)
        """
        # Prepara DataFrame padronizado
        tmp = df[['Date', ticker]].dropna().reset_index(drop=True)
        tmp.columns = ['date', 'actual']
        self.full_df = tmp.copy()
        self.df = tmp.copy()
        self.window = window
        self.ticker = ticker
        self.model = None
        self.model_path = model_path
        self.scaler_x = None
        self.scaler_y = None
        if scaler_x_path:
            self.load_scaler_x(scaler_x_path)
        if scaler_y_path:
            self.load_scaler_y(scaler_y_path)

    def load_model(self):
        if not self.model_path:
            raise ValueError("Model path not specified.")
        self.model = joblib.load(self.model_path)
        if getattr(self.model, 'theta', None) is None:
            raise RuntimeError("Modelo não contém `theta`. Salve após `normal_equation`.")
        print(f"Model loaded and validated from {self.model_path}")

    def load_scaler_x(self, scaler_path: str):
        """Carrega e valida o scaler usado em X"""
        self.scaler_x = joblib.load(scaler_path)
        if not hasattr(self.scaler_x, 'scale_'):
            raise ValueError("Scaler X não está ajustado.")
        expected = getattr(self.scaler_x, 'n_features_in_', None)
        if expected != self.window:
            raise ValueError(
                f"Scaler X treinado com {expected} features, janela atual={self.window}"
            )
        print(f"Scaler X loaded and validated for window={self.window}")

    def load_scaler_y(self, scaler_path: str):
        """Carrega e valida o scaler usado em y"""
        self.scaler_y = joblib.load(scaler_path)
        if not hasattr(self.scaler_y, 'data_min_'):
            raise ValueError("Scaler Y não parece ser MinMaxScaler ou não ajustado.")
        print("Scaler Y loaded and validated.")

    def create_windows(self) -> tuple[np.ndarray, list[pd.Timestamp]]:
        """
        Gera X e as datas correspondentes (t).
        Para cada data t, X contém os preços [P_{t-window+1}, ..., P_t].
        A previsão será para o dia t+1.
        """
        prices = self.df['actual'].values
        dates = self.df['date'].values
        n = len(prices)

        if n < self.window:
            raise ValueError(f"Série muito curta para criar uma janela: len={n}, window={self.window}")

        X, dates_t = [], []
        # O loop agora começa em 'window - 1' para a primeira janela completa
        # e vai até 'n - 1' para poder prever o último dia.
        for i in range(self.window - 1, n - 1):
            # A janela de features agora é [P_{i-window+1}, ..., P_i]
            start_index = i - (self.window - 1)
            end_index = i + 1
            X.append(prices[start_index:end_index])
            
            # A data associada a esta janela é a data do último preço (hoje)
            dates_t.append(dates[i])

        return np.array(X), dates_t


    def generate_predictions(self):
        if self.model is None:
            raise ValueError("Model not loaded. Run `load_model()` first.")
        
        X, dates_t = self.create_windows()
        
        # Normaliza os dados de entrada se um scaler_x for fornecido
        if self.scaler_x is not None:
            X_norm = self.scaler_x.transform(X)
        else:
            X_norm = X

        # Usa o método predict do modelo, que já lida com o bias
        y_pred_norm = self.model.predict(X_norm)

        # Desnormaliza as previsões se um scaler_y for fornecido
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_norm
        
        # O DataFrame agora começa a partir da primeira data com uma predição válida
        start_idx = self.window - 1
        end_idx = len(self.df) - 1
        
        self.df = pd.DataFrame({
            'date':        self.df['date'].iloc[start_idx:end_idx].values,
            'actual':      self.df['actual'].iloc[start_idx:end_idx].values,
            'predicted':   y_pred,
            'actual_next': self.df['actual'].iloc[start_idx + 1:].values
        })

    def simulate_trading(
        self,
        stop_loss: bool = False,
        initial_capital: float = 100000,
        shares_per_trade: int = 100,
        stop_type: str = 'percent',
        stop_value: float = 0.03,
        dead_zone_pct: float = 0.005
    ) -> dict:
        capital = initial_capital
        capital_history = [capital]
        hits = 0
        total_trades = 0
        profits = []
        stop_triggered = 0
        #dead_zone_count = 0

        for i in range(len(self.df) - 1):
            price_today = self.df.iloc[i]['actual']
            price_tomorrow = self.df.iloc[i + 1]['actual']
            predicted_price = self.df.iloc[i]['predicted']

            """diff_pct = abs(predicted_price - price_today) / price_today
            if diff_pct < dead_zone_pct:
                dead_zone_count += 1
                continue""" # we can remove this dead zone logic if not needed, i put it here to improve sharpe ratio performance

            limit = stop_value * price_today if stop_type == 'percent' else stop_value
            limit_amt = limit * shares_per_trade

            position = 'long' if predicted_price > price_today else 'short'
            pnl = ((price_tomorrow - price_today) if position == 'long'
                   else (price_today - price_tomorrow)) * shares_per_trade

            if stop_loss and pnl < -limit_amt:
                pnl = -limit_amt
                stop_triggered += 1

            capital += pnl
            profits.append(pnl)
            total_trades += 1
            if pnl > 0:
                hits += 1
            capital_history.append(capital)

        hit_rate = hits / total_trades if total_trades else 0
        total_return = (capital - initial_capital) / initial_capital
        sharpe_ratio = (np.mean(profits) / np.std(profits)
                        if len(profits) > 1 and np.std(profits) else 0)
        peak = np.maximum.accumulate(capital_history)
        max_drawdown = np.max((peak - capital_history) / peak)

        return {
            'total_return': total_return,
            'hit_rate': hit_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'total_trades': total_trades,
            'stop_triggered': stop_triggered,
            'predicted_prices': self.df['predicted'].tolist(),
            'today_prices': self.df['actual'].tolist(),
            'tomorrow_prices': self.df['actual'].shift(-1).tolist(),
            'dates': self.df['date'].tolist()
        }

    def simulate_buy_and_hold(
        self,
        initial_capital: float = 100000,
        shares: int = 100
    ) -> dict:
        df_bh = self.full_df
        if df_bh.empty:
            raise ValueError("DataFrame is empty. Ensure the data was loaded correctly.")

        price_buy = df_bh.iloc[0]['actual']
        price_sell = df_bh.iloc[-1]['actual']
        profit = (price_sell - price_buy) * shares
        final_capital = initial_capital + profit
        total_return = profit / initial_capital

        capital_history = [
            initial_capital + (df_bh.iloc[i]['actual'] - price_buy) * shares
            for i in range(len(df_bh))
        ]

        return {
            'total_return': total_return,
            'initial_price': price_buy,
            'final_price': price_sell,
            'final_capital': final_capital,
            'shares_held': shares,
            'days_held': len(df_bh),
        }
