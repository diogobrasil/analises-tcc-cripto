import numpy as np
import pandas as pd
import joblib

class ActionPredictionTrading:
    """
    Classe para criar previsões de preço, simular operações de trading
    e comparar com buy-and-hold, usando regressão linear via equação normal.
    """

    def __init__(self, df: pd.DataFrame, ticker: str, window: int = 3, model_path: str = None):
        """
        Args:
            df: DataFrame com colunas ['Date', ticker]
            ticker: nome da coluna de preços
            window: tamanho da janela de lag para features
            model_path: caminho para o arquivo do modelo serializado (joblib)
        """
        # Prepara DataFrame padronizado
        tmp = df[['Date', ticker]].dropna().reset_index(drop=True)
        tmp.columns = ['date', 'actual']
        self.full_df = tmp.copy()       # Série completa para buy-and-hold
        self.df = tmp.copy()            # Será fatiada para previsões
        self.window = window
        self.ticker = ticker
        self.model = None
        self.model_path = model_path
        self.scaler = None

    def create_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera matrizes X e y usando janelas deslizantes de tamanho self.window.
        
        Returns:
            X: array shape (n_samples, window)
            y: array shape (n_samples,)
        """
        prices = self.df['actual'].values
        n = len(prices)
        if n <= self.window:
            raise ValueError(f"Série muito curta: len={n}, window={self.window}")
        
        X, y = [], []
        for i in range(self.window, n):
            X.append(prices[i - self.window:i])
            y.append(prices[i])
        return np.array(X), np.array(y)

    def load_model(self):
        """
        Carrega o modelo serializado e verifica se 'theta' está presente.
        """
        if not self.model_path:
            raise ValueError("Model path not specified.")
        self.model = joblib.load(self.model_path)
        if getattr(self.model, 'theta', None) is None:
            raise RuntimeError("Modelo não contém `theta`. Verifique se salvou após `normal_equation`.")
        print(f"Model loaded and validated from {self.model_path}")

    def load_scaler(self, scaler_path: str):
        """
        Carrega e valida o scaler usado no treino.
        
        Args:
            scaler_path: caminho para o scaler serializado (joblib)
        """
        self.scaler = joblib.load(scaler_path)
        if not hasattr(self.scaler, 'scale_'):
            raise ValueError("Scaler não está ajustado. Ajuste-o com X de treino antes de salvar.")
        expected = getattr(self.scaler, 'n_features_in_', None)
        if expected is None or expected != self.window:
            raise ValueError(
                f"Scaler foi treinado com {expected} features, "
                f"mas a janela atual é de {self.window}."
            )
        print(f"Scaler loaded and validated for window={self.window}")

    def generate_predictions(self):
        """
        Gera previsões para cada janela e reconstrói self.df sem defasagem.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Run `load_model()` first.")

        X, _ = self.create_windows()
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # cálculo manual usando theta (bias + coefs)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X_bias @ self.model.theta

        # alinha datas e valores
        dates = self.df['date'].iloc[self.window:].reset_index(drop=True)
        actuals = self.df['actual'].iloc[self.window:].reset_index(drop=True)
        self.df = pd.DataFrame({
            'date': dates,
            'actual': actuals,
            'predicted': y_pred
        })

    def simulate_trading(
        self,
        stop_loss: bool = False,
        initial_capital: float = 100000,
        shares_per_trade: int = 100,
        stop_type: str = 'percent',
        stop_value: float = 0.02
    ) -> dict:
        """
        Simula operações long/short baseadas nas previsões,
        com stop-loss opcional.
        """
        capital = initial_capital
        capital_history = [capital]
        hits = 0
        total_trades = 0
        profits = []
        stop_triggered = 0

        for i in range(len(self.df) - 1):
            price_today = self.df.iloc[i]['actual']
            price_tomorrow = self.df.iloc[i + 1]['actual']
            prediction = self.df.iloc[i]['predicted']

            limit = stop_value * price_today if stop_type == 'percent' else stop_value
            limit_amt = limit * shares_per_trade

            if prediction > price_today:
                position = 'long'
            elif prediction < price_today:
                position = 'short'
            else:
                continue  # sem posição

            if position == 'long':
                pnl = (price_tomorrow - price_today) * shares_per_trade
            else:
                pnl = (price_today - price_tomorrow) * shares_per_trade

            if stop_loss and pnl < -limit_amt:
                pnl = -limit_amt
                stop_triggered += 1

            capital += pnl
            profits.append(pnl)
            total_trades += 1
            if pnl > 0:
                hits += 1

            capital_history.append(capital)

        hit_rate = hits / total_trades if total_trades > 0 else 0
        total_return = (capital - initial_capital) / initial_capital
        sharpe_ratio = (
            np.mean(profits) / np.std(profits)
            if len(profits) > 1 and np.std(profits) != 0 else 0
        )
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

    def simulate_buy_and_hold(
        self,
        initial_capital: float = 100000,
        shares: int = 100
    ) -> dict:
        """
        Compara buy-and-hold na série completa de preços (full_df).
        """
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
