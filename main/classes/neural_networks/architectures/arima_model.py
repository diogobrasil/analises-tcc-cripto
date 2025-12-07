import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

class ArimaModel:
    def __init__(self, arima_order=(1, 0, 1), garch_order=(1, 1), use_garch=True):
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.use_garch = use_garch
        self.arima_result = None
        self.garch_result = None

    # --- ALTERAÇÃO AQUI: Adicionado argumento 'exog' ---
    def fit(self, y_train, exog_train=None):
        try:
            # Passamos exog para o ARIMA
            model_arima = ARIMA(y_train, exog=exog_train, order=self.arima_order)
            self.arima_result = model_arima.fit()
        except Exception as e:
            raise ValueError(f"Erro ao ajustar ARIMA: {e}")

        # GARCH continua igual (ele opera só nos resíduos)
        if self.use_garch:
            try:
                residuals = self.arima_result.resid
                model_garch = arch_model(residuals, vol='Garch', p=self.garch_order[0], q=self.garch_order[1], dist='Normal')
                self.garch_result = model_garch.fit(disp='off', show_warning=False)
            except Exception:
                self.garch_result = None
        return self

    # --- ALTERAÇÃO AQUI: predict precisa receber o exog futuro ---
    def predict_next(self, steps=1, exog_future=None):
        if self.arima_result is None:
            raise ValueError("Modelo não treinado.")

        # Passamos exog para o forecast
        forecast_arima = self.arima_result.forecast(steps=steps, exog=exog_future)
        mu = forecast_arima.iloc[0] if isinstance(forecast_arima, pd.Series) else forecast_arima[0]

        sigma = 0.0
        if self.garch_result is not None:
            forecast_garch = self.garch_result.forecast(horizon=steps)
            var_pred = forecast_garch.variance.iloc[-1, 0]
            sigma = np.sqrt(var_pred)

        return mu, sigma

    def update(self, new_observations, new_exog=None):
        # Update com exog é complexo no statsmodels via append, 
        # para simplificar o teste, vamos ignorar o update online por enquanto 
        # ou apenas re-treinar se necessário.
        pass
