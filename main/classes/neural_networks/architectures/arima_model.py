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

    def fit(self, y_train, exog_train=None):
        """
        Ajusta o ARIMA (com ou sem Exog) e o GARCH nos resíduos.
        """
        try:
            # ARIMA aceita 'exog' como argumento opcional
            self.model_arima = ARIMA(y_train, exog=exog_train, order=self.arima_order)
            self.arima_result = self.model_arima.fit()
        except Exception as e:
            raise ValueError(f"Erro ao ajustar ARIMA: {e}")

        # GARCH opera nos resíduos do ARIMA (para modelar o risco/volatilidade)
        if self.use_garch:
            try:
                residuals = self.arima_result.resid
                # Escalar resíduos (x100) ajuda na convergência do GARCH em dados pequenos
                self.model_garch = arch_model(residuals * 100, vol='Garch', 
                                              p=self.garch_order[0], q=self.garch_order[1], 
                                              dist='Normal')
                self.garch_result = self.model_garch.fit(disp='off', show_warning=False)
            except Exception as e:
                print(f"Aviso: GARCH falhou ({e}). Seguindo sem ele.")
                self.garch_result = None
        return self

    def predict_next(self, steps=1, exog_future=None):
        """
        Realiza a previsão. Se houver exog, ele é obrigatório aqui.
        """
        if self.arima_result is None:
            raise ValueError("Modelo não treinado.")

        # Previsão da Média (ARIMAX)
        # exog_future deve ter tamanho igual a 'steps'
        forecast_arima = self.arima_result.forecast(steps=steps, exog=exog_future)
        
        # Tratamento seguro para diferentes tipos de retorno (Series, Array, Escalar)
        if isinstance(forecast_arima, pd.Series):
            mu = forecast_arima.values
        else:
            mu = np.array(forecast_arima)

        # Previsão da Volatilidade (GARCH)
        sigma = np.zeros(steps)
        if self.garch_result is not None:
            forecast_garch = self.garch_result.forecast(horizon=steps)
            var_pred = forecast_garch.variance.iloc[-1].values
            sigma = np.sqrt(var_pred) / 100 # Desfazendo a escala

        # Se pediu apenas 1 passo, retorna escalar. Se mais, retorna array.
        if steps == 1:
            return mu[0], sigma[0]
        return mu, sigma
