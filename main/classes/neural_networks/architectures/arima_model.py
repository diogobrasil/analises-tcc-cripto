from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ArimaModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.model_fit = None

    def fit(self, y_train):
        try:
            self.model = ARIMA(y_train, order=self.order)
            self.model_fit = self.model.fit()
            return self.model_fit
        except Exception as e:
            raise ValueError(f"Erro ao ajustar modelo ARIMA: {e}")

    def predict_one_step(self):
        if self.model_fit is None:
            raise ValueError("O modelo ARIMA não foi ajustado. Execute `fit` primeiro.")
        return self.model_fit.forecast(steps=1)[0]

    def update(self, new_value):
        if self.model_fit is None:
            raise ValueError("O modelo ARIMA não foi ajustado. Execute `fit` primeiro.")
        self.model_fit = self.model_fit.append(new_value, refit=False)
