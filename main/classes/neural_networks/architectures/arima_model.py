import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

# Suprimir avisos de convergência comuns em dados financeiros ruidosos
warnings.filterwarnings("ignore")

class ArimaModel:
    """
    Wrapper para um modelo Híbrido:
    - ARIMA para a média condicional (Retorno Esperado)
    - GARCH para a variância condicional (Volatilidade/Risco)
    """
    def __init__(self, arima_order=(1, 0, 1), garch_order=(1, 1), use_garch=True):
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.use_garch = use_garch
        
        self.arima_result = None
        self.garch_result = None

    def fit(self, y_train):
        """
        Ajusta o ARIMA nos dados e, opcionalmente, o GARCH nos resíduos.
        """
        # 1. Ajuste do ARIMA (Média)
        try:
            # Em dados de 15min (retornos), d=0 é o padrão.
            model_arima = ARIMA(y_train, order=self.arima_order)
            self.arima_result = model_arima.fit()
        except Exception as e:
            raise ValueError(f"Erro ao ajustar ARIMA: {e}")

        # 2. Ajuste do GARCH (Volatilidade)
        if self.use_garch:
            try:
                # Pegamos os resíduos do ARIMA
                residuals = self.arima_result.resid
                # Escalar resíduos pode ajudar na convergência do GARCH (multiplicar por 100)
                # Aqui usaremos o padrão. Model='Garch', dist='t' (Student's t para caudas gordas)
                model_garch = arch_model(residuals, vol='Garch', 
                                         p=self.garch_order[0], q=self.garch_order[1], 
                                         dist='Normal') # Use 't' se tiver arch instalado completo
                self.garch_result = model_garch.fit(disp='off', show_warning=False)
            except Exception as e:
                print(f"Aviso: Falha ao ajustar GARCH ({e}). Seguindo apenas com ARIMA.")
                self.garch_result = None

        return self

    def predict_next(self, steps=1):
        """
        Retorna a previsão de retorno (mu) e volatilidade (sigma) para o próximo passo.
        """
        if self.arima_result is None:
            raise ValueError("Modelo não treinado. Execute .fit() primeiro.")

        # Previsão da Média
        forecast_arima = self.arima_result.forecast(steps=steps)
        mu = forecast_arima.iloc[0] if isinstance(forecast_arima, pd.Series) else forecast_arima[0]

        sigma = 0.0
        if self.garch_result is not None:
            # Previsão da Variância
            # horizon=steps
            forecast_garch = self.garch_result.forecast(horizon=steps)
            var_pred = forecast_garch.variance.iloc[-1, 0]
            sigma = np.sqrt(var_pred)

        return mu, sigma

    def update(self, new_observations):
        """
        Atualiza o modelo com novos dados sem reestimar parâmetros (filtro de Kalman).
        Isso é mais rápido que refazer o fit().
        """
        if self.arima_result is None:
            raise ValueError("Modelo não iniciado.")
        
        # O statsmodels permite append/extend
        self.arima_result = self.arima_result.append(new_observations, refit=False)
        
        # Para o GARCH, em um setup simples de produção, muitas vezes re-treinamos
        # ou apenas ignoramos a atualização online complexa neste exemplo simplificado.
        # Se for crítico, o ideal é re-treinar o GARCH periodicamente.
