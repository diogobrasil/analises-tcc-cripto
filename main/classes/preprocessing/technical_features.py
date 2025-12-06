import pandas as pd
import numpy as np
import logging

class TechnicalFeatures:
    """
    Calcula indicadores técnicos avançados focados em Estacionariedade para Machine Learning.
    Inclui:
    1. Retornos (Log Returns)
    2. Price Action (Candle Body, Shadows, Range)
    3. Tendência (SMA Ratios)
    4. Momentum (RSI)
    5. Volatilidade e Volume
    6. Sazonalidade (Hora)
    """
    
    def __init__(self, df: pd.DataFrame):
        # Trabalhamos com cópia para não afetar o original
        self.df = df.copy()
        # Garante nomes de colunas padrão (minúsculo) para evitar erros de case-sensitive
        self.df.columns = [c.lower() for c in self.df.columns]
        
    def get_features(self) -> pd.DataFrame:
        """
        Calcula todos os indicadores e retorna o DF enriquecido.
        """
        logging.info("Gerando indicadores técnicos (Price Action + Trend + Momentum)...")
        
        # -----------------------------------------------------------
        # 1. Target Base: Log Returns (A feature mais importante)
        # -----------------------------------------------------------
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        
        # -----------------------------------------------------------
        # 2. Price Action & Candlestick Physics (Usando OHLC)
        # -----------------------------------------------------------
        
        # A. Amplitude Total (Range) Normalizada
        # (High - Low) / Close -> Indica volatilidade intraday pura
        self.df['candle_range'] = (self.df['high'] - self.df['low']) / self.df['close']
        
        # B. Retorno do Corpo (Intraday Momentum)
        # (Close - Open) / Open -> A força líquida do candle
        self.df['body_return'] = (self.df['close'] - self.df['open']) / self.df['open']
        
        # C. Pavio Superior (Pressão de Venda)
        # Distância entre High e o topo do corpo (seja open ou close)
        # Quanto maior, mais rejeição de alta houve.
        upper_wick = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['upper_shadow'] = upper_wick / self.df['close']
        
        # D. Pavio Inferior (Pressão de Compra)
        # Distância entre o fundo do corpo e o Low
        # Quanto maior, mais rejeição de baixa houve (martelo).
        lower_wick = self.df[['open', 'close']].min(axis=1) - self.df['low']
        self.df['lower_shadow'] = lower_wick / self.df['close']

        # -----------------------------------------------------------
        # 3. Tendência: Distância da Média (SMA Ratio)
        # -----------------------------------------------------------
        # Se > 1.0: Preço acima da média. Se < 1.0: Abaixo.
        for window in [9, 21, 50]:
            sma = self.df['close'].rolling(window=window).mean()
            self.df[f'sma_ratio_{window}'] = self.df['close'] / sma

        # -----------------------------------------------------------
        # 4. Momentum: RSI (Relative Strength Index)
        # -----------------------------------------------------------
        self.df['rsi_14'] = self._calculate_rsi(window=14)
        
        # -----------------------------------------------------------
        # 5. Volatilidade e Risco
        # -----------------------------------------------------------
        # Desvio padrão móvel dos retornos (Volatilidade Histórica recente)
        self.df['volatility_20'] = self.df['log_return'].rolling(window=20).std()
        
        # -----------------------------------------------------------
        # 6. Volume Relativo
        # -----------------------------------------------------------
        # Detecta spikes de volume. Volume / Média(20).
        # Tenta usar tick_volume se existir (comum em Forex/B3 via MT5), senão usa volume real
        vol_col = 'tick_volume' if 'tick_volume' in self.df.columns else 'volume'
        if vol_col in self.df.columns:
            vol_sma = self.df[vol_col].rolling(window=20).mean()
            # Adiciona 1e-9 para evitar divisão por zero se volume for 0
            self.df['vol_rel'] = self.df[vol_col] / (vol_sma + 1e-9)
        
        # -----------------------------------------------------------
        # 7. Sazonalidade (Tempo)
        # -----------------------------------------------------------
        # Converte hora em Seno/Cosseno para o modelo entender que 23h é perto de 00h
        # Tenta pegar do index se for DatetimeIndex, ou da coluna 'time'
        if isinstance(self.df.index, pd.DatetimeIndex):
            timestamps = self.df.index
        elif 'time' in self.df.columns:
            timestamps = pd.to_datetime(self.df['time'])
        else:
            timestamps = None
            
        if timestamps is not None:
            # Ciclo diário (24h)
            self.df['hour_sin'] = np.sin(2 * np.pi * timestamps.hour / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * timestamps.hour / 24)
        
        # -----------------------------------------------------------
        # Limpeza Final
        # -----------------------------------------------------------
        # Removemos os NaNs gerados pelo cálculo de janelas (ex: SMA 50 gera 50 NaNs iniciais)
        original_len = len(self.df)
        self.df.dropna(inplace=True)
        dropped = original_len - len(self.df)
        if dropped > 0:
            logging.info(f"Feature Engineering: Dropados {dropped} candles iniciais (warmup).")
        
        return self.df

    def _calculate_rsi(self, window=14):
        """Cálculo manual do RSI vetorializado"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        # Evita divisão por zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)