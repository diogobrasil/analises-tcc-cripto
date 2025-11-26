import logging
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# Importações absolutas
from classes.neural_networks.training.train_linear_regression import (
    load_data,
    create_window_data,
    split_data_time_anchored
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ResultsVisualizer:
    def __init__(self, metadata_path: str):
        self.meta_path = Path(metadata_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata não encontrado: {metadata_path}")
        
        self.base_dir = self.meta_path.parent
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run(self):
        config = self.metadata['config']
        artifacts = self.metadata['artifact_paths']
        train_cfg = config['training']
        
        # 1. Carregar Modelo e Scalers
        model, scaler_X, scaler_y = self._load_artifacts(artifacts)
        
        # 2. Recriar Dados de Teste
        # Nota: X_test e y_test virão no formato usado no treino (ex: Log Returns)
        X_test, y_test, test_timestamps, df_full = self._recreate_test_data(config)
        
        # 3. Predizer (na escala normalizada)
        logging.info("Gerando predições...")
        X_test_n = scaler_X.transform(X_test)
        y_pred_n = model.predict(X_test_n)
        
        # 4. Desnormalizar para a escala real (Log Returns Brutos)
        y_pred_raw = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()
        # y_test já está como Log Returns brutos
        
        # 5. Se for modelo de Retornos, precisamos RECONSTRUIR o preço para visualizar
        use_returns = train_cfg.get('use_returns', False)
        
        if use_returns:
            logging.info("Modelo de Retornos detectado: Reconstruindo preços...")
            y_true_plot, y_pred_plot = self._reconstruct_price_path(
                y_true_returns=y_test,
                y_pred_returns=y_pred_raw,
                timestamps=test_timestamps,
                df_original=df_full,
                target_col=config['data']['target_col']
            )
            ylabel = "Preço Reconstruído (R$)"
        else:
            # Se for modelo de preço, plota direto
            y_true_plot = y_test
            y_pred_plot = y_pred_raw
            ylabel = "Preço (R$)"

        # 6. Plotar
        self.plot_comparison(y_true_plot, y_pred_plot, test_timestamps, ylabel=ylabel, title_suffix="Visão Geral")
        self.plot_comparison(y_true_plot, y_pred_plot, test_timestamps, ylabel=ylabel, title_suffix="Zoom (150 candles)", zoom_slice=150)

    def _recreate_test_data(self, config):
        logging.info("Recriando pipeline de dados...")
        data_cfg = config['data']
        train_cfg = config['training']

        # Carrega o DF completo para podermos buscar os preços originais depois
        df = load_data(data_cfg['filepath'], data_cfg['date_col'], data_cfg.get('timezone', 'UTC'))
        
        # Cria janelas (aqui o target vira retorno se use_returns=True)
        X, y, timestamps = create_window_data(
            df, data_cfg['target_col'], train_cfg['window_size'],
            filter_cross_day=train_cfg.get('filter_cross_day', True),
            use_returns=train_cfg.get('use_returns', False)
        )

        _, X_test, _, y_test, _ = split_data_time_anchored(
            X, y, timestamps,
            test_ratio=train_cfg['test_split_ratio'],
            embargo_str=train_cfg.get('embargo', "0s")
        )
        
        test_timestamps = timestamps[-len(y_test):]
        
        return X_test, y_test, test_timestamps, df

    def _reconstruct_price_path(self, y_true_returns, y_pred_returns, timestamps, df_original, target_col):
        """
        Reconstroi o preço: Preço_t = Preço_{t-1} * exp(Retorno_t)
        Isso é uma simulação "One-Step Ahead".
        """
        # Precisamos dos preços REAIS anteriores aos timestamps de teste para aplicar o retorno
        # Como y_target é t+1, precisamos do preço em t.
        # timestamps aponta para o momento da feature (t).
        
        # Pegamos os preços originais alinhados com os timestamps de teste
        # Atenção: Se create_window_data removeu gaps, precisamos usar .loc para garantir alinhamento
        price_t_real = df_original.loc[timestamps][target_col].values
        
        # Reconstrução (Inverso do Log Return):
        # Retorno = ln(P_next / P_t)  -->  P_next = P_t * exp(Retorno)
        
        # 1. Preço Real Futuro (para conferir)
        # Nota: y_true_returns são os retornos reais. Se aplicarmos ao preço t, temos o preço t+1 real.
        price_next_real = price_t_real * np.exp(y_true_returns)
        
        # 2. Preço Previsto Futuro
        # Aplicamos o retorno que o modelo chutou sobre o preço real atual
        price_next_pred = price_t_real * np.exp(y_pred_returns)
        
        return price_next_real, price_next_pred

    def _load_artifacts(self, artifacts):
        # ... (Mantido igual)
        model_path = self.base_dir / Path(artifacts['model']).name
        scaler_X_path = self.base_dir / Path(artifacts['scaler_X']).name
        scaler_y_path = self.base_dir / Path(artifacts['scaler_y']).name

        return (
            joblib.load(model_path),
            joblib.load(scaler_X_path),
            joblib.load(scaler_y_path)
        )

    def plot_comparison(self, y_true, y_pred, timestamps, ylabel, title_suffix="", zoom_slice=None):
        df_plot = pd.DataFrame({'Real': y_true, 'Previsto': y_pred}, index=timestamps)
        
        if zoom_slice:
            df_plot = df_plot.iloc[-zoom_slice:]
        
        plt.figure(figsize=(14, 7))
        plt.plot(df_plot.index, df_plot['Real'], label='Real', color='blue', alpha=0.6)
        
        # Linha pontilhada para previsão
        plt.plot(df_plot.index, df_plot['Previsto'], label='Previsto (One-Step)', color='red', linestyle='--', alpha=0.8)
        
        # Indica visualmente se acertou a direção (Opcional, mas legal)
        # Se quiser poluir menos o gráfico, remova essa lógica
        
        plt.title(f'Resultado do Modelo - {title_suffix}')
        plt.xlabel('Data/Hora')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=str, required=True)
    args = parser.parse_args()
    
    viz = ResultsVisualizer(args.meta)
    viz.run()