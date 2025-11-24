import logging
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


from classes.neural_networks.training.train_linear_regression import (
    load_data,
    create_window_data,
    split_data_time_anchored
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ResultsVisualizer:
    """
    Classe responsável por carregar metadados de treinamento,
    reconstruir o cenário de teste e plotar os resultados.
    """
    
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
        """Executa o fluxo completo de carregamento e visualização."""
        config = self.metadata['config']
        artifacts = self.metadata['artifact_paths']
        
        # 1. Recriar Dados
        X_test, y_test, test_timestamps = self._recreate_test_data(config)
        
        # 2. Carregar Modelo e Scalers
        model, scaler_X, scaler_y = self._load_artifacts(artifacts)
        
        # 3. Predizer
        y_pred_real, y_true_real = self._predict_and_inverse(model, scaler_X, scaler_y, X_test, y_test)
        
        # 4. Plotar
        self.plot_comparison(y_true_real, y_pred_real, test_timestamps, title_suffix="Visão Geral")
        self.plot_comparison(y_true_real, y_pred_real, test_timestamps, title_suffix="Zoom (150 candles)", zoom_slice=150)

    def _recreate_test_data(self, config):
        logging.info("Recriando pipeline de dados...")
        data_cfg = config['data']
        train_cfg = config['training']

        df = load_data(data_cfg['filepath'], data_cfg['date_col'], data_cfg.get('timezone', 'UTC'))
        
        X, y, timestamps = create_window_data(
            df, data_cfg['target_col'], train_cfg['window_size'],
            filter_cross_day=train_cfg.get('filter_cross_day', True)
        )

        _, X_test, _, y_test, _ = split_data_time_anchored(
            X, y, timestamps,
            test_ratio=train_cfg['test_split_ratio'],
            embargo_str=train_cfg.get('embargo', "0s")
        )
        
        test_timestamps = timestamps[-len(y_test):]
        return X_test, y_test, test_timestamps

    def _load_artifacts(self, artifacts):
        logging.info("Carregando artefatos...")
        # Resolve nomes dos arquivos baseado na pasta do metadata
        model_path = self.base_dir / Path(artifacts['model']).name
        scaler_X_path = self.base_dir / Path(artifacts['scaler_X']).name
        scaler_y_path = self.base_dir / Path(artifacts['scaler_y']).name

        return (
            joblib.load(model_path),
            joblib.load(scaler_X_path),
            joblib.load(scaler_y_path)
        )

    def _predict_and_inverse(self, model, scaler_X, scaler_y, X_test, y_test):
        logging.info("Gerando predições...")
        X_test_n = scaler_X.transform(X_test)
        y_pred_n = model.predict(X_test_n)
        
        y_pred_real = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()
        # y_test já está na escala real (foi separado antes de normalizar no treino)
        return y_pred_real, y_test

    def plot_comparison(self, y_true, y_pred, timestamps, title_suffix="", zoom_slice=None):
        df_plot = pd.DataFrame({'Real': y_true, 'Previsto': y_pred}, index=timestamps)
        
        if zoom_slice:
            df_plot = df_plot.iloc[-zoom_slice:]
        
        plt.figure(figsize=(14, 7))
        plt.plot(df_plot.index, df_plot['Real'], label='Real', color='blue', alpha=0.6)
        plt.plot(df_plot.index, df_plot['Previsto'], label='Previsto', color='red', linestyle='--', alpha=0.8)
        
        plt.title(f'Resultado do Modelo - {title_suffix}')
        plt.xlabel('Data/Hora')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- Bloco de Execução (Script Runner) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualizador de Resultados")
    parser.add_argument('--meta', type=str, required=True, help="Caminho do JSON de metadados")
    
    args = parser.parse_args()
    
    viz = ResultsVisualizer(args.meta)
    viz.run()