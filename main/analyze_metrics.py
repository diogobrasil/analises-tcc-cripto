import argparse
import json
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

# Adiciona o diretório atual ao path
sys.path.append(str(Path.cwd()))

from main.classes.visualization.results_visualizer import ResultsVisualizer

def analyze_model(metadata_path):
    print(f"--- ANALISANDO MODELO: {Path(metadata_path).name} ---")
    
    viz = ResultsVisualizer(metadata_path)
    meta = viz.metadata
    config = meta['config']
    
    # Carrega modelo
    model, scaler_y = viz._load_artifacts(meta['artifact_paths'])
    
    # Recria dados
    print("Recriando dados de teste...")
    X_test, y_test, _, _ = viz._recreate_test_data(config)
    
    # Predição
    y_pred_n = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_n.reshape(-1, 1)).ravel()
    
    # --- HIT RATIO ---
    dir_real = np.where(y_test > 0, 1, 0)
    dir_pred = np.where(y_pred > 0, 1, 0)
    
    acc = accuracy_score(dir_real, dir_pred)
    baseline = max(dir_real.mean(), 1 - dir_real.mean())
    
    print(f"\n{'='*50}")
    print(f"🎯 ASSERTIVIDADE (HIT RATIO)")
    print(f"{'='*50}")
    print(f"Modelo:   {acc:.2%}")
    print(f"Baseline: {baseline:.2%}")
    
    if acc > 0.5:
        print("✅ Modelo tem viés preditivo (>50%)")
    else:
        print("❌ Modelo aleatório ou pior (<50%)")

    # --- FEATURE IMPORTANCE (UNIVERSAL) ---
    print(f"\n{'='*50}")
    print(f"🧠 O CÉREBRO DO MODELO")
    print(f"{'='*50}")
    
    try:
        # Tenta recuperar nomes das features do metadata
        feature_names = meta.get('training_info', {}).get('feature_names', None)
        
        # Estratégia para extrair importância dependendo do modelo
        importances = None
        
        # 1. Tenta XGBoost
        if hasattr(model, 'named_steps') and 'xgbregressor' in model.named_steps:
            print("Detectado: XGBoost")
            xgb_model = model.named_steps['xgbregressor']
            importances = xgb_model.feature_importances_
            
        # 2. Tenta Ridge/Linear
        elif hasattr(model, 'named_steps') and 'ridge' in model.named_steps:
            print("Detectado: Ridge Regression")
            ridge_model = model.named_steps['ridge']
            importances = np.abs(ridge_model.coef_) # Pega magnitude absoluta
            
        else:
            print("Tipo de modelo não identificado automaticamente no pipeline.")

        # Se conseguiu extrair, mostra
        if importances is not None:
            if feature_names is None or len(feature_names) != len(importances):
                print("Aviso: Nomes das features não batem com array de importância. Usando índices.")
                feature_names = [f"Feat_{i}" for i in range(len(importances))]
            
            df_imp = pd.DataFrame({'Feature': feature_names, 'Importancia': importances})
            df_imp = df_imp.sort_values(by='Importancia', ascending=False).head(15)
            
            print(f"{'FEATURE':<30} | {'IMPORTÂNCIA'}")
            print("-" * 45)
            for _, row in df_imp.iterrows():
                print(f"{row['Feature']:<30} | {row['Importancia']:.5f}")
                
    except Exception as e:
        print(f"Erro ao analisar features: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', type=str, required=True)
    args = parser.parse_args()
    analyze_model(args.meta)