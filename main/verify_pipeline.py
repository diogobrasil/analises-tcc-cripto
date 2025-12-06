import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from classes.neural_networks.training.train_linear_regression import run_training_pipeline
from classes.visualization.results_visualizer import ResultsVisualizer

def create_dummy_data(filepath):
    dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
    data = {
        'date': dates,
        'open': np.random.rand(200) * 100,
        'high': np.random.rand(200) * 100 + 5,
        'low': np.random.rand(200) * 100 - 5,
        'close': np.random.rand(200) * 100,
        'volume': np.random.randint(100, 1000, 200)
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Dummy data created at {filepath}")

def run_verification():
    base_dir = Path("temp_verification")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()

    data_path = base_dir / "dummy_data.csv"
    model_dir = base_dir / "models"
    
    create_dummy_data(data_path)

    config = {
        "data": {
            "filepath": str(data_path),
            "date_col": "date",
            "target_col": "close",
            "timezone": "UTC"
        },
        "training": {
            "window_size": 5,
            "test_split_ratio": 0.2,
            "filter_cross_day": False,
            "use_returns": True # Test with returns to check reconstruction
        },
        "output": {
            "model_dir": str(model_dir),
            "version_tag": "v1"
        }
    }

    print("Running pipeline...")
    try:
        run_training_pipeline(config)
        print("Pipeline execution successful!")
        
        # Check artifacts
        metadata_path = model_dir / f"close_metadata_v1.json"
        
        if metadata_path.exists():
            print("Metadata found. Running Visualizer...")
            # We can't easily check the plot in headless, but we can check if it runs without error
            # We will mock plt.show to avoid blocking
            import matplotlib.pyplot as plt
            plt.show = lambda: None
            
            viz = ResultsVisualizer(str(metadata_path))
            viz.run()
            print("Visualizer ran successfully!")
        else:
            print("Metadata MISSING!")
                
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if base_dir.exists():
            shutil.rmtree(base_dir)
            print("Cleanup done.")

if __name__ == "__main__":
    run_verification()
