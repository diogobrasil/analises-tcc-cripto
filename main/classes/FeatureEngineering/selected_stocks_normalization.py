import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pathlib import Path

# 1. Get the current script's directory
script_dir = Path(__file__).resolve().parent

# 2. Define the relative path to the script's directory
base_path = script_dir / "../../datasets/b3_dados/processed"

# 3. Check if the directory exists
if not base_path.exists():
    raise FileNotFoundError(f"Directory not found: {base_path}")

# 4. Path to the input file
file_path = base_path / "acoes_concat.csv"

# 5. Check if the file exists
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

# 6. Load the data
stocks_data = pd.read_csv(file_path)

# 7. Filter for the selected stocks
selected_stocks = ["VALE3", "BRAP3", "BBAS3", "ITUB4", "ELET3", "EQTL3"]
filtered_data = stocks_data[selected_stocks]

# 8. Apply Min-Max Normalization
min_max_scaler = MinMaxScaler()
min_max_normalized = pd.DataFrame(min_max_scaler.fit_transform(filtered_data), columns=selected_stocks)

# 9. Apply Z-Score Normalization
z_score_scaler = StandardScaler()
z_score_normalized = pd.DataFrame(z_score_scaler.fit_transform(filtered_data), columns=selected_stocks)

# 10. Apply Robust Scaler Normalization
robust_scaler = RobustScaler()
robust_normalized = pd.DataFrame(robust_scaler.fit_transform(filtered_data), columns=selected_stocks)

# 11. Combine all normalized data for comparison
normalized_data = pd.concat([
    min_max_normalized.add_suffix('_min_max'),
    z_score_normalized.add_suffix('_z_score'),
    robust_normalized.add_suffix('_robust')
], axis=1)

# 12. Path to save the output file
output_path = base_path / "selected_stocks_normalized.csv"

# 13. Create the output directory if it doesn't exist
output_path.parent.mkdir(parents=True, exist_ok=True)

# 14. Save the normalized data to a CSV file
normalized_data.to_csv(output_path, index=False)

print(f"Normalized data saved to: {output_path}")