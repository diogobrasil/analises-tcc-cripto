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

# 7. Ensure the date column exists and sort by date
if 'Date' not in stocks_data.columns:
    raise KeyError("The input data does not contain a 'Date' column.")
stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])  # Ensure the date is in datetime format
stocks_data = stocks_data.sort_values(by='Date')  # Sort by date

# 8. Filter for the selected stocks along with the Date column
selected_stocks = ["Date", "ITUB4", "BBAS3", "CYRE3", "TEND3", "DIRR3", "ELET3", "EQTL3", "CMIG4", "PETR3", "VALE3", "BRAP3"]

filtered_data = stocks_data[selected_stocks]

# 9. Save the Date column for later use
dates = filtered_data['Date']

# 10. Drop the Date column before normalization
data_to_normalize = filtered_data.drop(columns=['Date'])

# 11. Apply Min-Max Normalization
min_max_scaler = MinMaxScaler()
min_max_normalized = pd.DataFrame(min_max_scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns)

# 12. Apply Z-Score Normalization
z_score_scaler = StandardScaler()
z_score_normalized = pd.DataFrame(z_score_scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns)

# 13. Apply Robust Scaler Normalization
robust_scaler = RobustScaler()
robust_normalized = pd.DataFrame(robust_scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns)

# 14. Combine all normalized data for comparison and re-add the Date column
normalized_data = pd.concat([
    dates.reset_index(drop=True),
    min_max_normalized.add_suffix('_min_max'),
    z_score_normalized.add_suffix('_z_score'),
    robust_normalized.add_suffix('_robust')
], axis=1)

# 15. Path to save the output file
output_path = base_path / "selected_stocks_normalized.csv"

# 16. Create the output directory if it doesn't exist
output_path.parent.mkdir(parents=True, exist_ok=True)

# 17. Save the normalized data to a CSV file
normalized_data.to_csv(output_path, index=False)

print(f"Normalized data with date saved to: {output_path}")
