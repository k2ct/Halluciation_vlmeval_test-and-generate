import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python inspect_predictions.py <xlsx_or_tsv_file>")
    sys.exit(1)

file_path = sys.argv[1]

if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
elif file_path.endswith(".tsv"):
    df = pd.read_csv(file_path, sep="\t")
else:
    raise ValueError("Only .xlsx or .tsv is supported")

print("Shape:", df.shape)
print("\nColumns:")
for i, c in enumerate(df.columns.tolist()):
    print(f"[{i}] {c}")

print("\nFirst 3 rows:")
print(df.head(3).to_dict(orient="records"))