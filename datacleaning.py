from pathlib import Path
import pandas as pd

INPUT = Path("Dataset") / "STS Art Dataset.xlsx"
OUTPUT = Path("Dataset") / "STS Art Dataset - Cleaned.xlsx"

def trim_text_cell(x):
    if isinstance(x, str):
        if len(x) <= 37:
            return ""
        return x[10: len(x) - 27]
    return x

def clean_workbook(input_path: Path, output_path: Path):
    sheets = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
    cleaned = {}
    summary = []

    for name, df in sheets.items():
        df_before = df.copy()
        df_clean = df_before.applymap(trim_text_cell)
        cleaned[name] = df_clean

        text_cols = df_before.select_dtypes(include=['object', 'string']).columns
        if len(text_cols) > 0:
            before_text = df_before[text_cols].fillna("").astype(str)
            after_text  = df_clean[text_cols].fillna("").astype(str)
            changed = (before_text != after_text).sum().sum()
            total_text_cells = before_text.size - (before_text == "").sum().sum()  # non-empty text cells before
        else:
            changed = 0
            total_text_cells = 0

        summary.append({
            "sheet": name,
            "rows": len(df_before),
            "cols": df_before.shape[1],
            "text_columns_processed": len(text_cols),
            "total_text_cells_before": int(total_text_cells),
            "changed_text_cells": int(changed)
        })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df_clean in cleaned.items():
            df_clean.to_excel(writer, sheet_name=name, index=False)

    print("Cleaning completed.")
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}\n")
    print("Per-sheet summary:")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT.resolve()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clean_workbook(INPUT, OUTPUT)
