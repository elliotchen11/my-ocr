import os
import json
from typing import Any, Dict, List
import pandas as pd

INPUT_FOLDER = "json"                  # flat input folder
OUTPUT_FOLDER = "output_results/xlsx"  # output folder
OUTPUT_XLSX = os.path.join(OUTPUT_FOLDER, "PATH_merged_col.xlsx")
SHEET_NAME = "by_column"

def flatten_json(obj: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Recursively flatten a JSON-like object using head_subhead keys.
    - Dicts: recurse with joined keys.
    - Lists: stored as JSON strings.
    - Scalars: kept as-is.
    """
    items: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_json(v, new_key, sep=sep))
            elif isinstance(v, list):
                try:
                    items[new_key] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    items[new_key] = str(v)
            else:
                items[new_key] = v
    else:
        items[parent_key or "value"] = obj

    return items

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Read & flatten each file
    file_columns: Dict[str, Dict[str, Any]] = {}
    all_keys: set = set()

    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not fname.lower().endswith(".json"):
            continue

        fpath = os.path.join(INPUT_FOLDER, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid JSON: {fname} ({e})")
            continue

        if isinstance(data, list):
            flat = {"_root_list": json.dumps(data, ensure_ascii=False)}
        elif isinstance(data, dict):
            flat = flatten_json(data)
        else:
            flat = {"_root_value": data}

        file_columns[fname] = flat
        all_keys.update(flat.keys())

    if not file_columns:
        print("No valid JSON files found.")
        return

    # Build a DataFrame with keys as rows and files as columns
    keys_sorted = sorted(all_keys)
    table: Dict[str, List[Any]] = {"_key": keys_sorted}

    for fname, flat in file_columns.items():
        col_vals = [flat.get(k, "") for k in keys_sorted]
        table[fname] = col_vals

    df = pd.DataFrame(table, columns=["_key"] + sorted(file_columns.keys()))

    # Write to a single Excel file (one sheet)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=SHEET_NAME)

    print(f"✅ Wrote {OUTPUT_XLSX} [{SHEET_NAME}] with {len(df)} rows (keys) and {len(df.columns)-1} JSON columns.")

if __name__ == "__main__":
    main()
