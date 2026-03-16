import os
import json
from typing import Any, Dict, List
import pandas as pd

INPUT_FOLDER = "json"                  # <- flat input folder
OUTPUT_FOLDER = "output_results/xlsx"  # <- new output folder
OUTPUT_XLSX = os.path.join(OUTPUT_FOLDER, "PATH_merged_row.xlsx")

def flatten_json(obj: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Recursively flattens a JSON-like object (dict) using `parent_child` keys.
    - Dicts -> recurse with joined keys.
    - Lists -> keep as JSON string (stable single column).
    - Scalars -> kept as-is.
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

    rows: List[Dict[str, Any]] = []
    headers: set = set()

    # Read & flatten each file
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

        flat["_source_file"] = fname
        rows.append(flat)
        headers.update(flat.keys())

    if not rows:
        print("No valid JSON files found.")
        return

    headers = ["_source_file"] + sorted(h for h in headers if h != "_source_file")

    normalized = []
    for r in rows:
        normalized.append({h: r.get(h, "") for h in headers})

    df = pd.DataFrame(normalized, columns=headers)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="merged")

    print(f"✅ Wrote {OUTPUT_XLSX} with {len(df)} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    main()
