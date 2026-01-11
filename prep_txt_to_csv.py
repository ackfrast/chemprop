import sys
import pandas as pd

in_path = sys.argv[1] if len(sys.argv) > 1 else "ex1_ans.txt"
out_path = sys.argv[2] if len(sys.argv) > 2 else "chemprop_data.csv"

rows = []
with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        # 可能的格式：
        # (1) SMILES value
        # (2) idx SMILES value   (如果檔案裡真的有行號)
        if len(parts) < 2:
            continue

        if parts[0].isdigit() and len(parts) >= 3:
            smiles = parts[1]
            val = parts[2]
        else:
            smiles = parts[0]
            val = parts[1]

        try:
            y = float(val)
        except ValueError:
            continue

        rows.append((smiles, y))

df = pd.DataFrame(rows, columns=["smiles", "target"])
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with {len(df)} rows")
print(df.head())

