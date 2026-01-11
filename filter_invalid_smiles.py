import sys
import pandas as pd
from rdkit import Chem

in_path  = sys.argv[1]
out_path = sys.argv[2]
bad_path = sys.argv[3] if len(sys.argv) > 3 else None

df = pd.read_csv(in_path)

good_rows = []
bad_rows = []

for i, (smi, y) in enumerate(zip(df["smiles"], df["target"])):
    smi = str(smi).strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        bad_rows.append((i, smi, y))
    else:
        good_rows.append((smi, y))

out = pd.DataFrame(good_rows, columns=["smiles", "target"])
out.to_csv(out_path, index=False)

print(f"{in_path}: total={len(df)}  valid={len(out)}  invalid={len(bad_rows)}")
print("First 5 invalid examples:")
for row in bad_rows[:5]:
    print(row)

if bad_path:
    pd.DataFrame(bad_rows, columns=["row", "smiles", "target"]).to_csv(bad_path, index=False)
    print("Wrote invalid list:", bad_path)

