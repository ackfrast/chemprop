import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

DATA = "furan_pool.csv"   # 若你還沒做 furan_pool，就改回 chemprop_data.csv
OUT  = "train_sim_mix.csv"

TARGET_SMILES = "OCCc1ccco1"

# 你想要的總量與比例
TOTAL_WANT = 20000
SIM_FRAC = 0.7

RADIUS = 2
NBITS = 2048
SEED_BG = 42

df = pd.read_csv(DATA)

# target fp
t_mol = Chem.MolFromSmiles(TARGET_SMILES)
assert t_mol is not None, "Target SMILES invalid"
t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, RADIUS, nBits=NBITS)

# compute similarity
sims = []
valid_rows = []
for idx, smi in enumerate(df["smiles"].astype(str)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS)
    sims.append(DataStructs.TanimotoSimilarity(fp, t_fp))
    valid_rows.append(idx)

dfv = df.iloc[valid_rows].copy()
dfv["sim"] = sims

# === 核心修正：自動縮小 TOTAL，避免 rest 不夠抽 ===
N = len(dfv)
if N == 0:
    raise RuntimeError("No valid molecules after RDKit parsing. Check DATA file.")

TOTAL = min(TOTAL_WANT, N)  # 不會超過資料量
n_sim = int(TOTAL * SIM_FRAC)
n_sim = min(n_sim, N)       # 安全
top_sim = dfv.sort_values("sim", ascending=False).head(n_sim)

rest = dfv.drop(top_sim.index)
n_bg = TOTAL - len(top_sim)

# 如果 rest 不夠抽背景，直接把背景縮到 rest 全部
if len(rest) < n_bg:
    n_bg = len(rest)

bg = rest.sample(n=n_bg, random_state=SEED_BG) if n_bg > 0 else rest.head(0)

train = pd.concat(
    [top_sim.drop(columns=["sim"]), bg.drop(columns=["sim"])],
    ignore_index=True
).sample(frac=1.0, random_state=SEED_BG).reset_index(drop=True)

train.to_csv(OUT, index=False)

print("DATA rows (valid):", N)
print("Wrote:", OUT)
print("TOTAL:", len(train))
print("SIM:", len(top_sim), "BG:", len(bg))
print("Top-5 similarities:", top_sim["sim"].head().to_list())
print(train.head())
