import csv
from math import comb

csv_path = r"benchmarks\full_suite\full_benchmarks_20260625_211350.csv"

bypass = {}
pipeline = {}

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["problem_id"]
        acc = float(row["accuracy"])
        method = row["method"]
        if row["dataset"] != "gsm8k":
            continue
        if method == "CPPTAI":
            bypass[pid] = acc
        elif method == "CPPTAI_pipeline":
            pipeline[pid] = acc

common = sorted(set(bypass) & set(pipeline))
n = len(common)

b01 = sum(1 for pid in common if bypass[pid] == 0 and pipeline[pid] == 1)
b10 = sum(1 for pid in common if bypass[pid] == 1 and pipeline[pid] == 0)
b00 = sum(1 for pid in common if bypass[pid] == 0 and pipeline[pid] == 0)
b11 = sum(1 for pid in common if bypass[pid] == 1 and pipeline[pid] == 1)

disc = b01 + b10

def mcnemar_exact_p(b01, b10):
    n_disc = b01 + b10
    if n_disc == 0:
        return 1.0
    k_obs = min(b01, b10)
    p = sum(comb(n_disc, k) * (0.5 ** n_disc) for k in range(k_obs + 1))
    return min(1.0, 2 * p)

p_val = mcnemar_exact_p(b01, b10)

bypass_acc = sum(bypass[pid] for pid in common) / n
pipe_acc   = sum(pipeline[pid] for pid in common) / n

print(f"Coppie paired: {n}")
print(f"  b11 (entrambi ok):     {b11}")
print(f"  b00 (entrambi sbag):   {b00}")
print(f"  b01 (bypass0, pipe1):  {b01}  <- pipeline recupera")
print(f"  b10 (bypass1, pipe0):  {b10}  <- pipeline perde")
print(f"  Discordanti totali:    {disc}")
print(f"  McNemar exact p:       {p_val:.8f}")
print(f"\n  Bypass acc:   {bypass_acc:.4f} ({bypass_acc*100:.1f}%)")
print(f"  Pipeline acc: {pipe_acc:.4f} ({pipe_acc*100:.1f}%)")
print(f"  Delta:        +{(pipe_acc - bypass_acc)*100:.1f} pp")
