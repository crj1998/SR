

import numpy as np
import matplotlib.pyplot as plt

import pickle

plt.style.use("ggplot")
datapath = "/root/rjchen/workspace/outputs/cache/super_width_upsample.pkl"
with open(datapath, "rb") as f:
    data = pickle.load(f)

COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "olive", "cyan"]
i = 0
plt.figure(figsize=(5, 5), dpi=200)
plt.plot(np.linspace(15, 80, 500), np.linspace(15, 80, 500), color="gray", ls="--")
for k, v in data.items():
    print(k, np.mean(v))
    # if k in ["7", "14", "28", "56", "112"]:
    if k != "baseline":
        plt.scatter(data["baseline"], v, label=f"d={k}", marker="o", s=5, alpha=0.3, color=COLORS[i])
        i += 1
plt.xlabel("BICUBIC (dB)")
plt.ylabel("FRSCNN (dB)")
plt.legend()
plt.savefig("out.png", dpi=200)


"""
python plot/width.py
"""