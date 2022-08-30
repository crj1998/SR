import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# profile params macs psnr
def load_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    PSNRs = []
    Params = []
    MACs = []

    for row in data["profile"]:
        PSNRs.append(row["psnr"])
        Params.append(row["params"]/10**3)
        MACs.append(row["macs"]/10**6)
    return PSNRs, Params, MACs

# PSNRs, Params, MACs = load_data("../outputs/cache/profile.json")
# aPSNRs, aParams, aMACs = load_data("../outputs/cache/anchor_profile.json")

# fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)

# axes[0].plot(Params, PSNRs, label="Vanilla")
# axes[0].plot(aParams, aPSNRs, label="Upsample")
# axes[0].set_xlabel("Params (k)")
# axes[0].set_ylabel("PSNR (dB)")

# axes[1].plot(MACs, PSNRs, label="Vanilla")
# axes[1].plot(aMACs, aPSNRs, label="Upsample")
# axes[1].set_xlabel("MACs (M)")
# axes[1].set_ylabel("PSNR (dB)")

# plt.legend()
# plt.tight_layout()
# plt.savefig("out.png")


def load_upsample_psnr(filepath):
    data = np.load(filepath)
    return data

data = load_upsample_psnr("../outputs/cache/valid_bicubic_psnr.npy")

print(data.mean(), data.std())
exit()
quartile = np.percentile(data, [30, 70])

fig, ax = plt.subplots(1, 1, dpi=160)
nums, bins, _ = ax.hist(data, bins=60, width=0.75, range=(20, 80), color="C1")
ax.set_xlim(18, 82)
ax.set_xlabel("PSNR (dB)")
ax.set_ylabel("Num (k)")
ax.set_yticks([2000, 4000, 6000, 8000, 10000, 12000])
ax.set_yticklabels(["2k", "4k", "6k", "8k", "10k", "12k"])
trans = ax.get_xaxis_transform()
x = np.linspace(bins.min()-5, bins.max()+5, 1000)
ax.fill_between(x, 0, 1, alpha=0.2, transform=trans, facecolor='green', where= x >= quartile[1], label="Easy (0~30%)")
ax.fill_between(x, 0, 1, alpha=0.2, transform=trans, facecolor='orange', where= (quartile[0] <= x) & (x <= quartile[1]), label="Medium (30~70%)")
ax.fill_between(x, 0, 1, alpha=0.2, transform=trans, facecolor='red', where= x <= quartile[0], label="Hard (70~100%)")

plt.legend()
plt.tight_layout()
plt.savefig("out.png", dpi=160)

