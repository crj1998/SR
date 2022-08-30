import torch
import torch.nn as nn

weights = "/root/rjchen/workspace/outputs/SwinIR3dev/full_sandwich_X2/best.pth"
# weights = "/root/rjchen/workspace/outputs/SwinIR3dev/full_random_X2/best.pth"

state_dict = torch.load(weights)

for k, v in state_dict.items():
    print(k, v.shape)
import numpy as np
import matplotlib.pyplot as plt

# w = state_dict["layers.0.blocks.0.attn.k.weight"].cpu().numpy()
w = state_dict["layers.0.blocks.1.mlp.fc2.weight"].cpu().numpy()
bound = np.abs(w).max()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.matshow(w, vmin=-bound, vmax=bound, cmap="bwr")
ax.set_xticks(np.arange(0, w.shape[1]+16, 16), np.arange(0, w.shape[1]+16, 16))
ax.set_yticks(np.arange(0, w.shape[0]+16, 16), np.arange(0, w.shape[0]+16, 16))
plt.savefig("w.png")