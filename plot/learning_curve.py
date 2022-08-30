import os, time, csv
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# '/root/rjchen/workspace/outputs/SwinIR2/depth2_dim32_X2/record.csv'
def load_data(filename, fields):
    res = {k: [] for k in fields}
    with open(filename, 'r', newline='') as csvfile:
        # reader = csv.reader(csvfile, delimiter=',')
        # for row in reader:
        #     print(row)
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            for k in fields:
                res[k].append(eval(row[k]))
    return res

data = load_data('/root/rjchen/workspace/outputs/SwinIR2/depth2_dim32_X2/record.csv', ["Epoch", "Loss","Train","Valid"])
print(data)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(data["Epoch"], data["Valid"], label=f"depth")
plt.legend()
plt.grid()
plt.savefig("out.png")
plt.close()