import matplotlib.pyplot as plt
import numpy as np

attn = np.random.dirichlet(np.ones(8), size=8)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(attn, cmap="Blues")
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("attention_matrix.png", dpi=150, bbox_inches="tight")
print("saved")
