
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

matrix = df.pivot_table(
    index="target_tokens_approx",
    columns="depth_pct",
    values="correct",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    matrix.values,
    vmin=0,
    vmax=1,
    cmap="RdYlGn",
    aspect="auto"
)

ax.set_xticks(range(len(matrix.columns)))
ax.set_xticklabels(matrix.columns)
ax.set_yticks(range(len(matrix.index)))
ax.set_yticklabels(matrix.index)

ax.set_xlabel("Needle depth (%)")
ax.set_ylabel("Approximate target tokens")
ax.set_title("LLM Context Retrieval Accuracy")

fig.colorbar(im, ax=ax, label="Accuracy")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=160)
plt.show()
