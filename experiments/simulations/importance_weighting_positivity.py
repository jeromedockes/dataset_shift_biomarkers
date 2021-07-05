from matplotlib import pyplot as plt
import seaborn as sns

from config import FIGURES_DIR


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(2.3, .75))

axes[0].bar([0.0, 1.0], [0.9, 0.1])
axes[0].bar([0.0, 1.0], [0.5, 0.5], color="#00000000", edgecolor="k")
axes[1].bar([0.0, 1.0], [1.0, 0.0])
axes[1].text(1.0, 0.5, "??", ha="center", fontsize=15)
axes[0].arrow(
    0.0,
    0.87,
    0.0,
    -0.22,
    length_includes_head=True,
    head_width=0.07,
    color="k",
)
axes[0].arrow(
    1.0, 0.12, 0.0, 0.22, length_includes_head=True, head_width=0.07, color="k"
)

for ax in axes:
    ax.set_xticks([-.1, 1.1])
    ax.set_xticklabels([r"Men", r"Women"])
    ax.tick_params(axis="x", bottom=False, pad=0)
    # labels = ax.get_xticklabels()
    # labels[0].set_ha("right")
    # labels[1].set_ha("center")

sns.despine()

axes[0].set_title("OK")
axes[1].set_title("Hopeless")

fig.savefig(
    FIGURES_DIR / "importance_weighting_positivity.pdf", bbox_inches="tight"
)
