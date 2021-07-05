import sys
import itertools
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm


FIGURES_DIR = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "ukbiobank"
    / Path(sys.argv[0]).stem
)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


TAB_COLORS = [cm.tab10(i) for i in range(10)]
TAB_COLORS = [cm.Pastel2(i) for i in range(10)]
TAB_COLORS = [cm.Pastel1(i) for i in range(10)]
TAB_COLORS = [cm.Set2(i) for i in range(10)]

data_dir = Path(__file__).parent / "data"

score_files = data_dir.glob("scores_*.csv")
scores = []
for sf in score_files:
    scores.append(pd.read_csv(str(sf)))

df = pd.concat(scores, ignore_index=False)
df.rename(columns={df.columns[0]: "split_index"}, inplace=True)


def estimator_name(model, exponent, regress_out):
    if exponent != 0 and regress_out:
        return ""
    if regress_out:
        strategy = " + regress-out"
    elif exponent != 0:
        strategy = " + reweighting"
    else:
        strategy = ""
    return f"{model.upper()}{strategy}"


df["estimator"] = [
    estimator_name(e, x, r)
    for (e, x, r) in df.loc[
        :, ["model", "weight_exponent", "regress_out_age"]
    ].values
]
df = df[df["estimator"] != ""]
df.set_index(["split_index", "estimator"], inplace=True)
print(df.head())
df = (
    df.drop(columns=["model"])
    .stack()
    .reset_index()
    .rename(columns={"level_2": "task", 0: "auc"})
)

task_order = "young_young old_young old_old young_old".split()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 3),
    gridspec_kw=dict(wspace=0.03, hspace=0.05, width_ratios=[5, 6]),
)
axes = axes.ravel()
kwargs = {
    "width": 0.6,
    "x": "auc",
    "y": "task",
    "hue": "estimator",
    "saturation": 0.9,
}
for ax, (tasks, (estimator, palette)) in zip(
    axes,
    itertools.product(
        [task_order[:2], task_order[2:]],
        [("SVC", TAB_COLORS[:3]), ("GB", TAB_COLORS[3:])],
    ),
):
    ax_df = df[
        df["estimator"].str.startswith(estimator) & df["task"].isin(tasks)
    ]
    sns.boxplot(
        data=ax_df,
        ax=ax,
        order=tasks,
        palette=palette,
        hue_order=[
            f"{estimator}",
            f"{estimator} + reweighting",
            f"{estimator} + regress-out",
        ],
        **kwargs,
    )

axes[2].set_xlabel("AUC for linear SVC")
axes[3].set_xlabel("AUC for Gradient Boosting")


def add_bands(ax, color="gray", alpha=0.1):
    for ytick in ax.get_yticks()[::2]:
        ax.axhspan(
            ytick - 0.5, ytick + 0.5, zorder=0, color=color, alpha=alpha
        )


def format_auc(value, pos):
    return f"{value:.2}"[1:]


for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(-0.02, 1.05),
        handlelength=1.
    )
    add_bands(ax)
    ax.set_ylim((1.5, -0.5))
    ax.set_ylabel("")


for ax in axes[1::2]:
    ax.set_yticks([])
    ax.set_xlim((0.76, 0.82))

for ax in axes[::2]:
    ax.set_xlim((0.66, 0.71))
    ax.set_yticklabels(
        [tl.get_text().replace("_", " → ") for tl in ax.get_yticklabels()]
    )
for ax in axes[:2]:
    ax.set_xticks([])
    ax.set_xlabel("")
for ax in axes[2:]:
    ax.get_legend().remove()
    ax.xaxis.set_major_formatter(format_auc)

axes[3].xaxis.get_ticklabels()[0].set_ha("left")
axes[2].xaxis.get_ticklabels()[-1].set_ha("right")

fig.suptitle(
    "Predicting smoking status in the UKBiobank "
    "(10-fold CV, n train = 90K, n test = 9K)"
)
fig.savefig(
    str(FIGURES_DIR / "ukb_smoking_prediction.pdf"), bbox_inches="tight"
)
