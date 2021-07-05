import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from sklearn.linear_model import LinearRegression

from datashift import datasets, plotting

from config import FIGURES_DIR

X2_NOISE = 0.5
X1_NOISE = 0.5
NONLINEAR_MODEL = "svc_rbf"
# NONLINEAR_MODEL = "svc_poly"
# NONLINEAR_MODEL = "poly"
LINEAR_MODEL = "svc_lin"
ROT = np.pi / 4
LINEAR_STYLE = "-"
NONLINEAR_STYLE = "--"
# COLORMAP = "inferno"
COLORMAP = "gray"
UNSHARE_MIDDLE_COL = True
# NONLINEAR_MODEL = "gradient_boosting"


def show_sample(sample, ax, cols=["x1", "x2"], colors="z", s=None, norm=None):
    ecols = ["lightgreen", "lightblue"]
    ecols = plotting.TAB_COLORS[2], plotting.TAB_COLORS[0]
    # ecols = ["k", "k"]
    for y, m in zip((0, 1), ("o", "o")):
        part = sample[sample["y"] == y]
        ss = None if s is None else s[sample["y"] == y] * 3000
        kwargs = {}
        if colors is not None:
            kwargs = {"c": part["z"]}
        ax.scatter(
            part[cols[0]].values,
            part[cols[1]].values,
            marker=m,
            edgecolors=ecols[y],
            s=ss,
            norm=norm,
            linewidth=1,
            cmap=COLORMAP,
            **kwargs
        )


def regress_out_z(sample, linreg=None):
    if linreg is None:
        linreg = LinearRegression().fit(
            sample["z"].values[:, None], sample.loc[:, ["x1", "x2"]].values
        )
    residuals = sample.loc[:, ["x1", "x2"]].values - linreg.predict(
        sample["z"].values[:, None]
    )
    sample["x1_residual"] = residuals[:, 0]
    sample["x2_residual"] = residuals[:, 1]
    return linreg


fig, ax = plt.subplots(
    2,
    3,
    figsize=(6, 4.5),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.05, "hspace": 0.02},
)

if UNSHARE_MIDDLE_COL:
    gx = ax[0, 0].get_shared_x_axes()
    gy = ax[0, 0].get_shared_y_axes()
    for g in gx, gy:
        g.remove(ax[0, 1])
        g.remove(ax[1, 1])
    ax[0, 1].get_shared_x_axes().join(ax[0, 1], ax[1, 1])
    ax[0, 1].get_shared_y_axes().join(ax[0, 1], ax[1, 1])

parabola0 = datasets.ParabolasGenerator(
    z_loc=1.5, z_scale=2.0, x2_noise=X2_NOISE, x1_noise=X1_NOISE, rot=ROT
)
df = parabola0.sample()
vmin, vmax = df["z"].min(), df["z"].max()
norm = Normalize(vmin, vmax)
fig1, ax1 = plt.subplots()
ax1.scatter(df["z"], df["x1"], c=df["y"])
fig1.savefig("/tmp/fig1.pdf")
linreg = regress_out_z(df)
show_sample(df, ax[0, 0], norm=norm)
nonlinear = plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[0, 0],
    NONLINEAR_MODEL,
    linestyles=NONLINEAR_STYLE,
)
linear = plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[0, 0],
    LINEAR_MODEL,
    color="k",
    linestyles=LINEAR_STYLE,
)
show_sample(df, ax[0, 1], cols=["x1_residual", "x2_residual"], norm=norm)
nonlinear_residual = plotting.show_separation(
    df.loc[:, ["x1_residual", "x2_residual"]].values,
    df["y"].values[:, None],
    ax[0, 1],
    NONLINEAR_MODEL,
    linestyles=NONLINEAR_STYLE,
)
linear_residual = plotting.show_separation(
    df.loc[:, ["x1_residual", "x2_residual"]].values,
    df["y"].values,
    ax[0, 1],
    LINEAR_MODEL,
    color="k",
    linestyles=LINEAR_STYLE,
)
parabola1 = datasets.ParabolasGenerator(
    z_loc=-0.8, z_scale=1.0, x2_noise=X2_NOISE, x1_noise=X1_NOISE, rot=ROT
)

weights = parabola1.score(df["z"].values) / parabola0.score(df["z"].values)
weights /= weights.sum()
show_sample(df, ax[0, 2], s=weights, colors="z", norm=norm)
linear_weighted = plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[0, 2],
    LINEAR_MODEL,
    color="k",
    linestyles=LINEAR_STYLE,
    sample_weight=weights,
)
nonlinear_weighted = plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[0, 2],
    NONLINEAR_MODEL,
    color="k",
    linestyles=NONLINEAR_STYLE,
    sample_weight=weights,
)
# linear_weighted_residuals = plotting.show_separation(
#     df.loc[:, ["x1_residual", "x2_residual"]].values,
#     df["y"].values,
#     ax[0, 1],
#     LINEAR_MODEL,
#     color="k",
#     linestyles="solid",
#     sample_weight=weights,
# )
df = parabola1.sample()
regress_out_z(df, linreg)
show_sample(df, ax[1, 0], norm=norm)
plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[1, 0],
    nonlinear,
    linestyles=NONLINEAR_STYLE,
)
plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[1, 0],
    linear,
    color="k",
    linestyles=LINEAR_STYLE,
)
plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[1, 2],
    linear_weighted,
    color="k",
    linestyles=LINEAR_STYLE,
    sample_weight=weights,
)
plotting.show_separation(
    df.loc[:, ["x1", "x2"]].values,
    df["y"].values,
    ax[1, 2],
    nonlinear_weighted,
    color="k",
    linestyles=NONLINEAR_STYLE,
    sample_weight=weights,
)
show_sample(df, ax[1, 1], cols=["x1_residual", "x2_residual"], norm=norm)
plotting.show_separation(
    df.loc[:, ["x1_residual", "x2_residual"]].values,
    df["y"].values,
    ax[1, 1],
    nonlinear_residual,
    linestyles=NONLINEAR_STYLE,
)
plotting.show_separation(
    df.loc[:, ["x1_residual", "x2_residual"]].values,
    df["y"].values,
    ax[1, 1],
    linear_residual,
    color="k",
    linestyles=LINEAR_STYLE,
)
# plotting.show_separation(
#     df.loc[:, ["x1_residual", "x2_residual"]].values,
#     df["y"].values,
#     ax[1, 1],
#     linear_weighted_residuals,
#     color="k",
#     linestyles="solid",
# )
show_sample(df, ax[1, 2], colors="z", norm=norm)
fontsize = 10
ax[0, 0].set_ylabel("Source data", fontsize=fontsize)
ax[0, 0].set_title("No correction", fontsize=fontsize)
ax[0, 2].set_title("Reweighting samples", fontsize=fontsize)
ax[0, 1].set_title("Regressing age out", fontsize=fontsize)
ax[1, 0].set_ylabel(
    "Target data:\nshifted age distribution ", fontsize=fontsize
)
for axis in ax.ravel():
    axis.set_xticks([])
    axis.set_yticks([])

arrow_ax = fig.add_axes((0.07, 0.1, 0.05, 0.8))
arrow_ax.set_facecolor("none")
arrow = FancyArrowPatch(
    (0.38, 0.61),
    (0.4, 0.49),
    transform=arrow_ax.transAxes,
    connectionstyle="arc3,rad=0.3",
    arrowstyle="simple,head_length=1.2, head_width=1., tail_width=0.2",
    mutation_scale=7,
    color="#505050",
)
arrow_ax.add_patch(arrow)
arrow_ax.set_xticks([])
arrow_ax.set_yticks([])
for sp in arrow_ax.spines.values():
    sp.set_visible(False)


xlim = ax[1, 2].get_xlim()
ylim = ax[1, 2].get_ylim()
ax[1, 2].plot([1000, 1000], [0, 0], linestyle="-", color="k")
ax[1, 2].plot([1000, 1000], [0, 0], linestyle="--", color="k")
ax[1, 2].set_xlim(*xlim)
ax[1, 2].set_ylim(*ylim)
ax[1, 2].legend(
    ["Simple (linear SVM)", "Flexible (SVM RBF)"],
    frameon=False,
    title="Predictive model",
    loc="lower left",
    bbox_to_anchor=(-3.7, 0.1)
)


cbar_ax = fig.add_axes((-.1, 0.5, 0.02, 0.1))
cbar_ax.set_xticks([])
cbar_ax.set_yticks([])
img = np.linspace(0, 1, 50)[:, None]
cbar_ax.imshow(img, aspect="auto", cmap=COLORMAP)
cbar_ax.text(0.0, -4, "Age", va="bottom", ha="center")

fig.savefig("/tmp/fig.pdf")
fig.savefig(str(FIGURES_DIR / "parabolas.pdf"), bbox_inches="tight",
transparent=True)

fig.savefig("/tmp/fig.svg")
fig.savefig(str(FIGURES_DIR / "parabolas.svg"), bbox_inches="tight",
transparent=True)
