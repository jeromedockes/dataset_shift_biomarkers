import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

from config import FIGURES_DIR
from datashift.plotting import TAB_COLORS

CLUSTER_STD = 1.5

fig, ax = plt.subplots(
    2,
    1,
    sharex="all",
    sharey="all",
    gridspec_kw={"hspace": 0.2},
    figsize=(1.8, 3.5),
)
c1, c2 = TAB_COLORS[12], TAB_COLORS[14]

ALPHA = 0.5
x, y = make_blobs(
    shuffle=False,
    cluster_std=CLUSTER_STD,
    n_samples=300,
    centers=2,
    random_state=0,
)
x_, y_ = x[70:-70], y[70:-70]
naive_bayes = GaussianNB().fit(x, y)
priors = np.asarray([10, 150])
priors = priors / priors.sum()
naive_bayes_uniform = GaussianNB(priors=priors).fit(x, y)

marker_size = 15
ax[0].scatter(
    *x_[y_ == 0].T,
    color=c1,
    edgecolor=c1,
    marker="o",
    alpha=ALPHA,
    s=marker_size,
)
ax[0].scatter(
    *x_[y_ == 1].T,
    color=c2,
    edgecolor=c2,
    marker="^",
    alpha=ALPHA,
    s=marker_size,
)

g = np.mgrid[:100, :100] * 0.2 - 4.5
z = naive_bayes.predict_proba(np.c_[g[0].ravel(), g[1].ravel()])[:, 1].reshape(
    g[0].shape
)
ax[0].contour(g[0], g[1], z, levels=[0.5], colors=["k"], linestyles="-")
ax[1].contour(g[0], g[1], z, levels=[0.5], colors=["k"], linestyles="-")

z = naive_bayes_uniform.predict_proba(np.c_[g[0].ravel(), g[1].ravel()])[
    :, 1
].reshape(g[0].shape)
ax[0].contour(g[0], g[1], z, levels=[0.5], colors=["k"], linestyles="--")
ax[1].contour(g[0], g[1], z, levels=[0.5], colors=["k"], linestyles="--")

x_, y_ = x[140:], y[140:]
ax[1].scatter(
    *x_[y_ == 0].T,
    color=c1,
    edgecolor=c1,
    marker="o",
    alpha=ALPHA,
    s=marker_size,
)
ax[1].scatter(
    *x_[y_ == 1].T,
    color=c2,
    edgecolor=c2,
    marker="^",
    alpha=ALPHA,
    s=marker_size,
)


ax[0].set_title("Balanced dataset")
ax[1].set_title("Target dataset")

ax[1].plot([1000, 1000], [0, 0], linestyle="-", color="k")
ax[1].plot([1000, 1000], [0, 0], linestyle="--", color="k")
ax[1].legend(
    ["Original fit", "Corrected for label shift"],
    frameon=False,
    title="Decision boundary",
    loc=(0, -0.55),
)

ax[0].set_xlim(x[:, 0].min() - 0.2, x[:, 0].max() + 0.2)
ax[0].set_ylim(x[:, 1].min() - 0.2, x[:, 1].max() + 0.2)

for ax_ in ax:
    # ax_.set_aspect(.7)
    ax_.set_xticks([])
    ax_.set_yticks([])

fig.savefig(str(FIGURES_DIR / "label_shift.pdf"), bbox_inches="tight")
