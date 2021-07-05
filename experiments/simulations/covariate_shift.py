from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from datashift import datasets
from datashift.plotting import show_fit, TAB_COLORS

from config import FIGURES_DIR


# def squared(x):
#     return x ** 2

squared = None

xs_mean, xs_std = 0.0, 2.0
xt_mean, xt_std = 1.5, 1.0
xs_dist = stats.norm(xs_mean, xs_std)
xt_dist = stats.norm(xt_mean, xt_std)
x_source, y_source = datasets.quadratic_additive_noise(
    xs_mean, xs_std, n_points=60
)
weights = xt_dist.pdf(x_source) / xs_dist.pdf(x_source)
# weights = weights ** .2
weighted_line = np.poly1d(np.polyfit(x_source, y_source, 1, w=weights))
weighted_poly = np.poly1d(np.polyfit(x_source, y_source, 4, w=weights))
poly_4 = np.poly1d(np.polyfit(x_source, y_source, 4))
x_target, y_target = datasets.quadratic_additive_noise(
    xt_mean, xt_std, n_points=30
)

fig, ax = plt.subplots(
    3,
    2,
    sharex=True,
    sharey=True,
    figsize=(4.2, 3.0),
    gridspec_kw={"height_ratios": [1.0, 1.0, 0.4], "wspace": .04, "hspace": .08},
)
line = show_fit(
    x_source, y_source, ax[0, 0], true_function=squared, line_color="k"
)
show_fit(
    x_source,
    y_source,
    ax[0, 1],
    weighted_line,
    true_function=squared,
    line_color="k",
)
x = np.linspace(-5, 5, 100)
ax[0, 1].plot(x, weighted_line(x), color="k", linestyle="-")
ax[1, 0].plot(x, poly_4(x), color="k", linestyle="--")
ax[1, 1].plot(x, weighted_poly(x), color="k", linestyle="--")
ax[0, 1].plot(x, weighted_poly(x), color="k", linestyle="--")
ax[0, 0].plot(x, poly_4(x), color="k", linestyle="--")
# poly_4 = show_fit(
#     x_source, y_source, ax[0, 0], degree=4, true_function=squared
# )
show_fit(
    x_target,
    y_target,
    ax[1, 0],
    poly=line,
    scatter_color=TAB_COLORS[6],
    line_color="k",
    true_function=squared,
)
show_fit(
    x_target,
    y_target,
    ax[1, 1],
    poly=weighted_line,
    degree=4,
    scatter_color=TAB_COLORS[6],
    line_color="k",
    true_function=squared,
)
x = np.linspace(-10, 10, 200)
ys = xs_dist.pdf(x)
yt = xt_dist.pdf(x)
ax[2, 1].plot(x, yt / ys * 8, color="k")
ax[2, 1].set_xlabel("Importance weighting function")
# ax[2, 1].plot(x, y, color=TAB_COLORS[4])

for sp in ax[2, 0].spines.values():
    sp.set_visible(False)

for ax_ in ax.ravel():
    ax_.set_xticks([])
    ax_.set_yticks([])

ax[0, 0].set_ylabel("Source dataset")
ax[1, 0].set_ylabel("Target dataset")
ax[0, 0].set_title("Uniform weights")
ax[0, 1].set_title("Importance weighting")
ax[2, 0].plot([0, 0], [0, 0], linestyle="-", color="k")
ax[2, 0].plot([0, 0], [0, 0], linestyle="--", color="k")
ax[0, 0].set_xlim(x_source.min() - .5, x_source.max() + .5)
ax[0, 0].set_ylim(y_source.min() - 1, y_source.max() + 1)
ax[2, 0].legend(
    ["Degree 1", "Degree 4"], frameon=False, title="Best fit on source data",
    loc=(0, -.65)
)
fig.savefig(str(FIGURES_DIR / "covariate_shift.pdf"), bbox_inches="tight")
