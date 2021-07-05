import numpy as np
from matplotlib import pyplot as plt

from config import FIGURES_DIR

N = 1000


def make_data(n=N):
    rng = np.random.RandomState(0)
    Y = rng.randn(n)
    # L = rng.binomial(1, .2, 100)
    L = rng.randn(n) * 1.3
    X = Y + L + rng.randn(n) / 10
    return X, Y, L


def filter_data(V, n=N // 10, q=0.8):
    rng = np.random.RandomState(0)
    p = np.exp(3 * V)
    p[p < np.quantile(p, 0.5)] = 0
    p -= p.min()
    # p[V < np.quantile(V, .5)] = 0
    # p = np.exp(p)
    # p = V > V.mean()
    p = p / p.sum()
    return rng.choice(np.arange(len(V)), size=n, replace=False, p=p)


X, Y, L = make_data()
line = np.poly1d(np.polyfit(X, Y, 1))
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(4, 8))
fig, ax = [], []
kwargs = {}
for i in range(3):
    fig_ = plt.figure(figsize=(2, 1.5))
    ax_ = fig_.add_subplot(111, **kwargs)
    fig.append(fig_)
    ax.append(ax_)
    kwargs = {"sharex": ax_, "sharey": ax_}

node_size = 15
g_alpha = .4

ax[0].scatter(X, Y, s=node_size, alpha=g_alpha)
ax[1].scatter(X, Y, s=node_size, alpha=g_alpha)
ax[2].scatter(X, Y, s=node_size, alpha=g_alpha)

filter_Z = filter_data(np.random.RandomState(1).randn(len(X)))
line_Z = np.poly1d(np.polyfit(X[filter_Z], Y[filter_Z], 1))
o_alpha = .7
ax[0].scatter(X[filter_Z], Y[filter_Z], s=node_size, alpha=o_alpha)

filter_L = filter_data(L)
line_L = np.poly1d(np.polyfit(X[filter_L], Y[filter_L], 1))
ax[1].scatter(X[filter_L], Y[filter_L], s=node_size, alpha=o_alpha)

filter_X = filter_data(X)
line_X = np.poly1d(np.polyfit(X[filter_X], Y[filter_X], 1))
ax[2].scatter(X[filter_X], Y[filter_X], s=node_size, alpha=o_alpha)

xr = X.min(), X.max()
ax[0].plot(xr, line(xr), color="k", linestyle="--")
ax[0].plot(xr, line_Z(xr), color="k")
ax[1].plot(xr, line(xr), color="k", linestyle="--")
ax[1].plot(xr, line_L(xr), color="k")
ax[2].plot(xr, line(xr), color="k", linestyle="--")
ax[2].plot(xr, line_X(xr), color="k")

ylim = Y.min() - .7, Y.max() + .7
ax[0].set_ylim(*ylim)

# ax[0].set_title("General population: X := Y + M + ϵ")
ax[0].set_title("Uniform selection")
# ax[1].set_title("Selection based on $M$, a cause of $X$")
ax[1].set_title("Selection based on $M$")
ax[2].set_title("Selection based on $X$")

for ax_ in ax:
    ax_.set_xticks([])
    ax_.set_yticks([])
    ax_.set_ylabel("$Y$")

ax[1].set_xlabel("$X$")

for i in range(3):
    fig[i].savefig(
        str(FIGURES_DIR / f"selection_bias_{i + 1}.pdf"), bbox_inches="tight"
    )
