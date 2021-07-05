import numpy as np
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from sklearn.base import clone

TAB_COLORS = [cm.tab20(i) for i in range(20)]
ESTIMATORS = {
    "logistic": LogisticRegression(),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=500, random_state=0
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(100, 100, 100, 100), random_state=0
    ),
    "svc_rbf": SVC(probability=True, random_state=0, C=.2),
    # "svc_rbf": SVC(probability=True, random_state=0, C=10),
    "svc_lin": SVC(kernel="linear", probability=True, random_state=0),
    "svc_poly": SVC(kernel="poly", degree=2, probability=True, random_state=0),
    "random_forest": RandomForestClassifier(random_state=0),
    "poly": Pipeline(
        [("poly", PolynomialFeatures()), ("reg", LogisticRegression())]
    ),
}


def show_separation(
    X,
    y,
    ax,
    estimator="logistic",
    weights=None,
    color="k",
    linestyles="solid",
    sample_weight=None,
):
    if isinstance(estimator, str):
        kwargs = {}
        if sample_weight is not None:
            if estimator == "poly":
                kwargs = {"reg__sample_weight": sample_weight}
            else:
                kwargs = {"sample_weight": sample_weight}
        estimator = clone(ESTIMATORS[estimator])
        estimator.fit(X, y, **kwargs)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    # xmin, xmax = X[:, 0].min(), X[:, 0].max()
    # ymin, ymax = X[:, 1].min(), X[:, 1].max()
    step = 0.02
    # step = .1
    g = np.mgrid[xmin:xmax:step, ymin:ymax:step]
    pred = estimator.predict_proba(np.c_[g[0].ravel(), g[1].ravel()])[
        :, 1
    ].reshape(g[0].shape)
    ax.contour(
        g[0],
        g[1],
        pred,
        levels=[0.5],
        zorder=1,
        colors=[color],
        linestyles=linestyles,
    )
    return estimator


def show_fit(
    x,
    y,
    ax,
    poly="fit",
    degree=1,
    weights=None,
    xlim=(-5, 5),
    ylim=(-3, 15),
    scatter_color=TAB_COLORS[0],
    line_color=TAB_COLORS[2],
    true_function=None,
    linestyle="-"
):
    if weights is None:
        weights = np.ones(len(x))
    if poly == "fit":
        poly = np.poly1d(np.polyfit(x, y, degree, w=weights))
    ax.scatter(x, y, color=scatter_color, alpha=0.7, s=11.0, zorder=5)
    x = np.linspace(xlim[0] - 10, xlim[1] + 10, 200)
    ax.plot(x, poly(x), color=line_color, linestyle=linestyle)
    if true_function is not None:
        ax.plot(x, true_function(x), color="k", linestyle="--", zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return poly


def show_sample(sample, ax, cols=["x1", "x2"], colors="z", s=None, norm=None,
                cmap="inferno"):

    ecols = ["lightgreen", "lightblue"]
    ecols = TAB_COLORS[0], TAB_COLORS[4]
    # ecols = ["k", "k"]
    for y, m in zip((0, 1), ("o", "^")):
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
            linewidth=.2,
            cmap=cmap,
            **kwargs
        )
