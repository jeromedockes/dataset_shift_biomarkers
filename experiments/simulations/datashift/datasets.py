import numpy as np
from scipy.stats import multivariate_normal, norm
import pandas as pd
from sklearn.utils import check_random_state


def quadratic_additive_noise(
    x_mean=0, x_std=1.0, n_points=100, noise_level=1.0, random_state=0
):
    """y = x**2 + e"""
    rng = check_random_state(random_state)
    x = rng.normal(x_mean, x_std, size=n_points)
    noise = rng.normal(size=n_points) * noise_level
    y = x ** 2 + noise
    return x, y


def blobs(mean=[0], cov=[1.0], size=100, random_state=0):
    X = np.empty((len(mean) * size, len(mean[0])), dtype=float)
    y = np.empty(len(mean) * size)
    for i, (m, c) in enumerate(zip(mean, cov)):
        ii = i * size
        distrib = multivariate_normal(mean=m, cov=c)
        X[ii : ii + size] = distrib.rvs(size=size, random_state=random_state)
        y[ii : ii + size] = i
    return X, y


def parabola(
    rot=0, offset=0, c=0.2, noise=1.0, size=200, x=None, mode="laplace", loc=0
):
    rng = np.random.RandomState(0)
    if x is None:
        if mode == "uniform":
            xmin, xmax = -5, 5
            x = rng.uniform(xmin, xmax, size)
        elif mode == "laplace":
            x = rng.laplace(loc=loc, size=size)
        else:
            raise ValueError("mode not understood: {}".format(mode))
    y = c * x ** 2 + offset + noise * rng.normal(size=size)
    A = np.asarray([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
    return A.dot([x, y]).T


def parabola_multiclass(size=100, **kwargs):
    Xt = parabola(np.pi / 4, size=size, **kwargs)
    yt = np.zeros(len(Xt), dtype=int)
    Xf = parabola(np.pi / 4, 3.0, size=size, **kwargs)
    yf = np.ones(len(Xf), dtype=int)
    X = np.concatenate((Xt, Xf), axis=0)
    y = np.concatenate((yt, yf), axis=0)
    return X, y


class BlobsGenerator:
    def __init__(
            self, random_state=0, class_1_mean=(7., 0.), z_direction=(1, 1),
            z_noise=.3
    ):
        self.rng = check_random_state(random_state)
        self.class_0_mean = (0, 0)
        self.class_1_mean = class_1_mean
        self.z_direction = np.asarray(
            (np.cos(z_direction), np.sin(z_direction)))
        self.z_noise = z_noise

    def sample(self, size=200):
        y = self.rng.binomial(1, 0.5, size=size)
        n0 = (y == 0).sum()
        n1 = (y == 1).sum()
        X = np.empty((size, 2))
        X[y == 0] = self.rng.multivariate_normal(
            self.class_0_mean, np.eye(2), size=n0)
        X[y == 1] = self.rng.multivariate_normal(
            self.class_1_mean, np.eye(2), size=n1)
        z = self.z_direction.dot(X.T) + self.rng.normal(0., self.z_noise, size)
        return pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "z": z, "y": y})


class ParabolasGenerator:
    def __init__(
        self,
        z_loc=0,
        z_scale=1.0,
        c=0.3,
        rot=np.pi / 4,
        class_offset=3.0,
        x1_noise=0.7,
        x2_noise=1.0,
        y_noise=0.2,
        random_state=0,
    ):
        self.rng = check_random_state(random_state)
        self.z_loc = z_loc
        self.z_scale = z_scale
        self.c = c
        self.rot = rot
        self.class_offset = class_offset
        self.x1_noise = x1_noise
        self.x2_noise = x2_noise
        self.y_noise = y_noise
        self.A = np.asarray(
            [
                [np.cos(self.rot), np.sin(self.rot)],
                [-np.sin(self.rot), np.cos(self.rot)],
            ]
        )

    def sample(self, size=200):
        z = self.rng.normal(self.z_loc, self.z_scale, size=size)
        y = (z > 0).astype(int)
        y_noise = self.rng.binomial(1, self.y_noise, size=size)
        y = y * (1 - y_noise) + (1 - y) * y_noise
        x1 = z + self.rng.normal(0.0, self.x1_noise, size=size)
        offset = self.class_offset * y
        x2 = (
            self.c * x1 ** 2
            + offset
            + self.x2_noise * self.rng.normal(size=size)
        )
        x1, x2 = self.A.dot([x1, x2])
        return pd.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y})

    def score(self, z, normalize=True):
        distrib = norm(self.z_loc, self.z_scale)
        scores = distrib.pdf(z)
        if normalize:
            scores /= scores.sum()
        return scores
