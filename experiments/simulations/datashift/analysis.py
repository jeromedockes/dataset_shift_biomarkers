from sklearn.linear_model import LinearRegression


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
