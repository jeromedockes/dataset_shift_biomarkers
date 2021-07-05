"""This script was used to generate the cross-validation scores.

It is included for information but relies on UKBiobank data, which cannot be
shared (and some boilerplate code for loading it).

"""
import argparse
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone, BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ukbiobank import models, preprocessing, utils


class RegressOut(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = clone(estimator)

    def fit(self, X, y, var_to_remove, *args, **kwargs):
        self.imputer_ = IterativeImputer(
            add_indicator=0,
            random_state=0,
            max_iter=3,
            n_nearest_features=5,
        )

        X = self.imputer_.fit_transform(X)
        self.linreg_ = LinearRegression().fit(var_to_remove, X)
        X_ortho = X - self.linreg_.predict(var_to_remove)
        self.estimator_ = clone(self.estimator).fit(
            X_ortho, y, *args, **kwargs
        )

    def decision_function(self, X, var_to_remove):
        X = self.imputer_.transform(X)
        X_ortho = X - self.linreg_.predict(var_to_remove)
        return self.estimator_.decision_function(X_ortho)


def get_features(features_file):
    return pd.read_csv(features_file)


def load_data(data_file, features_file, n_features=None):
    data = pd.read_parquet(str(data_file)).astype("float")
    is_first_oc = (
        int(data.columns[3].split("-")[0])
        in preprocessing.get_first_occurrence_date_fields()
    )
    data = data[data.iloc[:, 1].notnull()]
    if is_first_oc:
        visit = data.iloc[:, 0]
        data = data[~(visit.values > data.iloc[:, 3].values)]
    data = data.iloc[6000:]
    data = data.sample(frac=1.0, random_state=0, axis=0)
    age = data.iloc[:, 1]
    sex = data.iloc[:, 2]
    y = data.iloc[:, 3]
    if is_first_oc:
        y = y.notnull()
    else:
        y = y.astype(bool)
    feat = get_features(features_file)
    x = data.loc[:, feat["feature_name"].values]
    if n_features is not None:
        x = x.iloc[:, :n_features]
    return {
        "age": age,
        "sex": sex,
        "X": x,
        "y": y,
        "target_name": data.columns[3],
    }


def prepare_populations(ages, p, quantile):
    bins = pd.qcut(
        ages, [0.0, quantile, 1 - quantile, 1.0], labels=[0, 1, 2]
    ).astype(int)
    size = np.unique(bins.values, return_counts=True)[1].min()
    young_idx = np.where(bins.values == 0)[0][:size]
    old_idx = np.where(bins.values == 2)[0][:size]
    n_swap = int(len(young_idx) * p)
    young_idx_mixed = np.concatenate([old_idx[:n_swap], young_idx[n_swap:]])
    old_idx_mixed = np.concatenate([young_idx[:n_swap], old_idx[n_swap:]])
    rng = np.random.default_rng(0)
    rng.shuffle(young_idx_mixed)
    rng.shuffle(old_idx_mixed)
    return {
        "young": ages.index.values[young_idx_mixed],
        "young_age_bins": bins.iloc[young_idx_mixed],
        "old": ages.index.values[old_idx_mixed],
        "old_age_bins": bins.iloc[old_idx_mixed],
        "preferred_bin_proba": 1 - p,
        "least_preferred_bin_proba": p,
    }


def get_train_test(samples, cv=5):
    split_pos = KFold(cv).split(np.arange(len(samples["young"])))
    split_indexes = []
    p1, p2 = (
        samples["preferred_bin_proba"],
        samples["least_preferred_bin_proba"],
    )

    young_to_old_w = np.zeros(len(samples["young"]))
    young_to_old_w[samples["young_age_bins"] == 0] = p2 / p1
    young_to_old_w[samples["young_age_bins"] == 2] = p1 / (p2 + 1e-8)
    young_to_old_w /= young_to_old_w.mean() + 1e-8

    old_to_young_w = np.zeros(len(samples["old"]))
    old_to_young_w[samples["old_age_bins"] == 2] = p2 / p1
    old_to_young_w[samples["old_age_bins"] == 0] = p1 / (p2 + 1e-8)
    old_to_young_w /= old_to_young_w.mean() + 1e-8

    for train, test in split_pos:
        split_info = {
            "train_young": samples["young"][train],
            "train_old": samples["old"][train],
            "test_young": samples["young"][test],
            "test_old": samples["old"][test],
            "young_to_old_train_w": young_to_old_w[train],
            "old_to_young_train_w": old_to_young_w[train],
            "young_to_old_test_w": young_to_old_w[test],
            "old_to_young_test_w": old_to_young_w[test],
        }
        split_indexes.append(split_info)
    return split_indexes


def get_scores(
    model, split_info, data, weight_exponent=0.0, regress_out_age=False
):
    scores = {}
    for train_pop, test_pop in itertools.product(
        ["young", "old"], ["young", "old"]
    ):
        model_ = clone(model)
        train, test = (
            split_info[f"train_{train_pop}"],
            split_info[f"test_{test_pop}"],
        )
        assert not set(train).intersection(test)
        kwargs = {}
        if train_pop != test_pop:
            weights = np.power(
                split_info[f"{train_pop}_to_{test_pop}_train_w"],
                weight_exponent,
            )
            weights /= weights.mean()
            key = "sample_weight"
            if isinstance(model_.estimator, Pipeline):
                key = "linearsvc__sample_weight"
            kwargs[key] = weights
        if regress_out_age:
            model_ = RegressOut(model_)
            kwargs["var_to_remove"] = data["age"].loc[train].values[:, None]
        model_.fit(
            data["X"].loc[train].values, data["y"].loc[train].values, **kwargs
        )
        kwargs = (
            {"var_to_remove": data["age"].loc[test].values[:, None]}
            if regress_out_age
            else {}
        )
        pred = model_.decision_function(data["X"].loc[test].values, **kwargs)
        auc = roc_auc_score(data["y"].loc[test].values, pred)
        scores[f"{train_pop}_{test_pop}"] = auc
    return scores


def run_cv(
    *,
    data,
    features,
    n_features,
    out,
    target_name,
    proba,
    quantile,
    estimator_name,
    jobs,
    weight_exponent,
    cv,
    regress_out_age,
):
    call_args = dict(
        data=data,
        features=features,
        n_features=n_features,
        target_name=target_name,
        proba=proba,
        quantile=quantile,
        estimator_name=estimator_name,
        weight_exponent=weight_exponent,
        cv=cv,
        regress_out_age=regress_out_age,
    )
    data = load_data(data, features, n_features)
    root_dir = Path(out)
    target_name = target_name
    if target_name is None:
        target_name = data["target_name"]
    out_dir = (
        root_dir
        / f"cv_{target_name}_n_features_{n_features}_swap_{proba}_q_{quantile}"
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"storing results in {out_dir}")
    samples = prepare_populations(data["age"], p=proba, quantile=quantile)
    splits = get_train_test(samples, cv=cv)
    if estimator_name == "gb":
        estimator = models.get_hist_gradient_boosting_classifier_cv()
    else:
        assert estimator_name == "svc"
        estimator = (
            models.get_linear_svc_cv()
            if regress_out_age
            else models.get_imputer_linear_svc_cv()
        )
    all_scores = joblib.Parallel(n_jobs=jobs)(
        joblib.delayed(get_scores)(
            estimator,
            sp,
            data,
            weight_exponent=weight_exponent,
            regress_out_age=regress_out_age,
        )
        for sp in splits
    )
    all_scores = pd.DataFrame(all_scores)
    all_scores.index.name = "split_index"
    all_scores["model"] = estimator_name
    all_scores["regress_out_age"] = regress_out_age
    all_scores["weight_exponent"] = weight_exponent
    fname = "scores_{}{}{}.csv".format(
        estimator_name,
        f"_weights_exponent_{weight_exponent}",
        "_regress_out_age" if regress_out_age else "",
    )
    all_scores.to_csv(out_dir / fname)
    info = utils.get_script_info()
    info["results_file"] = fname
    (out_dir / utils.format_string("params-{now}.json")).write_text(
        json.dumps(info)
    )
    info["parameters"] = call_args

    print(all_scores)

    print(f"results saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("features", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument(
        "-e", "--estimator_name", type=str, default="gb", choices=["gb", "svc"]
    )
    parser.add_argument("-f", "--n_features", type=int, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("-p", "--proba", type=float, default=0.1)
    parser.add_argument("-q", "--quantile", type=float, default=0.1)
    parser.add_argument("-x", "--weight_exponent", type=float, default=1.0)
    parser.add_argument("-t", "--target_name", type=str, default=None)
    parser.add_argument("-c", "--cv", type=int, default=5)
    parser.add_argument("-r", "--regress_out_age", action="store_true")
    args = parser.parse_args()
    run_cv(**args.__dict__)
