"""Fixed baselines and models; all fitted state comes from training rows."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class Majority:
    def fit(self, x, y):
        self.frequency = np.bincount(y, minlength=3) / len(y)
        self.majority = int(np.argmax(self.frequency))
        return self

    def predict(self, x):
        return np.full(len(x), self.majority, dtype=int)

    def predict_proba(self, x):
        return np.tile(self.frequency, (len(x), 1))


class ImbalanceRule:
    def fit(self, x, y):
        return self

    def predict(self, x):
        z = np.asarray(x["imbalance_5"])
        return np.where(z < -.1, 0, np.where(z > .1, 2, 1))

    def predict_proba(self, x):
        # Deterministic rule, not calibrated probabilities.
        return np.eye(3)[self.predict(x)]


def fit_models(x, y, seed=42):
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Training sample must contain down, flat and up classes")
    models = {
        "majority": Majority(),
        "imbalance_rule": ImbalanceRule(),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        "xgboost": XGBClassifier(n_estimators=120, max_depth=3, learning_rate=.05,
                                  min_child_weight=20, subsample=.8, colsample_bytree=.8,
                                  reg_lambda=5., n_jobs=1, random_state=seed,
                                  objective="multi:softprob", num_class=3, eval_metric="mlogloss"),
    }
    for model in models.values():
        model.fit(x, np.asarray(y, dtype=int))
    return models
