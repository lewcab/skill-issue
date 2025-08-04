"""
Parent Class for all models.
"""

from typing import Callable

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from constants import SEED


class Model:
    def __init__(self, model: Callable, x: pd.DataFrame, y: pd.Series):
        """
        Initializes the Model class.
        This class serves as a base for all machine learning models.
        """
        self.model = model
        self.x = x
        self.y = y
        self.solutions = []

    def k_fold(self, params: dict, k: int = 5) -> float:
        """
        Trains and tests a machine learning model with given parameters.
        :param params: Hyperparameters for the model.
        :param k: Number of folds.
        """
        try:
            m = self.model(**params, random_state=SEED)
        except Exception:
            m = self.model(**params)

        kf = StratifiedKFold(n_splits=k)
        kf.get_n_splits(self.x, self.y)

        accs = []
        for i, (i_train, i_test) in enumerate(kf.split(self.x, self.y)):
            m.fit(self.x.iloc[i_train], self.y.iloc[i_train])
            score = m.score(self.x.iloc[i_test], self.y.iloc[i_test])
            accs.append(score)

        acc_mean = pd.Series(accs).mean()

        return acc_mean

    def train_val_split(self, params: dict, ratio: float = 0.8) -> float:
        """
        Splits the data into training and validation sets, trains the model, and returns the accuracy.
        :param params: Hyperparameters for the model.
        :param ratio: Ratio of training data to total data.
        """
        m = self.model(**params, random_state=SEED)

        train_size = int(len(self.x) * ratio)
        x_train, x_val = self.x[:train_size], self.x[train_size:]
        y_train, y_val = self.y[:train_size], self.y[train_size:]

        m.fit(x_train, y_train)
        score = m.score(x_val, y_val)

        return score

