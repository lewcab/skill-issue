"""
Main script for training and evaluating machine learning models on match data.
"""
import os
from math import sqrt

import pandas as pd

from constants import LOSSES, CRITERIA, LAYER_SIZES, ACTIVATIONS, SOLVERS, LEARNING_RATES
from data_processor import DataProcessor
from models.gbc import GBC
from models.nn import NN

data_files = [
    "data/match-data.csv",
]
data = DataProcessor(
    data_files,
    normalize=2,
)
X, Y = data.x, data.y

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
print(X)
print(X.describe())
print()


def main():
    print("skill issue ...")
    optimize_nn()
    optimize_gbc()


def optimize_gbc():
    """
    Optimizes the Gradient Boosting Classifier using a genetic algorithm.
    """
    print("Optimizing Gradient Boosting Classifier (GBC) ...")
    gbc_space = [
        range(len(LOSSES)), # loss
        [10 ** -i for i in range(1, 5)], # learning_rate
        range(100, 301, 10), # n_estimators
        [0.1 * i for i in range(1, 11)], # subsample
        range(len(CRITERIA)), # criterion
        range(20, 61), # max_depth
        range(2, 31), # min_samples_split
        range(1, 16), # min_samples_leaf
        range(round(sqrt(X.shape[1]) * 0.75), X.shape[1] + 1), # max_features
    ]

    for i, param in enumerate(gbc_space):
        print(f"Parameter {i}: {param}")

    gbc = GBC(X, Y, gbc_space)
    gbc.run()

    write_solutions_to_csv(gbc.solutions, "gbc_solutions.csv")


def optimize_nn():
    """
    Optimizes a neural network model using a genetic algorithm.
    """
    print("Optimizing Neural Network (NN) ...")
    nn_space = [
        range(len(LAYER_SIZES)), # layer size
        range(len(ACTIVATIONS)), # activation
        range(len(SOLVERS)), # solver
        range(len(LEARNING_RATES)), # learning_rate
        [10 ** -x for x in range(1, 6)], # learning_rate_init
        [10 ** -x for x in range(1, 7)], # alpha
        range(800, 2001, 100), # max_iter
    ]

    for i, param in enumerate(nn_space):
        print(f"Parameter {i}: {param}")

    nn = NN(X, Y, nn_space)
    nn.run()

    write_solutions_to_csv(nn.solutions, "nn_solutions.csv")


def write_solutions_to_csv(solutions: list, filename: str):
    """
    Writes the solutions to a CSV file.
    :param solutions: List of dictionaries containing the model parameters and their accuracies.
    :param filename: The name of the file to write the solutions to.
    """
    if not os.path.exists('solutions'):
        os.makedirs('solutions')

    path = f"solutions/{filename}"
    df = pd.DataFrame(solutions)
    df.to_csv(path, index=False)
    print(f"Solutions written to {path}")


if __name__ == '__main__':
    main()
