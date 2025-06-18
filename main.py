# General Imports
import matplotlib.pyplot as plt
import numpy as np
import requests

# scikit-learn Imports
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from data_processor import DataProcessor


def main():
    print("skill issue ...")

    data_files = [
        "data/match-data.csv",
    ]
    data = DataProcessor(data_files)

    n_samples = len(data.x_train) + len(data.x_test)
    n_features = data.x_train.shape[1] if data.x_train.size else 0

    print(f"Data loaded and processed. ({n_samples}, {n_features})", end='\n\n')
    print(data)

    # Setting up the Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.1,
        max_depth=4,
    )

    gbc.fit(data.x_train, data.y_train)
    print("Gradient Boosting Classifier trained.", end='\n\n')
    print("Training score:", gbc.score(data.x_train, data.y_train))
    print("Testing score:", gbc.score(data.x_test, data.y_test), end='\n\n')

    # Setting up the Multi-layer Perceptron Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
    )
    mlp.fit(data.x_train, data.y_train)

    print("Multi-layer Perceptron Classifier trained.", end='\n\n')
    print("Training score:", mlp.score(data.x_train, data.y_train))
    print("Testing score:", mlp.score(data.x_test, data.y_test), end='\n\n')


if __name__ == '__main__':
    main()
