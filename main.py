# General Imports
import matplotlib.pyplot as plt
import numpy as np
import requests

# scikit-learn Imports
from sklearn.neural_network import MLPRegressor

from data_processor import DataProcessor


def main():
    print("skill issue ...")

    data_files = [
        "data/match-data.csv",
    ]
    data = DataProcessor(data_files)

    print("Data loaded and processed.", end='\n\n')
    print("Features (x):", data.x.shape)
    print(data.x, end='\n\n')
    print("Outcomes (y):", data.y.shape)
    print(data.y)


if __name__ == '__main__':
    main()
