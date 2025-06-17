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
    print(data.x)
    print(data.y)


if __name__ == '__main__':
    main()
