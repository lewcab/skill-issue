"""
Data Processor Class
This module defines a DataProcessor class that reads match data from CSV files
and processes it into numpy arrays for training machine learning models.
"""

import csv
import numpy as np

from numpy import array

# Constants
COLUMNS_IGNORE = [
    'MatchId',
    'Tournament',
    'DateTime_UTC',
    'Team1',
    'Team2',
]


class DataProcessor:
    def __init__(self, filenames: list[str], train_ratio: float = 0.8) -> None:
        """
        Initializes the DataProcessor with a list of filenames.
        Reads the data from the CSV files and collects it into two numpy arrays:
        - data_x: Statistics (features) for each match.
        - data_y: Match outcomes (Winner column).
        :param filenames: List of CSV file paths to read data from.
        """
        self.x_train = array([])
        self.y_train = array([])

        self.x_test = array([])
        self.y_test = array([])

        for filename in filenames:
            self._process_file(filename, train_ratio)

    def _process_file(self, filename: str, train_ratio: float = 0.8) -> None:
        """
        Reads a single CSV file and extracts match data and outcomes.
        :param filename: Path to the CSV file.
        """
        raw_data = []

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract features (all columns except 'Winner')
                row = [
                    float(value) for key, value in row.items()
                    if key not in COLUMNS_IGNORE
                ]
                # TODO: temp fix for matches with odd data...
                if row[-1] == 0:  # Ensure the last feature is not zero
                    continue
                raw_data.append(row)

        raw_data = np.array(raw_data)
        np.random.shuffle(raw_data)

        # Split data into features (x) and outcomes (y)
        # First column is the outcome (Winner)
        x = raw_data[:, 1:]
        y = raw_data[:, 0]

        # Normalize features to the range [0, 1]
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

        # Convert y to binary values (0 or 1)
        y -= 1


        # Split into training and testing sets
        split_index = int(len(x) * train_ratio)
        self.x_train = np.append(self.x_train, x[:split_index], axis=0) if self.x_train.size else x[:split_index]
        self.y_train = np.append(self.y_train, y[:split_index]) if self.y_train.size else y[:split_index]

        self.x_test = np.append(self.x_test, x[split_index:], axis=0) if self.x_test.size else x[split_index:]
        self.y_test = np.append(self.y_test, y[split_index:]) if self.y_test.size else y[split_index:]

    def __str__(self) -> str:
        """
        Returns a string representation of the DataProcessor object.
        Displays the shapes of the training and testing datasets.
        """
        return (
            f"DataProcessor:\n"
            f"+--- Training data shape: x={self.x_train.shape}, y={self.y_train.shape} ---+\n"
            f"{str(self.x_train)}\n{self.y_train}\n\n"
            f"+--- Testing data shape: {self.x_test.shape}, {self.y_test.shape} ---+\n"
            f"{str(self.x_test)}\n{self.y_test}\n\n"
        )
