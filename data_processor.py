"""
Data Processor Class
This module defines a DataProcessor class that reads match data from CSV files
and processes it into numpy arrays for training machine learning models.
"""

import csv

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
    def __init__(self, filenames: list[str]):
        """
        Initializes the DataProcessor with a list of filenames.
        Reads the data from the CSV files and collects it into two numpy arrays:
        - data_x: Statistics (features) for each match.
        - data_y: Match outcomes (Winner column).
        :param filenames: List of CSV file paths to read data from.
        """
        self.x = array([])
        self.y = array([])
        for filename in filenames:
            self._process_file(filename)


    def _process_file(self, filename: str) -> None:
        """
        Reads a single CSV file and extracts match data and outcomes.
        :param filename: Path to the CSV file.
        """
        x = []
        y = []

        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract features (all columns except 'Winner')
                features = [
                    float(value) for key, value in row.items()
                    if key != 'Winner' and key not in COLUMNS_IGNORE
                ]
                x.append(features)

                # Extract outcome (Winner column)
                y.append(int(row['Winner']))

        self.x = array(x, dtype=float)
        self.y = array(y, dtype=int)
