"""
Data Processor Class
This module defines a DataProcessor class that reads match data from CSV files
and processes it into pandas DataFrames for training machine learning models.
"""

import csv

import pandas as pd
import re

from constants import COLUMNS_META, COLUMNS_IGNORE, SEED, TEAM_1_WIN, TEAM_2_WIN


def calculate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates additional features for the match data.
    :param data: Raw match data as a pandas DataFrame.
    :return: Processed DataFrame with additional features.
    """
    # Create a new DataFrame to store the calculated differences
    calculated_data = pd.DataFrame()

    # Iterate over the columns to find team 1 and team 2 statistics
    for column in data.columns:
        if column == 'Winner':
            # Keep the Winner column as is
            calculated_data[column] = data[column]
            continue

        match = re.match(r"(\d)_(\d)_(.+)", column)
        if match:
            team, role, stat = match.groups()
            if int(team) == 1:
                # Find the corresponding team 2 column
                team_2_column = f"2_{role}_{stat}"
                if team_2_column in data.columns:
                    # Calculate the difference and add a new column
                    diff_column = f"{role}_{stat}_diff"
                    calculated_data[diff_column] = (
                            data[column] - data[team_2_column]
                    )

    return calculated_data


def get_kda(data: pd.DataFrame) -> pd.DataFrame:
    """
    For each player (indicated by 1_, 2_, ..., 5_), calculates the KDA (Kills, Deaths, Assists) in new `kda` column

    """
    for t in range(1, 3):  # For each team
        for i in range(1, 6):  # For each player role
            kills_col = f"{t}_{i}_kills"
            deaths_col = f"{t}_{i}_deaths"
            assists_col = f"{t}_{i}_assists"

            kda_col = f"{t}_{i}_kda"
            data[kda_col] = (data[kills_col] + data[assists_col]) / (data[deaths_col].replace(0, 1))  # Avoid division by zero

            # drop the individual kills, deaths, and assists columns
            data.drop(columns=[kills_col, deaths_col, assists_col], inplace=True)

    return data


def flip_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicates the data and flips the teams to create a balanced dataset.
    :param data: Original match data as a pandas DataFrame.
    :return: DataFrame with duplicated and flipped data.
    """
    flipped_data = data.copy()
    flipped_data['Winner'] = flipped_data['Winner'].apply(
        lambda x: 2 if x == 1 else 1
    )

    flipped_columns = {}
    for column in flipped_data.columns:
        if column.startswith('1_'):
            flipped_columns[column] = column.replace('1_', '2_', 1)
        elif column.startswith('2_'):
            flipped_columns[column] = column.replace('2_', '1_', 1)

    flipped_data.rename(columns=flipped_columns, inplace=True)
    return flipped_data


def is_feature(column_name: str) -> bool:
    """
    Returns True if the column name is a feature.
    """
    if column_name in COLUMNS_META:
        return False
    if column_name.endswith(COLUMNS_IGNORE):
        return False
    return True


class DataProcessor:
    def __init__(self, filenames: list[str], normalize: int = 0) -> None:
        """
        Initializes the DataProcessor with a list of filenames.
        Reads the data from the CSV files and collects it into two pandas DataFrames:
        - data_x: Statistics (features) for each match.
        - data_y: Match outcomes (Winner column).
        :param filenames: List of CSV file paths to read data from.
        """
        self.x, self.y = pd.DataFrame(), pd.Series()
        self.norm = {}

        raw_data = self._read_csv_files(filenames)
        self._process_data(raw_data, normalize)

    def _read_csv_files(self, filenames: list[str]) -> pd.DataFrame:
        """
        Reads and combines data from multiple CSV files.
        :param filenames: List of CSV file paths.
        :return: Combined raw data as a pandas DataFrame.
        """
        raw_data = []
        for filename in filenames:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    raw_data.append({
                        key: float(value) for key, value in row.items() if is_feature(key)
                    })
        return pd.DataFrame(raw_data)

    def _process_data(self, raw_data: pd.DataFrame, normalize: bool) -> None:
        """
        Processes raw data into features and labels.
        :param raw_data: Combined raw data as a pandas DataFrame.
        :param normalize: Whether to normalize the features.
        """
        raw_data = get_kda(raw_data)
        raw_data = pd.concat([raw_data, flip_data(raw_data)], ignore_index=True)
        raw_data.drop(columns=COLUMNS_IGNORE, inplace=True, errors='ignore')
        raw_data = calculate_data(raw_data)

        # Shuffle data
        raw_data = raw_data.sample(frac=1, random_state=SEED)

        x, y = raw_data.drop(columns=['Winner']), raw_data['Winner']
        if normalize == 1:
            # Normalize the features to the range [0, 1] using min-max normalization
            x = (x - x.min()) / (x.max() - x.min())
        elif normalize == 2:
            # Normalize the features to [-1, 1] using decimal scale normalization
            for column in x.columns:
                abs_col = x[column].abs()
                col_max = abs_col.max()
                for i in range(10):
                    if (col_max / (10 ** i)) < 1:
                        x[column] = x[column] / (10 ** i)
                        self.norm[column] = 10 ** i
                        break


        # Normalize the labels such that 1 means Team1 wins, 0 means Team1 loses
        y = y.apply(lambda x: TEAM_1_WIN if x == 1 else TEAM_2_WIN)

        self.x = x.reindex(sorted(x.columns), axis=1)
        self.y = y

    def __str__(self) -> str:
        """
        Returns a string representation of the DataProcessor object.
        Displays the shapes of the datasets.
        """
        return (
            f"DataProcessor:\n"
            f"+--- Data shape: x={self.x.shape}, y={self.y.shape} ---+\n"
            f"{self.x}\n{self.y}\n"
        )
