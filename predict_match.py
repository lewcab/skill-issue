"""
This script predicts the outcome of a specific match between two teams.
"""
import sys

import pandas as pd
from mwrogue.esports_client import EsportsClient
from sklearn.ensemble import GradientBoostingClassifier

from constants import COLUMNS_IGNORE, TEAM_1_WIN, TEAM_2_WIN, SEED
from data_collector import get_stats
from data_processor import DataProcessor, calculate_data, get_kda, is_feature


def main():
    validate_arguments()

    data_files = ['data/match-data.csv']
    model, norm = train_model(data_files)

    team1, team2 = sys.argv[1], sys.argv[2]
    print(f"Predicting match between {team1} and {team2}...")

    client = EsportsClient("lol")
    teams_data = fetch_match_data(client, team1, team2)
    teams_data = teams_data.loc[:, [col for col in teams_data.columns if is_feature(col)]]
    teams_data = get_kda(teams_data)
    teams_data = calculate_data(teams_data)
    for col in teams_data.columns:
        if col in norm:
            teams_data[col] = teams_data[col] / norm[col]
        else:
            print(f"Warning: Column {col} not found in normalization data, skipping normalization.")
    teams_data = teams_data.reindex(sorted(teams_data.columns), axis=1)
    print(f"Data for prediction: {teams_data.shape}")

    predict_match(model, teams_data, team1, team2)


def validate_arguments():
    """
    Validates the command-line arguments.
    Exits the script if the arguments are invalid.
    """
    if len(sys.argv) != 4:
        print("Usage: python predict_match.py <team1> <team2>")
        sys.exit(1)


def fetch_match_data(client, team1, team2):
    """
    Fetches match statistics for the given teams.
    :param client: EsportsClient instance.
    :param team1: Name of the first team.
    :param team2: Name of the second team.
    :return: pandas DataFrame of match statistics.
    """
    stats = get_stats(client, team1, team2)
    if not stats:
        print("Failed to retrieve match statistics.")
        sys.exit(1)

    teams_data = {
        key: value for key, value in stats.items()
        if key not in COLUMNS_IGNORE and key != 'Winner'
    }
    return pd.DataFrame([teams_data])


def train_model(data_files):
    """
    Trains a GradientBoostingClassifier using the provided data files.
    :param data_files: List of file paths containing match data.
    :return: Trained GradientBoostingClassifier instance.
    """
    data = DataProcessor(data_files, normalize=2)
    print(f"Training GradientBoostingClassifier... \n\tx={data.x.shape}, y={data.y.shape}")

    model = GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.01,
        n_estimators=200,
        subsample=0.7,
        criterion='squared_error',
        max_depth=26,
        min_samples_split=26,
        min_samples_leaf=9,
        max_features=18,
        random_state=SEED,
    )
    model.fit(data.x, data.y)
    return model, data.norm


def predict_match(model, teams_data, team1, team2):
    """
    Predicts the outcome of a match between two teams.
    :param model: Trained GradientBoostingClassifier instance.
    :param teams_data: pandas DataFrame of match statistics.
    :param team1: Name of the first team.
    :param team2: Name of the second team.
    """
    prediction = model.predict_proba(teams_data)[0]
    winner = model.predict(teams_data)
    print(f"Prediction for {team1} vs {team2}:")
    print(f"\t{team1}: {round(prediction[TEAM_1_WIN] * 100)}%")
    print(f"\t{team2}: {round(prediction[TEAM_2_WIN] * 100)}%")
    print(f"Winner: {team1 if winner[0] == TEAM_1_WIN else team2} ({'Team 1' if winner[0] == TEAM_1_WIN else 'Team 2'})")


if __name__ == '__main__':
    main()
