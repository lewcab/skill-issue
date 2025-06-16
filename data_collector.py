"""
Collects a list of Matches, tracking useful features for training a model.
Writes the data into a CSV file.
"""

import csv
import os
from datetime import datetime, timedelta
from time import sleep, time
from typing import Callable

from mwrogue.esports_client import EsportsClient
from numpy import array

# Definitions
HISTORY_LENGTH = 10     # Number of past matches to consider for team statistics
OUTPUT_NAME = "data/match_data.csv"


def main():
    start_time = time()

    client = EsportsClient("lol")
    print("Connected to the League of Legends esports client...")

    tournaments = get_tournaments(
        client,
        limit=10,
    )
    print("Collecting match data for the following tournaments:")
    print(tournaments)

    get_matches(
        client,
        tournaments,
    )

    end_time = time()
    print(f"Data collection completed in {end_time - start_time:.2f} seconds.")


def get_tournaments(client: EsportsClient, region: str = None, limit: int = 500) -> array:
    """
    Fetches a list of official tournaments that have already started.
    :param client: EsportsClient instance to interact with the API.
    :param region: Optional region filter for tournaments.
        Ex. "Americas", "EMEA", "Asia Pacific", "Korea", "China", "Japan", "International"
    :param limit: Maximum number of tournaments to return. Should be less than 500.
    :return: Array of tournament names.
    """
    if limit > 500:
        raise ValueError("Limit must be less than or equal to 500.")

    today = datetime.now()
    today = today.strftime("%Y-%m-%d")
    where = f"T.DateStart < '{today}' AND T.IsOfficial=1 AND T.TournamentLevel='Primary'"
    if region is not None: where += f" AND T.Region='{region}'"
    data = client.cargo_client.query(
        tables='Tournaments=T',
        fields='T.Name',
        where=where,
        order_by='T.DateStart DESC',
        limit=limit,
    )
    return array([t['Name'] for t in data])


def get_matches(client: EsportsClient, tournaments: list[str]) -> None:
    """
    Fetches matches for the given tournaments and collects team statistics.
    :param client: EsportsClient instance to interact with the API.
    :param tournaments: List of tournament names to fetch matches for.
    :return: List of match data with team statistics.
    """
    for tournament in tournaments:
        data = []
        print(f"Fetching matches for tournament: {tournament}")
        matches = get_tournament_matches(client, tournament)
        print(f"Found {len(matches)} matches for tournament: {tournament}")
        for m in matches:
            sleep(2)    # Rate limiting to avoid hitting API limits
            print("+-----------------------------------------------------------------+")

            team_1 = m['Team1']
            team_2 = m['Team2']
            match_datetime = m['DateTime UTC']

            print(f"Processing match: {team_1} vs {team_2} at {match_datetime}")
            try:
                team_1_stats = get_team_stats(client, team_1, match_datetime)
                team_2_stats = get_team_stats(client, team_2, match_datetime)
            except Exception as e:
                print(f"Error fetching stats for match {team_1} vs {team_2}: {e}")
                continue
            if len(team_1_stats) == 0 or len(team_2_stats) == 0:
                print(f"Skipping match {team_1} vs {team_2} due to missing stats.")
                continue

            match_data = {
                'MatchId': m['MatchId'],
                'Tournament': tournament,
                'DateTime_UTC': match_datetime,
                'Winner': int(m['Winner']),
                'Team1': team_1,
                'Team2': team_2,
            }
            for stat, val in team_1_stats.items():
                match_data[f'Team1_{stat}'] = val
            for stat, val in team_2_stats.items():
                match_data[f'Team2_{stat}'] = val
            data.append(match_data)

            print(f"\nTeam 1 stats in past {HISTORY_LENGTH} games:\n{team_1_stats}\n")
            print(f"Team 2 stats in past {HISTORY_LENGTH} games:\n{team_2_stats}")

        print(f"Finished processing {len(matches)} matches for tournament: {tournament}")
        write_to_csv(data)

    pass


def get_tournament_matches(client: EsportsClient, tournament: str) -> array:
    """
    Fetches all matches for a specific tournament.
    :param client: EsportsClient instance to interact with the API.
    :param tournament: Name of the tournament to fetch matches for.
    """
    data = client.cargo_client.query(
        tables="ScoreboardGames=SG",
        fields="SG.MatchId, SG.DateTime_UTC, SG.Team1, SG.Team2, SG.Winner",
        where=f"SG.Tournament='{tournament}'",
        order_by="SG.MatchID DESC",
        limit=500,
    )

    return array(data)

def get_team_stats(client: EsportsClient, team_name: str, match_datetime: str) -> dict:
    """
    Fetches the last HISTORY_LENGTH matches for a given team and calculates statistics.
    :param client: EsportsClient instance to interact with the API.
    :param team_name: Name of the team to fetch statistics for.
    :param match_datetime: The datetime of the match to consider as the cutoff for historical data.
    :return: Array of statistics for the team in the last HISTORY_LENGTH matches.
    """
    # change match_datetime to 6 hours before the match
    match_datetime = datetime.strptime(match_datetime, "%Y-%m-%d %H:%M:%S")
    match_datetime = match_datetime.strftime("%Y-%m-%d %H:%M:%S")
    match_datetime = datetime.strptime(match_datetime, "%Y-%m-%d %H:%M:%S") - timedelta(hours=6)
    match_datetime = match_datetime.strftime("%Y-%m-%d %H:%M:%S")
    data = client.cargo_client.query(
        tables="MatchSchedule=MS, ScoreboardGames=SG",
        join_on="MS.MatchId=SG.MatchId",
        fields=
            "MS.DateTime_UTC, SG.Tournament, MS.MatchId, SG.Gamelength_Number, SG.WinTeam, "
            "MS.Team1, MS.Team2, SG.Team1Players, SG.Team2Players, "
            "SG.Team1Gold, SG.Team2Gold, SG.Team1Kills, SG.Team2Kills, "
            "SG.Team1Towers, SG.Team2Towers, SG.Team1Inhibitors, SG.Team2Inhibitors, "
            "SG.Team1Dragons, SG.Team2Dragons, SG.Team1Barons, SG.Team2Barons, "
            "SG.Team1RiftHeralds, SG.Team2RiftHeralds, SG.Team1VoidGrubs, SG.Team2VoidGrubs",
        where=f"(MS.Team1='{team_name}' OR MS.Team2='{team_name}') AND MS.DateTime_UTC < '{match_datetime}'",
        order_by="MS.DateTime_UTC DESC",
        limit=HISTORY_LENGTH,
    )

    if len(data) == 0:
        print(f"No matches found for team '{team_name}' before {match_datetime}.")
        return {}

    stats = {}
    count = 0
    for i, match in enumerate(data):
        if match['Team1'] == team_name:
            team = 'Team1'
        else:
            team = 'Team2'

        game_length = get_stat(match, 'Gamelength Number', float, 1)

        # Aggregate statistics
        stats['win'] = stats.get('win', 0) + (1 if match['WinTeam'] == team_name else 0)
        stats['game_length'] = stats.get('game_length', 0) + game_length
        stats['gold'] = stats.get('gold', 0) + get_stat(match, f"{team}Gold", int)
        stats['kills'] = stats.get('kills', 0) + get_stat(match, f"{team}Kills", int)
        stats['dragons'] = stats.get('dragons', 0) + get_stat(match, f"{team}Dragons", int)
        stats['barons'] = stats.get('barons', 0) + get_stat(match, f"{team}Barons", int)
        stats['towers'] = stats.get('towers', 0) + get_stat(match, f"{team}Towers", int)
        stats['inhibitors'] = stats.get('inhibitors', 0) + get_stat(match, f"{team}Inhibitors", int)
        stats['heralds'] = stats.get('heralds', 0) + get_stat(match, f"{team}RiftHeralds", int)
        stats['grubs'] = stats.get('grubs', 0) + get_stat(match, f"{team}VoidGrubs", int)
        stats['gpm'] = stats.get('gpm', 0) + (get_stat(match, f"{team}Gold", int) / game_length)
        stats['kpm'] = stats.get('kpm', 0) + (get_stat(match, f"{team}Kills", int) / game_length)

        count += 1

    averages = {k: v / count for k, v in stats.items()}

    return averages


def get_player_stats(client: EsportsClient, player_name: str, match_datetime: str) -> dict:
    """
    Fetches the last HISTORY_LENGTH matches for a given player and calculates statistics.
    :param client: EsportsClient instance to interact with the API.
    :param player_name: Name of the player to fetch statistics for.
    :param match_datetime: The datetime of the match to consider as the cutoff for historical data.
    :return: Dictionary of statistics for the player in the last HISTORY_LENGTH matches.
    """
    # TODO: implement ...
    pass


def get_stat(match: dict, stat: str, cast: Callable, fallback: float = 0.0):
    stat = match.get(stat)
    return cast(stat) if stat is not None else fallback


def write_to_csv(data: list[dict]) -> None:
    """
    Writes the collected match data to a CSV file.
    :param data: List of match data dictionaries to write to the CSV file.
    """
    if not data or len(data) == 0:
        print("No data to write to CSV.")
        return

    # check if the output directory exists, if not create it
    exists = os.path.exists(os.path.dirname(OUTPUT_NAME))
    if not exists:
        os.makedirs(os.path.dirname(OUTPUT_NAME))

    with open(OUTPUT_NAME, 'a', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        for row in data:
            writer.writerow(row)

    print(f"{len(data)} rows written to {OUTPUT_NAME} successfully.")


if __name__ == '__main__':
    main()
