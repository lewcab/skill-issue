"""
Collects a list of Matches, tracking useful features for training a model.
Writes the data into a CSV file.
"""

# General Imports
from datetime import datetime, timedelta
from numpy import mean
from time import sleep
from typing import Callable

# API Imports
from mwrogue.esports_client import EsportsClient

# Definitions
HISTORY_LENGTH = 10     # Number of matches to consider for each stat aggregation


def main():
    client = EsportsClient("lol")
    print("Connected to the League of Legends esports client...")

    tournaments = [
        "MSI 2024",
    ]
    matches = get_matches(client, tournaments)


def get_matches(client: EsportsClient, tournaments: list[str]):
    data = []
    for tournament in tournaments:
        print(f"Fetching matches for tournament: {tournament}")
        matches = get_tournament_matches(client, tournament)
        print(f"Found {len(matches)} matches for tournament: {tournament}")
        sleep(2)    # Rate limiting to avoid hitting API limits
        for m in matches:
            team_1 = m['Team1']
            team_2 = m['Team2']
            match_datetime = m['DateTime UTC']

            print(f"Processing match: {team_1} vs {team_2} at {match_datetime}")

            team_1_stats = get_team_stats(client, team_1, match_datetime)
            team_2_stats = get_team_stats(client, team_2, match_datetime)
            print(f"\nTeam 1 stats in past {HISTORY_LENGTH} games:\n{team_1_stats}\n")
            print(f"Team 2 stats in past {HISTORY_LENGTH} games:\n{team_2_stats}")
            print("+-----------------------------------------------------------------+")

            sleep(2)    # Rate limiting to avoid hitting API limits

    return data


def get_tournament_matches(client: EsportsClient, tournament: str):
    data = client.cargo_client.query(
        tables="ScoreboardGames=SG",
        fields="SG.MatchId, SG.DateTime_UTC, SG.Team1, SG.Team2",
        where=f"SG.Tournament='{tournament}'",
        order_by="SG.MatchID DESC",
        limit=500,
    )

    return data

def get_team_stats(client: EsportsClient, team_name: str, match_datetime: str):
    # change match_datetime to 12 hours before the match
    match_datetime = datetime.strptime(match_datetime, "%Y-%m-%d %H:%M:%S")
    match_datetime = match_datetime.strftime("%Y-%m-%d %H:%M:%S")
    match_datetime = datetime.strptime(match_datetime, "%Y-%m-%d %H:%M:%S") - timedelta(hours=12)
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

    stats = []
    for i, match in enumerate(data):
        if match['Team1'] == team_name:
            team = 'Team1'
        else:
            team = 'Team2'

        if match['WinTeam'] == team_name: win = 1
        else: win = 0
        game_length = get_stat(match, 'Gamelength Number', float, 1)
        gold = get_stat(match, f"{team}Gold", int)
        kills = get_stat(match, f"{team}Kills", int)
        dragons = get_stat(match, f"{team}Dragons", int)
        barons = get_stat(match, f"{team}Barons", int)
        towers = get_stat(match, f"{team}Towers", int)
        inhibitors = get_stat(match, f"{team}Inhibitors", int)
        heralds = get_stat(match, f"{team}RiftHeralds", int)
        grubs = get_stat(match, f"{team}VoidGrubs", int)

        stats.append({
            'wr': win,
            'gold': gold,
            'gpm': gold / game_length,
            'kills': kills,
            'kpm': kills / game_length,
            'dragons': dragons,
            'barons': barons,
            'towers': towers,
            'inhibitors': inhibitors,
            'heralds': heralds,
            'grubs': grubs,
        })

    sums = {}
    counts = {}
    for stat in stats:
        for k, v in stat.items():
            sums[k] = sums.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1

    mean_stats = {k: v / counts[k] for k, v in sums.items()}

    print(data[0][f"{team}Players"])

    return mean_stats


def get_stat(match: dict, stat: str, cast: Callable, fallback: float = 0.0):
    stat = match.get(stat)
    return cast(stat) if stat is not None else fallback


if __name__ == '__main__':
    main()
