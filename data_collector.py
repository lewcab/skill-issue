"""
Collects a list of Matches, tracking useful features for training a model.
Writes the data into a CSV file.
"""

import csv
import os
from datetime import datetime, timedelta
from time import sleep, time
from typing import Callable

from mwclient import APIError
from mwrogue.esports_client import EsportsClient
from numpy import array

from constants import HISTORY_LENGTH, PLAYERS_PER_TEAM, OUTPUT_NAME


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

    n = get_matches(
        client,
        tournaments,
    )

    end_time = time()
    print(f"Collected {n} matches in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Data written to {OUTPUT_NAME}.")


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


def get_matches(client: EsportsClient, tournaments: list[str]) -> int:
    """
    Fetches matches for the given tournaments and collects team statistics.
    :param client: EsportsClient instance to interact with the API.
    :param tournaments: List of tournament names to fetch matches for.
    :return: List of match data with team statistics.
    """
    count = 0
    for tournament in tournaments:
        data = []
        matches = get_tournament_matches(client, tournament)
        if len(matches) == 0:
            print(f"No matches found for tournament: {tournament}")
            sleep(2)    # Rate limiting to avoid hitting API limits
            continue
        else:
            print(f"Found {len(matches)} matches for tournament: {tournament}")
        for m in matches:
            team_1 = m['Team1']
            team_2 = m['Team2']
            match_datetime = m['DateTime UTC']

            print("+-----------------------------------------------------------------+")
            print(f"Processing match: {team_1} vs {team_2} at {match_datetime}")
            try:
                match_data = get_stats(
                    client,
                    team_1, team_2, match_datetime,
                    tournament, m['MatchId'], m['Winner']
                )
            except APIError as e:
                if e.code == 'ratelimited':
                    sleep(30)   # Wait for a while before retrying
                match_data = get_stats(
                    client,
                    team_1, team_2, match_datetime,
                    tournament, m['MatchId'], m['Winner']
                )   # If this fails, it will raise an error and stop the script

            if len(match_data) == 0:
                continue

            data.append(match_data)

            print(f"\nTeam 1 ({match_data['Team1']}) stats in past {HISTORY_LENGTH} games:")
            for key, value in match_data.items():
                if key.startswith('Team1_'):
                    print(f"{key}: {value:.2f}", end=', ')
            print("\n")
            print(f"Team 2 ({match_data['Team2']}) stats in past {HISTORY_LENGTH} games:")
            for key, value in match_data.items():
                if key.startswith('Team2_'):
                    print(f"{key}: {value:.2f}", end=', ')
            print("\n")

        count += len(matches)
        print(f"Finished processing for tournament: {tournament}\n")
        write_to_csv(data)

    return count


def get_stats(
        client: EsportsClient,
        team_1: str, team_2: str, match_datetime: str = None,
        tournament: str = None, match_id: str = None, winner: str = -1
) -> dict:
    if match_datetime is None:
        match_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Fetch team stats
        sleep(2)  # Rate limiting to avoid hitting API limits
        team_1_stats = get_team_stats(client, team_1, match_datetime)
        team_2_stats = get_team_stats(client, team_2, match_datetime)
        if len(team_1_stats) == 0 or len(team_2_stats) == 0:
            print(f"Skipping match {team_1} vs {team_2} due to missing team stats.")
            return {}

        # Fetch player stats
        sleep(2)  # Rate limiting to avoid hitting API limits
        team_1_player_stats = get_player_stats(client, team_1, match_datetime)
        team_2_player_stats = get_player_stats(client, team_2, match_datetime)
        if len(team_1_player_stats) == 0 or len(team_2_player_stats) == 0:
            print(f"Skipping match {team_1} vs {team_2} due to missing player stats.")
            return {}

    except APIError as e:
        print(f"APIError fetching stats for match {team_1} vs {team_2}: {e}")
        raise e

    except Exception as e:
        print(f"Error fetching stats for match {team_1} vs {team_2}: {e}")
        return {}

    match_data = {
        'MatchId': match_id,
        'Tournament': tournament,
        'DateTime_UTC': match_datetime,
        'Winner': int(winner),
        'Team1': team_1,
        'Team2': team_2,
    }
    for stat, val in team_1_stats.items():
        match_data[f'Team1_{stat}'] = val
    for stat, val in team_2_stats.items():
        match_data[f'Team2_{stat}'] = val
    for stat, val in team_1_player_stats.items():
        match_data[f'Team1_{stat}'] = val
    for stat, val in team_2_player_stats.items():
        match_data[f'Team2_{stat}'] = val

    return match_data


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
    match_datetime = offset_datetime(match_datetime).strftime("%Y-%m-%d %H:%M:%S")
    team_name = team_name.replace("'", "''")  # Escape single quotes for SQL
    data = client.cargo_client.query(
        tables="MatchSchedule=MS, ScoreboardGames=SG",
        join_on="MS.MatchId=SG.MatchId",
        fields=
            "SG.Gamelength_Number, SG.WinTeam, MS.Team1, MS.Team2, "
            "SG.Team1Gold, SG.Team2Gold, SG.Team1Kills, SG.Team2Kills, "
            "SG.Team1Towers, SG.Team2Towers, SG.Team1Inhibitors, SG.Team2Inhibitors, "
            "SG.Team1Dragons, SG.Team2Dragons, SG.Team1Barons, SG.Team2Barons, "
            "SG.Team1RiftHeralds, SG.Team2RiftHeralds, SG.Team1VoidGrubs, SG.Team2VoidGrubs",
        where=
            f"(MS.Team1='{team_name}' OR MS.Team2='{team_name}') AND "
            f"MS.DateTime_UTC < '{match_datetime}'",
        order_by="SG.DateTime_UTC DESC",
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

        try:
            game_length = float(match.get('Gamelength Number'))
            stats['win'] = stats.get('win', 0) + (1 if match['WinTeam'] == team_name else 0)
            stats['game_length'] = stats.get('game_length', 0) + game_length
            stats['gold'] = stats.get('gold', 0) + int(match.get(f"{team}Gold"))
            stats['kills'] = stats.get('kills', 0) + int(match.get(f"{team}Kills"))
            stats['dragons'] = stats.get('dragons', 0) + int(match.get(f"{team}Dragons"))
            stats['barons'] = stats.get('barons', 0) + int(match.get(f"{team}Barons"))
            stats['towers'] = stats.get('towers', 0) + int(match.get(f"{team}Towers"))
            stats['inhibitors'] = stats.get('inhibitors', 0) + int(match.get(f"{team}Inhibitors"))
            stats['heralds'] = stats.get('heralds', 0) + int(match.get(f"{team}RiftHeralds"))
            stats['grubs'] = stats.get('grubs', 0) + int(match.get(f"{team}VoidGrubs"))
            stats['gpm'] = stats.get('gpm', 0) + int(match.get(f"{team}Gold")) / game_length
            stats['kpm'] = stats.get('kpm', 0) + int(match.get(f"{team}Kills")) / game_length
        except Exception as e:
            print(f"Error processing match {i+1}/{len(data)} for team '{team_name}': {e}")
            for d in data:
                print(d)
            return {}

        count += 1

    averages = {k: v / count for k, v in stats.items()}

    return averages


def get_player_stats(client: EsportsClient, team_name: str, match_datetime: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")) -> dict:
    """
    Fetches the last HISTORY_LENGTH matches for a given player and calculates statistics.
    :param client: EsportsClient instance to interact with the API.
    :param team_name: Name of the team to fetch statistics for.
    :param match_datetime: The datetime of the match to consider as the cutoff for historical data.
    :return: Dictionary of statistics for the player in the last HISTORY_LENGTH matches.
    """
    match_datetime = offset_datetime(match_datetime).strftime("%Y-%m-%d %H:%M:%S")
    team_name = team_name.replace("'", "''")  # Escape single quotes for SQL
    data = client.cargo_client.query(
        tables="MatchSchedule=MS, ScoreboardPlayers=SP",
        join_on="MS.MatchId=SP.MatchId",
        fields=
            "SP.Name, SP.Role, "
            "SP.Kills, SP.Deaths, SP.Assists, "
            "SP.Gold, SP.CS, SP.DamageToChampions",
        where=
            f"(MS.Team1='{team_name}' OR MS.Team2='{team_name}') AND "
            f"MS.DateTime_UTC < '{match_datetime}' AND "
            f"SP.Team='{team_name}'",
        order_by="SP.DateTime_UTC DESC",
        limit=HISTORY_LENGTH*PLAYERS_PER_TEAM,
    )

    stats = {}
    for match in data:
        role = match['Role']

        try:
            stats[f"{role}_count"] = stats.get(f"{role}_count", 0) + 1
            stats[f"{role}_kills"] = stats.get(f"{role}_kills", 0) + int(match.get("Kills"))
            stats[f"{role}_deaths"] = stats.get(f"{role}_deaths", 0) + int(match.get("Deaths"))
            stats[f"{role}_assists"] = stats.get(f"{role}_assists", 0) + int(match.get("Assists"))
            stats[f"{role}_gold"] = stats.get(f"{role}_gold", 0) + int(match.get("Gold"))
            stats[f"{role}_cs"] = stats.get(f"{role}_cs", 0) + int(match.get("CS"))
            stats[f"{role}_dmg"] = stats.get(f"{role}_dmg", 0) + int(match.get("DamageToChampions"))
        except Exception as e:
            print(f"Error processing player stats: {e}")
            for d in data:
                print(d)
            return {}

    # Verify data
    roles = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
    for k, v in stats.items():
        role, stat = k.split('_')
        if role not in roles:
            return {}
        if stat == 'count' and v != HISTORY_LENGTH:
            print(f"Warning: Incomplete data for role '{role}' in team '{team_name}'. Expected {HISTORY_LENGTH} matches, got {v}.")
            return {}
    for role in roles:
        del stats[f"{role}_count"]

    # Calculate averages
    averages = {k: v / HISTORY_LENGTH for k, v in stats.items()}

    return averages


def offset_datetime(dt: str, hours: int = -6) -> datetime:
    """
    Offsets a datetime string by a specified number of hours.
    :param dt: Datetime string in the format "YYYY-MM-DD HH:MM:SS".
    :param hours: Number of hours to offset (default is -6).
    :return: Offset datetime string in the same format.
    """
    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    dt += timedelta(hours=hours)
    return dt


def write_to_csv(data: list[dict]) -> None:
    """
    Writes the collected match data to a CSV file.
    :param data: List of match data dictionaries to write to the CSV file.
    """
    if not data or len(data) == 0:
        print("No data to write to CSV.")
        return

    # check if the output directory exists, if not create it
    if not os.path.exists(os.path.dirname(OUTPUT_NAME)):
        os.makedirs(os.path.dirname(OUTPUT_NAME))

    with open(OUTPUT_NAME, 'a', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if os.path.getsize(OUTPUT_NAME) == 0:
            writer.writeheader()

        for row in data:
            writer.writerow(row)

    print(f"{len(data)} rows written to {OUTPUT_NAME} successfully.")


if __name__ == '__main__':
    main()
