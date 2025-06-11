"""
Collects a list of Matches, tracking useful features for training a model.
Writes the data into a CSV file.
"""

# General Imports
import numpy as np
from collections import OrderedDict

# API Imports
from mwrogue.esports_client import EsportsClient


def main():
    site = EsportsClient("lol")
    print("Connected to the League of Legends esports client.")

    print("Fetching player stats...")
    player_stats = site.cargo_client.query(
        tables='Players=P',
        fields='P.Name, P.ID, P.Role, P.Team, P.Country, P.Age',
        where=
        'P.Country = "Canada" AND '
        'NOT P.IsRetired AND '
        'P.Name IS NOT NULL AND '
        'P.Team IS NOT NULL AND '
        'P.Role IN ("Top", "Jungle", "Mid", "Bot", "Support")',
        order_by='P.Age DESC',
        limit=100,
    )
    print(f"Fetched {len(player_stats)} player entries.")
    print("Player stats:")
    for player in player_stats:
        for key, value in player.items():
            print(f"{key}: {value},  ", end='')
        print()


if __name__ == '__main__':
    main()
