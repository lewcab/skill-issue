from datetime import datetime

SEED = 123
HISTORY_LENGTH = 10     # Number of past matches to consider for team statistics
PLAYERS_PER_TEAM = 5
OUTPUT_NAME = f"data/{datetime.now().strftime('%y%m%d_%H%M%S')}-match_data.csv"
COLUMNS_IGNORE = [
    'MatchId',
    'Tournament',
    'DateTime_UTC',
    'Team1',
    'Team2',
]