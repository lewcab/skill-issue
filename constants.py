from datetime import datetime

SEED = 123
HISTORY_LENGTH = 10
PLAYERS_PER_TEAM = 5
OUTPUT_NAME = f"data/{datetime.now().strftime('%y%m%d_%H%M%S')}-match_data.csv"
ROLE_TO_NUM = {
    'Top': 1,
    'Jungle': 2,
    'Mid': 3,
    'Bot': 4,
    'Support': 5,
}
TEAM_1_WIN = 1
TEAM_2_WIN = 0
COLUMNS_META = [
    'MatchId',
    'Tournament',
    'DateTime_UTC',
    'Team1',
    'Team2',
]
COLUMNS_IGNORE = (
    'grubs',
    'grubs_lead',
    '0_kills',
    'dragons',
    'barons',
    'towers',
    'inhibitors',
    'heralds',
    'gold',
    'gpm',
    'kpm',
    'cs',
)

# GBC non-numeric parameters
LOSSES = ['log_loss', 'exponential']
CRITERIA = ['friedman_mse', 'squared_error']

# NN non-numeric parameters
LAYER_SIZES = [(50, 50), (100, 50), (100, 100), (128, 64), (50, 25, 10)] # layer size
ACTIVATIONS = ['identity', 'logistic', 'tanh', 'relu'] # activation
SOLVERS = ['lbfgs', 'sgd', 'adam'] # solver
LEARNING_RATES = ['constant', 'invscaling', 'adaptive'] # learning_rate
