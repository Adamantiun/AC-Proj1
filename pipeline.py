from data_manip import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pycaret.classification import *

current_dir = os.path.dirname(os.path.abspath(__file__))

# PLAYERS_TEAMS
player_teams_input_path = 'original_data/players_teams.csv'
player_teams_column_pairs = [('fgMade', 'fgAttempted'), ('ftMade', 'ftAttempted'), ('threeMade', 'threeAttempted')]
player_teams_output_path = 'modified_data/players_teams.csv'
convert_columns_to_ratio(player_teams_input_path, player_teams_column_pairs, player_teams_output_path)

player_teams_columns_to_exclude = ['fgMade', 'fgAttempted', 'ftMade', 'ftAttempted', 'threeMade', 'threeAttempted']
exclude_columns(player_teams_output_path, player_teams_columns_to_exclude, player_teams_output_path)

# TEAMS
teams_input_path = 'original_data/teams.csv'
teams_columns_to_exclude = ['confW', 'confL', 'min', 'attend', 'arena', 'tmORB', 'tmDRB', 'tmTRB', 'opptmORB', 'opptmDRB', 'opptmTRB', 'divID', 'seeded']
teams_output_path = 'modified_data/teams.csv'

exclude_columns(teams_input_path, teams_columns_to_exclude, teams_output_path)

# PLAYERS
players_input_path = 'original_data/players.csv'
players_columns_to_exclude = ['firstseason', 'lastseason', 'height', 'weight', 'college', 'collegeOther', 'deathDate']
players_output_path = 'modified_data/players.csv'

exclude_columns(players_input_path, players_columns_to_exclude, players_output_path)

# AWARDS_PLAYERS
players_input_path = 'original_data/awards_players.csv'
players_columns_to_exclude = ['award', 'lgID']
players_output_path = 'modified_data/awards_players.csv'

exclude_columns(players_input_path, players_columns_to_exclude, players_output_path)

# TEAMS_POST
teams_input_path = 'original_data/teams_post.csv'
teams_columns_to_exclude = ['lgID']
teams_output_path = 'modified_data/teams_post.csv'

exclude_columns(teams_input_path, teams_columns_to_exclude, teams_output_path)


######################
# FUNÇÃO DO JOÃO QUE #
# DEIXA OS DADOS EM  #
# COMBINED_DATA.CSV  #
######################


data = pd.read_csv(os.path.join(current_dir, 'combined_data.csv'))

X = data.drop('playoff', axis=1)
y = data['playoff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

s = setup(data=data, target='playoff', session_id=123, normalize=True)
compare_models()
