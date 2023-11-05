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



awards_players_data = pd.read_csv(os.path.join(current_dir, 'modified_data/awards_players.csv'))
coaches_data = pd.read_csv(os.path.join(current_dir, 'modified_data/coaches.csv'))
players_data = pd.read_csv(os.path.join(current_dir, 'modified_data/players.csv'))
players_teams_data = pd.read_csv(os.path.join(current_dir, 'modified_data/players_teams.csv'))
series_post_data = pd.read_csv(os.path.join(current_dir, 'modified_data/series_post.csv'))
teams_data = pd.read_csv(os.path.join(current_dir, 'modified_data/teams.csv'))
teams_post_data = pd.read_csv(os.path.join(current_dir, 'modified_data/teams_post.csv'))

columns_to_use = ["name", "d_to", "GP", "d_stl", "d_blk", "o_asts", "o_reb", "d_reb", "o_ftm", "o_fta", "o_3pm", "o_3pa", "o_fgm", "o_fga", "o_pts", "won", "lost"]

all_turnovers = []
all_steals_blocks = []
all_assists = []
all_rebounds = []
all_ft_percentages = []
all_3p_percentage = []
all_fg_percentages = []
all_points_per_game = []
all_winning_percentage = []
all_playoff_appearances = []

combined_data = pd.concat([awards_players_data, coaches_data, players_data, players_teams_data, series_post_data, teams_data, teams_post_data], ignore_index=True)
combined_data.to_csv('combined_data.csv', index=False)

X = combined_data.drop('playoff', axis=1)
y = combined_data['playoff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

s = setup(data=combined_data, target='playoff', session_id=123, normalize=True)
compare_models()
