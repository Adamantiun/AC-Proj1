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

column_mapping = {'bioID': 'playerID'}
rename_columns(players_output_path, column_mapping, players_output_path)


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



# players_teams modificado para teams_stats

df = pd.read_csv('original_data/players_teams.csv')

df_teams = df.groupby(['year', 'tmID']).agg({
    'GP': 'sum',
    'GS': 'sum',
    'minutes': 'sum',
    'points': 'sum',
    'oRebounds': 'sum',
    'dRebounds': 'sum',
    'rebounds': 'sum',
    'assists': 'sum',
    'steals': 'sum',
    'blocks': 'sum',
    'turnovers': 'sum',
    'PF': 'sum',
    'fgAttempted': 'sum',
    'fgMade': 'sum',
    'ftAttempted': 'sum',
    'ftMade': 'sum',
    'threeAttempted': 'sum',
    'threeMade': 'sum',
    'dq': 'sum',
    'PostGP': 'sum',
    'PostGS': 'sum',
    'PostMinutes': 'sum',
    'PostPoints': 'sum',
    'PostoRebounds': 'sum',
    'PostdRebounds': 'sum',
    'PostRebounds': 'sum',
    'PostAssists': 'sum',
    'PostSteals': 'sum',
    'PostBlocks': 'sum',
    'PostTurnovers': 'sum',
    'PostPF': 'sum',
    'PostfgAttempted': 'sum',
    'PostfgMade': 'sum',
    'PostftAttempted': 'sum',
    'PostftMade': 'sum',
    'PostthreeAttempted': 'sum',
    'PostthreeMade': 'sum',
    'PostDQ': 'sum'
}).reset_index()

df_teams = df_teams.round(2)

df_teams.to_csv('2.0_data/teams_stats.csv', index=False)

msg = "team_stats.csv created\n"
print(msg)


# coaches to coachesWinRate

df_coaches = pd.read_csv('original_data/coaches.csv')

df_coaches['games'] = df_coaches['won'] + df_coaches['lost']

df_win_loss = df_coaches.groupby('coachID').agg({
    'games': 'sum',
    'won': 'sum'
}).reset_index()

df_win_loss['currentWinRate'] = df_win_loss['won'] / df_win_loss['games']

df_win_loss = df_win_loss.round(2)

df_result = df_win_loss[['coachID', 'games', 'currentWinRate']]

df_result.to_csv('2.0_data/coachesWinRate.csv', index=False)

msg = "coachesWinRate.csv created"
print(msg)


"""awards_players_data = pd.read_csv(os.path.join(current_dir, 'modified_data/awards_players.csv'))
coaches_data = pd.read_csv(os.path.join(current_dir, 'modified_data/coaches.csv'))
players_data = pd.read_csv(os.path.join(current_dir, 'modified_data/players.csv'))
players_teams_data = pd.read_csv(os.path.join(current_dir, 'modified_data/players_teams.csv'))
series_post_data = pd.read_csv(os.path.join(current_dir, 'modified_data/series_post.csv'))
teams_data = pd.read_csv(os.path.join(current_dir, 'modified_data/teams.csv'))
teams_post_data = pd.read_csv(os.path.join(current_dir, 'modified_data/teams_post.csv'))

columns_to_use = ["name", "d_to", "GP", "d_stl", "d_blk", "o_asts", "o_reb", "d_reb", "o_ftm", "o_fta", "o_3pm", "o_3pa", "o_fgm", "o_fga", "o_pts", "won", "lost"]

combined_data = pd.merge(awards_players_data, players_data, on='playerID', how='inner')
combined_data = pd.merge(players_teams_data, combined_data, on='playerID', how='inner')
combined_data = pd.merge(teams_data, combined_data, on='tmID', how='inner')
combined_data = pd.merge(coaches_data, combined_data, on='tmID', how='inner')
combined_data = pd.merge(teams_post_data, combined_data, on='tmID', how='inner')
combined_data = pd.merge(series_post_data, combined_data, on='year', how='inner')

combined_data['playoff'].fillna('NA', inplace=True)
combined_data.to_csv('combined_data.csv', index=False)
combined_data.drop(columns=['year_x', 'year_y'], inplace=True)



X = combined_data.drop('playoff', axis=1)
y = combined_data['playoff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

s = setup(data=combined_data, target='playoff', session_id=123, normalize=True)
compare_models()"""
