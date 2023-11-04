from data_manip import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pycaret.classification import *

current_dir = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(current_dir, 'combined_data.csv'))

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


# Important statistic concatenated into combined_data.csv

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

data = data[data['year'] == 10]
    
if 'd_to' in columns_to_use:
    data['Turnovers'] = (data['d_to']) / data['GP']
    turnovers_percentages = data.groupby('name')['Turnovers'].mean().reset_index()
    all_turnovers.append(turnovers_percentages)
    
if 'd_stl' in columns_to_use or 'd_blk' in columns_to_use:
    data['Steals and Blocks'] = (data['d_stl'] + data['d_blk']) / data['GP']
    steals_blocks_percentages = data.groupby('name')['Steals and Blocks'].mean().reset_index()
    all_steals_blocks.append(steals_blocks_percentages)
    
if 'o_asts' in columns_to_use:
    data['Assists'] = (data['o_asts']) / data['GP']
    assists_percentages = data.groupby('name')['Assists'].mean().reset_index()
    all_assists.append(assists_percentages)

if 'o_reb' in columns_to_use or 'd_reb' in columns_to_use:
    data['Rebounds'] = (data['o_reb'] + data['d_reb']) / data['GP']
    rebounds_percentages = data.groupby('name')['Rebounds'].mean().reset_index()
    all_rebounds.append(rebounds_percentages)

if 'o_ftm' in columns_to_use and 'o_fta' in columns_to_use:
    data['FT%'] = (data['o_ftm'] / data['o_fta']) * 100
    ft_percentages = data.groupby('name')['FT%'].mean().reset_index()
    all_ft_percentages.append(ft_percentages)

if 'o_3pm' in columns_to_use and 'o_3pa' in columns_to_use:
    data['3P%'] = (data['o_3pm'] / data['o_3pa']) * 100
    three_point_percentages = data.groupby('name')['3P%'].mean().reset_index()
    all_3p_percentage.append(three_point_percentages)

if 'o_fgm' in columns_to_use and 'o_fga' in columns_to_use:
    data['FG%'] = (data['o_fgm'] / data['o_fga']) * 100
    fg_percentages = data.groupby('name')['FG%'].mean().reset_index()
    all_fg_percentages.append(fg_percentages)

if 'o_pts' in columns_to_use:
    data['PPG'] = (data['o_pts'] / data['GP']).round()
    points_per_game = data[['name', 'PPG']]
    points_per_game.columns = ['name', 'PointsPerGame']
    all_points_per_game.append(points_per_game)

if 'won' in columns_to_use:
    data['wp'] = ((data['won'] / (data['won'] + data['lost'])) * 100).round()
    wp = data[['name', 'wp']]
    wp.columns = ['name', 'Winning Percentage']
    all_winning_percentage.append(wp)

#if 'playoff' in columns_to_use:
    #data = data[data['playoff'] == 'Y']
    ##playoff_appearances.columns = ['name', 'Playoff Appearances']
    #all_playoff_appearances.append(playoff_appearances)

final_turnovers_data = pd.concat(all_turnovers, ignore_index=True)
final_steals_blocks_data = pd.concat(all_steals_blocks, ignore_index=True)
final_assists_data = pd.concat(all_assists, ignore_index=True)
final_rebounds_data = pd.concat(all_rebounds, ignore_index=True)
final_ft_percentages_data = pd.concat(all_ft_percentages, ignore_index=True)
final_3p_percentage_data = pd.concat(all_3p_percentage, ignore_index=True)
final_fg_percentages_data = pd.concat(all_fg_percentages, ignore_index=True)
final_points_per_game_data = pd.concat(all_points_per_game, ignore_index=True)
final_winning_percentage_data = pd.concat(all_winning_percentage, ignore_index=True)
#final_playoff_appearances_data = pd.concat(all_playoff_appearances, ignore_index=True)

final_combined_data = final_turnovers_data.merge(final_steals_blocks_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_assists_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_rebounds_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_ft_percentages_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_3p_percentage_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_fg_percentages_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_points_per_game_data, on='name', how='outer')
final_combined_data = final_combined_data.merge(final_winning_percentage_data, on='name', how='outer')
#final_combined_data = final_combined_data.merge(final_playoff_appearances_data, on='name', how='outer')

final_combined_data.to_csv("combined_data.csv", index=False)

#print(final_combined_data)

#data = pd.read_csv(os.path.join(current_dir, 'combined_data.csv'))

X = data.drop('playoff', axis=1)
y = data['playoff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

s = setup(data=data, target='playoff', session_id=123, normalize=True)
compare_models()
