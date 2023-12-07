from data_manip import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pycaret.classification import *

def pipeline(teams, players_teams, coaches):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # players_teams modificado para teams_stats

    players_teams = pd.read_csv('original_data/players_teams.csv')

    players_teams = players_teams.groupby(['year', 'tmID']).agg({
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

    players_teams['fgRate'] = players_teams['fgMade'] / players_teams['fgAttempted']
    players_teams['ftRate'] = players_teams['ftMade'] / players_teams['ftAttempted']
    players_teams['threeRate'] = players_teams['threeMade'] / players_teams['threeAttempted']

    players_teams = players_teams.drop(['fgAttempted', 'fgMade'], axis=1)
    players_teams = players_teams.drop(['ftAttempted', 'ftMade'], axis=1)
    players_teams = players_teams.drop(['threeAttempted', 'threeMade'], axis=1)
    players_teams = players_teams.drop(['PostfgMade', 'PostfgAttempted'], axis=1)
    players_teams = players_teams.drop(['PostftMade', 'PostftAttempted'], axis=1)
    players_teams = players_teams.drop(['PostthreeMade', 'PostthreeAttempted'], axis=1)

    players_teams.to_csv('2.0_data/players_teams.csv', index=False)

    # coaches to coachesWinRate

    df_coaches = pd.read_csv('original_data/coaches.csv')

    df_coaches['coachWinRate'] = (df_coaches['won'] / (df_coaches['won'] + df_coaches['lost']))
    df_coaches = df_coaches.drop(["lgID", "stint", "won", "lost", "post_wins", "post_losses"], axis=1)
    df_coaches = df_coaches.round(2)

    df_coaches.to_csv('2.0_data/coaches.csv', index=False)

    ### TEAMS TO TEAMS 2.0 ###

    teams_2_0 = pd.read_csv('original_data/teams.csv')

    # Feature selection
    selected_features = ['playoff', 'year', 'tmID', 'GP', 'o_pts', 'd_pts', 'o_reb', 'd_reb', 'o_asts', 'o_stl', 'o_blk', 'o_fga', 'o_to' , 'confW', 'confL', 'homeW', 'homeL', 'awayW', 'awayL', 'd_fga', 'd_stl']

    teams_2_0 = teams_2_0[selected_features].copy()
    print(teams_2_0.head())

    teams_2_0['home_win_pct'] = teams_2_0['homeW'] / (teams_2_0['homeW'] + teams_2_0['homeL'])
    teams_2_0['away_win_pct'] = teams_2_0['awayW'] / (teams_2_0['awayW'] + teams_2_0['awayL'])
    teams_2_0['conf_win_rate'] = teams_2_0['confW'] / (teams_2_0['confW'] + teams_2_0['confL'])

    teams_2_0['scoring_efficiency'] = teams_2_0['o_pts'] / teams_2_0['o_fga']
    teams_2_0['reb_efficiency'] = teams_2_0['o_reb'] / teams_2_0['o_fga']
    #teams_2_0['defensive_pts_efficiency'] = teams_2_0['d_pts'] / teams_2_0['GP']
    teams_2_0['def_reb_efficiency'] = teams_2_0['d_reb'] / teams_2_0['d_fga']
    teams_2_0['off_reb_efficiency'] = teams_2_0['o_reb'] / teams_2_0['o_fga']
    teams_2_0['ast_to_ratio'] = teams_2_0['o_asts'] / teams_2_0['o_to']

    teams_2_0['steals_per_game'] = teams_2_0['o_stl'] / teams_2_0['GP']
    teams_2_0['blocks_per_game'] = teams_2_0['o_blk'] / teams_2_0['GP']

    teams_2_0 = teams_2_0.drop(['homeW','homeL', 'awayW', 'awayL', 'GP', 'o_pts', 'o_fga', 'o_asts', 'o_to', 'd_pts', 'd_fga', 'd_reb', 'd_stl', 'o_stl', 'o_blk', 'confW', 'confL'], axis=1)
    teams_2_0.to_csv('2.0_data/teams.csv', index=False)

    teams = pd.read_csv(os.path.join(current_dir, '2.0_data/teams.csv'))
    coachesWinRate = pd.read_csv(os.path.join(current_dir, '2.0_data/coaches.csv'))
    teams_stats = pd.read_csv(os.path.join(current_dir, '2.0_data/players_teams.csv'))

    combined_data = pd.merge(teams, coachesWinRate, on=['year', 'tmID'], how='outer')

    combined_data = combined_data.round(2)

    ## Dados treino ##
    X = combined_data[combined_data['year'] < 10]
    y = combined_data[combined_data['year'] == 10]

    X = X.drop('year', axis=1)
    y = y.drop('year', axis=1)

    combined_data.to_csv('combined_data.csv', index=False)

    s = setup(data=X, target='playoff', session_id=123, normalize=True)
    models = ['rf', 'et', 'gbc', 'lr', 'dt', 'svm', 'lda', 'ridge', 'ada', 'knn', 'nb', 'qda', 'dummy']
    for model in models:
        best = compare_models(include=[model])
        predictions = predict_model(best, data=y)

pipeline('a','a','a')
