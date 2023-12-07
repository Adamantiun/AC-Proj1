from data_manip import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pycaret.classification import *

def pipeline(teams, players_teams, coaches):
    '''
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
    '''

    players_teams['fgRate'] = players_teams['fgMade'] / players_teams['fgAttempted']
    players_teams['ftRate'] = players_teams['ftMade'] / players_teams['ftAttempted']
    players_teams['threeRate'] = players_teams['threeMade'] / players_teams['threeAttempted']

    players_teams = players_teams.drop(["fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade", 'stint', 'lgID', 'GP', 'GS', 'minutes', 'points', "oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF"], axis=1)

    #players_teams.to_csv('2.0_data/players_teams.csv', index=False)

    # coaches to coachesWinRate

    coaches['coachWinRate'] = (coaches['won'] / (coaches['won'] + coaches['lost']))
    coaches = coaches.drop(["lgID", "stint", "won", "lost", "post_wins", "post_losses"], axis=1)
    coaches = coaches.round(2)

    #coaches.to_csv('2.0_data/coaches.csv', index=False)

    ### TEAMS TO TEAMS 2.0 ###

    # Feature selection
    selected_features = ['playoff', 'year', 'tmID', 'GP', 'o_pts', 'd_pts', 'o_reb', 'd_reb', 'o_asts', 'o_stl', 'o_blk', 'o_fga', 'o_to' , 'confW', 'confL', 'homeW', 'homeL', 'awayW', 'awayL', 'd_fga', 'd_stl']

    teams = teams[selected_features].copy()
    print(teams.head())

    teams['home_win_pct'] = teams['homeW'] / (teams['homeW'] + teams['homeL'])
    teams['away_win_pct'] = teams['awayW'] / (teams['awayW'] + teams['awayL'])
    teams['conf_win_rate'] = teams['confW'] / (teams['confW'] + teams['confL'])

    teams['scoring_efficiency'] = teams['o_pts'] / teams['o_fga']
    teams['reb_efficiency'] = teams['o_reb'] / teams['o_fga']
    teams['defensive_pts_efficiency'] = teams['d_pts'] / teams['GP']
    teams['def_reb_efficiency'] = teams['d_reb'] / teams['d_fga']
    teams['off_reb_efficiency'] = teams['o_reb'] / teams['o_fga']
    teams['ast_to_ratio'] = teams['o_asts'] / teams['o_to']

    teams['steals_per_game'] = teams['o_stl'] / teams['GP']
    teams['blocks_per_game'] = teams['o_blk'] / teams['GP']

    teams = teams.drop(['homeW','homeL', 'awayW', 'awayL', 'GP', 'o_pts', 'o_fga', 'o_asts', 'o_to', 'd_pts', 'd_fga', 'd_reb', 'd_stl', 'o_stl', 'o_blk', 'confW', 'confL'], axis=1)
    #teams.to_csv('2.0_data/teams.csv', index=False)

    #teams = pd.read_csv(os.path.join(current_dir, '2.0_data/teams.csv'))
    #coachesWinRate = pd.read_csv(os.path.join(current_dir, '2.0_data/coaches.csv'))
    #teams_stats = pd.read_csv(os.path.join(current_dir, '2.0_data/players_teams.csv'))

    combined_data = pd.merge(teams, coaches, on=['year', 'tmID'], how='outer')
    combined_data = pd.merge(combined_data, players_teams, on=['year', 'tmID'], how='outer')

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

