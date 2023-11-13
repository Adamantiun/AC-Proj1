from data_manip import *
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pycaret.classification import *

current_dir = os.path.dirname(os.path.abspath(__file__))

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

df_teams['fgRate'] = df_teams['fgMade'] / df_teams['fgAttempted']
df_teams['ftRate'] = df_teams['ftMade'] / df_teams['ftAttempted']
df_teams['threeRate'] = df_teams['threeMade'] / df_teams['threeAttempted']
df_teams['postFgRate'] = df_teams['PostfgMade'] / df_teams['PostfgAttempted']
df_teams['postFtRate'] = df_teams['PostftMade'] / df_teams['PostftAttempted']
df_teams['postThreeRate'] = df_teams['PostthreeMade'] / df_teams['PostthreeAttempted']

df_teams = df_teams.drop(['fgAttempted', 'fgMade'], axis=1)
df_teams = df_teams.drop(['ftAttempted', 'ftMade'], axis=1)
df_teams = df_teams.drop(['threeAttempted', 'threeMade'], axis=1)
df_teams = df_teams.drop(['PostfgMade', 'PostfgAttempted'], axis=1)
df_teams = df_teams.drop(['PostftMade', 'PostftAttempted'], axis=1)
df_teams = df_teams.drop(['PostthreeMade', 'PostthreeAttempted'], axis=1)

# now adding average player awards
df_awards = pd.read_csv('original_data/awards_players.csv')
df_players = pd.read_csv('original_data/players.csv')
df_players_teams = pd.read_csv('original_data/players_teams.csv')
# Processar os prêmios dos jogadores
df_awards_summed = df_awards.groupby('playerID').size().reset_index(name='totalAwards')
# Juntar os jogadores com os prêmios
df_players_with_awards = pd.merge(df_players, df_awards_summed, how='left', left_on='bioID', right_on='playerID')
df_players_with_awards['totalAwards'] = df_players_with_awards['totalAwards'].fillna(0).astype(int)
df_players_with_awards = df_players_with_awards.drop(columns='playerID')
# Juntar os jogadores com as equipes
df_merged_teams = pd.merge(df_players_teams, df_players_with_awards, how='left', left_on='playerID', right_on='bioID')
# Calcular a média de prêmios por equipe em cada ano
df_teams_avg_awards = df_merged_teams.groupby(['tmID', 'year'])['totalAwards'].mean().reset_index().rename(columns={'totalAwards': 'avgAwards'})
# Juntar as equipes com a média de prêmios
df_teams = pd.merge(df_teams, df_teams_avg_awards, how='left', on=['tmID', 'year'])

# adicionar dados de series_post.csv
df_series_post = pd.read_csv('original_data/series_post.csv')
# Criar uma coluna 'maxRound' usando um dicionário de mapeamento
round_mapping = {'FR': 1, 'CF': 2, 'F': 3}
df_series_post['maxRound'] = df_series_post['round'].map(round_mapping)
# Criar um DataFrame auxiliar para 'tmIDWinner'
df_winner = df_series_post.groupby(['year', 'tmIDWinner']).agg(sp_winCount=('round', 'count'), sp_maxRound=('maxRound', 'max')).reset_index()
# Criar um DataFrame auxiliar para 'tmIDLoser'
df_loser = df_series_post.groupby(['year', 'tmIDLoser']).agg(sp_lossCount=('round', 'count')).reset_index()
# Juntar os DataFrames auxiliares para 'tmIDWinner' e 'tmIDLoser'
df_relevant_series_post = pd.merge(df_winner, df_loser, how='outer', left_on=['year', 'tmIDWinner'], right_on=['year', 'tmIDLoser'])
# Preencher valores nulos com 0 e remover linhas com 'tmID' nulo
df_relevant_series_post['sp_lossCount'] = df_relevant_series_post['sp_lossCount'].fillna(0)
df_relevant_series_post = df_relevant_series_post.dropna(subset=['tmIDWinner'])
# Calcular a winrate
df_relevant_series_post['sp_winRate'] = (df_relevant_series_post['sp_winCount'] / (df_relevant_series_post['sp_winCount'] + df_relevant_series_post['sp_lossCount'])).fillna(0)
# Selecionar as colunas relevantes
df_relevant_series_post = df_relevant_series_post[['year', 'tmIDWinner', 'sp_winRate', 'sp_maxRound']]
# Fazer o join com o DataFrame df_relevant_series_post
df_teams = pd.merge(df_teams, df_relevant_series_post, how='left', left_on=['year', 'tmID'], right_on=['year', 'tmIDWinner'])
# Preencher os valores nulos com 0 para as colunas 'sp_winRate' e 'sp_maxRound'
df_teams['sp_winRate'] = df_teams['sp_winRate'].fillna(0)
df_teams['sp_maxRound'] = df_teams['sp_maxRound'].fillna(0)

# coaches to coachesWinRate

df_coaches = pd.read_csv('original_data/coaches.csv')
df_coaches['coachWinRate'] = (df_coaches['won'] / (df_coaches['won'] + df_coaches['lost']))
df_coaches['coachTotalGames'] = (df_coaches['won'] + df_coaches['lost'])
df_coaches = df_coaches.round(2)
df_coaches = df_coaches.drop(["lgID", "stint", "won", "lost", "post_wins", "post_losses", "coachID"], axis=1)

df_teams = pd.merge(df_teams, df_coaches, how='left', on=['year', 'tmID'])

df_teams = df_teams.round(2)

df_teams.to_csv('2.0_data/teams_stats.csv', index=False)

msg = "team_stats.csv created\n"
print(msg)

# coaches to coachesWinRate

df_coaches = pd.read_csv('original_data/coaches.csv')

df_coaches['coachWinRate'] = (df_coaches['won'] / (df_coaches['won'] + df_coaches['lost']))

df_coaches = df_coaches.round(2)

df_coaches = df_coaches.drop(["lgID", "stint", "won", "lost", "post_wins", "post_losses"], axis=1)

df_coaches.to_csv('2.0_data/coachesWinRate.csv', index=False)

msg = "coachesWinRate.csv created"
print(msg)

### TEAMS TO TEAMS 2.0 ###

teams = pd.read_csv('original_data/teams.csv')
teams_post = pd.read_csv('original_data/teams_post.csv')

# Feature selection
selected_features = ['playoff', 'year', 'tmID', 'won', 'lost', 'GP', 'o_pts', 'd_pts', 'o_reb', 'd_reb', 'o_asts', 'o_stl', 'o_blk', 'o_fga', 'o_to' , 'confW', 'confL', 'homeW', 'homeL', 'awayW', 'awayL', 'd_fga', 'd_stl']

# Create teams 2.0 dataframe with selected features
teams_2_0 = teams[selected_features].copy()

# Calculate win rates
teams_2_0['home_win_pct'] = teams_2_0['homeW'] / (teams_2_0['homeW'] + teams_2_0['homeL'])
teams_2_0['away_win_pct'] = teams_2_0['awayW'] / (teams_2_0['awayW'] + teams_2_0['awayL'])
teams_2_0['conf_win_rate'] = teams_2_0['confW'] / (teams_2_0['confW'] + teams_2_0['confL'])

# Efficiencies
teams_2_0['scoring_efficiency'] = teams_2_0['o_pts'] / teams_2_0['o_fga']
teams_2_0['reb_efficiency'] = teams_2_0['o_reb'] / teams_2_0['o_fga']
teams_2_0['defensive_pts_efficiency'] = teams_2_0['d_pts'] / teams_2_0['GP']
teams_2_0['def_reb_efficiency'] = teams_2_0['d_reb'] / teams_2_0['d_fga']
teams_2_0['off_reb_efficiency'] = teams_2_0['o_reb'] / teams_2_0['o_fga']
teams_2_0['ast_to_ratio'] = teams_2_0['o_asts'] / teams_2_0['o_to']

# Per games
teams_2_0['steals_per_game'] = teams_2_0['o_stl'] / teams_2_0['GP']
teams_2_0['blocks_per_game'] = teams_2_0['o_blk'] / teams_2_0['GP']

# Add playoff information if available
if 'W' in teams_post.columns and 'L' in teams_post.columns:
    total_playoff_games = teams_post['W'] + teams_post['L']
    teams_2_0['playoff_win_pct'] = teams_post['W'] / total_playoff_games if not total_playoff_games.empty and (total_playoff_games > 0).all() else 0
else:
    teams_2_0['playoff_win_pct'] = 0

teams_2_0['playoff_win_pct'].fillna(0, inplace=True)
teams_2_0.to_csv('2.0_data/teams.csv', index=False)

teams_input_path = '2.0_data/teams.csv'
teams_columns_to_exclude = ['homeW', 'homeL', 'awayW', 'awayL', 'o_pts', 'o_fga', 'o_asts', 'o_to', 'won', 'GP', 'd_pts', 'd_fga', 'd_reb', 'd_stl', 'o_stl', 'o_blk', 'confW', 'confL']
teams_output_path = '2.0_data/teams.csv'
exclude_columns(teams_input_path, teams_columns_to_exclude, teams_output_path)

teams = pd.read_csv(os.path.join(current_dir, '2.0_data/teams.csv'))
coachesWinRate = pd.read_csv(os.path.join(current_dir, '2.0_data/coachesWinRate.csv'))
teams_stats = pd.read_csv(os.path.join(current_dir, '2.0_data/teams_stats.csv'))

combined_data = pd.merge(teams, teams_stats, on='tmID', how='outer')
#combined_data = pd.merge(coachesWinRate, combined_data, on='tmID', how='outer')

combined_data['playoff'].fillna('NA', inplace=True)
combined_data.to_csv('combined_data.csv', index=False)
combined_data.drop(columns=['year_x', 'year_y'], inplace=True)

X = combined_data.drop('playoff', axis=1)
y = combined_data['playoff']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

s = setup(data=combined_data, target='playoff', session_id=123, normalize=True)
compare_models()
