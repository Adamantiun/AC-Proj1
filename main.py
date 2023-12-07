from pipeline import *

### ORIGINAL DATA ###
players_teams = pd.read_csv('original_data/players_teams.csv')
coaches = pd.read_csv('original_data/coaches.csv')
teams = pd.read_csv('original_data/teams.csv')
year_to_predict = 10

pipeline(teams, players_teams, coaches, year_to_predict)

### COMPETITION DATA ###
players_teams = pd.read_csv('competition_data/players_teams.csv')
coaches = pd.read_csv('competition_data/coaches.csv')
teams = pd.read_csv('competition_data/teams.csv')
year_to_predict = 11

#pipeline(teams, players_teams, coaches, year_to_predict)

