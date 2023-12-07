import pandas as pd
from pipeline import *

players_teams = pd.read_csv('original_data/players_teams.csv')
coaches = pd.read_csv('original_data/coaches.csv')
teams = pd.read_csv('original_data/teams.csv')

pipeline(teams, players_teams, coaches)

