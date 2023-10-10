import pandas as pd
import matplotlib.pyplot as plt

"""teams_df = pd.read_csv('./modified_data/teams.csv')

# Tabela com a contagem de vezes que cada equipa foi aos playoffs

playoff_teams = teams_df[teams_df['playoff'] == 'Y']

playoff_count = playoff_teams['name'].value_counts().reset_index()
playoff_count.columns = ['Name', 'Playoff Appearances']

playoff_count = playoff_count.sort_values(by='Playoff Appearances', ascending=False)
print(playoff_count)

plt.figure(figsize=(10, 6))
plt.bar(playoff_count['Name'], playoff_count['Playoff Appearances'])
plt.xlabel('Equipa')
plt.ylabel('Nº de Aparições nos Playoffs')
plt.title('Nº de Aparições nos Playoffs/Equipa')
plt.xticks(rotation=90) 
plt.tight_layout()

plt.show()"""

# Média de idades das jogadoras por equipas

players_df = pd.read_csv('./modified_data/players.csv')
players_teams_df = pd.read_csv('./modified_data/players_teams.csv')
teams_df = pd.read_csv('./modified_data/teams.csv')

# Elimine as linhas em players.csv onde os primeiros 4 dígitos de birthDate são todos iguais a 0
players_df = players_df[~players_df['birthDate'].str.startswith('0000')]

# Calcule a idade dos jogadores com base em birthDate
players_df['birthYear'] = players_df['birthDate'].str[:4].astype(int)
players_df['idade'] = 2023 - players_df['birthYear']

# Junte os DataFrames com base em bioID/playerID
merged_df = players_teams_df.merge(players_df, left_on='playerID', right_on='bioID')

# Calcule a média de idades dos jogadores por tmID
team_avg_age = merged_df.groupby('tmID')['idade'].mean().reset_index()

# Passo 5: Junte com os nomes das equipas a partir de teams.csv usando tmID
team_avg_age = team_avg_age.merge(teams_df, left_on='tmID', right_on='tmID')

plt.figure(figsize=(10, 6))
plt.bar(team_avg_age['name'], team_avg_age['idade'])
plt.xlabel('Nome da Equipe')
plt.ylabel('Média de Idades dos Jogadores')
plt.title('Média de Idades dos Jogadores por Equipe')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
