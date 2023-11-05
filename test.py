import pandas as pd
import matplotlib.pyplot as plt

"""df = pd.read_csv('./modified_data/teams.csv')

df = df[(df['playoff'] == 'Y') & (df['year'] >= 8)]

result = df['name'].value_counts().reset_index()
result.columns = ['Team', 'Playoff Appearances']

result = result.sort_values(by='Playoff Appearances', ascending=False)

print(result)


players_df = pd.read_csv('./modified_data/players.csv')
players_teams_df = pd.read_csv('./modified_data/players_teams.csv')
teams_df = pd.read_csv('./modified_data/teams.csv')

players_df = players_df[~players_df['birthDate'].str.startswith('0000')]

players_df['birthYear'] = players_df['birthDate'].str[:4].astype(int)
players_df['idade'] = 2023 - players_df['birthYear']

players_teams_df = players_teams_df[players_teams_df['year'] == 10]

merged_df = players_teams_df.merge(players_df, left_on='playerID', right_on='bioID')

team_avg_age = merged_df.groupby('tmID')['idade'].mean().reset_index()

team_avg_age = team_avg_age.merge(teams_df, left_on='tmID', right_on='tmID')

plt.figure(figsize=(10, 6))
plt.bar(team_avg_age['name'], team_avg_age['idade'])
plt.xlabel('Team')
plt.ylabel('Average age')
plt.title('Average players age per team (Year 10)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


data = pd.read_csv("./modified_data/teams.csv")

last_10_years = data[data['year'] >= data['year'].max() - 9]

team_records = last_10_years.groupby('name')[['won', 'lost']].sum().reset_index()

team_records['win_percentage'] = (team_records['won'] / (team_records['won'] + team_records['lost'])) * 100

team_records = team_records.sort_values(by='win_percentage', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(team_records['name'], team_records['win_percentage'])
plt.xlabel('Win Percentage (%)')
plt.ylabel('Team')
plt.title('Win Percentage of All Teams in the Last 10 Years')
plt.grid(axis='x')

plt.show()


df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['Winning Percentage'] = (df['won'] / (df['won'] + df['lost'])) * 100

results= df[['name', 'Winning Percentage']]

results = results.sort_values(by='Winning Percentage', ascending=False)

print(results)


df = pd.read_csv("./modified_data/teams.csv")

df= df[df['year'] == 10]

df['PPG'] = (df['o_pts'] / df['GP']).round()

result = df[['name', 'PPG']]
result.columns= ['Team', 'PointsPerGame']

result = result.sort_values(by='PointsPerGame', ascending=False)

print(result)


df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['FG%'] = (df['o_fgm'] / df['o_fga']) * 100

fg_percentages = df.groupby('name')['FG%'].mean().reset_index()
result = fg_percentages.sort_values(by='FG%', ascending=False)


print(result)


df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['3P%'] = (df['o_3pm'] / df['o_3pa']) * 100

fg_percentages = df.groupby('name')['3P%'].mean().reset_index()
result = fg_percentages.sort_values(by='3P%', ascending=False)

print(result)


df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['FT%'] = (df['o_ftm'] / df['o_fta']) * 100

fg_percentages = df.groupby('name')['FT%'].mean().reset_index()
result = fg_percentages.sort_values(by='FT%', ascending=False)

print(result)



df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['Rebounds'] = (df['o_reb'] + df['d_reb'])/ df['GP']

fg_percentages = df.groupby('name')['Rebounds'].mean().reset_index()
result = fg_percentages.sort_values(by='Rebounds', ascending=False)

print(result)



df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['Assists'] = (df['o_asts'])/ df['GP']

fg_percentages = df.groupby('name')['Assists'].mean().reset_index()
result = fg_percentages.sort_values(by='Assists', ascending=False)

print(result)



df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['Steals and Blocks'] = (df['d_stl'] + df['d_blk']) / df['GP']

fg_percentages = df.groupby('name')['Steals and Blocks'].mean().reset_index()
result = fg_percentages.sort_values(by='Steals and Blocks', ascending=False)

print(result)



df = pd.read_csv("./modified_data/teams.csv")

df = df[df['year'] == 10]

df['Turnovers'] = (df['d_to']) / df['GP']

fg_percentages = df.groupby('name')['Turnovers'].mean().reset_index()
result = fg_percentages.sort_values(by='Turnovers', ascending=False)

print(result)"""


file_paths = ["./modified_data/teams.csv"] 

columns_to_use = ["name", "d_to", "GP", "d_stl", "d_blk", "o_asts", "o_reb", "d_reb", "o_ftm", "o_fta", "o_3pm", "o_3pa", "o_fgm", "o_fga", "o_pts", "won", "lost", "playoff"]

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

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df = df[df['year'] == 10]
    
    if 'd_to' in columns_to_use:
        df['Turnovers'] = (df['d_to']) / df['GP']
        turnovers_percentages = df.groupby('name')['Turnovers'].mean().reset_index()
        all_turnovers.append(turnovers_percentages)
    
    if 'd_stl' in columns_to_use or 'd_blk' in columns_to_use:
        df['Steals and Blocks'] = (df['d_stl'] + df['d_blk']) / df['GP']
        steals_blocks_percentages = df.groupby('name')['Steals and Blocks'].mean().reset_index()
        all_steals_blocks.append(steals_blocks_percentages)
    
    if 'o_asts' in columns_to_use:
        df['Assists'] = (df['o_asts']) / df['GP']
        assists_percentages = df.groupby('name')['Assists'].mean().reset_index()
        all_assists.append(assists_percentages)

    if 'o_reb' in columns_to_use or 'd_reb' in columns_to_use:
        df['Rebounds'] = (df['o_reb'] + df['d_reb']) / df['GP']
        rebounds_percentages = df.groupby('name')['Rebounds'].mean().reset_index()
        all_rebounds.append(rebounds_percentages)

    if 'o_ftm' in columns_to_use and 'o_fta' in columns_to_use:
        df['FT%'] = (df['o_ftm'] / df['o_fta']) * 100
        ft_percentages = df.groupby('name')['FT%'].mean().reset_index()
        all_ft_percentages.append(ft_percentages)

    if 'o_3pm' in columns_to_use and 'o_3pa' in columns_to_use:
        df['3P%'] = (df['o_3pm'] / df['o_3pa']) * 100
        three_point_percentages = df.groupby('name')['3P%'].mean().reset_index()
        all_3p_percentage.append(three_point_percentages)

    if 'o_fgm' in columns_to_use and 'o_fga' in columns_to_use:
        df['FG%'] = (df['o_fgm'] / df['o_fga']) * 100
        fg_percentages = df.groupby('name')['FG%'].mean().reset_index()
        all_fg_percentages.append(fg_percentages)

    if 'o_pts' in columns_to_use:
        df['PPG'] = (df['o_pts'] / df['GP']).round()
        points_per_game = df[['name', 'PPG']]
        points_per_game.columns = ['name', 'PointsPerGame']
        all_points_per_game.append(points_per_game)

    if 'won' in columns_to_use:
        df['wp'] = ((df['won'] / (df['won'] + df['lost'])) * 100).round()
        wp = df[['name', 'wp']]
        wp.columns = ['name', 'Winning Percentage']
        all_winning_percentage.append(wp)

    if 'playoff' in columns_to_use:
        df = df[df['playoff'] == 'Y']
        playoff_appearances = df['name'].value_counts().reset_index()
        playoff_appearances.columns = ['name', 'Playoff Appearances']
        all_playoff_appearances.append(playoff_appearances)

final_turnovers_df = pd.concat(all_turnovers, ignore_index=True)
final_steals_blocks_df = pd.concat(all_steals_blocks, ignore_index=True)
final_assists_df = pd.concat(all_assists, ignore_index=True)
final_rebounds_df = pd.concat(all_rebounds, ignore_index=True)
final_ft_percentages_df = pd.concat(all_ft_percentages, ignore_index=True)
final_3p_percentage_df = pd.concat(all_3p_percentage, ignore_index=True)
final_fg_percentages_df = pd.concat(all_fg_percentages, ignore_index=True)
final_points_per_game_df = pd.concat(all_points_per_game, ignore_index=True)
final_winning_percentage_df = pd.concat(all_winning_percentage, ignore_index=True)
final_playoff_appearances_df = pd.concat(all_playoff_appearances, ignore_index=True)

final_combined_df = final_turnovers_df.merge(final_steals_blocks_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_assists_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_rebounds_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_ft_percentages_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_3p_percentage_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_fg_percentages_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_points_per_game_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_winning_percentage_df, on='name', how='outer')
final_combined_df = final_combined_df.merge(final_playoff_appearances_df, on='name', how='outer')
final_combined_df['Playoff prediction'] = 0

final_combined_df.to_csv("combined_data.csv", index=False)

print(final_combined_df)