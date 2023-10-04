import pandas as pd

def exclude_columns(input_path, columns_to_exclude, output_path):
    df = pd.read_csv(input_path)
    df = df.drop(columns=columns_to_exclude)
    df.to_csv(output_path, index=False)
    print(f"Columns {', '.join(columns_to_exclude)} removed and new file saved as '{output_path}'" + '\n')

def convert_columns_to_ratio(input_path, column_pairs, output_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_path)

    # Convert specified column pairs to "fgRatio" columns
    for pair in column_pairs:
        made_column, attempted_column = pair
        df[f'{made_column}_{attempted_column}_Ratio'] = df[made_column] / df[attempted_column]

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)

    print(f"Columns {', '.join([f'{pair[0]}/{pair[1]}' for pair in column_pairs])} converted to 'fgRatio' columns and new file saved as '{output_path}'" + '\n')

# PLAYERS_TEAMS
teams_input_path = 'original_data/players_teams.csv'
teams_column_pairs = [('fgMade', 'fgAttempted'), ('ftMade', 'ftAttempted'), ('threeMade', 'threeAttempted')]
teams_output_path = 'modified_data/players_teams.csv'

convert_columns_to_ratio(teams_input_path, teams_column_pairs, teams_output_path)

# TEAMS
teams_input_path = 'original_data/teams.csv'
teams_columns_to_exclude = ['confW', 'confL', 'min', 'attend', 'arena']
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
