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
        df[f'{made_column}_{attempted_column}_Ratio'] = round(df[made_column] / df[attempted_column], 2)
        df[f'{made_column}_{attempted_column}_Ratio'].replace('NaN', 0)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_path, index=False)

    print(f"Columns {', '.join([f'{pair[0]}/{pair[1]}' for pair in column_pairs])} converted to ratio columns and new file saved as '{output_path}'" + '\n')


