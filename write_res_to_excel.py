"""
Script to summarize data in stats files into a excel spreadsheet
"""
import pandas as pd
import ast
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--stats-path',
    dest='stats_path',
    required=False,
    type=str,
    default='stats/',
    help="The stats path to load from"
)
args = parser.parse_args()

files = os.listdir( args.stats_path )

# Create an empty list to store all dataframes
dfs = []

for file1 in files:
    if not file1.endswith('.py'):
        with open(os.path.join(args.stats_path, file1), 'r', encoding='utf-8') as file:
            data_str = file.read()
        
        # Convert string representation of dictionary to dictionary object
        try:
            data_dict = ast.literal_eval(data_str)
        except e:
            print("Skipping file {}, exception: {}".format( file1, e ))
            continue

        # Extract relevant data
        net_fname = data_dict['net_fname']
        prop_fname = data_dict['prop_fname']
        result = data_dict['result']
        net_size = data_dict['net_size']
        df = pd.DataFrame({'net_fname': [net_fname],
                       'prop_fname': [prop_fname],
                       'result': [result],
                       'net_size': [net_size]
                       })
        
        # Append the dataframe to the list
        dfs.append(df)

# Concatenate all dataframes into a single dataframe
final_df = pd.concat(dfs, ignore_index=True)

# Write the concatenated dataframe to an Excel file
excel_file = 'output.xlsx'
final_df.to_excel(excel_file, index=False)
print(f"Data written to {excel_file} successfully.")
