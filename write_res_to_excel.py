import pandas as pd
import ast
import os

directory = input("Enter directory path: ")
files = os.listdir(directory)

# Create an empty list to store all dataframes
dfs = []

for file1 in files:
    if not file1.endswith('.py'):
        with open(os.path.join(directory, file1), 'r', encoding='utf-8') as file:
            data_str = file.read()
        
        # Convert string representation of dictionary to dictionary object
        data_dict = ast.literal_eval(data_str)

        choice = 2

        # Extract relevant data
        net_fname = data_dict['net_fname']
        prop_fname = data_dict['prop_fname']
        result = data_dict['result']
        time = data_dict['time']
        net_size = data_dict['net_size']
        if choice==1:
            no_of_refine_steps = data_dict['no_of_refine_steps']
            time_for_solver_calls = data_dict['time_for_solver_calls']
            last_solver_time = time_for_solver_calls[-1]
            times_for_each_refine_step = data_dict['times_for_each_refine_step']

        # Create a DataFrame
        if choice==1:
            df = pd.DataFrame({'net_fname': [net_fname],
                            'prop_fname': [prop_fname],
                            'result': [result],
                            'time': [time],
                            'net_size': [net_size],
                            'no_of_refine_steps': [no_of_refine_steps],
                            'time_for_solver_calls':[time_for_solver_calls],
                            'times_for_each_refine_step':[times_for_each_refine_step], 
                            'last_solver_time':[last_solver_time]
                            })
        else:
            df = pd.DataFrame({'net_fname': [net_fname],
                           'prop_fname': [prop_fname],
                           'result': [result],
                           'time': [time],
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
