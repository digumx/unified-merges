"""
A small script to convert the standard .vnnlib property files into our .prop
property files
"""
import re
import os
import os.path
import argparse


def extract_variable_index(variable):
    match_x = re.match(r'X_(\d+)', variable)
    match_y = re.match(r'Y_(\d+)', variable)
    if match_x:
        return int(match_x.group(1))
    elif match_y:
        return int(match_y.group(1))
    else:
        return None


def parse_assertions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    bounds = {}

    for line in lines:
        match_lower = re.match(r'\(assert \(\>= (\w+) (\d+\.\d+)\)\)', line)
        match_upper = re.match(r'\(assert \(<= (\w+) (\d+\.\d+)\)\)', line)
        if match_lower:
            variable, value = match_lower.groups()
            index = extract_variable_index(variable)
            if str(index) not in bounds:
                bounds[str(index)] = {}
            bounds[str(index)]['Lower'] = float(value)
        elif match_upper:
            variable, value = match_upper.groups()
            index = extract_variable_index(variable)
            if str(index) not in bounds:
                bounds[str(index)] = {}
            bounds[str(index)]['Upper'] = float(value)


    return bounds

def parse_output_asserts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    constraints = []

    for line in lines:
        match = re.match(r'\(and \((>= (\w+) (\w+))\)\)', line.strip())
        match1 = re.match(r'\(and \((<= (\w+) (\w+))\)\)', line.strip())
        if match:
            op, variable1, variable2 = match.groups()[:1][0].split()
            constraints.append((op, variable1, variable2))
        if match1:
            op, variable1, variable2 = match.groups()[:3]
            constraints.append((op, variable1, variable2))

    return constraints


def convert_to_output_format(constraints):
    output = {'output': []}

    for op, variable1, variable2 in constraints:
        index1 = extract_variable_index(variable1)
        index2 = extract_variable_index(variable2)
        lower_bound = 0
        upper_bound = 0

        if str(op) == '>=' and index1 is not None and index2 is not None:
            output['output'].append(([(1, index1), (-1, index2)], {'Lower': lower_bound}))
        if str(op) == '<=' and index1 is not None and index2 is not None:
            output['output'].append(([(1, index1), (-1, index2)], {'Upper': lower_bound}))
    
    return output



def write_output_file(output_file_path, bounds, output_constraints):
    with open(output_file_path, 'w') as output_file:
        output_file.write("{\n\t'input' : [\n")

        for variable, values in bounds.items():
            output_file.write(f"\t\t({variable}, {values}),\n")

        output_file.write("\t]\n,")

        output_file.write("\t'output' :")

        for variable, values in output_constraints.items():
            output_file.write(f"\t{values}\n")
        
        output_file.write(",\n}")



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-ip_dir', 
        dest='input_dir',
        required=True,
        type=str,
        help="The input directory for getting the vnnlib files."
    )

    parser.add_argument('-op_dir', 
        dest='output_dir',
        required=True,
        type=str,
        help="The output directory for dumping the prop files."
    )  

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    for itm in os.listdir( input_dir ):
        if not itm.endswith( '.vnnlib' ):
            continue
    
        input_file_path = os.path.join( input_dir, itm )
        print("Handling: ", input_file_path)
        output_file_path = os.path.join( output_dir, "{}.prop".format( itm ))

        bounds = parse_assertions(input_file_path)
        constraints = parse_output_asserts(input_file_path)
        output_constraints = convert_to_output_format(constraints)
        write_output_file(output_file_path, bounds, output_constraints)

if __name__ == "__main__":
    main()
