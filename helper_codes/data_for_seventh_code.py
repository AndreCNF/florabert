import os
import numpy as np

input_dir = r'C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\FloraBERT_feature_counts\normalized_tpm_data' 
output_dir = r'C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\florabert\data\raw\gene_expression\Normalized_TPMs_each_NAM_line_aligned_against_own_refgen'
input_files = [filename for filename in os.listdir(input_dir)]
dic_for_run = {}
runs = []

with open(r"C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\florabert\helper_files\Ordering_Runs.txt", 'r') as file:
    cultivar = ""
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if not line.startswith("E"):
            cultivar = line
        else:
            dic_for_run[line] = [cultivar]
            runs.append(line)

runs.reverse()

with open(r"C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\florabert\helper_codes_outputs\unique_org.tsv", 'r') as file:
    lines = file.readlines()
    for line in lines[1 : ]:
        parts = line.strip().split()
        if parts[0] in dic_for_run:
            dic_for_run[parts[0]].append(parts[2])

# print(runs, "\n")
# print(dic_for_run, "\n")
# use the above print statements to visualize dictionary and list created and identifying bugs

i = 0
dic_for_cultivar = {}

# Process each input file
for input_file in input_files:
    # Construct the full path of the input file
    input_path = os.path.join(input_dir, input_file)

    run = runs[i]
    cultivar = dic_for_run[run][0]
    organism_part = dic_for_run[run][1]

    if cultivar not in dic_for_cultivar:
        dic_for_cultivar[cultivar] = []
    # Construct the corresponding output file name

    # Read the input TPM file into a list of lines
    with open(input_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()  # Assuming tab-separated values
        gene_id, tpm_value = parts[0], float(parts[1])
        dic_for_cultivar[cultivar].append(f'{gene_id}\t{cultivar}\t{organism_part}\t{tpm_value:.6f}\n')
    
    i += 1

for cultivar in dic_for_cultivar:
    output_file = os.path.join(output_dir, f'{cultivar}_TPM_expression_counts_aligned_against_own_genome.txt')
    with open(output_file, 'w') as file:
        file.writelines(dic_for_cultivar[cultivar])