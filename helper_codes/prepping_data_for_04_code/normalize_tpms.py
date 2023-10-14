import os
import numpy as np

# Directory containing the input files
input_dir = r'path\to\tpm_data'  # Replace with the actual directory path

# Directory where the output files will be saved
output_dir = r'path\to\normalized_tpm_data'  # Replace with the desired output directory path

# List all files in the input directory that start with 'tpm_' and end with '.tabular'
input_files = [filename for filename in os.listdir(input_dir) if filename.startswith('tpm') and filename.endswith('.tabular')]

# Process each input file
for input_file in input_files:
    # Construct the full path of the input file
    input_path = os.path.join(input_dir, input_file)

    # Construct the corresponding output file name
    output_file = os.path.join(output_dir, f'normalized_{input_file}')

    # Read the input TPM file into a list of lines
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # Process the lines to apply the log transformation
    offset = 0.001
    output_lines = []
    for line in lines:
        parts = line.strip().split()  # Assuming tab-separated values
        gene_id, tpm_value = parts[0], float(parts[1])
        log2_tpm = np.log2(tpm_value + offset)
        output_lines.append(f'{gene_id}\t{log2_tpm:.6f}\n')

    # Write the transformed data to the output file
    with open(output_file, 'w') as file:
        file.writelines(output_lines)
