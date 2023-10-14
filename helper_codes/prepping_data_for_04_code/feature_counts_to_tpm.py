import os
import re

def calculate_tpm(count_file, length_file, output_file):
    # Read counts from the counts file
    counts = {}
    with open(count_file, 'r') as f:
        for line in f:
            if not line.startswith('Geneid'):
                parts = line.strip().split('\t')
                gene_id = parts[0]
                count = float(parts[1])
                counts[gene_id] = count

    # Read lengths from the length file
    lengths = {}
    with open(length_file, 'r') as f:
        for line in f:
            if not line.startswith('Geneid'):
                gene_id, length = line.strip().split('\t')
                lengths[gene_id] = int(length)

    # Calculate TPM values and write them to the output file
    with open(output_file, 'w') as f:
        for gene_id, count in counts.items():
            length = lengths.get(gene_id, 1.0)  # Default to 1 if length is missing
            rpkm = (count / length) / (sum(counts.values()) / 1e6)
            f.write(f"{gene_id}\t{rpkm:.6f}\n")

# Specify the directory path with raw string literal
directory_path = r'path\to\datasets'

# Iterate through files following the naming convention
file_sets = [f for f in os.listdir(directory_path) if re.match(r'featureCounts_on_data_\d+_and_data_\d+__Counts_\d+.tabular', f)]

for i, file_set in enumerate(file_sets):
    # Extract the set number from the file name using regular expressions
    set_match = re.search(r'featureCounts_on_data_(\d+)_and_data_(\d+)__Counts_(\d+).tabular', file_set)
    # print(type(set_match.group(1)))
    if set_match:
        set_number = f"{set_match.group(1)}_and_data_{set_match.group(2)}"
        count_file = os.path.join(directory_path, file_set)
        # Construct the corresponding length file name
        length_file_name = f"featureCounts_on_data_{set_number}__Feature_lengths_{str(int(set_match.group(3)) + 2)}.tabular"
        length_file = os.path.join(directory_path, length_file_name)
        
        # Calculate TPM for this file set and write it to a TPM expression count file
        tpm_output_file = os.path.join(r"path\to\tpm_data", f"tpm_{set_number}.tabular")
        calculate_tpm(count_file, length_file, tpm_output_file)
        print("File written")
