import os
import re

def calculate_tpm(combined_file, output_file):
    # Read counts and lengths from the combined file
    counts = {}
    lengths = {}
    
    with open(combined_file, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split()
        for line in lines[1:]:
            parts = line.strip().split()
            gene_id = parts[0]
            read_count = float(parts[1])
            length = float(parts[2])
            counts[gene_id] = read_count
            lengths[gene_id] = length

    # Calculate TPM values and write them to the output file
    with open(output_file, 'w') as f:
        f.write(f"Geneid\tTPM\n")
        for gene_id, read_count in counts.items():
            length = lengths.get(gene_id, 1.0)  # Default to 1 if length is missing
            rpkm = (read_count / length) / (sum(counts.values()) / 1e6)
            f.write(f"{gene_id[5 : ]}\t{rpkm:.6f}\n")

# Specify the path to the combined file
combined_file_path = r'path\to\exception_file.tabular'

# Specify the output file path
output_file_path = r'path\to\tpm_expression_counts.tabular'

# Calculate TPM and write it to the output file
calculate_tpm(combined_file_path, output_file_path)
