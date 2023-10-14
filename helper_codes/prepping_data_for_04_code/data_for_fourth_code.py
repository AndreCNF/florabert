import os
import numpy as np
import glob

input_dir = r'path\to\normalized_tpm_data' 
output_dir = r'path\to\Normalized_TPMs_each_NAM_line_aligned_against_own_refgen'
dic_for_cult = {}

with open(r"path\to\Ordering_Runs.txt", 'r') as file:
    cultivar = ""
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if not line.startswith("E"):
            cultivar = line
            dic_for_cult[cultivar] = []
        else:
            dic_for_cult[cultivar].append(line)

with open(r"path\to\unique_org.tsv", 'r') as file:
    lines = file.readlines()
    for line in lines[1 : ]:
        parts = line.strip().split()
        run = parts[0]
        org_part = parts[2]
        for e in dic_for_cult:
            if run in dic_for_cult[e]:
                dic_for_cult[e][dic_for_cult[e].index(run)] = org_part

for cultivar in dic_for_cult:
    path = os.path.join(input_dir, cultivar)
    input_files = glob.glob(f"{path}/*.tabular")
    org_parts = dic_for_cult[cultivar]
    output_lines = []
    output_line = "Cultivar\torganism_part"
    with open(input_files[0], 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.strip().split() 
        gene_id, tpm_value = parts[0], float(parts[1])
        output_line += f"\t{gene_id}"

    output_lines.append(output_line + "\n")
    output_line = f"{cultivar}\t{org_parts[0]}"
    
    for line in lines:
        parts = line.strip().split() 
        gene_id, tpm_value = parts[0], float(parts[1])
        output_line += f"\t{tpm_value}"
    
    j = 1

    output_lines.append(output_line + "\n")

    for i in range(1, len(input_files)):
        output_line = f"{cultivar}\t{org_parts[j]}"
        with open(input_files[i], 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split() 
            gene_id, tpm_value = parts[0], float(parts[1])
            output_line += f"\t{tpm_value}"

        output_lines.append(output_line + "\n")
        j += 1
    
    with open(os.path.join(output_dir, f"{cultivar}_TPM_expression_counts_aligned_against_own_genome.txt"), 'w') as file:
        file.writelines(output_lines)
    
    print(cultivar, "done")