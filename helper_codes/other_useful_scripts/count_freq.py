""" 
    Check frequency of organism parts in txt file outputs
    update tissues under config.py as needed (keep highest freq. tissues/org_parts only)
"""

from module.florabert import config, utils, dataio
import pandas as pd

GENEX_DIR = (
    config.data_raw
    / "gene_expression"
    / "Normalized_TPMs_each_NAM_line_aligned_against_own_refgen"
)
genex_files = list(
        GENEX_DIR.glob("*_TPM_expression_counts_aligned_against_own_genome.txt")
    )

dic_for_org_parts = {}

for genex_file in genex_files:
    print(genex_file)
    df_genex = pd.read_csv(genex_file, sep="\t")
    for e in df_genex["organism_part"]:
        if e not in dic_for_org_parts:
            dic_for_org_parts[e] = 1
        else:
            dic_for_org_parts[e] += 1

print(dic_for_org_parts)