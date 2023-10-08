import pandas as pd

# Read the two TSV files
df_cultivar = pd.read_csv("output_organism_part.tsv", sep="\t")
df_cultivar_2 = pd.read_csv("output_organism_part_2.tsv", sep="\t")

# Combine the two DataFrames vertically
combined_df = pd.concat([df_cultivar, df_cultivar_2], axis=0)

# Save the combined DataFrame to a new TSV file
combined_df.to_csv("combined_organism_output.tsv", sep="\t", index=False)
