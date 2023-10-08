import pandas as pd

# Sample data for reference genomes
refgen_data = [
    ["Zm-B73-REFERENCE-NAM-5.0", "Zm00001eb.1"],
    ["Zm-B97-REFERENCE-NAM-1.0", "Zm00018ab.1"],
    ["Zm-CML52-REFERENCE-NAM-1.0", "Zm00019ab.1"],
    ["Zm-CML69-REFERENCE-NAM-1.0", "Zm00020ab.1"],
    ["Zm-CML103-REFERENCE-NAM-1.0", "Zm00021ab.1"],
    ["Zm-CML228-REFERENCE-NAM-1.0", "Zm00022ab.1"],
    ["Zm-CML247-REFERENCE-NAM-1.0", "Zm00023ab.1"],
    ["Zm-CML277-REFERENCE-NAM-1.0", "Zm00024ab.1"],
    ["Zm-CML322-REFERENCE-NAM-1.0", "Zm00025ab.1"],
    ["Zm-CML333-REFERENCE-NAM-1.0", "Zm00026ab.1"],
    ["Zm-HP301-REFERENCE-NAM-1.0", "Zm00027ab.1"],
    ["Zm-Il14H-REFERENCE-NAM-1.0", "Zm00028ab.1"],
    ["Zm-Ki3-REFERENCE-NAM-1.0", "Zm00029ab.1"],
    ["Zm-Ki11-REFERENCE-NAM-1.0", "Zm00030ab.1"],
    ["Zm-Ky21-REFERENCE-NAM-1.0", "Zm00031ab.1"],
    ["Zm-M37W-REFERENCE-NAM-1.0", "Zm00032ab.1"],
    ["Zm-M162W-REFERENCE-NAM-1.0", "Zm00033ab.1"],
    ["Zm-Mo18W-REFERENCE-NAM-1.0", "Zm00034ab.1"],
    ["Zm-Ms71-REFERENCE-NAM-1.0", "Zm00035ab.1"],
    ["Zm-NC350-REFERENCE-NAM-1.0", "Zm00036ab.1"],
    ["Zm-NC358-REFERENCE-NAM-1.0", "Zm00037ab.1"],
    ["Zm-Oh7B-REFERENCE-NAM-1.0", "Zm00038ab.1"],
    ["Zm-Oh43-REFERENCE-NAM-1.0", "Zm00039ab.1"],
    ["Zm-P39-REFERENCE-NAM-1.0", "Zm00040ab.1"],
    ["Zm-Tx303-REFERENCE-NAM-1.0", "Zm00041ab.1"],
    ["Zm-Tzi8-REFERENCE-NAM-1.0", "Zm00042ab.1"]
]

# Read SRA data from a TSV file
sra_df = pd.read_csv(r"C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\filereport_read_run_PRJEB36014_tsv.txt", sep="\t")

# Create a DataFrame for reference genomes
refgen_df = pd.DataFrame(refgen_data, columns=["refgen", "annot"])

# Function to extract cultivar and organism_part from sra_ftp
def extract_info(ftp_url):
    parts = ftp_url.split("/")
    cultivar = parts[-1].split("_")[0]
    organism_part = parts[-1].split("_")[2]
    refgen = ""
    annot = ""
    for e in refgen_data:
        if cultivar.lower() in e[0].lower():
            refgen = e[0]
            annot = e[1]
            break
    return cultivar, organism_part, refgen, annot

# Apply the function to create new columns in sra_df
sra_df["cultivar"], sra_df["organism_part"], sra_df["refgen"], sra_df["annot"] = zip(*sra_df["submitted_ftp"].map(extract_info))

result_df_cultivar = sra_df[["run_accession", "cultivar", "refgen", "annot"]]
result_df_organism_part = sra_df[["run_accession", "cultivar", "organism_part"]]

# Save DataFrames to TSV files
result_df_cultivar.to_csv("output_cultivar_2.tsv", sep="\t", index=False)
result_df_organism_part.to_csv("output_organism_part_2.tsv", sep="\t", index=False)
