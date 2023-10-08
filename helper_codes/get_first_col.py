# Open the TSV file for reading and the TXT file for writing
with open('filereport_read_run_PRJEB35943_tsv.txt', 'r') as tsv_file, open('sra_accession_b73.txt', 'w') as txt_file:
    for line in tsv_file:
        columns = line.strip().split('\t')
        if columns:
            # Write the first column to the TXT file
            txt_file.write(columns[0] + '\n')
