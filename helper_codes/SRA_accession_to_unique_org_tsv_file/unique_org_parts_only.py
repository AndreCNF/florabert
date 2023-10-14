# Open the TSV file for reading and the TXT file for writing
checked = {}
with open('combined_organism_output.tsv', 'r') as tsv_file, open('unique_org.tsv', 'w') as out_file:
    for line in tsv_file:
        columns = line.strip().split()
        if columns[1] not in checked:
            checked[columns[1]] = [columns[2]]
            out_file.write(columns[0] + "\t" + columns[1] + "\t" + columns[2] + "\n")
        elif columns[2] not in checked[columns[1]]:
            checked[columns[1]].append(columns[2])
            out_file.write(columns[0] + "\t" + columns[1] + "\t" + columns[2] + "\n")