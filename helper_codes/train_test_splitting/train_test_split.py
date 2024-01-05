# resources not enough; using only one db at a time; will still result in 90-10 train-test split of all db's combined
import csv
import random

# Open the CSV file
with open(r"path\to\florabert\data\processed\combined\ensembl.csv") as csvfile: # change ensembl to other db's names as needed
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header if it exists
    sequences = [row[2] for row in reader]  # Assuming the sequence is in the 3rd column (index 2)

random.shuffle(sequences)
l = len(sequences)
print(l)
train_l = (int)(l * 0.9) # update value of 0.9 to change train-test split ratio
# Write the sequences to a text file
with open(r'path\to\florabert\data\final\transformer\seq\all_seqs_train.txt', 'a') as file: # change to w to write from scratch
    for sequence in sequences[: train_l]:
        file.write(sequence + '\n')
with open(r'path\to\florabert\data\final\transformer\seq\all_seqs_test.txt', 'a') as file: # change to w to write from scratch
    for sequence in sequences[train_l : ]:
        file.write(sequence + '\n')