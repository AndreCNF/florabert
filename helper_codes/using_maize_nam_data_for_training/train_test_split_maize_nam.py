# resources not enough; using only maize_nam
import csv
import random

# Open the CSV file
with open(f'C:\\Users\\GURDARSH VIRK\\OneDrive\\Desktop\\PS 3-1 FloraBERT\\florabert\\data\\processed\\combined\\maize_nam.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header if it exists
    sequences = [row[2] for row in reader]  # Assuming the sequence is in the 3rd column (index 2)

random.shuffle(sequences)
l = len(sequences)
train_l = (int)(l * 0.7)
# Write the sequences to a text file
with open(r'C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\florabert\data\final\transformer\seq\all_seqs_train.txt', 'w') as file:
    for sequence in sequences[: train_l]:
        file.write(sequence + '\n')
with open(r'C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\florabert\data\final\transformer\seq\all_seqs_test.txt', 'w') as file:
    for sequence in sequences[train_l : ]:
        file.write(sequence + '\n')