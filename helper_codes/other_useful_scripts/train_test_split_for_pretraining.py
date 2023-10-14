import random

# Input file with all sequences
input_file = r"path\to\all_seqs_train_sample.txt"
# Output files for train and test splits
train_output_file = r"path\to\all_seqs_train.txt"
test_output_file = r"path\to\all_seqs_test.txt"

# Initialize lists to hold train and test sequences
train_sequences = []
test_sequences = []

# Open the input file and read lines
with open(input_file, "r") as file:
    lines = file.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

# Calculate the split point
split_point = int(0.7 * len(lines))

# Split lines into train and test
train_lines = lines[:split_point]
test_lines = lines[split_point:]

# Write train sequences to the train output file
with open(train_output_file, "w") as train_file:
    train_file.writelines(train_lines)

# Write test sequences to the test output file
with open(test_output_file, "w") as test_file:
    test_file.writelines(test_lines)
