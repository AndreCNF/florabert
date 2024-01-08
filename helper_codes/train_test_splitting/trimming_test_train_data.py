# trim the data if needed
import random

# Function to read data from file and split into three parts
def split_data(filename, split_parts, file_type):
    with open(filename, 'r') as file:
        lines = file.readlines()
        random.shuffle(lines)  # Shuffle the lines

        total_lines = len(lines)
        part_size = total_lines // split_parts

        for i in range(split_parts):
            with open(f'path\to\\florabert\\data\\final\\transformer\\seq\\split_{file_type}\\all_seqs_{file_type}_{i + 1}.txt', 'w') as file:
                file.writelines(lines[(i * part_size) : ((i + 1) * part_size)])

split_data(r'path\to\florabert\data\final\transformer\seq\full_test_train\full_all_seqs_train.txt', 4, 'train')
split_data(r'path\to\florabert\data\final\transformer\seq\full_test_train\full_all_seqs_test.txt', 4, 'test')