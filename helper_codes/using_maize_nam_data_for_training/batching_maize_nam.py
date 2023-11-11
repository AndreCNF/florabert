import random
with open(r"C:\Users\GURDARSH VIRK\OneDrive\Desktop\PS 3-1 FloraBERT\florabert\data\final\transformer\seq\all_seqs_test.txt", 'r') as file:
    lines = file.readlines()
    random.shuffle(lines)
    n = len(lines)
    i = 0
    while i < 101:
        w_lines = lines[i * n//100 : (i + 1) * n // 100]
        with open(f"C:\\Users\\GURDARSH VIRK\\OneDrive\\Desktop\\PS 3-1 FloraBERT\\florabert\\data\\final\\transformer\\seq\\batched_test\\all_seqs_test_{i + 1}.txt", "w") as out:
            out.writelines(w_lines)
        i += 1
