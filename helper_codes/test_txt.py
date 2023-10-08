with open("all_seqs_train_sample.txt", 'r') as txt:
    i = 0
    for line in txt:
        print(line)
        if i == 100:
            break
        i += 1