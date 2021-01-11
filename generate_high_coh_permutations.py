import os, random, csv, sys
from nltk.tokenize import sent_tokenize
import itertools
import numpy as np

corpus = sys.argv[1]
root_dir = 'data/' + corpus + '/'
in_dir = root_dir + 'text/'
out_dir = root_dir + 'text_permute/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def is_orig_permutation(orig_sents, perm_sents):
    for index, sent in enumerate(orig_sents):
        if sent != perm_sents[index]:
            return False
    return True

# which texts to permute
if corpus == 'Clinton' or corpus == 'Enron' or corpus == 'Yelp' or corpus == 'Dummy':
    title_row = ["text_id","subject","text","ratingA1","ratingA2","ratingA3","labelA","ratingM1","ratingM2","ratingM3","ratingM4","ratingM5","labelM"]
elif corpus == 'Yahoo':
    title_row = ["text_id","question_title","question","text","ratingA1","ratingA2","ratingA3","labelA","ratingM1","ratingM2","ratingM3","ratingM4","ratingM5","labelM"]
splits = ['train','test']
high_coh_texts = {}
total = 0
for split in splits:
    in_file = open(root_dir + corpus + '_' + split + '.csv','r')
    out_file = open(root_dir + corpus + '_' + split + '_perm.csv', 'w')
    writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting = csv.QUOTE_ALL)
    writer.writerow(title_row)
    reader = csv.DictReader(in_file)
    for row in reader:
        if row['labelA'] == '3':
            high_coh_texts[row['text_id']] = 1
            # print(row)
            writer.writerow([row[key] for key in row])
        total += 1
    out_file.close()
print(len(high_coh_texts))
print("total %d" %total)

# read orig texts
count = 0
num_files = 0
for filename in os.listdir(in_dir):
    if not filename.endswith(".txt"):
        continue
    # read sentences and tokenize at sentence boundaries
    sentences = []
    text_id = filename.split(".")[0]
    if text_id not in high_coh_texts:
        continue
    with open(in_dir + filename, 'r') as in_file:
        orig_lines = in_file.readlines()
        for line in orig_lines:
            sentences.extend(sent_tokenize(line))
    # remove empty lines (don't matter for permutations)
    new_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent != "":
            new_sentences.append(sent)
    sentences = new_sentences
    if len(sentences) == 1: # no possible permutations
        continue
    out_file_orig = open(out_dir + text_id + "_sent.txt", "w")
    num_files += 1
    count += 1
    for sent in sentences:
        out_file_orig.write(sent + "\n")
    out_file_orig.close()
    # create 20 permutations
    num_permutations = 0
    used_permutations = {}
    found_duplicate = False
    if len(sentences) < 6: # generate all permutations
        all_permutations = list(itertools.permutations(sentences))
        random.shuffle(all_permutations)
        for perm in all_permutations: 
            if num_permutations >= 20:
                break
            if not found_duplicate:
                if is_orig_permutation(sentences, perm):
                    found_duplicate = True
                    continue
            out_file_perm = open(out_dir + text_id + ".perm-" + str(num_permutations+1) + ".txt", "w")
            num_files += 1
            for sent in perm:
                out_file_perm.write(sent + "\n") 
            num_permutations += 1
            out_file_perm.close()
    else: # need to sample permutations
        while num_permutations < 20:
            permutation = np.random.permutation(len(sentences))
            permutation_str = [str(num) for num in permutation]
            permutation_idx_str = ",".join(permutation_str)
            if permutation_idx_str not in used_permutations:
                out_file_perm = open(out_dir + text_id + ".perm-" + str(num_permutations+1) + ".txt", "w")
                num_files += 1
                for sent_idx in permutation:
                    out_file_perm.write(sentences[sent_idx] + "\n")
                out_file_perm.close()
                num_permutations += 1
                used_permutations[permutation_idx_str] = 1 
            
print(count)
print(num_files)
