import csv, os, sys

corpus = sys.argv[1]
corpus_dir = 'data/' + corpus + '/'
text_dir = corpus_dir + 'text/'
if not os.path.exists(text_dir):
    os.makedirs(text_dir)
splits = ['train', 'test']
for split in splits:
    with open(corpus_dir + corpus + '_' + split + '.csv','r') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            text_id = row['text_id']
            filename = text_id + '.txt'
            if os.path.exists(text_dir + filename):
                continue
            out_file = open(text_dir + filename, 'w')
            out_file.write(row['text'])
            out_file.close()
