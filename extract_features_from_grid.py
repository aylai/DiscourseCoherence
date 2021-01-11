import itertools
import os, sys

corpus = sys.argv[1]
seq_len = int(sys.argv[2])
salience_threshold = int(sys.argv[3])
syntax_opt = int(sys.argv[4])
is_permute_arg = sys.argv[5]
is_permute = False
if is_permute_arg == 'true':
    is_permute = True
append_str = ''
if is_permute:
    append_str = '_permute'

in_dir = 'data/'+corpus+'/'
if not os.path.isdir(in_dir + 'features' + append_str + '/'):
    os.mkdir(in_dir + 'features' + append_str + '/')
feat_dir = in_dir + 'features' + append_str + '/seq_' + str(seq_len) + '_sal_' + str(salience_threshold) + '_syn_' + str(syntax_opt) + '/'
if not os.path.isdir(feat_dir):
    os.mkdir(feat_dir)
print(feat_dir)
for filename in os.listdir(in_dir + 'grid' + append_str + '/'):
    if not filename.endswith("grid"):
        continue
    filename_base = filename.rsplit(".", 1)[0]
    out_file = open(feat_dir + filename_base + ".feat", "w")
    with open(in_dir + 'grid' + append_str + '/' + filename, "r") as in_file:
        # read grid
        sequences = []
        frequencies = []
        for line in in_file:
            line = line.strip()
            tokens = line.split()
            try:
                frequency = int(tokens[-1])
            except ValueError:
                print(line)
                frequency = 0
            frequencies.append(frequency)
            sequence = "".join(tokens[1:-1])
            sequence = "<" + sequence + ">"  # add start and end tokens
            sequences.append(sequence)
        in_file.close()

        # compute feature vector
        if syntax_opt == 1:  # syntax on
            labels = ['s', 'o', 'x', '-']
        else:  # syntax off (ignore entity roles)
            labels = ['x', '-']
        feature_vector = []
        for salience_class in [0, 1]:
            if salience_threshold == 1 and salience_class == 1: # only one salience class
                break
            for i in range(seq_len):  # over possible sequence lengths
                seq_len = i + 1  # shortest seq is length 2
                num_total_sequences = 0
                for sent_index, sentence in enumerate(sequences):
                    if salience_class == 0 and frequencies[sent_index] >= salience_threshold:
                        num_total_sequences += len(sentence) - seq_len + 1
                    elif salience_class == 1 and frequencies[sent_index] < salience_threshold:
                        num_total_sequences += len(sentence) - seq_len + 1
                total_prob = 0
                seq_minus_one = {}
                for possible_seq in itertools.product(labels, repeat=seq_len):
                    possible_seq_tok = "".join(possible_seq)
                    seq_minus_one[possible_seq_tok[:-1]] = 1
                    num_occurrences = 0
                    for sent_index, sentence in enumerate(sequences):
                        sentence_temp = sentence
                        if syntax_opt == 0:
                            sentence_temp = sentence_temp.replace('s', 'x')
                            sentence_temp = sentence_temp.replace('o', 'x')
                        if salience_class == 0 and frequencies[sent_index] >= salience_threshold:
                            num_occurrences += sum(sentence_temp[j:].startswith(possible_seq_tok) for j in range(len(sentence_temp)))
                        elif salience_class == 1 and frequencies[sent_index] < salience_threshold:
                            num_occurrences += sum(sentence_temp[j:].startswith(possible_seq_tok) for j in range(len(sentence_temp)))
                    feature_prob = 0
                    if num_total_sequences > 0:
                        feature_prob = float(num_occurrences) / num_total_sequences
                    feature_vector.append(feature_prob)
                    total_prob += feature_prob
                # add start and end tokens
                for shorter_seq in seq_minus_one:
                    possible_seq_toks = ["<" + shorter_seq, shorter_seq + ">"]
                    for possible_seq_tok in possible_seq_toks:
                        num_occurrences = 0
                        for sent_index, sentence in enumerate(sequences):
                            sentence_temp = sentence
                            if syntax_opt == 0:
                                sentence_temp = sentence_temp.replace('s', 'x')
                                sentence_temp = sentence_temp.replace('o', 'x')
                            if salience_class == 0 and frequencies[sent_index] >= salience_threshold:
                                num_occurrences += sum(sentence_temp[j:].startswith(possible_seq_tok) for j in range(len(sentence_temp)))
                            elif salience_class == 1 and frequencies[sent_index] < salience_threshold:
                                num_occurrences += sum(sentence[j:].startswith(possible_seq_tok) for j in range(len(sentence)))
                        feature_prob = 0
                        if num_total_sequences > 0:
                            feature_prob = float(num_occurrences) / num_total_sequences
                        feature_vector.append(feature_prob)
                        total_prob += feature_prob
        for val in feature_vector:
            out_file.write(str(val) + " ")
        out_file.close()
