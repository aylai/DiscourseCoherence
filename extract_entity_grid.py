# takes csv files, parses them, and extracts entity grid
from pycorenlp import StanfordCoreNLP
import os, json, sys

corpus = sys.argv[1]
in_dir = 'data/' + corpus + '/'
nlp = StanfordCoreNLP('http://localhost:9000')  # requires you have the Stanford CoreNLP server running: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started

if not os.path.exists(in_dir + 'parsed/'):
    os.makedirs(in_dir + 'parsed/')
if not os.path.exists(in_dir + 'grid/'):
    os.makedirs(in_dir + 'grid/')


def update_noun_types(dep_type, np_words, curr_nouns_type):
    for word in np_words:
        if word not in curr_nouns_type:
            curr_nouns_type[word] = dep_type
        if curr_nouns_type[word] == "x" or curr_nouns_type[word] == "o":
            curr_nouns_type[word] = dep_type
    return curr_nouns_type


def get_np(dependency, const_parse):
    target_id = dependency['dependent']
    index = 0
    nouns = []
    for line in const_parse.splitlines():
        if ")" not in line:
            continue
        tokens = line.strip().split(") (")
        num_tokens = len(tokens)  # remove phrase label
        phrase_start_idx = index + 1
        index += num_tokens
        phrase_end_idx = index + 1
        if target_id <= index and tokens[0].startswith("(NP"):
            for token in tokens:
                if token.startswith("(NP"):
                    token = token[3:].strip()
                while token.startswith("("):
                    token = token[1:] 
                while token.endswith(")"):
                    token = token[:-1].strip()
                word = token.split(None, 1)[1]  # remove POS tag
                if token.startswith("NN"):
                    nouns.append(word.lower())
                elif token.startswith("PRP "):
                    nouns.append(word.lower())
                elif token.startswith("DT") and len(tokens) == 1:
                    nouns.append(word.lower())  # is noun phrase, only one DT word (this, all) in the phrase
            break
    return nouns, phrase_start_idx, phrase_end_idx


# read all text files, parse and extract entity grid
for filename in os.listdir(in_dir + "text/"):
    with open(in_dir + "text/" + filename,'r') as in_file:
        if not filename.endswith(".txt"):
            continue
        nouns_list = []
        nouns_dict = {}
        sent_annotations = []
        text_id = filename.rsplit(".", 1)[0]
        const_out = open(in_dir + "parsed/" + text_id + ".const_parse", "w")
        dep_out = open(in_dir + "parsed/" + text_id + ".dep_parse", "w")
        grid_out = open(in_dir + "grid/" + text_id + ".grid", "w")
        # read text document
        document_lines = []
        for line in in_file:
            line = line.strip()
            if line == "":
                continue
            if isinstance(line, str):
                document_lines.append(line)
        document = " ".join(document_lines)
        try:
            output = nlp.annotate(document, properties={
                'annotators': 'tokenize,ssplit,pos,depparse,parse',
                'outputFormat': 'json'
            })
        except:
            print('Failed to parse file %s' % filename)
            continue
        if output == 'CoreNLP request timed out. Your document may be too long.':
            print('Timed out when attempting to parse file %s' % filename)
            continue
        for sent in output['sentences']:
            sent_idx = sent['index'] + 1
            const_out.write(sent['parse'] + "\n")
            json.dump(sent['basicDependencies'], dep_out)
            dep_out.write("\n")
            curr_nouns_type = {}
            for token in sent['tokens']:
                # collect all nouns and pronouns
                if token['pos'].startswith("NN") or token['pos'] == 'PRP':
                    token_str = token['word'].lower()
                    curr_nouns_type[token_str] = "x"
                    if token_str not in nouns_dict:
                        nouns_list.append(token_str) 
                        nouns_dict[token_str] = 0
                    nouns_dict[token_str] += 1
            # find highest-ranked role of entity in this sentence (subj > obj > other)
            for dep in sent['basicDependencies']:
                dep_type = ""
                if dep['dep'] == 'nsubj' or dep['dep'] == 'nsubjpass':
                    dep_type = "s"
                elif dep['dep'] == 'dobj':
                    dep_type = "o"
                if dep_type != "":
                    np, phrase_start_idx, phrase_end_idx = get_np(dep, sent['parse'])
                    curr_nouns_type = update_noun_types(dep_type, np, curr_nouns_type)
            sent_annotations.append(curr_nouns_type)

        # output entity grid
        for noun in nouns_list:
            grid_out.write(noun + " ")
            for sent_ann in sent_annotations:
                if noun in sent_ann: 
                    grid_out.write(sent_ann[noun] + " ")
                else:
                    grid_out.write("- ")
            grid_out.write(str(nouns_dict[noun]) + "\n")  # entity frequency (salience count)
        grid_out.close()
        const_out.close()
        dep_out.close()
