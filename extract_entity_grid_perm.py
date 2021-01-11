# takes csv files, parses them, and extracts entity grid
from pycorenlp import StanfordCoreNLP
import os, json, sys

nlp = StanfordCoreNLP('http://localhost:9000')
corpus = sys.argv[1]

in_dir = 'data/' + corpus + '/'

if not os.path.exists(in_dir + 'parsed_permute/'):
    os.makedirs(in_dir + 'parsed_permute/')
if not os.path.exists(in_dir + 'grid_permute/'):
    os.makedirs(in_dir + 'grid_permute/')

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
        index += num_tokens
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
    return nouns

# read all text files, parse and extract entity grid
for filename in os.listdir(in_dir + "text_permute/"):
    if not filename.endswith("_sent.txt"):
        continue  # original files only
    with open(in_dir + "text_permute/" + filename, 'r') as in_file:
        # process original sentence order file
        nouns_list = []
        nouns_dict = {}
        sent_annotations = []
        text_id = filename.rsplit("_", 1)[0]
        const_out_filename = in_dir + "parsed_permute/" + text_id + ".0.const_parse"
        dep_out_filename = in_dir + "parsed_permute/" + text_id + ".0.dep_parse"
        grid_out_filename = in_dir + "parsed_permute/" + text_id + ".0.grid"
        if os.path.exists(const_out_filename) and os.path.exists(dep_out_filename) and os.path.exists(
                grid_out_filename):
            continue
        const_out = open(in_dir + "parsed_permute/" + text_id + ".0.const_parse", "w")
        const_lines = {}
        dep_out = open(in_dir + "parsed_permute/" + text_id + ".0.dep_parse", "w")
        dep_lines = {}
        grid_out = open(in_dir + "grid_permute/" + text_id + ".0.grid", "w")
        grid_lines = {}
        for line in in_file:  # sentences in original order
            line = line.strip()
            const_lines[line] = []
            dep_lines[line] = []
            grid_lines[line] = []
            if line.strip() == "":  # not sure if this ever fires (I might have removed line breaks in these files -- for entity grid only)
                const_out.write("\n\n")
                dep_out.write("\n\n")
                continue
            output = nlp.annotate(line, properties={
                'annotators': 'tokenize,ssplit,pos,depparse,parse',
                'outputFormat': 'json'
            })
            for sent in output['sentences']:
                const_out.write(sent['parse'] + "\n")
                const_lines[line].append(sent['parse'])
                json.dump(sent['basicDependencies'], dep_out)
                dep_out.write("\n")
                dep_lines[line].append(sent['basicDependencies'])
                curr_nouns_type = {}
                for token in sent['tokens']:
                    if token['pos'].startswith("NN") or token['pos'] == 'PRP':
                        token_str = token['word'].lower()
                        curr_nouns_type[token_str] = "x"
                        if token_str not in nouns_dict:
                            nouns_list.append(token_str)
                            nouns_dict[token_str] = 0
                        nouns_dict[token_str] += 1
                for dep in sent['basicDependencies']:
                    dep_type = ""
                    if dep['dep'] == 'nsubj' or dep['dep'] == 'nsubjpass':
                        dep_type = "s"
                    elif dep['dep'] == 'dobj':
                        dep_type = "o"
                    if dep_type != "":
                        np = get_np(dep, sent['parse'])
                        curr_nouns_type = update_noun_types(dep_type, np, curr_nouns_type)
                sent_annotations.append(curr_nouns_type)
                grid_lines[line].append(curr_nouns_type)

        for noun in nouns_list:
            grid_out.write(noun + " ")
            for sent_ann in sent_annotations:
                if noun in sent_ann:
                    grid_out.write(sent_ann[noun] + " ")
                else:
                    grid_out.write("- ")
            grid_out.write(str(nouns_dict[noun]) + "\n")  # frequency for salience feature
        grid_out.close()
        const_out.close()
        dep_out.close()
        for i in range(1, 21):
            filename_perm = text_id + ".perm-" + str(i)
            if not os.path.exists(in_dir + "text_permute/" + filename_perm + ".txt"):
                continue
            const_out = open(in_dir + "parsed_permute/" + filename_perm + ".const_parse", "w")
            dep_out = open(in_dir + "parsed_permute/" + filename_perm + ".dep_parse", "w")
            grid_out = open(in_dir + "grid_permute/" + filename_perm + ".grid", "w")
            sent_annotations = []
            with open(in_dir + "text_permute/" + filename_perm + ".txt", "r") as in_file:
                for line in in_file:
                    line = line.strip()
                    for parse in const_lines[line]:
                        const_out.write(parse + "\n")
                    for parse in dep_lines[line]:
                        json.dump(parse, dep_out)
                        dep_out.write("\n")
                    for grid_line in grid_lines[line]:
                        sent_annotations.append(grid_line)
            for noun in nouns_list:
                grid_out.write(noun + " ")
                for sent_ann in sent_annotations:
                    if noun in sent_ann:
                        grid_out.write(sent_ann[noun] + " ")
                    else:
                        grid_out.write("- ")
                grid_out.write(str(nouns_dict[noun]) + "\n")  # saliance frequency feature
            grid_out.close()
            const_out.close()
            dep_out.close()