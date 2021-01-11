import itertools
import os, sys
import numpy as np

role_weights = {'s': 3, 'o': 2, 'x': 1}


def compute_avg_outdeg(matrix):
    out_degree_list = []
    for sent in matrix:
        out_degree = 0
        for weight in sent:
            out_degree += weight
        out_degree_list.append(out_degree)
    return np.mean(out_degree_list)


corpus = sys.argv[1]
is_permute_arg = sys.argv[2]
is_permute = False
if is_permute_arg == 'true':
    is_permute = True
append_str = ''
if is_permute:
    append_str = '_permute'

root_dir = 'data/'+corpus+'/'
in_dir = root_dir + 'grid' + append_str + '/'
out_dir = root_dir + 'graph' + append_str + '/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# process all grid files (*.grid)
for filename in os.listdir(in_dir):
    if not filename.endswith("grid"):
        continue
    filename_base = filename.rsplit(".", 1)[0] # assumes no periods '.' in grid name
    out_file_u = open(out_dir + filename_base + ".graph_u", "w")
    out_file_u_dist = open(out_dir + filename_base + ".graph_u_dist", "w")
    out_file_w = open(out_dir + filename_base + ".graph_w", "w")
    out_file_w_dist = open(out_dir + filename_base + ".graph_w_dist", "w")
    out_file_syn = open(out_dir + filename_base + ".graph_syn", "w")
    out_file_syn_dist = open(out_dir + filename_base + ".graph_syn_dist", "w")
    with open(in_dir + filename, "r") as in_file:
        matrix_u = []
        matrix_u_dist = []
        matrix_w = []
        matrix_w_dist = []
        matrix_syn = []
        matrix_syn_dist = []
        for line in in_file:  # for all entities in text
            line = line.strip()
            tokens = line.split()
            try:
                count = int(tokens[-1])
                sentence_roles = tokens[1:-1]
            except ValueError:
                sentence_roles = tokens[1:]  # remove frequency count and word
            while sentence_roles[0] not in {'-', 'x', 's', 'o'}:
                sentence_roles = sentence_roles[1:]
            num_sentences = len(sentence_roles)
            if matrix_u == []:
                # initialize adjacency matrices
                for i in range(num_sentences):
                    list_i = []
                    for j in range(num_sentences):
                        list_i.append(0)
                    matrix_u.append(list(list_i))  # copy list
                    matrix_u_dist.append(list(list_i))
                    matrix_w.append(list(list_i))
                    matrix_w_dist.append(list(list_i))
                    matrix_syn.append(list(list_i))
                    matrix_syn_dist.append(list(list_i))
            # find sentences that contain this entity
            sentence_indices = []
            for index, role in enumerate(sentence_roles):
                if role != "-":
                    sentence_indices.append(index)
            for pair in itertools.combinations(sentence_indices, 2):  # get all sentence pairs
                first_sent = min(pair)
                second_sent = max(pair)
                matrix_u[first_sent][second_sent] = 1  # binary
                matrix_u_dist[first_sent][second_sent] = 1 / (second_sent - first_sent)
                matrix_w[first_sent][second_sent] += 1  # count
                matrix_w_dist[first_sent][second_sent] += 1 / (second_sent - first_sent)
                matrix_syn[first_sent][second_sent] += role_weights[sentence_roles[first_sent]] * role_weights[sentence_roles[second_sent]]
                matrix_syn_dist[first_sent][second_sent] += role_weights[sentence_roles[first_sent]] * role_weights[sentence_roles[second_sent]] / (second_sent - first_sent)
        # print graph score to files
        out_file_u.write(str(compute_avg_outdeg(matrix_u)) + "\n")
        out_file_u_dist.write(str(compute_avg_outdeg(matrix_u_dist)) + "\n")
        out_file_w.write(str(compute_avg_outdeg(matrix_w)) + "\n")
        out_file_w_dist.write(str(compute_avg_outdeg(matrix_w_dist)) + "\n")
        out_file_syn.write(str(compute_avg_outdeg(matrix_syn)) + "\n")
        out_file_syn_dist.write(str(compute_avg_outdeg(matrix_syn_dist)) + "\n")
        # print graph adjacency matrix
        for i in range(num_sentences):
            for j in range(num_sentences):
                out_file_u.write(str(matrix_u[i][j]) + " ")
                out_file_u_dist.write(str(matrix_u_dist[i][j]) + " ")
                out_file_w.write(str(matrix_w[i][j]) + " ")
                out_file_w_dist.write(str(matrix_w_dist[i][j]) + " ")
                out_file_syn.write(str(matrix_syn[i][j]) + " ")
                out_file_syn_dist.write(str(matrix_syn_dist[i][j]) + " ")
            out_file_u.write("\n")
            out_file_u_dist.write("\n")
            out_file_w.write("\n")
            out_file_w_dist.write("\n")
            out_file_syn.write("\n")
            out_file_syn_dist.write("\n")
        out_file_u.close()
        out_file_u_dist.close()
        out_file_w.close()
        out_file_w_dist.close()
        out_file_syn.close()
        out_file_syn_dist.close()
