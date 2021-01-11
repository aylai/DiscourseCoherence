import os, csv, sys
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

corpus = sys.argv[1]
evaluation = sys.argv[2]
graph_type = sys.argv[3]
if evaluation == 'class':
    threshold1 = float(sys.argv[4])
    threshold2 = float(sys.argv[5])
if evaluation == 'minority':
    threshold1 = float(sys.argv[4])


def compute_corr(test_labels, test_scores):
    all_labels = []
    all_scores = []
    for test_id in test_labels:
        all_labels.append(test_labels[test_id])
        all_scores.append(test_scores[test_id])
    mse = mean_squared_error(all_labels, all_scores)
    corr = spearmanr(all_labels, all_scores)[0]
    return mse, corr


def compute_fscore(threshold, train_labels, train_scores):
    tp = 0
    fp = 0
    fn = 0
    for train_id in train_labels:
        label = train_labels[train_id]
        score = train_scores[train_id]
        if score < threshold:
            pred = 1
        else:
            pred = 0
        if pred == label:
            if label == 1:
                tp += 1
        else: # incorrect prediction
            if pred == 1:
                fp += 1
            else:
                fn += 1
    precision = 0
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    recall = 0
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    f05 = 0  # compute F0.5 score
    if (precision + recall) > 0:
        f05 = (1.25 * precision * recall) / (1.25 * precision + recall)
    return precision, recall, f05

def evaluate_perm(test_scores_orig, test_scores_perm):
    num_correct = 0
    num_total = 0
    for test_id in test_scores_orig:
        orig_score = test_scores_orig[test_id]
        for perm_id in test_scores_perm[test_id]:
            perm_score = test_scores_perm[test_id][perm_id]
            if orig_score > perm_score:
                num_correct += 1
            num_total += 1
    return num_correct, num_total


in_dir = 'data/' + corpus + '/'
# read all test data
test_ids = []
test_labels = {}
if evaluation == 'perm':
    in_filename = in_dir + corpus + '_test_perm.csv'
else:
    in_filename = in_dir + corpus + '_test.csv'
with open(in_filename,'r') as in_file:
    reader = csv.DictReader(in_file)
    for row in reader:
        test_ids.append(row['text_id'])
        label = None
        if evaluation == 'class':
            label = int(row['labelA'])
        elif evaluation == 'score_pred':
            labels = [int(row['ratingA1']), int(row['ratingA2']), int(row['ratingA3'])]
            label = np.mean(labels)
        elif evaluation == 'minority':
            num_low_judgments = 0
            if row['ratingA1'] == '1':
                num_low_judgments += 1
            if row['ratingA2'] == '1':
                num_low_judgments += 1
            if row['ratingA3'] == '1':
                num_low_judgments += 1
            if num_low_judgments >= 2:
                label = 1
            else:
                label = 0
        test_labels[row['text_id']] = label
test_scores = {}
test_scores_perm = {}
test_scores_orig = {}
num_correct = 0
num_total = 0
for test_id in test_ids:
    if evaluation == 'perm':
        orig_filename = in_dir + 'graph_permute/' + test_id + '.0.graph_' + graph_type
        if not os.path.exists(orig_filename):
            continue  # no valid permutations
        with open(in_dir + 'graph_permute/' + test_id + '.0.graph_' + graph_type, 'r') as in_file:
            for line in in_file:
                score = float(line.strip())
                test_scores_orig[test_id] = score
                test_scores_perm[test_id] = {}
                break
        # read permutations
        for i in range(1, 21):
            perm_filename = in_dir + 'graph_permute/' + test_id + '.perm-' + str(i) + '.graph_' + graph_type
            if not os.path.exists(perm_filename):
                continue
            with open(perm_filename, 'r') as in_file:
                for line in in_file:
                    score = float(line.strip())
                    test_scores_perm[test_id][i] = score
                    break
    else:
        with open(in_dir + 'graph/' + test_id + '.graph_' + graph_type, 'r') as in_file:
            for line in in_file:
                score = float(line.strip())
                test_scores[test_id] = score
                if evaluation == 'class':
                    if score < threshold1:
                        pred_label = 1
                    elif score < threshold2:
                        pred_label = 2
                    else:
                        pred_label = 3
                    gold_label = test_labels[test_id]
                    if gold_label == pred_label:
                        num_correct += 1
                    num_total += 1
                break
if evaluation == 'class':
    print("Results on test:\nAccuracy: %0.2f" % (100 * (num_correct / num_total)))
elif evaluation == 'minority':
    precision, recall, fscore = compute_fscore(threshold1, test_labels, test_scores)
    print("Results on test:\nPrecision: %0.2f  Recall: %0.2f  F0.5: %0.2f" % (precision, recall, fscore))
elif evaluation == 'score_pred':
    mse, corr = compute_corr(test_labels, test_scores)
    print("Results on test:\nSpearman corr: %0.3f  MSE: %0.3f" % (corr, mse))
elif evaluation == 'perm':
    num_correct, num_total = evaluate_perm(test_scores_orig, test_scores_perm)
    print("Results on test:\nAccuracy: %0.2f" % (100 * (num_correct / num_total)))
