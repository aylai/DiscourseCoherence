from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import os, csv, random, sys
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

corpus = sys.argv[1]
feature_dirname = sys.argv[2]
evaluation = sys.argv[3]


def evaluate_fscore(labels, predictions):
    tp = 0
    fp = 0
    fn = 0
    for idx, label in enumerate(labels):
        pred = predictions[idx]
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

def read_features(text_ids, labels_dict):
    instances = []
    labels = []
    for text_id in text_ids:
        if evaluation == 'perm':
            orig_instance = []
            orig_filename = in_dir + 'features_permute/' + feature_dirname + '/' + text_id + '.0.feat'
            if not os.path.exists(orig_filename):
                continue  # file without valid permutations
            with open(in_dir + 'features_permute/' + feature_dirname + '/' + text_id + '.0.feat', 'r') as in_file:
                for line in in_file:
                    line = line.strip().split()
                    for val in line:
                        orig_instance.append(float(val))
            for j in range(1, 21):
                other_doc_instance = []
                filename = in_dir + 'features_permute/' + feature_dirname + '/' + text_id + '.perm-' + str(j) + '.feat'
                if not os.path.exists(filename):
                    continue
                with open(in_dir + 'features_permute/' + feature_dirname + '/' + text_id + '.perm-' + str(j) + '.feat',
                          'r') as in_file:
                    for line in in_file:
                        line = line.strip().split()
                        for val in line:
                            other_doc_instance.append(float(val))
                # randomly order documents
                doc_order = random.randint(1, 2)
                if doc_order == 1:  # doc1 = orig document
                    feat = np.asarray(orig_instance) - np.asarray(other_doc_instance)
                    label = 1
                else:
                    feat = np.asarray(other_doc_instance) - np.asarray(orig_instance)
                    label = 2
                instances.append(feat)
                labels.append(label)
        else:
            instance = []
            with open(in_dir + 'features/' + feature_dirname + '/' + text_id + '.feat','r') as in_file:
                for line in in_file:
                    line = line.strip().split()
                    for val in line:
                        instance.append(float(val))
            labels.append(labels_dict[text_id])
            instances.append(instance)
    return instances, labels


in_dir = 'data/' + corpus + '/'
train_ids = []
train_labels_dict = {}
eval_ids = []
eval_labels_dict = {}
splits = ['train', 'test']
for split in splits:
    if evaluation == 'perm':
        in_filename = in_dir + corpus + '_' + split + '_perm.csv'
    else:
        in_filename = in_dir + corpus + '_' + split + '.csv'
    with open(in_filename, 'r') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            text_id = row['text_id']
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
            if split == 'train':
                train_ids.append(text_id)
                train_labels_dict[text_id] = label
            elif split == 'test':
                eval_ids.append(text_id)
                eval_labels_dict[text_id] = label
# read features
train_instances, train_labels = read_features(train_ids, train_labels_dict)
train_arr = np.array(train_instances)
eval_instances, eval_labels = read_features(eval_ids, eval_labels_dict)
eval_arr = np.array(eval_instances)
# shuffle training data
indices = [idx for idx in range(len(train_instances))]
random.shuffle(indices)
shuffle_train_instances = [train_instances[idx] for idx in indices]
shuffle_train_labels = [train_labels[idx] for idx in indices]
# train and evaluate model
if evaluation == 'class' or evaluation == 'minority' or evaluation == 'perm':
    clf = RandomForestClassifier()
elif evaluation == 'score_pred':
    clf = RandomForestRegressor()
clf.fit(np.array(shuffle_train_instances), np.array(shuffle_train_labels))
# predictions = clf.predict(np.array(eval_instances))
if evaluation == 'class' or evaluation == 'perm':
    accuracy = clf.score(np.array(eval_instances), np.array(eval_labels))
    print("Results on test:\nAccuracy: %0.2f" % (accuracy * 100))
elif evaluation == 'score_pred':
    predictions = clf.predict(np.array(eval_instances))
    mse = mean_squared_error(eval_labels, predictions)
    corr = spearmanr(eval_labels, predictions)[0]
    print("Results on test:\nSpearman corr: %0.3f  MSE: %0.3f" % (corr, mse))
if evaluation == 'minority':
    predictions = clf.predict(np.array(eval_instances))
    eval_precision, eval_recall, eval_fscore = evaluate_fscore(eval_labels, predictions)
    print("Results on test:\nPrecision: %0.2f  Recall: %0.2f  F0.5: %0.2f" % (eval_precision, eval_recall, eval_fscore))

