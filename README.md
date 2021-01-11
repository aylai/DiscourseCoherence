# DiscourseCoherenceDev

## Dependencies

This code is written in Python. The dependencies are:

* Python3 (with recent versions of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/))
* [Pytorch](http://pytorch.org/) (tested on Pytorch 0.3.1)
* [scikit-learn](http://scikit-learn.org/stable/)
* NLTK >= 3
* [pycorenlp](https://github.com/smilli/py-corenlp)
* [progressbar2](https://pypi.org/project/progressbar2/)

## Evaluation

All models can be trained for 4 different evaluation tasks:
- 'class': 3-class classification (low, medium, high coherence)
- 'score_pred': mean score prediction
- 'perm': binary ranking of original vs. permuted texts (requires text permutation files)
- 'minority': binary classification of low coherence vs. all other texts

## Data Directory Structure

The GDCD data is available by request (see https://github.com/aylai/GCDC-corpus for details). To run the preprocessing scripts, you will have to create a directory for each corpus in 'data/' containing the train and test csv files. For the Yelp data, you will need to download the data separately (https://www.yelp.com/dataset) and add the corresponding review titles and texts to the incomplete csv file (the CSV header should match the fields in the Clinton and Enron CSVs).


## Preprocessing

'corpus' refers to the corpus name: {Yahoo, Clinton, Enron, Yelp}

*1) Extract texts from CSV to separate files.* Required for entity grid and entity graph models, as well as generating text permutations for evaluation.

Input: data/[corpus]/[corpus]_train.csv and data/[corpus]/[corpus]_test.csv files. Output: data/[corpus]/text/ directory containing all individual text files.

```
python3 csv_to_text_files.py [corpus]
```

*2) Generate permutation text files (20 per text).* Only generates permutations for high-coherence texts (label = 3). Required for evaluating any model on the binary permutation ranking task (can skip this step for all other experiments).

Input: data/[corpus]/[corpus]_train.csv, data/[corpus]/[corpus]_test.csv, and data/[corpus]/text/ files. Output: data/[corpus]/text_permute directory containing original and permuted text files for all high-coherence texts.

```
python3 generate_high_coh_permutations [corpus]
```

*3) Extract entity grid files (requires Stanford CoreNLP for parsing).* Required for entity grid and entity graph models.

This step requires running the Stanford CoreNLP server (with Java 8, not Java 9). More details here: https://github.com/smilli/py-corenlp and here: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started. You will probably need to run the server with -timeout 50000 (or possibly higher) instead of -timeout 15000 in order to process the longest documents in this dataset.

**Original files only:**

Input: data/[corpus]/text/ files. Output: data/[corpus]/parsed/ and data/[corpus]/grid/ files.


```
python3 extract_entity_grid.py [corpus]
```

**Permuted files:**

Input: data/[corpus]/text_permute/ files. Output: data/[corpus]/parsed_permute/ and data/[corpus]/grid_permute/ files


```
python3 extract_entity_grid_perm.py [corpus]
```

*4) Extract entity graph files from entity grid files.* Extracts 6 different types of entity graphs: {unweighted, weighted, and syntax-sensitive} with or without distance discounting. Specify 'true' or 'false' for 'is_permutation' argument. Required for entity graph model.

Input: data/[corpus]/grid[_permute]/ files. Output: data/[corpus]/graph[_permute] files.

```
python3 extract_graph_from_grid.py [corpus] [is_permutation]
```

*5) Extract features from entity grid files.* Required for entity grid model. Must specify:
- 'seq_len' the number of sequential sentences over which to compute features (e.g. 2, 3, 4)
- 'salience_threshold' the threshold for salient vs. non-salient entities (e.g. 2, 3, 4 occurrences); specify '1' for only one saliance class
- 'syntax_opt' 1 to use syntactic roles (s, o, x, -); 0 to ignore syntactic roles (x, -)
- 'is_permutation': 'true' if using permuted text files, 'false' if using original text files only

Input: data/[corpus]/grid[_permute]/ files. Output: data/[corpus]/features[_permute]/[feature_set]

```
python3 extract_features_from_grid.py [corpus] [seq_len] [salience_threshold] [syntax_opt] [is_permutation]
```

## Models

### Entity grid

Train a random forest classifier on entity grid features. 'feature_set' specifies the name of the feature directory in data/[corpus]/features[_permute]. 'evaluation' specifies the task: 'class', 'score_pred', 'minority', 'perm'.

```
python3 entity_grid.py [corpus] [feature_set] [evaluation]
```

### Entity graph

Use entity graph outdegree values to evaluate on different tasks. Must specify graph type: [u, u_dist, w, w_dist, syn, syn_dist].

Thresholds (any real numbers):
- 'class': must specify 'threshold1' and 'threshold2'
- 'minority': must specify 'threshold1'
- 'perm': no threshold
- 'score_pred': no threshold

```
python3 entity_graph.py [corpus] [evaluation] [graph_type] [opt:threshold1] [opt:threshold2]
```

### Neural clique

Train 3-class classification model on Yahoo data with clique size = 7 sentences:
```
python3 main.py --model_name yahoo_class_model --train_corpus Yahoo --model_type clique --task class --clique 7
```

See main.py for other parameters.

### Neural SentAvg

Train 3-class classification model on Yahoo data:
```
python3 main.py --model_name yahoo_class_model --train_corpus Yahoo --model_type sent_avg --task class
```

See main.py for other parameters.

**Note:** the SentAvg model cannot be trained for the binary permutation ranking task (because all sentence order permutations have the same score).

### Neural ParSeq

Train 3-class classification model on Yahoo data:
```
python3 main.py --model_name yahoo_class_model --train_corpus Yahoo --model_type par_seq --task class
```

See main.py for other parameters.

**Note:** the ParSeq model currently cannot be trained for the binary permutation ranking task.