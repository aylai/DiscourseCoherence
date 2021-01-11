import torch
import torch.nn as nn
import numpy as np
import os
from DocumentWithCliques import DocumentWithCliques
from DocumentWithParagraphs import DocumentWithParagraphs
import random
from torch.autograd import Variable
from nltk import word_tokenize
from nltk import sent_tokenize
import csv

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
is_cuda = torch.cuda.is_available()


class Data(object):

    def __init__(self, params):
        self.params = params
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_to_idx['<pad>'] = 0
        self.idx_to_word[0] = '<pad>'
        self.word_embeds = None

    def read_orig_doc(self, filename, data_type, for_clique):
        sentences = []
        with open(filename, "r") as in_file:
            for line in in_file:
                line = line.strip()
                if data_type == "BL":
                    line = line.split(None, 1)[1] # remove sent ID
                if not self.params['case_sensitive']:
                    line = line.lower()
                sentences.append(line)
        if for_clique:
            for i in range(int(self.params['clique_size'] / 2)):
                sentences.insert(0, "<d>")
                sentences.append("</d>")
        return sentences

    def read_perm_doc(self, filename, sentences, data_type, for_clique):
        sentence_indices = []
        with open(filename, "r") as in_file:
            for line in in_file:
                line = line.strip()
                if data_type == "BL":
                    line = line.split(None, 1)[1]
                if not self.params['case_sensitive']:
                    line = line.lower()
                sentence_indices.append(sentences.index(line))
        if for_clique:
            for i in range(int(self.params['clique_size'] / 2)):
                sentence_indices.insert(0, 0)  # start pad
                sentence_indices.append(len(sentences) - 1)
        return sentence_indices

    # read my Yahoo/Clinton/Enron data for 3-way classification (full train/test)
    def read_data_class(self, params, split):
        # corpus = params['data_dir'].rsplit('/', 2)[1]
        if split == 'train' or split == 'train_nodev':
            corpus = params['train_corpus']
        elif split == 'test':
            corpus = params['test_corpus']
        documents = []
        add_new_words = False
        if self.word_embeds is None and split == "train":
            add_new_words = True
        filename = corpus + '_' + split + '.csv'
        with open(params['data_dir'] + corpus + '/' + filename,'r') as in_file:
            reader = csv.DictReader(in_file)
            for row in reader:
                text = row['text']
                if not self.params['case_sensitive']:
                    text = text.lower()
                text_id = row['text_id']
                if params['task'] == 'score_pred':
                    labels = [int(row['ratingA1']), int(row['ratingA2']), int(row['ratingA3'])]
                    label = np.mean(labels)
                # elif params['eval_minority']:
                elif params['task'] == 'minority':
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
                else:
                    label = int(row['labelA'])
                    label = label - 1 # zero-indexing
                if params['model_type'] == 'clique':
                    orig_sentences = []
                    for par in text.splitlines():
                        par = par.strip()
                        if par == "":
                            continue
                        orig_sentences.extend(sent_tokenize(par))
                    for i in range(int(self.params['clique_size'] / 2)):
                        orig_sentences.insert(0, "<d>")
                        orig_sentences.append("</d>")
                    doc = DocumentWithCliques(orig_sentences, self.params['clique_size'], None, text_id, label)
                    for sent in doc.orig_sentences:
                        sent_idx = []
                        for token in sent:
                            idx = self.add_token_to_index(token, add_new_words)
                            sent_idx.append(idx)
                        doc.index_sentences.append(sent_idx)
                elif params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq':
                    doc = DocumentWithParagraphs(text, label, id=text_id)
                    # index words
                    doc_indexed = []
                    for para in doc.text:
                        para_indexed = []
                        for sent in para:
                            sent_indexed = []
                            for word in sent:
                                sent_indexed.append(self.add_token_to_index(word, add_new_words))
                            para_indexed.append(sent_indexed)
                        doc_indexed.append(para_indexed)
                    doc.text_indexed = doc_indexed
                documents.append(doc)
        return documents

    # read my Yahoo/Clinton/Enron data for binary ranking permutation task (cross-validation fold)
    def read_data_perm(self, params, split):
        # corpus = params['data_dir'].rsplit('/', 2)[1]
        if split == 'train' or split == 'train_nodev':
            corpus = params['train_corpus']
        elif split == 'dev':
            corpus = params['train_corpus']
        elif split == 'test':
            corpus = params['test_corpus']
        documents = []
        add_new_words = False
        if self.word_embeds is None and split == "train":
            add_new_words = True
        # get list of files in this split
        filename = corpus + '_' + split + '_perm.csv'
        text_ids = []
        with open(params['data_dir'] + corpus + '/' + filename, 'r') as in_file:
            reader = csv.DictReader(in_file)
            for row in reader:
                text_ids.append(row['text_id'])
        for text_id in text_ids:
            # read orig file
            if not os.path.exists(params['data_dir'] + corpus + '/text_permute/' + text_id + '_sent.txt'):
                print(text_id + " not found in permutation data.")
                continue
            orig_sentences = self.read_orig_doc(params['data_dir'] + corpus + '/text_permute/' + text_id + '_sent.txt', "mine", params['model_type']=='clique')
            perm_docs = []
            for i in range(1,21):
                filename_perm = params['data_dir'] + corpus + '/text_permute/' + text_id + '.perm-' + str(i) + '.txt'
                if not os.path.exists(filename_perm):
                    continue
                perm_docs.append(self.read_perm_doc(filename_perm, orig_sentences, "mine", params['model_type']=='clique'))
            if len(perm_docs) == 0:
                continue  # document has no permutations (is only a single sentence) -- remove from data
            if params['model_type'] == 'clique':
                doc = DocumentWithCliques(orig_sentences, self.params['clique_size'], perm_docs, text_id)
                for sent in doc.orig_sentences:
                    sent_idx = []
                    for token in sent:
                        idx = self.add_token_to_index(token, add_new_words)
                        sent_idx.append(idx)
                    doc.index_sentences.append(sent_idx)
            elif params['model_type'] == 'sent_avg' or params['model_type'] == 'par_seq':
                # note this loses paragraph info (not useful for permutations task)
                doc = DocumentWithParagraphs("\n".join(orig_sentences), None, orig_sentences, perm_docs, text_id)
                # index words
                doc_indexed = []
                for para in doc.text:
                    para_indexed = []
                    for sent in para:
                        sent_indexed = []
                        for word in sent:
                            sent_indexed.append(self.add_token_to_index(word, add_new_words))
                        para_indexed.append(sent_indexed)
                    doc_indexed.append(para_indexed)
                doc.text_indexed = doc_indexed
            documents.append(doc)
        return documents

    def add_token_to_index(self, token, add_new_words):
        if token not in self.word_to_idx and add_new_words:  # add to vocab
            idx = len(self.word_to_idx)
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
        elif token not in self.word_to_idx and not add_new_words:  # replace with UNK token
            if 'unk' not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx['unk'] = idx
                self.idx_to_word[idx] = 'unk'
            return self.word_to_idx['unk']
        return self.word_to_idx[token]

    def create_cliques(self, documents, task, limit=None): # create cliques of k sentences
        items = []
        labels = []
        for doc in documents:
            doc.create_cliques_orig()
            for clique in doc.orig_cliques:
                temp_item = []
                for sent in clique:
                    # temp_item.append(Variable(LongTensor(list(sent))).view(1, -1))
                    temp_item.append(list(sent))
                items.append(temp_item)
                if task == 'perm':
                    labels.append(1) # coherent clique
                elif task == 'class' or task == 'score_pred' or task == 'minority':
                    labels.append(doc.label)
            if task == 'perm':
                doc.create_cliques_neg()
                for clique in doc.neg_cliques:
                    temp_item = []
                    for sent in clique:
                        temp_item.append(list(sent))
                    items.append(temp_item)
                    labels.append(0) # incoherent clique
                doc.create_cliques_perm()
        if limit is not None and limit < len(items):
            indices = list(range(len(items)))
            random.shuffle(indices)
            indices = indices[:limit]
            new_items = []
            new_labels = []
            for i in indices:
                new_items.append(items[i])
                new_labels.append(labels[i])
            items = new_items
            labels = new_labels
        return items, labels

    def retrieve_doc_cliques_by_label(self, document, task, limit=None): # create cliques of k sentences
        items_pos = []
        items_neg = []
        document.create_cliques_orig()
        document.create_cliques_neg()
        for clique in document.orig_cliques:
            temp_item = []
            for sent in clique:
                # temp_item.append(Variable(LongTensor(list(sent))).view(1, -1))
                temp_item.append(list(sent))
            items_pos.append(temp_item)
        if task == 'perm':
            for perm_doc in document.perm_cliques:
                perm_temp = []
                for clique in perm_doc:
                    temp_item = []
                    for sent in clique:
                        # temp_item.append(Variable(LongTensor(list(sent))).view(1, -1))
                        temp_item.append(list(sent))
                    perm_temp.append(temp_item)
                items_neg.append(perm_temp)
        return items_pos, items_neg

    def retrieve_doc_sents_by_label(self, document, limit=None): # create cliques of k sentences
        items_pos = []
        items_neg = []
        orig_sentences = document.get_sentences()
        for sent in orig_sentences:
            # items_pos.append(Variable(LongTensor(list(sent))).view(1, -1))
            items_pos.append(list(sent))
        for perm_doc in document.permutation_indices:
            doc_neg = []
            for sent_idx in perm_doc:
                # doc_neg.append(Variable(LongTensor(list(orig_sentences[sent_idx]))).view(1, -1))
                doc_neg.append(list(orig_sentences[sent_idx]))
            items_neg.append(doc_neg)
        return [items_pos], items_neg

    def create_doc_sents(self, documents, split_type, task, limit=-1):
        items = []
        labels = []
        ids = []
        for doc in documents:
            doc_items = []
            if split_type == 'paragraph':
                for paragraph in doc.get_paragraphs():
                    par_sentences = []
                    for sent in paragraph:
                        par_sentences.append(sent)
                    doc_items.append(par_sentences)
            if split_type == 'sentence':
                if task == 'class' or task == 'score_pred' or task == 'minority':
                    for sent in doc.get_sentences():
                        doc_items.append(sent)
                elif task == 'perm':
                    orig_sentences = doc.get_sentences()
                    perm_count = 1
                    for perm in doc.permutation_indices:
                        # create permuted doc
                        doc_items = []
                        for sent_idx in perm:
                            doc_items.append(orig_sentences[sent_idx])
                        items.append(doc_items)
                        labels.append(0) # permuted
                        ids.append(doc.id+".0")
                        # create orig doc for each permuted doc
                        doc_items = []
                        for sent in orig_sentences:
                            doc_items.append(sent)
                        items.append(doc_items)
                        labels.append(1)
                        ids.append(doc.id+"."+str(perm_count))
                        perm_count += 1
            if task != "perm":
                items.append(doc_items)
                labels.append(doc.label)
                ids.append(doc.id)
        if -1 < limit < len(items):
            indices = list(range(len(items)))
            random.shuffle(indices)
            indices = indices[:limit]
            new_items = []
            new_labels = []
            new_ids = []
            for i in indices:
                new_items.append(items[i])
                new_labels.append(labels[i])
                new_ids.append(ids[i])
            items = new_items
            labels = new_labels
            ids = new_ids
        return items, labels, ids

    def load_vectors(self):
        print("\nLoading vectors:")
        if self.params['vector_type'] == 'glove':
            data = []
            for line in open(self.params['vector_path']):
                tokens = line.split()
                if len(tokens) != 301:
                    continue
                word = tokens[0]
                vector_len = len(tokens) - 1
                for t in tokens[1:]:
                    data.append(float(t))
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
            data_arr = np.reshape(data, newshape=(int(len(data)/vector_len), vector_len))
            # add pad array at index 0
            data_arr = np.concatenate((np.random.rand(1, vector_len), data_arr), 0)
            # add OOV array
            data_arr = np.concatenate((data_arr, np.random.rand(1, vector_len)), 0)
            idx = len(self.word_to_idx)
            self.word_to_idx['unk'] = idx
            self.idx_to_word[idx] = 'unk'
            # add doc start pad array
            data_arr = np.concatenate((data_arr, np.random.rand(1, vector_len)), 0)
            idx = len(self.word_to_idx)
            self.word_to_idx['<d>'] = idx
            self.idx_to_word[idx] = '<d>'
            # add doc end pad array
            data_arr = np.concatenate((data_arr, np.random.rand(1, vector_len)), 0)
            idx = len(self.word_to_idx)
            self.word_to_idx['</d>'] = idx
            self.idx_to_word[idx] = '</d>'
            self.word_embeds = nn.Embedding(data_arr.shape[0], data_arr.shape[1])
            if USE_CUDA:
                self.word_embeds = self.word_embeds.cuda()
            self.word_embeds.weight.data.copy_(torch.from_numpy(data_arr))
            self.word_embeds.weight.requires_grad = False
            print("loading: done")
            return self.word_embeds, vector_len
        else:
            print("unrecognized vector type")

    def rand_vectors(self, vocab_size):
        if 'unk' not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx['unk'] = idx
            self.idx_to_word[idx] = 'unk'
        if '<d>' not in self.word_to_idx:
            # add doc start pad
            idx = len(self.word_to_idx)
            self.word_to_idx['<d>'] = idx
            self.idx_to_word[idx] = '<d>'
        if '</d>' not in self.word_to_idx:
            # add doc end pad array
            idx = len(self.word_to_idx)
            self.word_to_idx['</d>'] = idx
            self.idx_to_word[idx] = '</d>'
        self.word_embeds = nn.Embedding(len(self.word_to_idx), self.params['embedding_dim'])
        if is_cuda:
            self.word_embeds = self.word_embeds.cuda()
        return self.word_embeds

    def get_batch(self, data, labels, indices, model_type, clique_size=0):
        batch = []
        batch_labels = []
        if model_type == 'clique':
            for i in range(clique_size):
                batch.append([])
        for idx in indices:
            batch_labels.append(labels[idx])
            if model_type == 'sent_avg' or model_type == 'par_seq':
                batch.append(data[idx])
            elif model_type == 'clique':
                for i in range(clique_size):
                    batch[i].append(data[idx][i])
        return batch, batch_labels

    def reverse_index(self, sorted_index):
        rev_index = []
        for val in sorted_index:
            rev_index.append(0)
        for idx, val in enumerate(sorted_index):
            rev_index[val] = idx
        return rev_index

    def reorder_list(self, data_list, reorder_idx):
        new_data_list = []
        for idx in reorder_idx:
            new_data_list.append(data_list[idx])
        return new_data_list

    def pad_to_batch(self, batch, word_to_idx, model_type, clique_size=0):  # batch is list of (sequence, label)
        if model_type == 'par_seq':
            input_var = []
            input_len = []
            reverse_index = []
            for doc in batch:
                doc_var = []
                doc_len = []
                doc_index = []
                for par in doc:
                    # batch_lengths = LongTensor([seq[0].size(0) for seq in par])
                    batch_lengths = LongTensor([len(seq) for seq in par])
                    sorted_lengths, original_index = torch.sort(batch_lengths, 0, descending=True)
                    doc_index.append(LongTensor(self.reverse_index(original_index)))
                    sorted_batch = sorted(par, key=lambda b: len(b), reverse=True)
                    x = sorted_batch
                    max_x = max([len(s) for s in x])
                    x_p = []
                    for i in range(len(par)):
                        if len(x[i]) < max_x:
                            x_p.append(torch.cat([Variable(LongTensor(x[i])).view(1,-1),
                                                  Variable(
                                                      LongTensor([word_to_idx['<pad>']] * (max_x - len(x[i])))).view(
                                                      1, -1)], 1))
                        else:
                            x_p.append(Variable(LongTensor(x[i])).view(1,-1))
                    input_var_temp = torch.cat(x_p)
                    doc_var.append(input_var_temp)
                    doc_len.append([list(map(lambda s: s == 0, t.data)).count(False) for t in input_var_temp])
                input_var.append(doc_var)
                input_len.append(doc_len)
                reverse_index.append(doc_index)
        if model_type == 'sent_avg':
            input_var = []
            input_len = []
            reverse_index = []
            for doc in batch:
                batch_lengths = LongTensor([len(seq) for seq in doc])
                sorted_lengths, original_index = torch.sort(batch_lengths, 0, descending=True)
                reverse_index.append(LongTensor(self.reverse_index(original_index)))
                sorted_batch = sorted(doc, key=lambda b: len(b), reverse=True)
                x = sorted_batch
                max_x = max([len(s) for s in x])
                x_p = []
                for i in range(len(doc)):
                    if len(x[i]) < max_x:
                        x_p.append(
                            torch.cat([Variable(LongTensor(x[i])).view(1,-1),
                                       Variable(LongTensor([word_to_idx['<pad>']] * (max_x - len(x[i])))).view(1,
                                                                                                                  -1)],
                                      1))
                    else:
                        x_p.append(Variable(LongTensor(x[i])).view(1,-1))
                input_var_temp = torch.cat(x_p)
                input_var.append(input_var_temp)
                input_len.append([list(map(lambda s: s == 0, t.data)).count(False) for t in input_var_temp])
        elif model_type == 'clique':
            # list of lists for each sentence-batch in a clique
            input_var = []
            input_len = []
            reverse_index = []
            for i in range(clique_size):
                batch_lengths = LongTensor([len(seq) for seq in batch[i]])
                sorted_lengths, original_index = torch.sort(batch_lengths, 0, descending=True)

                reverse_index.append(LongTensor(self.reverse_index(original_index)))
                x = sorted(batch[i], key=lambda b: len(b), reverse=True)
                max_x = max([len(s) for s in x])
                x_p = []
                for i in range(len(batch[i])):
                    if len(x[i]) < max_x:
                        x_p.append(
                            torch.cat(
                                [Variable(LongTensor(x[i])).view(1, -1), Variable(LongTensor([word_to_idx['<pad>']] * (max_x - len(x[i])))).view(1, -1)],
                                1))
                    else:
                        x_p.append(Variable(LongTensor(x[i])).view(1, -1))
                input_var.append(torch.cat(x_p))
                input_len.append(list(sorted_lengths))
        return input_var, input_len, reverse_index