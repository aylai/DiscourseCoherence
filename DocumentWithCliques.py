import torch
from nltk import word_tokenize
import random

is_cuda = torch.cuda.is_available()


class DocumentWithCliques(object):

    def __init__(self, orig_sentences, clique_size, permutation_indices=None, id = '', label=None):
        self.id = id
        self.clique_size = clique_size
        self.orig_sentences = []
        for sent in orig_sentences:
            sent = sent.strip()
            if sent == "":
                continue
            if sent == "<d>" or sent == "</d>":
                self.orig_sentences.append([sent])
            else:
                self.orig_sentences.append(word_tokenize(sent))
        self.permutation_indices = []
        if permutation_indices is not None:
            self.permutation_indices = permutation_indices  # index into orig_sentences
        self.index_sentences = [] # token-indexed version of self.orig_sentences
        self.orig_full_sequence = None
        self.perm_full_sequences = None
        self.label = label

    # turn full doc into flat sequence of word indices
    def get_orig_full_sequence(self):
        if self.orig_full_sequence is not None:
            return self.orig_full_sequence
        self.orig_full_sequence = []
        for sent in self.index_sentences:
            self.orig_full_sequence.extend(sent)
        return self.orig_full_sequence

    # turn all doc permutations into sentence lists of word indices
    def get_perm_index_sentences(self):
        if self.perm_full_sequences is not None:
            return self.get_perm_full_sequences
        self.perm_doc_sentences = []
        for perm in self.permutation_indices:
            doc_temp = []
            for sent_idx in perm:
                doc_temp.append(self.index_sentences[sent_idx])
            self.perm_doc_sentences.append(doc_temp)
        return self.perm_doc_sentences

    # turn all doc permutations into flat sequences of word indices
    def get_perm_full_sequences(self):
        if self.perm_full_sequences is not None:
            return self.get_perm_full_sequences
        self.perm_full_sequences = []
        for perm in self.permutation_indices:
            doc_temp = []
            for sent_idx in perm:
                doc_temp.extend(self.index_sentences[sent_idx])
            self.perm_full_sequences.append(doc_temp)
        return self.perm_full_sequences

    def create_cliques_orig(self): # assume self.index_sentences is non-empty
        self.orig_cliques = []
        self.orig_cliques_index = []
        for i in range(len(self.index_sentences) - self.clique_size + 1):
            clique = []
            clique_index = []
            for j in range(self.clique_size):
                clique.append(self.index_sentences[i + j])
                clique_index.append(i+j)
            self.orig_cliques.append(clique)
            self.orig_cliques_index.append(clique_index)

    # randomly create negative cliques from the original document sentences
    def create_cliques_neg(self):
        self.neg_cliques = []
        for orig_clique in self.orig_cliques_index: # negative example for each window: replace center sentence
            if len(self.orig_cliques_index) == 1:
                break # no possible negative cliques for this doc
            valid_sentences = {}
            for sent_idx in orig_clique:
                valid_sentences[sent_idx] = 1
            valid_sentences[0] = 1  # don't allow <d> pad
            valid_sentences[len(self.index_sentences) - 1] = 1  # don't allow </d> pad
            if len(valid_sentences) == len(self.index_sentences):
                continue # no possible negative cliques for this positive clique
            center_idx = int(len(orig_clique) / 2)
            new_sent = random.randrange(len(self.index_sentences))
            while new_sent in valid_sentences:
                new_sent = random.randrange(len(self.index_sentences))
            neg_clique = []
            for sent_idx in orig_clique:
                neg_clique.append(self.index_sentences[sent_idx])
            neg_clique[center_idx] = self.index_sentences[new_sent]
            self.neg_cliques.append(neg_clique)

    # create cliques for predefined permutations of this document
    def create_cliques_perm(self):
        self.perm_cliques = []
        for perm in self.permutation_indices:
            cliques = []
            for i in range(len(perm) - self.clique_size + 1):
                clique = []
                for j in range(self.clique_size):
                    clique.append(self.index_sentences[perm[i + j]])
                cliques.append(clique)
            self.perm_cliques.append(cliques)
