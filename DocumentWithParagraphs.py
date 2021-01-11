from nltk import sent_tokenize, word_tokenize


class DocumentWithParagraphs(object):

    def __init__(self, text_with_line_breaks, label=None, orig_sentences=None, permutation_indices=None, id=''):
        self.id = id
        self.text = []
        self.text_indexed = []
        self.label = label
        lines = text_with_line_breaks.splitlines()
        for line in lines:
            line = line.strip()
            if line != "": # this is a paragraph
                paragraph = []
                sents = sent_tokenize(line)
                for sent in sents:
                    words = word_tokenize(sent)
                    paragraph.append(words)
                self.text.append(paragraph)
        self.orig_sentences = []
        if orig_sentences is not None:
            self.orig_sentences = orig_sentences
        self.permutation_indices = []
        if permutation_indices is not None:
            self.permutation_indices = permutation_indices

    def get_paragraphs(self):
        return self.text_indexed

    def get_sentences(self):
        sentences = []
        for paragraph in self.text_indexed:
            for sent in paragraph:
                sentences.append(sent)
        return sentences

    def get_words(self):
        words = []
        for paragraph in self.text_indexed:
            for sent in paragraph:
                for word in sent:
                    words.append(word)
        return words