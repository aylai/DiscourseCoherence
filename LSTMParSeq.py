import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

# todo this whole class
class LSTMParSeq(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMParSeq, self).__init__()
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.lstm_dim = params['lstm_dim']
        self.dropout = params['dropout']
        self.embeddings = data_obj.word_embeds
        self.word_lstm = nn.LSTM(self.embedding_dim, self.lstm_dim)
        self.word_lstm_hidden = None
        self.sent_lstm = nn.LSTM(self.lstm_dim, self.lstm_dim)
        self.sent_lstm_hidden = None
        self.par_lstm = nn.LSTM(self.lstm_dim, self.lstm_dim)
        self.par_lstm_hidden = None
        self.hidden_layer = nn.Linear(self.lstm_dim, self.hidden_dim)
        if params['task'] == 'perm':
            num_labels = 2
        elif params['task'] == 'minority':
            num_labels = 2
        elif params['task'] == 'class':
            num_labels = 3
        elif params['task'] == 'score_pred':
            num_labels = 1
        self.predict_layer = nn.Linear(self.hidden_dim, num_labels)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform(m.weight)
        if USE_CUDA:
            self.hidden_layer = self.hidden_layer.cuda()
            self.predict_layer = self.predict_layer.cuda()

    def init_hidden(self, batch_size):
        if USE_CUDA:
            return (Variable(torch.zeros(1, batch_size, self.lstm_dim).cuda()),
                    Variable(torch.zeros(1, batch_size, self.lstm_dim)).cuda())
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_dim)),
                    Variable(torch.zeros(1, batch_size, self.lstm_dim)))

    def forward(self, inputs, input_lengths, original_index):
        doc_vecs = None
        for i in range(len(inputs)): # loop over docs
            par_vecs = None
            for j in range(len(inputs[i])): # loop over paragraphs
                doc_batch_size = len(inputs[i][j]) # number of sents
                self.word_lstm_hidden = self.init_hidden(doc_batch_size)
                seq_tensor = self.embeddings(inputs[i][j])
                # pack
                packed_input = pack_padded_sequence(seq_tensor, input_lengths[i][j], batch_first=True)
                packed_output, (ht, ct) = self.word_lstm(packed_input, self.word_lstm_hidden)
                # reorder
                final_output = ht[-1]
                odx = original_index[i][j].view(-1, 1).expand(len(input_lengths[i][j]), final_output.size(-1))
                output_unsorted = torch.gather(final_output, 0, Variable(odx))
                # LSTM to produce paragraph vector from sentence vectors
                output_unsorted = output_unsorted.unsqueeze(1)
                self.sent_lstm_hidden = self.init_hidden(output_unsorted.size(1)) # batch size 1
                output_pars, (ht, ct) = self.sent_lstm(output_unsorted, self.sent_lstm_hidden)
                final_output = ht[-1]
                # append paragraph vector to batch
                if par_vecs is None:
                    par_vecs = final_output
                else:
                    par_vecs = torch.cat([par_vecs, final_output], dim=0)
            # LSTM over paragraph vectors to create document vector
            par_vecs = par_vecs.unsqueeze(1)
            self.par_lstm_hidden = self.init_hidden(par_vecs.size(1)) # batch size 1
            output_doc, (ht, ct) = self.par_lstm(par_vecs, self.par_lstm_hidden)
            final_output = ht[-1]
            # append doc vector to batch
            if doc_vecs is None:
                doc_vecs = final_output
            else:
                doc_vecs = torch.cat([doc_vecs, final_output], dim=0)
        doc_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(doc_vecs))), p=self.dropout, training=self.training)
        coherence_pred = self.predict_layer(doc_vectors)
        if self.task != 'score_pred':
            coherence_pred = F.softmax(coherence_pred, dim=0)
        return coherence_pred
