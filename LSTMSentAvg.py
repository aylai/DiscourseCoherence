import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class LSTMSentAvg(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMSentAvg, self).__init__()
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.lstm_dim = params['lstm_dim']
        self.dropout = params['dropout']
        self.embeddings = data_obj.word_embeds
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim)
        self.hidden = None
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
        lstm_out = None  # document vectors
        for i in range(len(inputs)):  # loop over docs
            doc_batch_size = len(inputs[i])  # number of sents
            self.hidden = self.init_hidden(doc_batch_size)
            seq_tensor = self.embeddings(inputs[i])
            # pack
            packed_input = pack_padded_sequence(seq_tensor, input_lengths[i], batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)
            # reorder
            final_output = ht[-1]
            odx = original_index[i].view(-1, 1).expand(len(input_lengths[i]), final_output.size(-1))
            output_unsorted = torch.gather(final_output, 0, Variable(odx))
            # sum sentence vectors
            output_sum = torch.sum(output_unsorted, 0).unsqueeze(0)
            if lstm_out is None:
                lstm_out = output_sum
            else:
                lstm_out = torch.cat([lstm_out, output_sum], dim=0)
        doc_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(lstm_out))), p=self.dropout, training=self.training)
        coherence_pred = self.predict_layer(doc_vectors)
        if self.task != 'score_pred':
            coherence_pred = F.softmax(coherence_pred, dim=0)
        return coherence_pred
