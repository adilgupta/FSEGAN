"""
Define NN architecture, forward function and loss function
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from generic_model import generic_model


class LSTM(generic_model):

    def __init__(self, config, weights=None):

        super(LSTM, self).__init__(config)

        self.feat_dim, self.hidden_dim, self.num_phones, self.num_layers = config['feat_dim'], config['hidden_dim'], config['num_phones'], \
                                                       config['num_layers']

        if config['bidirectional']:
            self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.3,
                                bidirectional=True, batch_first=True)
            # 1 for pad token, 2 for bidirectional
            self.hidden2phone = nn.Linear(self.hidden_dim * 2, self.num_phones + 1)
        else:
            self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=0.3,
                                bidirectional=False, batch_first=True)
            self.hidden2phone = nn.Linear(self.hidden_dim, self.num_phones + 1)  # for pad token

        loss, optimizer = config['train']['loss_func'], config['train']['optim']

        if loss == 'CEL':

            if config['weighted_loss'] and weights is not None:
                self.loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weights)).float())
                print("Using Weighted CEL")
            else:
                weights = np.append(np.ones((self.num_phones)) / self.num_phones, np.zeros((1)))
                self.loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())
                print("Using CEL")

            loss_found = True

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=config['train']['lr'], momentum=0.9)
            optim_found = True
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'])
            optim_found = True

        if loss_found == False or optim_found == False:
            print("Can't find desired loss function/optimizer")
            exit(0)

        # Load mapping

        try:
            fname = './lstm_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)

            self.weights = np.array([x[1] for x in self.phone_to_id.values()])

            assert len(self.phone_to_id) == config['num_phones'] + 1  # 1 for pad token

        except:
            print("Can't find phone mapping")
            exit(0)

    def init_hidden(self):

        hidden = next(self.parameters()).data.new(self.num_layers, self.batch_size, self.hidden_dim)
        cell = next(self.parameters()).data.new(self.num_layers, self.batch_size, self.hidden_dim)

        if self.config_file['cuda'] and torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()

        return (hidden, cell)

    def forward(self, x, X_lengths):

        batch_size, seq_len, _ = x.size()
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)
        # now run through LSTM
        X, _ = self.lstm(X)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.hidden2phone(X)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, max(X_lengths), self.num_phones + 1)
        return X

    def calculate_loss(self, outputs, labels, lens):

        cur_len_max = outputs.shape[1]
        labels = labels[:, :cur_len_max]
        flat_labels = labels.contiguous().view(-1)
        flat_pred = outputs.contiguous().view(-1, self.num_phones + 1)

        return self.loss_func(flat_pred, flat_labels)


class GRU(generic_model):

    def __init__(self, config, weights=None):

        super(GRU, self).__init__(config)

        feat_dim, hidden_dim, num_phones, num_layers = config['feat_dim'], config['hidden_dim'], config['num_phones'], \
                                                       config['num_layers']

        num_layers = 3
        linear_hidden = 256

        if config['bidirectional']:
            self.lstm = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.3,
                               bidirectional=True, batch_first=True)
            self.hidden2hidden = nn.Linear(hidden_dim * 2, linear_hidden)  # 1 for pad token, 2 for bididrectional
        else:
            self.lstm = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.3,
                               bidirectional=False, batch_first=True)
            self.hidden2hidden = nn.Linear(hidden_dim, linear_hidden)  # for pad token

        self.hidden2phone = nn.Linear(linear_hidden, num_phones + 1)

        self.num_phones = num_phones
        self.hidden_dim, self.num_layers = hidden_dim, num_layers

        loss, optimizer = config['train']['loss_func'], config['train']['optim']

        if loss == 'CEL':

            if config['weighted_loss'] and weights is not None:
                self.loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weights)).float())
                print("Using Weighted CEL")
            else:
                weights = np.append(np.ones((self.num_phones)) / self.num_phones, np.zeros((1)))
                self.loss_func = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float())
                print("Using CEL")

            loss_found = True

        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=config['train']['lr'], momentum=0.9)
            optim_found = True
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=config['train']['lr'])
            optim_found = True

        if loss_found == False or optim_found == False:
            print("Can't find desired loss function/optimizer")
            exit(0)
            """
        # Load mapping
        try:
            fname = config['dir']['dataset'] + 'lstm_mapping.json'
            with open(fname, 'r') as f:
                self.phone_to_id = json.load(f)

            self.weights = np.array([x[1] for x in self.phone_to_id.values()])

            assert len(self.phone_to_id) == config['num_phones'] + 1  # 1 for pad token

        except:
            print("Can't find phone mapping")
            exit(0)
"""
    def init_hidden(self):

        hidden = next(self.parameters()).data.new(self.num_layers, self.batch_size, self.hidden_dim)
        cell = next(self.parameters()).data.new(self.num_layers, self.batch_size, self.hidden_dim)

        if self.config_file['cuda'] and torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()

        return (hidden, cell)

    def forward(self, x, X_lengths):

        batch_size, seq_len, _ = x.size()
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)
        # now run through LSTM
        X, _ = self.lstm(X)
        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])
        # run through actual linear layer
        X = self.hidden2hidden(X)
        X = self.hidden2phone(X)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size, max(X_lengths), self.num_phones + 1)
        return X

    def calculate_loss(self, outputs, labels, lens):

        cur_len_max = outputs.shape[1]
        labels = labels[:, :cur_len_max]
        flat_labels = labels.contiguous().view(-1)
        flat_pred = outputs.contiguous().view(-1, self.num_phones + 1)

        return self.loss_func(flat_pred, flat_labels)
