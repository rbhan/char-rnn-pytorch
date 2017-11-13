import torch.nn as nn
from torch.autograd import Variable


class CharRNNModel(nn.Module):
    def __init__(self, config):
        super(CharRNNModel, self).__init__()

        # Parameters
        self.num_layers = config.num_layers
        self.rnn_size = config.rnn_size
        self.rnn_model = config.rnn_model

        # Layers (containing weights)
        self.embedding = nn.Embedding(
            config.vocab_size, self.rnn_size,
        )

        if self.rnn_model == "LSTM":
            rnn_cell = nn.LSTM
        elif self.rnn_model == "GRU":
            rnn_cell = nn.GRU
        else:
            raise KeyError()

        self.rnn = rnn_cell(
            self.rnn_size,
            self.rnn_size,
            self.num_layers,
            batch_first=True,
            dropout=(1 - config.keep_prob),
        )
        self.fc1 = nn.Linear(self.rnn_size, config.vocab_size)

    def forward(self, x, hidden):
        """
        Inputs come in with dimensions (from data loaders):
            [batch_size, seq_length]
        """

        # Embed
        embedded = self.embedding(x, hidden)

        # Push through RNN
        lstm_out, hidden = self.rnn(embedded)

        # Apply Linear layer
        output = self.fc1(lstm_out)

        return (
            output,
            hidden,
        )

    def init_hidden(self, batch_size):
        """Initialize hidden weights"""
        # LSTMs require two sets of hidden states
        weight = next(self.parameters()).data

        h = Variable(
            weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
        )
        if self.rnn_model == "LSTM":
            c = Variable(
                weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
            )
            return h, c
        elif self.rnn_model == "GRU":
            return h
        else:
            raise KeyError(self.rnn_model)
