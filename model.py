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
            dropout=(1 - config.keep_prob),
        )
        self.fc1 = nn.Linear(self.rnn_size, config.vocab_size)

    def forward(self, x, hidden):
        """
        Inputs come in with dimensions (from data loaders):
            [batch_size, seq_length, embedding_dim]
        However, PyTorch RNN cells work with the following convention:
            [seq_length, batch_size, embedding_dim]

        Because of that, as well as other PyTorch-specific dimension
        requirements, I'll do some dimension-hacking within the model.
        Would be nice to find better solutions for these
        """

        # Switch to [seq_length, batch_size, embedding_dim]
        x = x.permute(1, 0)

        # Embed
        embedded = self.embedding(x)

        # Push through RNN
        lstm_out, hidden = self.rnn(embedded)

        # Apply Fully-Connected layer. We temporarily compress to dim:
        #    [seq_length * batch_size, embedding_dim]
        # to apply FC1
        output = self.fc1(
            lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        ).view(lstm_out.size(0), lstm_out.size(1), self.fc1.out_features)

        # Switch to back to [batch_size, seq_length, embedding_dim]
        return (
            output.permute(1, 0, 2),
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
