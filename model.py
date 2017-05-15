import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CharRNNModel(nn.Module):
    def __init__(self, config):
        super(CharRNNModel, self).__init__()

        # Parameters
        self.num_layers = config.num_layers
        self.rnn_size = config.rnn_size

        # Layers (containing weights)
        self.embedding = nn.Embedding(
            config.vocab_size, self.rnn_size,
        )
        self.rnn = nn.LSTM(
            self.rnn_size,
            self.rnn_size,
            self.num_layers,
            dropout=(1-config.input_keep_prob),
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

        # Apply Log-Softmax. We temporarily switch the embedding dimension
        # first:
        #    [num_chars, seq_length, batch_size]
        # This is because Softmax normalizes over the possible characters,
        # so it needs to be the first dimension.
        log_softmax_output = \
            F.log_softmax(output.permute(2, 1, 0)).permute(2, 1, 0)

        return (
            log_softmax_output.permute(1, 0, 2),
            hidden,
        )

    def init_hidden(self, batch_size):
        """Initialize hidden weights"""
        # LSTMs require two sets of hidden states
        weight = next(self.parameters()).data
        return (
            Variable(
                weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
            ),
            Variable(
                weight.new(self.num_layers, batch_size, self.rnn_size).zero_()
            )
        )
