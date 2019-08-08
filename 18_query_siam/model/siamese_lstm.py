import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            batch_size,
            hidden_size,
            num_layers,
            bidir,
            dropout,
            activation
    ):
        super(LSTMEncoder, self).__init__()

        self.vocab_size  = vocab_size
        self.embed_size  = embed_size
        self.batch_size  = batch_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.bidir = bidir
        if self.bidir:
            self.direction = 2
        else: self.direction = 1

        self.activation = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "elu": nn.functional.elu,
            "none": lambda x: x,
        }[activation]

        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            dropout=self.dropout, num_layers=self.num_layers,
                            bidirectional=self.bidir, batch_first=True)


    def initHiddenCell(self, batch_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, batch_size, self.hidden_size))

        return rand_hidden.to(device), rand_cell.to(device)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        return output, hidden, cell


class Siamese_LSTM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            batch_size,
            hidden_size,
            num_layers,
            bidir,
            dropout,
            activation,
            pad_ind=0,
            add_feat_size=0
    ):

        super(Siamese_LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.bidir = bidir
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=pad_ind)

        self.encoder = LSTMEncoder(vocab_size, embed_size, batch_size, hidden_size,
                                   num_layers, bidir, dropout, activation)


        #self.fc_dim = fc_dim   # args 추가

        self.input_dim = 3 * self.encoder.direction * self.encoder.hidden_size + add_feat_size

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(self.input_dim / 2), 1)
        )

    def forward(self, sent1, sent2, add_feat, len1, len2):
        s1 = self.embedding(sent1)
        s2 = self.embedding(sent2)

        batch_size = sent1.size(0)

        # init hidden, cell
        h1, c1 = self.encoder.initHiddenCell(batch_size)
        h2, c2 = self.encoder.initHiddenCell(batch_size)

        v1, h1_n, c1 = self.encoder(s1, h1, c1)
        v2, h2_n, c2 = self.encoder(s2, h2, c2)

        num_layers = self.num_layers
        num_directions = 2 if self.bidir else 1
        hidden_size = self.hidden_size
        first_v1 = h1_n.view(num_layers, num_directions, batch_size, hidden_size)[num_layers - 1, 0, :, :]
        first_v2 = h2_n.view(num_layers, num_directions, batch_size, hidden_size)[num_layers - 1, 0, :, :]

        last_v1 = h1_n.view(num_layers, num_directions, batch_size, hidden_size)[num_layers - 1, -1, :, :]
        last_v2 = h2_n.view(num_layers, num_directions, batch_size, hidden_size)[num_layers - 1, -1, :, :]

        # utilize these two encoded vectors
        features = torch.cat((last_v1, last_v2, torch.abs(last_v1 - last_v2), first_v1, first_v2,
                        torch.abs(first_v1 - first_v2), add_feat), 1)

        output = self.classifier(features)
        output = torch.sum(output, dim=1).squeeze()

        return output
