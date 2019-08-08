import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.autograd import Variable


class CNN_Text(nn.Module):
    def __init__(self, config, vocab_size, n_label):
        super(CNN_Text, self).__init__()
        
        V = vocab_size
        D = config.embed_dim
        C = n_label
        Ci = 1
        Co = 128
        Ks = [2, 3, 4, 5]

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logits = self.fc1(x)  # (N, C)

        outputs = {
            "logits": logits,
            "predicted_intents": torch.topk(logits, 1)[1],
        }
        return outputs
