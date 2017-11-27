import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, 
                 input_size=32, 
                 vocab_size=10, 
                 emb_size=128,
                 inp_hidden=200, 
                 inp_layers=1, 
                 rnn_hidden=128,
                 rnn_layers=1,
                 out_hidden=128,
                 out_layers=1,
                 use_cuda=False):
        super().__init__()   
        
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.inp_hidden = inp_hidden
        self.inp_layers = inp_layers
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.out_hidden = out_hidden
        self.out_layers = out_layers

        self.X = None

        layers = [nn.Linear(input_size, inp_hidden), nn.ReLU(True)]
        for _ in range(inp_layers - 1):
            layers.extend([nn.Linear(inp_hidden, inp_hidden), nn.ReLU(True)])
        self.inp = nn.Sequential(*layers)
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, rnn_hidden, batch_first=True, num_layers=rnn_layers)
        layers = [nn.Linear(inp_hidden + rnn_hidden, out_hidden), nn.ReLU(True)]
        for _ in range(out_layers - 1):
            layers.extend([nn.Linear(out_hidden, out_hidden), nn.ReLU(True)])
        layers.append(nn.Linear(out_hidden, vocab_size))
        self.out_token  = nn.Sequential(*layers)
        self.use_cuda = use_cuda
    
    def given(self, X):
        self.X = X

    def forward(self, inp):
        x = self.emb(inp)
        o, _ = self.rnn(x)
        o = o.contiguous()
        X = self.inp(self.X)
        X = X.view(X.size(0), 1, X.size(1))
        X = X.repeat(1, o.size(1), 1)
        o = torch.cat((o, X), 2)
        #o = X
        o = o.view(o.size(0) * o.size(1), o.size(2))
        o = self.out_token(o)
        return o

    def next_token(self, inp, state):
        if self.use_cuda:
            inp = inp.cuda()
        x = self.emb(inp)
        _, state = self.rnn(x, state)
        #h, c = state #LSTM
        h = state # GRU
        
        h = h[-1] # last layer
        X = self.inp(self.X)
        X = X.repeat(h.size(0), 1)
        h = torch.cat((h, X), 1)
        #h = X
        o = self.out_token(h)
        return o, state


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=10, emb_size=128, hidden_size=128, num_layers=1, nb_features=1, use_cuda=False):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.out_token  = nn.Linear(hidden_size, vocab_size)
        self.out_value = nn.Linear(hidden_size, nb_features)
    
    def forward(self, inp):
        x = self.emb(inp)
        o, _ = self.rnn(x)
        o = o.contiguous()
        o = o.view(o.size(0) * o.size(1), o.size(2))
        o = self.out_token(o)
        return o

    def next_token(self, inp, state):
        if self.use_cuda:
            inp = inp.cuda()
        x = self.emb(inp)
        _, state = self.rnn(x, state)
        h = state
        h = h[-1] # last layer
        o = self.out_token(h)
        return o, state
    
    def next_value(self, inp, state):
        if self.use_cuda:
            inp = inp.cuda()
        x = self.emb(inp)
        _, state = self.rnn(x, state)
        h= state
        h = h[-1] # last layer
        o = self.out_value(h)
        return o, state

