import os
from functools import partial
import numpy as np
from clize import run
from collections import OrderedDict
from collections import defaultdict
import pandas as pd
from scipy.stats import spearmanr

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

"""
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.tree import *
"""

from sklearn.utils import shuffle

from grammaropt.grammar import as_str
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.grammar import Vectorizer
from grammaropt.grammar import NULL_SYMBOL
from grammaropt.grammar import build_grammar

from model import Model
from model import SimpleModel

def to_str(v):
    try:
        v = float(v)
        return str(v)
    except Exception:
        pass

    try:
        v = int(v)
        return str(v)
    except Exception:
        pass
    
    if v in ("True", "False"):
        return v
    
    return '\\"{}\\"'.format(v)


def _load_meta():
    return pd.read_csv('pmlb_metafeatures.csv')

def _load_runs():
    return pd.read_csv('sklearn-benchmark5-data-edited.tsv', names=['dataset', 'algo', 'hypers', 'acc1', 'acc2', 'acc3'], delim_whitespace=True)


# for reproducibility, otherwise, without ordering, its lake we have a different random seeed
# each run

def sort_func(k):
    return "0" * (10-len(k)) + k


def write_grammar():
    df = _load_runs()
    rule = defaultdict(lambda: OrderedDict())
    namespace = dict()
    slug = {}
    for algo, rows in df.groupby('algo'):
        for hyper in rows['hypers']:
            if hyper[-1] == ',':
                hyper = hyper[0:-1]
            for param in hyper.split(','):
                try:
                    key, value = param.split('=')
                except Exception:
                    print(hyper)
                    continue
                raw_key = key
                if key in namespace and namespace[key] != algo:
                    key = key + '_' + algo.lower()
                namespace[key] = algo
                slug[(key, algo)] = raw_key
                if key in rule[algo]:
                    rule[algo][key].add(value)
                else:
                    rule[algo][key] = set([value])
    grammar = []
    names = " / ".join(rule.keys())
    grammar.append('estimator = {}'.format(names))
    for algo, params in rule.items():
        param_names = list(params.keys())
        param_names = ['"{}" eq {}'.format(slug[(name, algo)], name) for name in param_names]
        param_names = ' cm '.join(param_names)
        grammar.append('{} = "{}" op {} cp'.format(algo, algo, param_names))
        for param, values in params.items():
            values = list(values)
            values = ['"{}"'.format(to_str(v)) for v in values]
            values = sorted(values, key=sort_func)
            values = ' / '.join(values)
            grammar.append('{} = {}'.format(param, values))
    grammar.append('eq = "="')
    grammar.append('op = "("')
    grammar.append('cp = ")"')
    grammar.append('cm = ","')
    grammar = '\n'.join(grammar)
    with open('grammar', 'w') as fd:
        fd.write(grammar)


def build_meta_dataset():
    meta = _load_meta()
    meta = meta.dropna(axis=1)
    runs = _load_runs()
    runs['acc'] = (runs['acc1'] + runs['acc2'] + runs['acc3']) / 3.0
    X = []
    Y = []
    H = []
    A = []
    D = []
    for dataset, rows in runs.groupby('dataset'):
        features = meta[meta['dataset'] == dataset].drop('dataset', axis=1).values[0]
        a, b  = rows['acc'].min(), rows['acc'].max()
        c = rows['acc']
        perfs = (c - a) / (b - a + (b==a))
        hypers = rows['hypers']
        algos = rows['algo']
        for perf, hyper, algo in zip(perfs, hypers, algos):
            if '=' not in hyper:
                continue
            if np.isnan(perf):
                continue
            if hyper[-1] == ',':
                hyper = hyper[-1]
            h = hyper.split(',')
            vals = []
            for hi in h:
                if '=' in hi:
                    k, v = hi.split('=')
                    v = to_str(v)
                    v = v.replace("\\", "")
                    vals.append('{}={}'.format(k, v))
            if len(vals) == 0:
                continue
            h = ','.join(vals)
            H.append(h)
            A.append(str(algo))
            D.append(dataset)
            X.append(features)
            Y.append(perf)

    np.savez('meta_dataset.npz', X=X, y=Y, H=H, A=A, D=D)



def acc(pred, true_classes):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true_classes).float().mean()
    return acc



def weights_init(m, ih_std=0.08, hh_std=0.08):
    if isinstance(m, nn.LSTM):
        m.weight_ih_l0.data.normal_(0, ih_std)
        m.weight_hh_l0.data.normal_(0, hh_std)
    elif isinstance(m, nn.GRU):
        m.weight_ih_l0.data.normal_(0, ih_std)
        m.weight_hh_l0.data.normal_(0, hh_std)
    elif isinstance(m, nn.Conv2d):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


def train(*, lr=1e-2, resume=False):
    batch_size = 64
    emb_size = 64
    inp_layers = 1
    inp_hidden = 128
    rnn_layers = 1
    rnn_hidden = 64
    out_layers = 2
    out_hidden = 128

    use_cuda = True
    gamma = 0.99
    nb_epochs = 10000

    grammar = build_grammar(open('grammar').read())
    data = np.load('meta_dataset.npz')
    X = data['X']
    H = data['H']
    A = data['A']
    Y = data['y']
    D = data['D']
    X, H, A, Y, D = shuffle(X, H, A, Y, D, random_state=42)
    C = np.array(["{}({})".format(a, h) for h, a in zip(H, A)])
    score_of = dict(((c, d), y)  for d, c, y in zip(D, C, Y))
    #f = (D == 'yeast') * (A == 'SGDClassifier')
    f = Y == 1.0
    X = np.log(1 + X)
    X = X[f]
    H = H[f]
    A = A[f]
    Y = Y[f]
    D = D[f]
    C = C[f]
    
    #train = np.array([d in train_datasets for d in D])
    #test = np.array([d in test_datasets for d in D])
    train = D != 'yeast'
    test = D == 'yeast'

    Xtrain = X[train]
    Ytrain = Y[train]
    Ctrain = C[train]

    Xtest = X[test]
    Dtest = D[test]
    
    print(Xtrain.shape, Xtest.shape)
    
    nb = 100
    X = X[0:nb]
    H = H[0:nb]
    A = A[0:nb]
    Y = Y[0:nb]
    D = D[0:nb]

    mu, std = X.mean(axis=0, keepdims=True), (1e-7 + X.std(axis=0, keepdims=True))
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    print(Xtrain.min(), Xtrain.max(), Xtest.min(), Xtest.max())
    
    vect = Vectorizer(grammar)
    vect._init()
    if resume:
        model = torch.load('rnn.th')
    else:
        model = Model(
            input_size=len(X[0]), 
            vocab_size=len(vect.tok_to_id), 
            emb_size=emb_size,
            inp_hidden=inp_hidden, 
            inp_layers=inp_layers, 
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            out_hidden=out_hidden,
            out_layers=out_layers,
            use_cuda=use_cuda)
        model.apply(partial(weights_init, ih_std=0.08, hh_std=0.08))
    if use_cuda:
        model = model.cuda()

    #optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    #optim = torch.optim.RMSprop(model.parameters(), lr=lr)
    #scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=30, verbose=True)
    scheduler = StepLR(optim, step_size=100, gamma=0.1)
    rnn = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=grammar, rnn=rnn)
    avg_loss = 0.
    avg_precision = 0.
    stats = []
    nupdates = 0
    avg_rho = 0
    #crit = nn.CrossEntropyLoss()
    for epoch in range(nb_epochs):
        scheduler.step(avg_loss)
        for i in range(0, len(Xtrain), batch_size):
            x = Xtrain[i:i + batch_size]
            x = x.astype('float32')
            r = Ytrain[i:i + batch_size]
            r = r.astype('float32')
            c = Ctrain[i:i + batch_size]
            x = np.array(x)
            r = np.array(r)
            c = vect.transform(c)
            c = [[0] + ci for ci in c]
            c = np.array(c)

            inp = c[:, 0:-1]
            out = c[:, 1:] 
            out_size = out.shape
            out = out.flatten()
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            out = torch.from_numpy(out).long()
            out = Variable(out)
            
            r0 = r
            r = torch.from_numpy(r).float() 
            r = r.repeat(1, inp.size(1))
            r = r.view(-1, 1)
            r = Variable(r)

            x = torch.from_numpy(x)
            x = Variable(x)
            
            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()
                x = x.cuda()
                r = r.cuda()

            model.zero_grad()
            model.given(x)
            y = model(inp)
            loss = nn.functional.nll_loss(nn.functional.log_softmax(y), out)           
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), 2)
            optim.step()

            precision = acc(y, out)
            p = torch.log(nn.Softmax()(y).gather(1, out.view(-1, 1)))
            p = p.view(out_size)
            p = p.mean(1).view(-1)
            p = p.data.cpu().numpy()
            
            rho, _ = spearmanr(p, r0)
            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            avg_precision = avg_precision * gamma + precision.data[0] * (1 - gamma)
            avg_rho = avg_rho * gamma + rho * (1 - gamma)

            stats.append({'loss': loss.data[0], 'precision': precision.data[0], 'rho': rho})

            if nupdates % 10 == 0:
                pd.DataFrame(stats).to_csv(os.path.join('.', 'stats.csv'))
                print('Epoch : {:05d} [{:06d}/{:06d}] Avg loss : {:.6f} Avg Precision : {:.6f} Avg rho : {:.6f}'.format(epoch, i, len(X), avg_loss, avg_precision, avg_rho))
            if nupdates % 100 == 0:
                torch.save(model, 'rnn.th')
                for idx in range(min(len(Xtest), 100)):
                    x = Xtest[idx:idx+1]
                    x = x.astype('float32')
                    x = torch.from_numpy(x)
                    x = Variable(x)
                    if use_cuda:
                        x = x.cuda()
                    model.given(x)
                    wl.walk()
                    code = as_str(wl.terminals)
                    print(code, score_of.get((code, Dtest[idx])))
            nupdates += 1


def train_simple(*, lr=1e-4, resume=False):
    batch_size = 64
    emb_size = 128
    hidden_size = 64
    num_layers = 1
    use_cuda = True
    gamma = 0.99
    nb_epochs = 10000
    grammar = build_grammar(open('grammar').read())
    data = np.load('meta_dataset.npz')
    X = data['X']
    H = data['H']
    A = data['A']
    Y = data['y']
    D = data['D']
    X, H, A, Y, D = shuffle(X, H, A, Y, D, random_state=42)
    C = np.array(["{}({})".format(a, h) for h, a in zip(H, A)])
    score_of = dict(((c, d), y)  for d, c, y in zip(D, C, Y))

    train = (Y==1.0)
    Ytrain = Y[train]
    Ctrain = C[train]
    
    #nb = 10
    #Ytrain = Ytrain[0:nb]
    #Ctrain = Ctrain[0:nb]

    vect = Vectorizer(grammar)
    vect._init()
    if resume:
        model = torch.load('rnn-simple.th')
    else:
        model = SimpleModel(
            vocab_size=len(vect.tok_to_id), 
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_cuda=use_cuda)
        model.apply(partial(weights_init, ih_std=0.08, hh_std=0.08))
    if use_cuda:
        model = model.cuda()
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    #optim = torch.optim.RMSprop(model.parameters(), lr=lr)
    rnn = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=grammar, rnn=rnn)
    avg_loss = 0.
    avg_precision = 0.
    stats = []
    nupdates = 0
    avg_rho = 0
    for epoch in range(nb_epochs):
        #scheduler.step(avg_loss)
        for i in range(0, len(Ytrain), batch_size):
            r = Ytrain[i:i + batch_size]
            r = r.astype('float32')
            c = Ctrain[i:i + batch_size]
            r = np.array(r)
            c = vect.transform(c)
            c = [[0] + ci for ci in c]
            c = np.array(c)
            inp = c[:, 0:-1]
            out = c[:, 1:] 
            out_size = out.shape
            out = out.flatten()
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            out = torch.from_numpy(out).long()
            out = Variable(out)
            
            r0 = r
            r = torch.from_numpy(r).float() 
            r = r.repeat(1, inp.size(1))
            r = r.view(-1, 1)
            r = Variable(r)

            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()
                r = r.cuda()

            model.zero_grad()
            y = model(inp)
            loss = nn.functional.nll_loss(r * nn.functional.log_softmax(y), out)           
            loss.backward()
            optim.step()

            precision = acc(y, out)
            p = torch.log(nn.Softmax()(y).gather(1, out.view(-1, 1)))
            p = p.view(out_size)
            p = p.mean(1).view(-1)
            p = p.data.cpu().numpy()
            
            rho, _ = spearmanr(p, r0)
            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            avg_precision = avg_precision * gamma + precision.data[0] * (1 - gamma)
            avg_rho = avg_rho * gamma + rho * (1 - gamma)

            stats.append({'loss': loss.data[0], 'precision': precision.data[0], 'rho': rho})

            if nupdates % 10 == 0:
                pd.DataFrame(stats).to_csv(os.path.join('.', 'stats.csv'))
                print('Epoch : {:05d} [{:06d}/{:06d}] Avg loss : {:.6f} Avg Precision : {:.6f} Avg rho : {:.6f}'.format(epoch, i, len(Ytrain), avg_loss, avg_precision, avg_rho))
            if nupdates % 100 == 0:
                torch.save(model, 'rnn-simple.th')
                for i in range(10):
                    wl.walk()
                    code = as_str(wl.terminals)
                    print(code, score_of.get((code, 'yeast')))
            nupdates += 1


if __name__ == '__main__':
    run([write_grammar, build_meta_dataset, train, train_simple])
