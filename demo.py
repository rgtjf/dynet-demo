from collections import Counter, defaultdict
from itertools import count
import random
import scipy.stats as meas
import dynet as dy
import numpy as np

# format of files: each line is "id sent1 sent2 label score"
train_file="./data/SICK_train.txt"
dev_file="./data/SICK_test_annotated.txt"

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def fw2i(self, word):
        if word not in self.w2i:
            return self.w2i['_UNK_']
        else:
            return self.w2i[word]

    def size(self): return len(self.w2i.keys())

def read(fname):
    """
    Read a POS-tagged file where each line is of the form "word1/tag2 word2/tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    with file(fname) as fh:
        fh.readline()
        for line in fh:
            line = line.strip().split('\t')
            id, sa, sb, score, label = line
            sa = sa.split()
            sb = sb.split()
            score = (eval(score) - 1) / 4.0
            yield sa, sb, score


train=list(read(train_file))
dev=list(read(dev_file))

words=[]
chars=set()
wc=Counter()

for sa, sb, score in train:
    for w in sa:
        words.append(w)
        chars.update(w)
        wc[w] += 1

words.append("_UNK_")
chars.add("_UNK_")
chars.add("<*>")


vw = Vocab.from_corpus([words])
vc = Vocab.from_corpus([chars])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
nchars  = vc.size()

ntopics = 200

# DyNet Starts

model = dy.Model()
trainer = dy.AdamTrainer(model)
# trainer = dy.MomentumSGDTrainer(model)

pW = model.add_parameters((100, 100)) # biliearn attention

pW0 = model.add_parameters((100, 100)) # mlp attention
pW1 = model.add_parameters((100, 100))
pb  = model.add_parameters(100)



pWMem = model.add_parameters((128, 128))
Mem = model.add_lookup_parameters((ntopics, 128))


WORDS_LOOKUP = model.add_lookup_parameters((nwords, 128))
CHARS_LOOKUP = model.add_lookup_parameters((nchars, 20))


# word-level LSTMs
fwdRNN = dy.LSTMBuilder(1, 128, 50, model) # layers, in-dim, out-dim, model
bwdRNN = dy.LSTMBuilder(1, 128, 50, model)

# char-level LSTMs
cFwdRNN = dy.LSTMBuilder(1, 20, 64, model)
cBwdRNN = dy.LSTMBuilder(1, 20, 64, model)

def word_rep(w, cf_init, cb_init):
    """
    if word in embedding, word_rep
    else char-lstm
    """
    if wc[w] > 0:
        w_index = vw.w2i[w]
        return WORDS_LOOKUP[w_index]
    else:
        pad_char = vc.fw2i("<*>")
        char_ids = [pad_char] + [vc.fw2i(c) for c in w] + [pad_char]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

def build_graph(sa, sb):
    dy.renew_cg()

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    cf_init = cFwdRNN.initial_state()
    cb_init = cBwdRNN.initial_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs_sa = [ word_rep(w, cf_init, cb_init) for w in sa]
    # wembs_sa = [ memory_attention(wemb) for wemb in wembs_sa]


    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs_sa)
    bw_exps = b_init.transduce(reversed(wembs_sa))


    # biLSTM states
    bi_exps_sa = [ dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps)) ]

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs_sb = [word_rep(w, cf_init, cb_init) for w in sb]
    # wembs_sb = [memory_attention(wemb) for wemb in wembs_sb]

    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs_sb)
    bw_exps = b_init.transduce(reversed(wembs_sb))

    # biLSTM states
    bi_exps_sb = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

    # rep_sa = bi_exps_sa[-1]
    # rep_sb = bi_exps_sb[-1]

    # rep_sa = bilinear_attention(bi_exps_sa, bi_exps_sb[-1])
    # rep_sb = bilinear_attention(bi_exps_sb, bi_exps_sa[-1])

    rep_sa = mlp_attention(bi_exps_sa, bi_exps_sb[-1])
    rep_sb = mlp_attention(bi_exps_sb, bi_exps_sa[-1])

    out_sc_exp = dy.exp(-dy.l1_distance(rep_sa, rep_sb))

    return out_sc_exp


def bilinear_attention(answers, question):
    """
    exps: sent_len, hdim
    exp: hdim
    """
    W = dy.parameter(pW) # hdim * hdim
    question = W * question # hdim
    probs = [ dy.dot_product(answer, question) for answer in answers ]
    probs = dy.softmax(dy.concatenate(probs))
    output_exp = dy.esum([answer*prob for answer, prob in zip(answers, probs)])
    return output_exp
# 77.49


def mlp_attention(answers, question):
    W0 = dy.parameter(pW0)
    W1 = dy.parameter(pW1)
    b = dy.parameter(pb)
    M = [  dy.tanh(W0 * answer + W1 * question) for answer in answers ]
    probs = [ dy.dot_product(m, b) for m in M ]
    probs = dy.softmax(dy.concatenate(probs))
    output_exp = dy.esum([answer * prob for answer, prob in zip(answers, probs)])
    return output_exp
# 77.90



def memory_attention(word):
    WMem = dy.parameter(pWMem)
    word = WMem * word  # hdim
    probs = [dy.dot_product(word, dy.lookup(Mem, idx)) for idx in range(ntopics)]
    probs = dy.softmax(dy.concatenate(probs))
    output_exp = dy.esum([ dy.lookup(Mem, idx) * probs[idx] for idx in range(ntopics)])
    return output_exp



def loss(sa, sb, ref_sc):
    out_sc_exp = build_graph(sa, sb)
    cost = dy.square(ref_sc - out_sc_exp)
    return cost

def predict(sa, sb):
    out_sc_exp = build_graph(sa, sb)
    out_sc = out_sc_exp.scalar_value()
    return out_sc

def eval(dev):
    preds, golds = [], []
    for i, test_example in enumerate(dev):
        sa, sb, ref_sc = test_example
        out_sc = predict(sa, sb)
        preds.append(out_sc)
        golds.append(ref_sc)
    pearsonr = meas.pearsonr(preds, golds)[0] * 100
    return pearsonr

cum_loss = 0
best_pearsonr = 0.0
for ITER in range(50):
    random.shuffle(train)
    for i, train_example in enumerate(train):
        sa, sb, ref_sc = train_example
        if i > 0 and i % 500 == 0:   # print status
            trainer.status()
            print(cum_loss)
            cum_loss = 0
        if i % 1000 == 0 or i == len(train)-1: # eval on dev
            pearsonr = eval(train)
            print("Train accuracy: %.2f %%" % pearsonr)
            pearsonr = eval(dev)
            if best_pearsonr < pearsonr:
                best_pearsonr = pearsonr
            print('Dev accuracy: %.2f %%, Best accuracy: %.2f %%' % (pearsonr, best_pearsonr))

        # train on sent
        loss_exp =  loss(sa, sb, ref_sc)
        cum_loss += loss_exp.scalar_value()
        loss_exp.backward()
        trainer.update()
    print "epoch %r finished" % ITER
    trainer.update_epoch(1.0)

