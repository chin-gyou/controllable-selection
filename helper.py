import math
import time
from config import *
import sentencepiece as spm
#from bpemb import BPEmb

def as_hours(s):
    h = math.floor(s / 3600)
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d:%d' % (h, m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_hours(s), as_hours(rs))

def find_trigram(beam, cind):
    in1, in2, bsize = [], [], beam.size(0)
    bigram = beam[:, cind-2:cind]#bsize*2
    for i in range(cind-2):
        mask = (torch.sum(bigram == beam[:, i:i+2], 1) == 2).float().nonzero() #bsize
        for j in mask:
            in1.append(int(j))
            in2.append(int(beam[int(j), i+2]))
    return (in1, in2)

def softmax_mask(w,m):
    max_value, _ = w.max(0, True)
    score = torch.exp(w-max_value)*m
    return score/(score.sum(0,True) + 1e-10)

def cleanid(oid):
    o = oid.replace('29946 1 18260 1', '1')
    #o = oid.replace('24913 1 4993 1', '1')
    return ' '.join(o.split())

def cleantxt(otxt):
    o = otxt.replace("\u2581 < unk >", "unk")
    return ' '.join(o.split())

def cleanidfile(fname):
    ids = open(fname, 'r', encoding='utf8').read().strip().split('\n')
    r = [cleanid(i)+'\n' for i in ids]
    with open(fname, 'w', encoding='utf8') as f:
        f.writelines(r)

def cleantxtfile(fname):
    ids = open(fname, 'r', encoding='utf8').read().strip().split('\n')
    r = [cleantxt(i)+'\n' for i in ids]
    with open(fname, 'w', encoding='utf8') as f:
        f.writelines(r)

#load vocabulary from extra file, return a dict with {word: index}
def load_vocab(vocab_file):
    vocab = {'PAD': PAD}
    texts = open(vocab_file, 'r', encoding='utf8').read().strip().split('\n')
    texts = [text.strip().split() for text in texts]
    vocab.update({l[0]: i+1 for i,l in enumerate(texts)})
    return vocab

"""
replace UNK token with the most focused table value word
v_file: table value text file
rfile: most focused position in the table value
of1: original decoded outputs
of2: new outputs after replacing
"""
def copy_replace(vf, rf, of1, of2):
    values = open(vf, 'r', encoding='utf8').read().strip().split('\n')    
    values = [s.strip().split(' ') for s in values]
    o1s = open(of1, 'r', encoding='utf8').read().strip().split('\n') 
    o1s = [s.strip().split(' ') for s in o1s]
    rs = open(rf, 'r', encoding='utf8').read().strip().split('\n')
    rs = [list(map(int, s.strip().split(' '))) for s in rs]
    results = []
    for v,o,r in zip(values, o1s, rs):
        n_o = [v[r[i]] if o[i] == 'UNK' and r[i]<len(v) else o[i] for i in range(len(o))]         
        results.append(' '.join(n_o)+'\n')
    with open(of2, 'w', encoding='utf8') as f:
        f.writelines(results)

def id2txt(ifile, ofile, wdict):
    ids = open(ifile, encoding='utf-8').readlines()
    wlines = []
    for line in ids:
        ws = line.strip().split()
        txts = [wdict[int(w)].strip('\u2581') for w in ws]
        wlines.append(' '.join(txts) + '\n')
    with open(ofile, 'w') as f:
        f.writelines(wlines)

def txt2id(ifile, ofile, wdict):
    lines = open(ifile, encoding='utf-8').readlines()
    wlines = []
    for line in lines:
        ws = line.strip().split() + ['END']
        ids = [str(wdict[w]) for w in ws]
        wlines.append(' '.join(ids) + '\n')
    with open(ofile, 'w') as f:
        f.writelines(wlines)

def txt2ids(ifile, ofile, model):
    sp = spm.SentencePieceProcessor()
    sp.load(model)
    ls = open(ifile, encoding='utf-8').readlines()
    #index 0 for padding, so id + 1
    ids = [' '.join([str(1+t) for t in sp.EncodeAsIds(l.strip())])+'\n' for l in ls]
    #ids = [' '.join([str(1+t) for t in model.encode_ids(l.strip())])+'\n' for l in ls]
    with open(ofile,'w', encoding='utf-8') as f:
        f.writelines(ids)

def segtxt(ifile, ofile, model):
    sp = spm.SentencePieceProcessor()
    sp.load(model)
    ls = open(ifile, encoding='utf-8').readlines()
    seg = [' '.join(sp.EncodeAsPieces(l))+'\n' for l in ls]
    with open(ofile,'w', encoding='utf-8') as f:
        f.writelines(seg)


def add_end(ifile):
    ls = open(ifile, encoding='utf-8').readlines()
    ls = [l.strip()+' 102\n' for l in ls]
    with open(ifile, 'w', encoding='utf-8') as f:
        f.writelines(ls)

def bp2words(ifile, ofile, model):
    lines = open(ifile, encoding = 'utf-8').readlines()
    sp = spm.SentencePieceProcessor()
    sp.load(model)
    wout = []
    for l in lines:
        d = sp.DecodePieces(l.strip().split(' '))
        wout.append(d+'\n')
    with open(ofile, 'w', encoding = 'utf-8') as f:
        f.writelines(wout)

#change matrix tensor to the format of key: list of sentences
def tensor2sent(t):
    l = t.tolist()
    keysents = {}
    for i, sent in enumerate(l):
        keysents[i] = [' '.join([str(w) for w in sent])]
    return keysents

if __name__ == '__main__':
    ins = ['../cnn-dailymail/train/article.txt', '../cnn-dailymail/train/title.txt', '../cnn-dailymail/valid/article.txt', '../cnn-dailymail/valid/title.txt', '../cnn-dailymail/test/article.txt', '../cnn-dailymail/test/title.txt']
    tns = ['../cnn-dailymail/train/article.n30k.txt', '../cnn-dailymail/train/title.n30k.txt', '../cnn-dailymail/valid/article.n30k.txt', '../cnn-dailymail/valid/title.n30k.txt', '../cnn-dailymail/test/article.n30k.txt', '../cnn-dailymail/test/title.n30k.txt']
    ids = ['../cnn-dailymail/train/article.n30k.id', '../cnn-dailymail/train/title.n30k.id', '../cnn-dailymail/valid/article.n30k.id', '../cnn-dailymail/valid/title.n30k.id', '../cnn-dailymail/test/article.n30k.id', '../cnn-dailymail/test/title.n30k.id']
    
    #vocab = load_vocab('../cnn-dailymail/nbpe30k.vocab')
    #vocab = {v:k for k,v in vocab.items()}
    #for txt, inf, idf in zip(ins, tns,ids):
    #    print(txt,inf)
    #    id2txt(idf, inf, vocab)
    #    cleanidfile(idf)
        #if txt == '../cnn-dailymail/train/article.txt':
        #    continue
    #    txt2ids(txt, idf, '../cnn-dailymail/bpe30k.model')
    #    segtxt(txt, inf, '../cnn-dailymail/bpe30k.model')
    #tids = ['../cnn-dailymail/train/title.30k.id', '../cnn-dailymail/valid/title.30k.id', '../cnn-dailymail/test/title.30k.id']
    #for t in tids:
    #    add_end(t)
#    bp2words('ooutputs', 'bpoutputs','../data/bpe30k.model')
    #txt2ids('../gigaword/DUC2004/article.txt', '../gigaword/DUC2004/article.n30k.id', '../gigaword/nbpe30k.model')
    #txt2ids('../gigaword/DUC2004/title0.txt', '../gigaword/DUC2004/title.n30k.id', '../gigaword/nbpe30k.model')
    #txt2ids('../cnn-dailymail/gigatest/title.txt', '../cnn-dailymail/gigatest/title.25k.id', BPEmb(lang="en", dim=100, vs = 25000))
    #cleanidfile('../cnn-dailymail/gigatest/article.n30k.id') 
    #cleanidfile('../cnn-dailymail/gigatest/title.n30k.id') 
    #cleantxtfile('../data/test/article.30k.txt') 
    segtxt('./dataset/gigaword/gigatest/title.txt', './dataset/gigaword/gigatest/title.30k.txt', './dataset/gigaword/bpe30k.model')
    #add_end('decoding/xsum/test2.id')
#     add_end('../data/valid/title.16k.txt')
#     add_end('../data/test/title.16k.txt')
#    modes = ['train','valid','test']
#    types = ['article','title']
#    for m in modes:
#        for t in types:
#            segtxt('../data/'+m+'/'+t+'.txt','../data/'+m+'/'+t+'.16k.txt', '../data/bpe16k.model')
